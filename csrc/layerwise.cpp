#include "layerwise.h"
#include <atomic>
#include <cstdio>
#include <fcntl.h>
#include <stdexcept>
#include <sys/eventfd.h>
#include <unistd.h>
#include <nvtx3/nvToolsExt.h>

namespace flexkv {

// ===== Event polling notification =====

void LayerwiseTransferGroup::notify_layer_batch(int start_layer,
                                                 int layers_this_batch) {
  if (!enable_eventfd_ || layer_eventfds_.empty()) return;

  int offset = current_counter_id_ * tp_size_ * num_layers_;
  int *eventfds_base = layer_eventfds_.data() + offset;

  for (int layer = start_layer;
       layer < start_layer + layers_this_batch; ++layer) {
    for (int tp_rank = 0; tp_rank < tp_size_; ++tp_rank) {
      int fd = eventfds_base[tp_rank * num_layers_ + layer];
      if (fd >= 0) {
        uint64_t val = 2;
        ssize_t ret = write(fd, &val, sizeof(val));
        (void)ret;
      }
    }
  }
}

void LayerwiseTransferGroup::event_polling_loop() {
  while (!poll_stop_.load(std::memory_order_acquire)) {
    int next = poll_next_batch_.load(std::memory_order_acquire);
    if (next >= (int)poll_batches_.size()) {
      break;
    }

    PollBatchInfo &batch = poll_batches_[next];
    if (batch.notified) {
      poll_next_batch_.fetch_add(1, std::memory_order_acq_rel);
      continue;
    }

    bool all_done = true;
    for (int g = 0; g < num_gpus_ && all_done; ++g) {
      cudaSetDevice(gpu_device_ids_[g]);
      cudaError_t err = cudaEventQuery(batch.per_gpu_events[g]);
      if (err == cudaErrorNotReady) {
        all_done = false;
      } else if (err != cudaSuccess) {
        all_done = false;
      }
    }

    if (all_done) {
      batch.notified = true;
      notify_layer_batch(batch.start_layer, batch.layers_this_batch);
      poll_next_batch_.fetch_add(1, std::memory_order_acq_rel);
    } else {
      std::this_thread::yield();
    }
  }
}

struct LayerCallbackData {
  int start_layer;
  int layers_this_batch;
  int expected_count;
  std::atomic<int> *counter;
  // Eventfd info for notification
  bool enable_eventfd;
  int tp_size;
  int num_layers;
  int *layer_eventfds;  // Pointer to eventfds array for current counter set
  // NVTX range id for CPU->GPU transfer
  nvtxRangeId_t *current_range_id_ptr;  // Pointer to current layer's range ID
  bool is_last_batch;  // Whether this is the last batch
  char next_range_name[64];  // Name for next layer's range (if not last batch)
  nvtxRangeId_t *next_range_id_ptr;  // Pointer to next layer's range ID storage
};

static void CUDART_CB layer_done_host_callback(void *userData) {
  LayerCallbackData *data = static_cast<LayerCallbackData *>(userData);
  int completed = data->counter->fetch_add(1) + 1;
  if (completed == data->expected_count) {
    // Notify via eventfd when all GPUs complete this layer batch
    if (data->enable_eventfd && data->layer_eventfds != nullptr) {
      // Signal each tp_rank's eventfd for completed layers
      for (int layer = data->start_layer; 
           layer < data->start_layer + data->layers_this_batch; ++layer) {
        for (int tp_rank = 0; tp_rank < data->tp_size; ++tp_rank) {
          int fd = data->layer_eventfds[tp_rank * data->num_layers + layer];
          if (fd >= 0) {
            // Write 2 to support both get_key_buffer and get_value_buffer waits
            uint64_t val = 2;
            ssize_t ret = write(fd, &val, sizeof(val));
          }
        }
      }
    }
    // End current NVTX range when all GPUs complete
    if (data->current_range_id_ptr != nullptr && *data->current_range_id_ptr != 0) {
      nvtxRangeEnd(*data->current_range_id_ptr);
    }
    // Start next layer's NVTX range (so it begins right after current layer ends)
    if (!data->is_last_batch && data->next_range_id_ptr != nullptr) {
      *data->next_range_id_ptr = nvtxRangeStartA(data->next_range_name);
    }
    delete data->counter;
  }
  delete data;
}

LayerwiseTransferGroup::LayerwiseTransferGroup(
    int num_gpus, const std::vector<std::vector<torch::Tensor>> &gpu_blocks,
    torch::Tensor &cpu_blocks,
    std::map<int, std::vector<std::string>> &ssd_files,
    int num_layers, torch::Tensor &gpu_kv_strides_tensor,
    torch::Tensor &gpu_block_strides_tensor,
    torch::Tensor &gpu_layer_strides_tensor,
    torch::Tensor &gpu_chunk_sizes_tensor, int iouring_entries,
    int iouring_flags, torch::Tensor &layer_eventfds_tensor, int tp_size,
    const std::vector<std::vector<torch::Tensor>> &indexer_gpu_blocks,
    torch::Tensor indexer_cpu_blocks,
    torch::Tensor indexer_gpu_kv_strides_tensor,
    torch::Tensor indexer_gpu_block_strides_tensor,
    torch::Tensor indexer_gpu_layer_strides_tensor,
    torch::Tensor indexer_gpu_chunk_sizes_tensor,
    std::map<int, std::vector<std::string>> indexer_ssd_files) {

  has_multi_group_ = false;
  num_gpus_ = num_gpus;
  num_layers_ = num_layers;
  tp_size_ = tp_size;
  current_counter_id_ = 0;

  // Initialize eventfds
  enable_eventfd_ = (layer_eventfds_tensor.numel() > 0);
  if (enable_eventfd_) {
    // layer_eventfds_tensor layout: [num_counters, tp_size, num_layers]
    // Index formula: counter_id * tp_size * num_layers + tp_rank * num_layers + layer
    int total_fds = layer_eventfds_tensor.numel();
    num_counters_ = total_fds / (tp_size * num_layers);
    
    int32_t *fds_ptr = layer_eventfds_tensor.data_ptr<int32_t>();
    layer_eventfds_.assign(fds_ptr, fds_ptr + total_fds);
    
    printf("[LayerwiseTransferGroup] Initialized with eventfds: "
           "tp_size=%d, num_counters=%d, num_layers=%d, total_fds=%d\n",
           tp_size_, num_counters_, num_layers_, total_fds);
  } else {
    num_counters_ = 0;
    printf("[LayerwiseTransferGroup] Initialized without eventfds\n");
  }

  gpu_kv_strides_in_bytes_ = new int64_t[num_gpus];
  gpu_block_strides_in_bytes_ = new int64_t[num_gpus];
  gpu_layer_strides_in_bytes_ = new int64_t[num_gpus];
  gpu_chunk_sizes_in_bytes_ = new int64_t[num_gpus];

  int64_t *kv_strides_ptr = gpu_kv_strides_tensor.data_ptr<int64_t>();
  int64_t *block_strides_ptr = gpu_block_strides_tensor.data_ptr<int64_t>();
  int64_t *layer_strides_ptr = gpu_layer_strides_tensor.data_ptr<int64_t>();
  int64_t *chunk_sizes_ptr = gpu_chunk_sizes_tensor.data_ptr<int64_t>();

  for (int i = 0; i < num_gpus; i++) {
    gpu_kv_strides_in_bytes_[i] = kv_strides_ptr[i];
    gpu_block_strides_in_bytes_[i] = block_strides_ptr[i];
    gpu_chunk_sizes_in_bytes_[i] = chunk_sizes_ptr[i];
    gpu_layer_strides_in_bytes_[i] = layer_strides_ptr[i];
  }

  num_tensors_per_gpu_ = gpu_blocks[0].size();
  cudaMallocHost((void **)&gpu_blocks_,
                 num_gpus_ * num_tensors_per_gpu_ * sizeof(void *));
  for (int i = 0; i < num_gpus_; ++i) {
    for (int j = 0; j < num_tensors_per_gpu_; ++j) {
      gpu_blocks_[i * num_tensors_per_gpu_ + j] = gpu_blocks[i][j].data_ptr();
    }
  }

  if (num_tensors_per_gpu_ == 1) {
    backend_type_ = BackendType::TRTLLM;
  } else if (num_tensors_per_gpu_ == num_layers) {
    backend_type_ = BackendType::VLLM;
  } else if (num_tensors_per_gpu_ == num_layers * 2) {
    backend_type_ = BackendType::SGLANG;
  } else {
    throw std::runtime_error("Unsupported GPU block type: " +
                             std::to_string(num_tensors_per_gpu_));
  }

  gpu_tensor_handlers_.reserve(num_gpus_);
  for (int i = 0; i < num_gpus_; i++) {
    int64_t **gpu_blocks_ptr =
        reinterpret_cast<int64_t **>(gpu_blocks_ + i * num_tensors_per_gpu_);
    gpu_tensor_handlers_.emplace_back(
        backend_type_, gpu_blocks_ptr, num_layers, gpu_kv_strides_in_bytes_[i],
        gpu_block_strides_in_bytes_[i], gpu_layer_strides_in_bytes_[i]);
  }

  cpu_blocks_ = cpu_blocks.data_ptr();

  // Get GPU device IDs from tensors (like tp_transfer_thread_group.cpp)
  gpu_device_ids_.resize(num_gpus_);
  for (int i = 0; i < num_gpus_; ++i) {
    gpu_device_ids_[i] = gpu_blocks[i][0].device().index();
  }

  // Create CUDA streams for each GPU
  streams_.resize(num_gpus_);
  events_.resize(num_gpus_);
  
  // Get highest priority (lowest value)
  int leastPriority, greatestPriority;
  cudaDeviceGetStreamPriorityRange(&leastPriority, &greatestPriority);
  
  for (int i = 0; i < num_gpus_; i++) {
    cudaSetDevice(gpu_device_ids_[i]);
    cudaStreamCreateWithPriority(&streams_[i], cudaStreamNonBlocking, greatestPriority);
    cudaEventCreate(&events_[i]);
  }

  // Initialize SSD IO context if ssd_files is not empty
  enable_ssd_ = !ssd_files.empty();
  if (enable_ssd_) {
    ioctx_ = std::make_unique<SSDIOCTX>(ssd_files, ssd_files.size(),
                                        iouring_entries, iouring_flags);
  }

  // Initialize indexer fuse support
  enable_indexer_ = !indexer_gpu_blocks.empty();
  if (enable_indexer_) {
    indexer_num_tensors_per_gpu_ = indexer_gpu_blocks[0].size();
    cudaMallocHost((void **)&indexer_gpu_blocks_,
                   num_gpus_ * indexer_num_tensors_per_gpu_ * sizeof(void *));
    for (int i = 0; i < num_gpus_; ++i) {
      for (int j = 0; j < indexer_num_tensors_per_gpu_; ++j) {
        indexer_gpu_blocks_[i * indexer_num_tensors_per_gpu_ + j] =
            indexer_gpu_blocks[i][j].data_ptr();
      }
    }

    indexer_cpu_blocks_ = indexer_cpu_blocks.data_ptr();

    indexer_gpu_kv_strides_in_bytes_ = new int64_t[num_gpus];
    indexer_gpu_block_strides_in_bytes_ = new int64_t[num_gpus];
    indexer_gpu_layer_strides_in_bytes_ = new int64_t[num_gpus];
    indexer_gpu_chunk_sizes_in_bytes_ = new int64_t[num_gpus];

    int64_t *idx_kv_strides_ptr = indexer_gpu_kv_strides_tensor.data_ptr<int64_t>();
    int64_t *idx_block_strides_ptr = indexer_gpu_block_strides_tensor.data_ptr<int64_t>();
    int64_t *idx_layer_strides_ptr = indexer_gpu_layer_strides_tensor.data_ptr<int64_t>();
    int64_t *idx_chunk_sizes_ptr = indexer_gpu_chunk_sizes_tensor.data_ptr<int64_t>();

    for (int i = 0; i < num_gpus; i++) {
      indexer_gpu_kv_strides_in_bytes_[i] = idx_kv_strides_ptr[i];
      indexer_gpu_block_strides_in_bytes_[i] = idx_block_strides_ptr[i];
      indexer_gpu_layer_strides_in_bytes_[i] = idx_layer_strides_ptr[i];
      indexer_gpu_chunk_sizes_in_bytes_[i] = idx_chunk_sizes_ptr[i];
    }

    // Determine indexer backend type from tensor count (symmetric with main KV)
    if (indexer_num_tensors_per_gpu_ == 1) {
      indexer_backend_type_ = BackendType::TRTLLM;
    } else if (indexer_num_tensors_per_gpu_ == num_layers) {
      indexer_backend_type_ = BackendType::VLLM;
    } else if (indexer_num_tensors_per_gpu_ == num_layers * 2) {
      indexer_backend_type_ = BackendType::SGLANG;
    } else {
      throw std::runtime_error("Unsupported indexer GPU block type: " +
                               std::to_string(indexer_num_tensors_per_gpu_));
    }

    // Build GTensorHandlers for indexer (symmetric with main KV)
    indexer_gpu_tensor_handlers_.reserve(num_gpus_);
    for (int i = 0; i < num_gpus_; i++) {
      int64_t **idx_gpu_blocks_ptr = reinterpret_cast<int64_t **>(
          indexer_gpu_blocks_ + i * indexer_num_tensors_per_gpu_);
      indexer_gpu_tensor_handlers_.emplace_back(
          indexer_backend_type_, idx_gpu_blocks_ptr, num_layers,
          indexer_gpu_kv_strides_in_bytes_[i],
          indexer_gpu_block_strides_in_bytes_[i],
          indexer_gpu_layer_strides_in_bytes_[i]);
    }

    fprintf(stderr, "[LayerwiseTransferGroup] Indexer fuse: enabled=true, "
           "num_tensors_per_gpu=%d, chunk_size=%ld bytes, backend=%s\n",
           indexer_num_tensors_per_gpu_, indexer_gpu_chunk_sizes_in_bytes_[0],
           indexer_backend_type_ == BackendType::SGLANG ? "SGLANG" :
           indexer_backend_type_ == BackendType::VLLM ? "VLLM" : "TRTLLM");
  }

  // Initialize indexer SSD IO context if indexer_ssd_files is not empty
  enable_indexer_ssd_ = !indexer_ssd_files.empty();
  if (enable_indexer_ssd_) {
    indexer_ioctx_ = std::make_unique<SSDIOCTX>(
        indexer_ssd_files, indexer_ssd_files.size(),
        iouring_entries, iouring_flags);
  }
}

LayerwiseTransferGroup::LayerwiseTransferGroup(
    int num_gpus,
    const std::vector<std::vector<std::vector<torch::Tensor>>>
        &gpu_blocks_per_group,
    torch::Tensor &cpu_blocks,
    std::map<int, std::vector<std::string>> &ssd_files,
    int num_original_layers,
    const std::vector<std::vector<std::pair<int, int>>> &layer_members,
    const std::vector<int> &group_num_layers,
    const std::vector<int64_t> &group_cpu_offset_bytes,
    const std::vector<int64_t> &group_ssd_offset_bytes,
    const std::vector<int64_t> &group_cpu_layer_strides,
    const std::vector<int64_t> &group_cpu_kv_strides,
    const std::vector<int64_t> &group_ssd_layer_strides,
    const std::vector<int64_t> &group_ssd_kv_strides,
    const std::vector<int64_t> &group_chunk_sizes,
    const std::vector<int64_t> &group_h2d_cpu_kv_strides,
    const std::vector<int64_t> &group_h2d_cpu_layer_strides,
    const std::vector<int64_t> &group_cpu_block_strides,
    const std::vector<int64_t> &group_cpu_tp_strides,
    const std::vector<int64_t> &group_gpu_kv_strides,
    const std::vector<int64_t> &group_gpu_block_strides,
    const std::vector<int64_t> &group_gpu_layer_strides,
    const std::vector<int64_t> &group_gpu_chunk_sizes,
    int iouring_entries, int iouring_flags,
    torch::Tensor &layer_eventfds_tensor, int tp_size) {

  if (num_gpus <= 0 || num_original_layers <= 0) {
    throw std::runtime_error(
        "[LayerwiseTransferGroup multi-group] num_gpus and "
        "num_original_layers must be positive");
  }

  num_gpus_ = num_gpus;
  num_layers_ = num_original_layers;
  num_original_layers_ = num_original_layers;
  tp_size_ = tp_size;
  current_counter_id_ = 0;
  has_multi_group_ = true;

  num_tensors_per_gpu_ = 0;
  gpu_blocks_ = nullptr;
  gpu_kv_strides_in_bytes_ = nullptr;
  gpu_block_strides_in_bytes_ = nullptr;
  gpu_layer_strides_in_bytes_ = nullptr;
  gpu_chunk_sizes_in_bytes_ = nullptr;

  enable_eventfd_ = (layer_eventfds_tensor.numel() > 0);
  if (enable_eventfd_) {
    int total_fds = layer_eventfds_tensor.numel();
    int fds_per_counter = tp_size * num_original_layers;
    if (fds_per_counter <= 0 || total_fds % fds_per_counter != 0) {
      throw std::runtime_error(
          "[LayerwiseTransferGroup multi-group] invalid eventfd tensor size");
    }
    num_counters_ = total_fds / fds_per_counter;
    int32_t *fds_ptr = layer_eventfds_tensor.data_ptr<int32_t>();
    layer_eventfds_.assign(fds_ptr, fds_ptr + total_fds);
  } else {
    num_counters_ = 0;
  }

  layer_members_ = layer_members;
  if (static_cast<int>(layer_members_.size()) != num_original_layers_) {
    throw std::runtime_error(
        "[LayerwiseTransferGroup multi-group] layer_members length " +
        std::to_string(layer_members_.size()) + " does not match " +
        std::to_string(num_original_layers_));
  }

  const int num_groups = static_cast<int>(group_num_layers.size());
  if (num_groups <= 0 ||
      static_cast<int>(gpu_blocks_per_group.size()) != num_groups) {
    throw std::runtime_error(
        "[LayerwiseTransferGroup multi-group] invalid group count");
  }
  auto require_group_count = [num_groups](const auto &values,
                                          const char *name) {
    if (static_cast<int>(values.size()) != num_groups) {
      throw std::runtime_error(
          std::string("[LayerwiseTransferGroup multi-group] ") + name +
          " size does not match group count");
    }
  };
  require_group_count(group_cpu_offset_bytes, "group_cpu_offset_bytes");
  require_group_count(group_ssd_offset_bytes, "group_ssd_offset_bytes");
  require_group_count(group_cpu_layer_strides, "group_cpu_layer_strides");
  require_group_count(group_cpu_kv_strides, "group_cpu_kv_strides");
  require_group_count(group_ssd_layer_strides, "group_ssd_layer_strides");
  require_group_count(group_ssd_kv_strides, "group_ssd_kv_strides");
  require_group_count(group_chunk_sizes, "group_chunk_sizes");
  require_group_count(group_h2d_cpu_kv_strides,
                      "group_h2d_cpu_kv_strides");
  require_group_count(group_h2d_cpu_layer_strides,
                      "group_h2d_cpu_layer_strides");
  require_group_count(group_cpu_block_strides, "group_cpu_block_strides");
  require_group_count(group_cpu_tp_strides, "group_cpu_tp_strides");

  const int flattened_gpu_size = num_groups * num_gpus;
  if (static_cast<int>(group_gpu_kv_strides.size()) != flattened_gpu_size ||
      static_cast<int>(group_gpu_block_strides.size()) != flattened_gpu_size ||
      static_cast<int>(group_gpu_layer_strides.size()) != flattened_gpu_size ||
      static_cast<int>(group_gpu_chunk_sizes.size()) != flattened_gpu_size) {
    throw std::runtime_error(
        "[LayerwiseTransferGroup multi-group] flattened GPU stride size "
        "does not match num_groups * num_gpus");
  }

  groups_.resize(num_groups);
  for (int gi = 0; gi < num_groups; ++gi) {
    GroupParams &group = groups_[gi];
    group.num_layers = group_num_layers[gi];
    group.cpu_offset_bytes = group_cpu_offset_bytes[gi];
    group.ssd_offset_bytes = group_ssd_offset_bytes[gi];
    group.cpu_layer_stride = group_cpu_layer_strides[gi];
    group.cpu_kv_stride = group_cpu_kv_strides[gi];
    group.ssd_layer_stride = group_ssd_layer_strides[gi];
    group.ssd_kv_stride = group_ssd_kv_strides[gi];
    group.chunk_size = group_chunk_sizes[gi];
    group.h2d_cpu_kv_stride = group_h2d_cpu_kv_strides[gi];
    group.h2d_cpu_layer_stride = group_h2d_cpu_layer_strides[gi];
    group.cpu_block_stride = group_cpu_block_strides[gi];
    group.cpu_tp_stride = group_cpu_tp_strides[gi];

    group.gpu_kv_strides.resize(num_gpus);
    group.gpu_block_strides.resize(num_gpus);
    group.gpu_layer_strides.resize(num_gpus);
    group.gpu_chunk_sizes.resize(num_gpus);
    for (int device_idx = 0; device_idx < num_gpus; ++device_idx) {
      int flat_idx = gi * num_gpus + device_idx;
      group.gpu_kv_strides[device_idx] = group_gpu_kv_strides[flat_idx];
      group.gpu_block_strides[device_idx] =
          group_gpu_block_strides[flat_idx];
      group.gpu_layer_strides[device_idx] =
          group_gpu_layer_strides[flat_idx];
      group.gpu_chunk_sizes[device_idx] = group_gpu_chunk_sizes[flat_idx];
    }

    if (static_cast<int>(gpu_blocks_per_group[gi].size()) != num_gpus) {
      throw std::runtime_error(
          "[LayerwiseTransferGroup multi-group] device count mismatch");
    }
    group.num_tensors_per_gpu =
        static_cast<int>(gpu_blocks_per_group[gi][0].size());
    if (group.num_tensors_per_gpu <= 0) {
      throw std::runtime_error(
          "[LayerwiseTransferGroup multi-group] empty GPU tensor list");
    }
    cudaMallocHost((void **)&group.gpu_blocks_flat,
                   num_gpus * group.num_tensors_per_gpu * sizeof(void *));
    for (int device_idx = 0; device_idx < num_gpus; ++device_idx) {
      if (static_cast<int>(gpu_blocks_per_group[gi][device_idx].size()) !=
          group.num_tensors_per_gpu) {
        throw std::runtime_error(
            "[LayerwiseTransferGroup multi-group] tensor count mismatch");
      }
      for (int tensor_idx = 0; tensor_idx < group.num_tensors_per_gpu;
           ++tensor_idx) {
        group.gpu_blocks_flat[device_idx * group.num_tensors_per_gpu +
                              tensor_idx] =
            gpu_blocks_per_group[gi][device_idx][tensor_idx].data_ptr();
      }
    }

    if (group.num_tensors_per_gpu == 1) {
      group.backend_type = BackendType::TRTLLM;
    } else if (group.num_tensors_per_gpu == group.num_layers) {
      group.backend_type = BackendType::VLLM;
    } else if (group.num_tensors_per_gpu == group.num_layers * 2) {
      group.backend_type = BackendType::SGLANG;
    } else {
      throw std::runtime_error(
          "[LayerwiseTransferGroup multi-group] unsupported tensor layout "
          "for group " + std::to_string(gi));
    }

    group.gpu_tensor_handlers.reserve(num_gpus);
    for (int device_idx = 0; device_idx < num_gpus; ++device_idx) {
      int64_t **gpu_blocks_ptr = reinterpret_cast<int64_t **>(
          group.gpu_blocks_flat +
          device_idx * group.num_tensors_per_gpu);
      group.gpu_tensor_handlers.emplace_back(
          group.backend_type, gpu_blocks_ptr, group.num_layers,
          group.gpu_kv_strides[device_idx],
          group.gpu_block_strides[device_idx],
          group.gpu_layer_strides[device_idx]);
    }
  }

  cpu_blocks_ = cpu_blocks.data_ptr();
  gpu_device_ids_.resize(num_gpus_);
  for (int device_idx = 0; device_idx < num_gpus_; ++device_idx) {
    gpu_device_ids_[device_idx] =
        gpu_blocks_per_group[0][device_idx][0].device().index();
  }

  streams_.resize(num_gpus_);
  events_.resize(num_gpus_);
  int least_priority, greatest_priority;
  cudaDeviceGetStreamPriorityRange(&least_priority, &greatest_priority);
  for (int device_idx = 0; device_idx < num_gpus_; ++device_idx) {
    cudaSetDevice(gpu_device_ids_[device_idx]);
    cudaStreamCreateWithPriority(&streams_[device_idx], cudaStreamNonBlocking,
                                 greatest_priority);
    cudaEventCreate(&events_[device_idx]);
  }

  enable_ssd_ = !ssd_files.empty();
  if (enable_ssd_) {
    ioctx_ = std::make_unique<SSDIOCTX>(ssd_files, ssd_files.size(),
                                        iouring_entries, iouring_flags);
  }
}

LayerwiseTransferGroup::~LayerwiseTransferGroup() {
  // Stop polling thread if running (only in POLLING mode)
  if (notify_mode_ == NotifyMode::POLLING) {
    poll_stop_.store(true, std::memory_order_release);
    if (poll_thread_.joinable()) {
      poll_thread_.join();
    }
  }

  for (int i = 0; i < num_gpus_; i++) {
    cudaSetDevice(gpu_device_ids_[i]);
    cudaStreamDestroy(streams_[i]);
    cudaEventDestroy(events_[i]);
  }

  if (gpu_blocks_ != nullptr) {
    cudaFreeHost(gpu_blocks_);
  }
  for (auto &group : groups_) {
    if (group.gpu_blocks_flat != nullptr) {
      cudaFreeHost(group.gpu_blocks_flat);
      group.gpu_blocks_flat = nullptr;
    }
    group.gpu_tensor_handlers.clear();
  }

  gpu_tensor_handlers_.clear();
  delete[] gpu_kv_strides_in_bytes_;
  delete[] gpu_block_strides_in_bytes_;
  delete[] gpu_layer_strides_in_bytes_;
  delete[] gpu_chunk_sizes_in_bytes_;

  // Clean up indexer resources
  if (enable_indexer_) {
    cudaFreeHost(indexer_gpu_blocks_);
    indexer_gpu_tensor_handlers_.clear();
    delete[] indexer_gpu_kv_strides_in_bytes_;
    delete[] indexer_gpu_block_strides_in_bytes_;
    delete[] indexer_gpu_layer_strides_in_bytes_;
    delete[] indexer_gpu_chunk_sizes_in_bytes_;
  }
}

void LayerwiseTransferGroup::layer_done_callback(int start_layer,
                                                 int layers_this_batch,
                                                 int expected_count,
                                                 nvtxRangeId_t *current_range_id_ptr,
                                                 bool is_last_batch,
                                                 const char *next_range_name,
                                                 nvtxRangeId_t *next_range_id_ptr,
                                                 int callbacks_per_gpu) {
  std::atomic<int> *counter = new std::atomic<int>(0);
  
  // Get eventfd pointer for current counter set
  int *eventfds_ptr = nullptr;
  if (enable_eventfd_ && num_counters_ > 0) {
    // Offset into layer_eventfds_ for current counter set
    int offset = current_counter_id_ * tp_size_ * num_layers_;
    eventfds_ptr = layer_eventfds_.data() + offset;
  }
  
  for (int i = 0; i < num_gpus_; ++i) {
    for (int callback_idx = 0; callback_idx < callbacks_per_gpu;
         ++callback_idx) {
      LayerCallbackData *data = new LayerCallbackData{
          start_layer, layers_this_batch, expected_count, counter,
          enable_eventfd_, tp_size_, num_layers_, eventfds_ptr,
          current_range_id_ptr, is_last_batch, {0}, next_range_id_ptr};
      if (next_range_name != nullptr) {
        snprintf(data->next_range_name, sizeof(data->next_range_name), "%s",
                 next_range_name);
      }
      cudaLaunchHostFunc(streams_[i], layer_done_host_callback, data);
    }
  }
}

void LayerwiseTransferGroup::layerwise_transfer(
    const torch::Tensor &ssd_block_ids, const torch::Tensor &cpu_block_ids_d2h,
    const int64_t ssd_layer_stride_in_bytes,
    const int64_t ssd_kv_stride_in_bytes, const int num_blocks_per_file,
    const int round_robin, const int num_threads_per_device,
    const torch::Tensor &gpu_block_id_tensor,
    const torch::Tensor &cpu_block_id_tensor,
    const int64_t cpu_kv_stride_in_bytes,
    const int64_t cpu_layer_stride_in_bytes,
    const int64_t cpu_block_stride_in_bytes,
    const int64_t cpu_chunk_size_in_bytes,
    const int64_t h2d_cpu_kv_stride_in_bytes,
    const int64_t h2d_cpu_layer_stride_in_bytes,
    const int64_t cpu_tp_stride_in_bytes, const int transfer_cta_num,
    const bool use_ce_transfer, const int num_layers,
    const int layer_granularity, const bool is_mla,
    const int counter_id,
    const torch::Tensor &indexer_gpu_block_id_tensor,
    const torch::Tensor &indexer_cpu_block_id_tensor,
    const int64_t indexer_cpu_block_stride_in_bytes,
    const int64_t indexer_cpu_layer_stride_in_bytes,
    const int64_t indexer_h2d_cpu_kv_stride_in_bytes,
    const int64_t indexer_h2d_cpu_layer_stride_in_bytes,
    const torch::Tensor &indexer_ssd_block_ids,
    const torch::Tensor &indexer_cpu_block_ids_d2h,
    const int64_t indexer_ssd_layer_stride_in_bytes,
    const int64_t indexer_ssd_kv_stride_in_bytes,
    const int64_t indexer_cpu_chunk_size_in_bytes,
    const int indexer_num_blocks_per_file,
    const std::string &mla_d2h_mode,
    const std::string &notify_mode) {

  // Resolve notification mode from the caller (sourced from
  // GLOBAL_CONFIG_FROM_ENV.layerwise_notify_mode on the Python side).
  notify_mode_ = (notify_mode == "polling") ? NotifyMode::POLLING
                                            : NotifyMode::HOSTFUNC;

  // Set current counter ID for eventfd notification
  current_counter_id_ = counter_id;

  int num_blocks = gpu_block_id_tensor.numel();
  int64_t *gpu_block_ids =
      static_cast<int64_t *>(gpu_block_id_tensor.data_ptr());
  int64_t *cpu_block_ids =
      static_cast<int64_t *>(cpu_block_id_tensor.data_ptr());
  void *cpu_ptr = cpu_blocks_;

  // Indexer block ids (may be empty if indexer is not enabled or not provided)
  bool do_indexer_transfer = enable_indexer_ &&
      indexer_gpu_block_id_tensor.defined() &&
      indexer_gpu_block_id_tensor.numel() > 0;
  int num_indexer_blocks = 0;
  int64_t *indexer_gpu_block_ids = nullptr;
  int64_t *indexer_cpu_block_ids = nullptr;
  if (do_indexer_transfer) {
    num_indexer_blocks = indexer_gpu_block_id_tensor.numel();
    indexer_gpu_block_ids =
        static_cast<int64_t *>(indexer_gpu_block_id_tensor.data_ptr());
    indexer_cpu_block_ids =
        static_cast<int64_t *>(indexer_cpu_block_id_tensor.data_ptr());
  }

  // Create CUDA events for timing each layer batch (on GPU 0)
  int num_batches = (num_layers + layer_granularity - 1) / layer_granularity;
  std::vector<cudaEvent_t> timing_events(num_batches + 1);  // +1 for start event
  std::vector<int> batch_start_layers(num_batches);
  std::vector<int> batch_layers_count(num_batches);
  
  cudaSetDevice(gpu_device_ids_[0]);
  for (int i = 0; i <= num_batches; ++i) {
    cudaEventCreate(&timing_events[i]);
  }

  // Prepare poll batches for event-based notification
  if (notify_mode_ == NotifyMode::POLLING && num_batches > 0) {
    poll_batches_.clear();
    poll_batches_.resize(num_batches);
    for (int b = 0; b < num_batches; ++b) {
      poll_batches_[b].start_layer = b * layer_granularity;
      poll_batches_[b].layers_this_batch =
          std::min(layer_granularity, num_layers - b * layer_granularity);
      poll_batches_[b].per_gpu_events.resize(num_gpus_);
      poll_batches_[b].notified = false;
      for (int g = 0; g < num_gpus_; ++g) {
        cudaSetDevice(gpu_device_ids_[g]);
        cudaEventCreateWithFlags(&poll_batches_[b].per_gpu_events[g],
                                 cudaEventDisableTiming);
      }
    }
  }

  // Record start event
  cudaEventRecord(timing_events[0], streams_[0]);

  // Allocate storage for NVTX range IDs (one per batch)
  std::vector<nvtxRangeId_t> h2d_range_ids(num_batches, 0);
  // Pre-generate all range names with data size info
  std::vector<std::string> h2d_range_names(num_batches);
  for (int b = 0; b < num_batches; ++b) {
    int sl = b * layer_granularity;
    int ltb = std::min(layer_granularity, num_layers - sl);
    // Calculate data size for this batch: chunk_size * 2 (K+V) * layers * num_blocks
    // H2D: every GPU reads a full KV copy regardless of mla_d2h_mode (the mode
    // only controls where on CPU it reads from, not how much).
    int64_t bytes_this_batch = 0;
    for (int g = 0; g < num_gpus_; ++g) {
      bytes_this_batch += gpu_chunk_sizes_in_bytes_[g] * 2 * ltb * num_blocks;
    }
    // Add indexer bytes if applicable
    int64_t indexer_bytes_this_batch = 0;
    if (do_indexer_transfer) {
      for (int g = 0; g < num_gpus_; ++g) {
        indexer_bytes_this_batch += indexer_gpu_chunk_sizes_in_bytes_[g] * ltb * num_indexer_blocks;
      }
    }
    double mb_this_batch = (bytes_this_batch + indexer_bytes_this_batch) / (1024.0 * 1024.0);
    char name[256];
    if (do_indexer_transfer) {
      snprintf(name, sizeof(name), "CPU->GPU Layer[%d,%d) KV:%.2fMB+Idx:%.2fMB",
               sl, sl + ltb, bytes_this_batch / (1024.0 * 1024.0),
               indexer_bytes_this_batch / (1024.0 * 1024.0));
    } else {
      snprintf(name, sizeof(name), "CPU->GPU Layer[%d,%d) %.2fMB", sl, sl + ltb,
               bytes_this_batch / (1024.0 * 1024.0));
    }
    h2d_range_names[b] = name;
  }

  // Start the first batch's NVTX range in main thread
  if (num_batches > 0) {
    h2d_range_ids[0] = nvtxRangeStartA(h2d_range_names[0].c_str());
  }

  // Step 0: SSD -> CPU transfer for ALL layers at once (before layerwise loop).
  // This is required because the CPU memory uses TP-divided layout where each rank's
  // data occupies a contiguous region [rank*tp_stride, (rank+1)*tp_stride). Per-layer-batch
  // SSD reads with full strides would land at wrong CPU positions for TP > 1.
  if (enable_ssd_ && ssd_block_ids.numel() > 0) {
    int num_ssd_blocks = ssd_block_ids.numel();
    int64_t ssd_bytes = cpu_chunk_size_in_bytes * 2 * num_layers * num_ssd_blocks;
    double ssd_mb = ssd_bytes / (1024.0 * 1024.0);
    char ssd_range_name[128];
    snprintf(ssd_range_name, sizeof(ssd_range_name),
             "SSD->CPU AllLayers[0,%d) %.2fMB", num_layers, ssd_mb);
    nvtxRangePushA(ssd_range_name);

    torch::Tensor all_layer_ids =
        torch::arange(0, num_layers,
                      torch::TensorOptions().dtype(torch::kInt32));
    transfer_kv_blocks_ssd(
        *ioctx_, all_layer_ids, reinterpret_cast<int64_t>(cpu_blocks_),
        ssd_block_ids, cpu_block_ids_d2h, cpu_layer_stride_in_bytes,
        cpu_kv_stride_in_bytes, ssd_layer_stride_in_bytes,
        ssd_kv_stride_in_bytes, cpu_chunk_size_in_bytes,
        cpu_block_stride_in_bytes,
        true, // is_read: SSD -> CPU
        num_blocks_per_file, round_robin, num_threads_per_device, is_mla);

    nvtxRangePop();
  }

  // Indexer SSD -> CPU transfer for ALL layers at once.
  if (enable_indexer_ssd_ && indexer_ssd_block_ids.defined() &&
      indexer_ssd_block_ids.numel() > 0) {
    int num_indexer_ssd_blocks = indexer_ssd_block_ids.numel();
    int64_t indexer_ssd_bytes = indexer_cpu_chunk_size_in_bytes * num_layers * num_indexer_ssd_blocks;
    double indexer_ssd_mb = indexer_ssd_bytes / (1024.0 * 1024.0);
    char idx_ssd_range_name[128];
    snprintf(idx_ssd_range_name, sizeof(idx_ssd_range_name),
             "Indexer SSD->CPU AllLayers[0,%d) %.2fMB", num_layers, indexer_ssd_mb);
    nvtxRangePushA(idx_ssd_range_name);

    torch::Tensor all_layer_ids =
        torch::arange(0, num_layers,
                      torch::TensorOptions().dtype(torch::kInt32));
    transfer_kv_blocks_ssd(
        *indexer_ioctx_, all_layer_ids,
        reinterpret_cast<int64_t>(indexer_cpu_blocks_),
        indexer_ssd_block_ids, indexer_cpu_block_ids_d2h,
        indexer_cpu_layer_stride_in_bytes,
        indexer_ssd_kv_stride_in_bytes,
        indexer_ssd_layer_stride_in_bytes,
        indexer_ssd_kv_stride_in_bytes,
        indexer_cpu_chunk_size_in_bytes,
        indexer_cpu_block_stride_in_bytes,
        true, // is_read: SSD -> CPU
        indexer_num_blocks_per_file, round_robin, num_threads_per_device,
        true /* is_mla: indexer always MLA */);

    nvtxRangePop();
  }

  int batch_idx = 0;
  for (int start_layer = 0; start_layer < num_layers;
       start_layer += layer_granularity) {
    int layers_this_batch =
        std::min(layer_granularity, num_layers - start_layer);

    batch_start_layers[batch_idx] = start_layer;
    batch_layers_count[batch_idx] = layers_this_batch;

    // Step 1: CPU -> GPU transfer
    // NVTX range for this batch was already started (by main thread for first batch,
    // or by previous batch's callback for subsequent batches)
    
    // Validate mla_d2h_mode once before the per-GPU loop
    std::string mode = mla_d2h_mode;
    if (is_mla && mode != "sharded" && mode != "all_write" && mode != "rank0_only") {
      fprintf(stderr, "[FlexKV] Warning: Invalid mla_d2h_mode='%s', using default 'sharded'\n",
              mode.c_str());
      mode = "sharded";
    }

    for (int i = 0; i < num_gpus_; ++i) {
      cudaSetDevice(gpu_device_ids_[i]);
      int64_t cpu_startoff_inside_chunks = i * cpu_tp_stride_in_bytes;
      int64_t gpu_startoff_inside_chunks = 0;
      int64_t chunk_size = gpu_chunk_sizes_in_bytes_[i];

      // Handle MLA D2H mode for H2D transfer (inlined logic)
      if (is_mla) {
        if (mode == "sharded") {
          cpu_startoff_inside_chunks = 0;
          gpu_startoff_inside_chunks = 0;
        } else if (mode == "all_write") {
          // Each rank's complete KV occupies num_blocks blocks on CPU.
          // Use cpu_block_stride (not gpu_chunk_size) because BLOCKFIRST's
          // block_stride includes all layers+kv_dims, while LAYERFIRST's
          // block_stride == chunk_size (same result).
          cpu_startoff_inside_chunks = i * num_blocks * cpu_block_stride_in_bytes;
          gpu_startoff_inside_chunks = 0;
        } else if (mode == "rank0_only") {
          cpu_startoff_inside_chunks = 0;
          gpu_startoff_inside_chunks = 0;
        }
      }

      switch (backend_type_) {
      case BackendType::VLLM:
        flexkv::transfer_kv_blocks<BackendType::VLLM>(
            num_blocks, start_layer, layers_this_batch, gpu_block_ids,
            gpu_tensor_handlers_[i], gpu_startoff_inside_chunks, cpu_block_ids,
            cpu_ptr, h2d_cpu_kv_stride_in_bytes, h2d_cpu_layer_stride_in_bytes,
            cpu_block_stride_in_bytes, cpu_startoff_inside_chunks, chunk_size,
            streams_[i], transfer_cta_num, true, use_ce_transfer, is_mla, false);
        break;
      case BackendType::TRTLLM:
        flexkv::transfer_kv_blocks<BackendType::TRTLLM>(
            num_blocks, start_layer, layers_this_batch, gpu_block_ids,
            gpu_tensor_handlers_[i], gpu_startoff_inside_chunks, cpu_block_ids,
            cpu_ptr, h2d_cpu_kv_stride_in_bytes, h2d_cpu_layer_stride_in_bytes,
            cpu_block_stride_in_bytes, cpu_startoff_inside_chunks, chunk_size,
            streams_[i], transfer_cta_num, true, use_ce_transfer, is_mla, false);
        break;
      case BackendType::SGLANG:
        flexkv::transfer_kv_blocks<BackendType::SGLANG>(
            num_blocks, start_layer, layers_this_batch, gpu_block_ids,
            gpu_tensor_handlers_[i], gpu_startoff_inside_chunks, cpu_block_ids,
            cpu_ptr, h2d_cpu_kv_stride_in_bytes, h2d_cpu_layer_stride_in_bytes,
            cpu_block_stride_in_bytes, cpu_startoff_inside_chunks, chunk_size,
            streams_[i], transfer_cta_num, true, use_ce_transfer, is_mla, false);
        break;
      }

      // Fused indexer CPU -> GPU transfer on the same stream
      // Uses transfer_kv_blocks (symmetric with main KV Step 2) instead of
      // hand-written cudaMemcpyAsync loops for backend-agnostic support.
      // Note: indexer uses ReplicatedLinear weights with 1 head (is_mla=true),
      // so all TP ranks hold identical data. No TP head-partitioning needed,
      // cpu_startoff is always 0 (unlike main KV which may offset by tp_stride).
      if (do_indexer_transfer) {
        int64_t idx_chunk_size = indexer_gpu_chunk_sizes_in_bytes_[i];
        // idx_cpu_startoff = 0: indexer data is not partitioned across TP ranks
        int64_t idx_cpu_startoff = 0;

        switch (indexer_backend_type_) {
        case BackendType::VLLM:
          flexkv::transfer_kv_blocks<BackendType::VLLM>(
              num_indexer_blocks, start_layer, layers_this_batch,
              indexer_gpu_block_ids, indexer_gpu_tensor_handlers_[i],
              0 /* gpu_startoff */, indexer_cpu_block_ids,
              indexer_cpu_blocks_,
              indexer_h2d_cpu_kv_stride_in_bytes,
              indexer_h2d_cpu_layer_stride_in_bytes,
              indexer_cpu_block_stride_in_bytes,
              idx_cpu_startoff, idx_chunk_size,
              streams_[i], transfer_cta_num, true /* h2d */,
              use_ce_transfer, true /* is_mla */, false /* sync */);
          break;
        case BackendType::TRTLLM:
          flexkv::transfer_kv_blocks<BackendType::TRTLLM>(
              num_indexer_blocks, start_layer, layers_this_batch,
              indexer_gpu_block_ids, indexer_gpu_tensor_handlers_[i],
              0 /* gpu_startoff */, indexer_cpu_block_ids,
              indexer_cpu_blocks_,
              indexer_h2d_cpu_kv_stride_in_bytes,
              indexer_h2d_cpu_layer_stride_in_bytes,
              indexer_cpu_block_stride_in_bytes,
              idx_cpu_startoff, idx_chunk_size,
              streams_[i], transfer_cta_num, true /* h2d */,
              use_ce_transfer, true /* is_mla */, false /* sync */);
          break;
        case BackendType::SGLANG:
          flexkv::transfer_kv_blocks<BackendType::SGLANG>(
              num_indexer_blocks, start_layer, layers_this_batch,
              indexer_gpu_block_ids, indexer_gpu_tensor_handlers_[i],
              0 /* gpu_startoff */, indexer_cpu_block_ids,
              indexer_cpu_blocks_,
              indexer_h2d_cpu_kv_stride_in_bytes,
              indexer_h2d_cpu_layer_stride_in_bytes,
              indexer_cpu_block_stride_in_bytes,
              idx_cpu_startoff, idx_chunk_size,
              streams_[i], transfer_cta_num, true /* h2d */,
              use_ce_transfer, true /* is_mla */, false /* sync */);
          break;
        }
      }
    }

    // Record event after this batch on GPU 0
    cudaSetDevice(gpu_device_ids_[0]);
    cudaEventRecord(timing_events[batch_idx + 1], streams_[0]);

    if (notify_mode_ == NotifyMode::POLLING) {
      // Record per-GPU events for polling thread
      for (int i = 0; i < num_gpus_; ++i) {
        cudaSetDevice(gpu_device_ids_[i]);
        cudaEventRecord(poll_batches_[batch_idx].per_gpu_events[i], streams_[i]);
      }
    } else {
      // NVTX: current range ends in callback, next range starts in callback
      bool is_last_batch = (batch_idx == num_batches - 1);
      const char *next_name = is_last_batch ? nullptr : h2d_range_names[batch_idx + 1].c_str();
      nvtxRangeId_t *next_id_ptr = is_last_batch ? nullptr : &h2d_range_ids[batch_idx + 1];

      layer_done_callback(start_layer, layers_this_batch,
                          num_gpus_,
                          &h2d_range_ids[batch_idx], is_last_batch,
                          next_name, next_id_ptr);
    }
    batch_idx++;
  }

  // POLLING mode: clean up any lingering thread, then start polling + sync.
  if (notify_mode_ == NotifyMode::POLLING) {
    // Defensive cleanup: stop any lingering polling thread from a prior call.
    poll_stop_.store(true, std::memory_order_release);
    if (poll_thread_.joinable()) {
      poll_thread_.join();
    }

    // Start polling thread — it writes eventfds as each batch completes.
    poll_stop_.store(false, std::memory_order_release);
    poll_next_batch_.store(0, std::memory_order_release);
    poll_thread_ = std::thread(&LayerwiseTransferGroup::event_polling_loop, this);

    // Block until all GPU work is complete. The polling thread will have
    // fired all eventfds by the time sync returns.
    for (int i = 0; i < num_gpus_; ++i) {
      cudaSetDevice(gpu_device_ids_[i]);
      cudaError_t err = cudaStreamSynchronize(streams_[i]);
      if (err != cudaSuccess) {
        poll_stop_.store(true, std::memory_order_release);
        if (poll_thread_.joinable()) poll_thread_.join();
        throw std::runtime_error("layerwise_transfer failed on GPU " +
                                 std::to_string(i) + ": " +
                                 cudaGetErrorString(err));
      }
    }
    poll_stop_.store(true, std::memory_order_release);
    if (poll_thread_.joinable()) {
      poll_thread_.join();
    }

    // Destroy poll batch events
    for (int b = 0; b < (int)poll_batches_.size(); ++b) {
      for (int g = 0; g < num_gpus_; ++g) {
        cudaSetDevice(gpu_device_ids_[g]);
        cudaEventDestroy(poll_batches_[b].per_gpu_events[g]);
      }
    }
  } else {
    for (int i = 0; i < num_gpus_; ++i) {
      cudaError_t err = cudaStreamSynchronize(streams_[i]);
      if (err != cudaSuccess) {
        throw std::runtime_error("layerwise_transfer failed on GPU " +
                                 std::to_string(i) + ": " +
                                 cudaGetErrorString(err));
      }
    }
  }

  // Calculate and print timing for each layer batch
  // chunk_size per GPU * num_gpus * 2 (K+V) * layers_this_batch * num_blocks
  // fprintf(stderr, "\n[LayerwiseTransfer] CPU->GPU Transfer Timing (num_blocks=%d):\n", num_blocks);
  float total_time_ms = 0.0f;
  int64_t total_bytes = 0;
  
  for (int i = 0; i < num_batches; ++i) {
    float elapsed_ms = 0.0f;
    cudaEventElapsedTime(&elapsed_ms, timing_events[i], timing_events[i + 1]);

    // Calculate bytes transferred for this batch
    // H2D: every GPU reads a full KV copy regardless of mla_d2h_mode
    int64_t bytes_this_batch = 0;
    for (int g = 0; g < num_gpus_; ++g) {
      bytes_this_batch += gpu_chunk_sizes_in_bytes_[g] * 2 * batch_layers_count[i] * num_blocks;
    }
    // Include indexer bytes
    int64_t indexer_bytes_batch = 0;
    if (do_indexer_transfer) {
      for (int g = 0; g < num_gpus_; ++g) {
        indexer_bytes_batch += indexer_gpu_chunk_sizes_in_bytes_[g] * batch_layers_count[i] * num_indexer_blocks;
      }
      bytes_this_batch += indexer_bytes_batch;
    }
    
    double bandwidth_gbps = (bytes_this_batch / (1024.0 * 1024.0 * 1024.0)) / (elapsed_ms / 1000.0);
    
    // fprintf(stderr, "  Layers [%d, %d): time=%.3f ms, size=%.2f MB, bandwidth=%.2f GB/s\n",
    //         batch_start_layers[i], 
    //         batch_start_layers[i] + batch_layers_count[i],
    //         elapsed_ms,
    //         bytes_this_batch / (1024.0 * 1024.0),
    //         bandwidth_gbps);
    
    total_time_ms += elapsed_ms;
    total_bytes += bytes_this_batch;
  }
  
  double total_bandwidth_gbps = (total_bytes / (1024.0 * 1024.0 * 1024.0)) / (total_time_ms / 1000.0);
  // fprintf(stderr, "  Total: time=%.3f ms, size=%.2f MB, avg_bandwidth=%.2f GB/s\n\n",
  //         total_time_ms, total_bytes / (1024.0 * 1024.0), total_bandwidth_gbps);
  // fflush(stderr);

  // Cleanup timing events
  cudaSetDevice(gpu_device_ids_[0]);
  for (int i = 0; i <= num_batches; ++i) {
    cudaEventDestroy(timing_events[i]);
  }
}

void LayerwiseTransferGroup::layerwise_transfer_multi_group(
    const torch::Tensor &ssd_block_ids,
    const torch::Tensor &cpu_block_ids_d2h,
    const int num_blocks_per_file, const int round_robin,
    const int num_threads_per_device,
    const torch::Tensor &gpu_block_id_tensor,
    const torch::Tensor &cpu_block_id_tensor,
    const int transfer_cta_num, const bool use_ce_transfer,
    const bool is_mla, const int counter_id,
    const std::string &notify_mode) {
  if (!has_multi_group_) {
    throw std::runtime_error(
        "layerwise_transfer_multi_group() called on a single-group instance");
  }
  if (counter_id < 0 ||
      (num_counters_ > 0 && counter_id >= num_counters_)) {
    throw std::runtime_error(
        "layerwise_transfer_multi_group() received invalid counter_id");
  }

  notify_mode_ = (notify_mode == "polling") ? NotifyMode::POLLING
                                            : NotifyMode::HOSTFUNC;
  current_counter_id_ = counter_id;

  int num_blocks = gpu_block_id_tensor.numel();
  int64_t *gpu_block_ids =
      static_cast<int64_t *>(gpu_block_id_tensor.data_ptr());
  int64_t *cpu_block_ids =
      static_cast<int64_t *>(cpu_block_id_tensor.data_ptr());

  // CPU and SSD use the same byte-flat BLOCKFIRST representation. Transfer
  // each block as one opaque blob so very small compressed groups never issue
  // sub-page IO independently.
  if (enable_ssd_ && ssd_block_ids.numel() > 0) {
    const int64_t block_stride = groups_[0].cpu_block_stride;
    torch::Tensor one_layer_id =
        torch::zeros({1}, torch::TensorOptions().dtype(torch::kInt32));
    transfer_kv_blocks_ssd(
        *ioctx_, one_layer_id, reinterpret_cast<int64_t>(cpu_blocks_),
        ssd_block_ids, cpu_block_ids_d2h,
        /*cpu_layer_stride_in_bytes=*/block_stride,
        /*cpu_kv_stride_in_bytes=*/0,
        /*ssd_layer_stride_in_bytes=*/block_stride,
        /*ssd_kv_stride_in_bytes=*/0,
        /*chunk_size_in_bytes=*/block_stride,
        /*block_stride_in_bytes=*/block_stride,
        /*is_read=*/true, num_blocks_per_file, round_robin,
        num_threads_per_device, /*is_mla=*/true);
  }

  std::vector<int> active_layers;
  active_layers.reserve(num_original_layers_);
  for (int original_layer = 0; original_layer < num_original_layers_;
       ++original_layer) {
    if (!layer_members_[original_layer].empty()) {
      active_layers.push_back(original_layer);
    }
  }

  std::vector<nvtxRangeId_t> h2d_range_ids(num_original_layers_, 0);
  std::vector<std::string> h2d_range_names(num_original_layers_);
  if (notify_mode_ == NotifyMode::HOSTFUNC) {
    for (int original_layer : active_layers) {
      char range_name[160];
      snprintf(range_name, sizeof(range_name),
               "CPU->GPU OrigLayer[%d] members=%zu", original_layer,
               layer_members_[original_layer].size());
      h2d_range_names[original_layer] = range_name;
    }
    if (!active_layers.empty()) {
      int first_layer = active_layers.front();
      h2d_range_ids[first_layer] =
          nvtxRangeStartA(h2d_range_names[first_layer].c_str());
    }
  } else {
    poll_batches_.clear();
    poll_batches_.resize(active_layers.size());
  }

  for (size_t active_idx = 0; active_idx < active_layers.size();
       ++active_idx) {
    int original_layer = active_layers[active_idx];
    const auto &members = layer_members_[original_layer];

    for (const auto &member : members) {
      int group_idx = member.first;
      int local_layer_id = member.second;
      if (group_idx < 0 || group_idx >= static_cast<int>(groups_.size())) {
        throw std::runtime_error(
            "multi-group layer member contains invalid group index");
      }
      const GroupParams &group = groups_[group_idx];
      if (local_layer_id < 0 || local_layer_id >= group.num_layers) {
        throw std::runtime_error(
            "multi-group layer member contains invalid local layer index");
      }

      for (int device_idx = 0; device_idx < num_gpus_; ++device_idx) {
        cudaSetDevice(gpu_device_ids_[device_idx]);
        int64_t cpu_startoff = device_idx * group.cpu_tp_stride;
        if (is_mla) {
          cpu_startoff = 0;
        }
        void *group_cpu_ptr =
            static_cast<char *>(cpu_blocks_) + group.cpu_offset_bytes;
        int64_t chunk_size = group.gpu_chunk_sizes[device_idx];

        switch (group.backend_type) {
        case BackendType::VLLM:
          flexkv::transfer_kv_blocks<BackendType::VLLM>(
              num_blocks, local_layer_id, 1, gpu_block_ids,
              group.gpu_tensor_handlers[device_idx], 0, cpu_block_ids,
              group_cpu_ptr, group.h2d_cpu_kv_stride,
              group.h2d_cpu_layer_stride, group.cpu_block_stride,
              cpu_startoff, chunk_size, streams_[device_idx], transfer_cta_num,
              true, use_ce_transfer, is_mla, false);
          break;
        case BackendType::TRTLLM:
          flexkv::transfer_kv_blocks<BackendType::TRTLLM>(
              num_blocks, local_layer_id, 1, gpu_block_ids,
              group.gpu_tensor_handlers[device_idx], 0, cpu_block_ids,
              group_cpu_ptr, group.h2d_cpu_kv_stride,
              group.h2d_cpu_layer_stride, group.cpu_block_stride,
              cpu_startoff, chunk_size, streams_[device_idx], transfer_cta_num,
              true, use_ce_transfer, is_mla, false);
          break;
        case BackendType::SGLANG:
          flexkv::transfer_kv_blocks<BackendType::SGLANG>(
              num_blocks, local_layer_id, 1, gpu_block_ids,
              group.gpu_tensor_handlers[device_idx], 0, cpu_block_ids,
              group_cpu_ptr, group.h2d_cpu_kv_stride,
              group.h2d_cpu_layer_stride, group.cpu_block_stride,
              cpu_startoff, chunk_size, streams_[device_idx], transfer_cta_num,
              true, use_ce_transfer, is_mla, false);
          break;
        }
      }
    }

    if (notify_mode_ == NotifyMode::POLLING) {
      PollBatchInfo &poll_batch = poll_batches_[active_idx];
      poll_batch.start_layer = original_layer;
      poll_batch.layers_this_batch = 1;
      poll_batch.notified = false;
      poll_batch.per_gpu_events.resize(num_gpus_);
      for (int device_idx = 0; device_idx < num_gpus_; ++device_idx) {
        cudaSetDevice(gpu_device_ids_[device_idx]);
        cudaEventCreateWithFlags(&poll_batch.per_gpu_events[device_idx],
                                 cudaEventDisableTiming);
        cudaEventRecord(poll_batch.per_gpu_events[device_idx],
                        streams_[device_idx]);
      }
    } else {
      bool is_last = active_idx + 1 == active_layers.size();
      int next_layer = is_last ? -1 : active_layers[active_idx + 1];
      const char *next_name =
          is_last ? nullptr : h2d_range_names[next_layer].c_str();
      nvtxRangeId_t *next_id =
          is_last ? nullptr : &h2d_range_ids[next_layer];
      int callbacks_per_gpu = static_cast<int>(members.size());
      layer_done_callback(
          original_layer, 1, callbacks_per_gpu * num_gpus_,
          &h2d_range_ids[original_layer], is_last, next_name, next_id,
          callbacks_per_gpu);
    }
  }

  if (notify_mode_ == NotifyMode::POLLING && !poll_batches_.empty()) {
    poll_stop_.store(true, std::memory_order_release);
    if (poll_thread_.joinable()) {
      poll_thread_.join();
    }
    poll_stop_.store(false, std::memory_order_release);
    poll_next_batch_.store(0, std::memory_order_release);
    poll_thread_ =
        std::thread(&LayerwiseTransferGroup::event_polling_loop, this);
  }

  for (int device_idx = 0; device_idx < num_gpus_; ++device_idx) {
    cudaSetDevice(gpu_device_ids_[device_idx]);
    cudaError_t error = cudaStreamSynchronize(streams_[device_idx]);
    if (error != cudaSuccess) {
      poll_stop_.store(true, std::memory_order_release);
      if (poll_thread_.joinable()) {
        poll_thread_.join();
      }
      throw std::runtime_error(
          "layerwise_transfer_multi_group failed on GPU " +
          std::to_string(device_idx) + ": " + cudaGetErrorString(error));
    }
  }

  if (notify_mode_ == NotifyMode::POLLING) {
    if (poll_thread_.joinable()) {
      poll_thread_.join();
    }
    poll_stop_.store(true, std::memory_order_release);
    for (auto &poll_batch : poll_batches_) {
      for (int device_idx = 0; device_idx < num_gpus_; ++device_idx) {
        cudaSetDevice(gpu_device_ids_[device_idx]);
        cudaEventDestroy(poll_batch.per_gpu_events[device_idx]);
      }
    }
  }
}

} // namespace flexkv
