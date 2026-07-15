#pragma once

#include <atomic>
#include <cuda_runtime.h>
#include <fcntl.h>
#include <map>
#include <memory>
#include <string>
#include <thread>
#include <torch/extension.h>
#include <utility>
#include <vector>
#include <sys/eventfd.h>
#include <unistd.h>
#include <nvtx3/nvToolsExt.h>

#include "gtensor_handler.cuh"
#include "transfer.cuh"
#include "transfer_ssd.h"

namespace flexkv {

// Per-group byte strides and GPU tensor metadata for heterogeneous layouts.
struct GroupParams {
  int num_layers;
  int64_t cpu_offset_bytes;
  int64_t ssd_offset_bytes;
  int64_t cpu_layer_stride;
  int64_t cpu_kv_stride;
  int64_t ssd_layer_stride;
  int64_t ssd_kv_stride;
  int64_t chunk_size;

  int64_t h2d_cpu_kv_stride;
  int64_t h2d_cpu_layer_stride;
  int64_t cpu_block_stride;
  int64_t cpu_tp_stride;

  std::vector<int64_t> gpu_kv_strides;
  std::vector<int64_t> gpu_block_strides;
  std::vector<int64_t> gpu_layer_strides;
  std::vector<int64_t> gpu_chunk_sizes;

  void **gpu_blocks_flat = nullptr;
  int num_tensors_per_gpu = 0;
  BackendType backend_type = BackendType::SGLANG;
  std::vector<GTensorHandler> gpu_tensor_handlers;
};

class LayerwiseTransferGroup {
public:
  LayerwiseTransferGroup(
      int num_gpus, const std::vector<std::vector<torch::Tensor>> &gpu_blocks,
      torch::Tensor &cpu_blocks,
      std::map<int, std::vector<std::string>> &ssd_files,
      int num_layers, torch::Tensor &gpu_kv_strides_tensor,
      torch::Tensor &gpu_block_strides_tensor,
      torch::Tensor &gpu_layer_strides_tensor,
      torch::Tensor &gpu_chunk_sizes_tensor, int iouring_entries,
      int iouring_flags, torch::Tensor &layer_eventfds_tensor, int tp_size,
      const std::vector<std::vector<torch::Tensor>> &indexer_gpu_blocks = {},
      torch::Tensor indexer_cpu_blocks = torch::Tensor(),
      torch::Tensor indexer_gpu_kv_strides_tensor = torch::Tensor(),
      torch::Tensor indexer_gpu_block_strides_tensor = torch::Tensor(),
      torch::Tensor indexer_gpu_layer_strides_tensor = torch::Tensor(),
      torch::Tensor indexer_gpu_chunk_sizes_tensor = torch::Tensor(),
      std::map<int, std::vector<std::string>> indexer_ssd_files = {});

  // Multi-group constructor. ``layer_members[orig]`` contains every
  // (group_idx, local_layer_id) member transferred for one original layer.
  LayerwiseTransferGroup(
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
      torch::Tensor &layer_eventfds_tensor, int tp_size);

  ~LayerwiseTransferGroup();

  // Layerwise transfer: SSD->CPU + CPU->GPU
  void layerwise_transfer(
      const torch::Tensor
          &ssd_block_ids, // SSD source block ids (for disk2host)
      const torch::Tensor
          &cpu_block_ids_d2h, // CPU dest block ids (for disk2host)
      const int64_t ssd_layer_stride_in_bytes,
      const int64_t ssd_kv_stride_in_bytes, const int num_blocks_per_file,
      const int round_robin, const int num_threads_per_device,
      const torch::Tensor
          &gpu_block_id_tensor, // GPU dest block ids (for host2device)
      const torch::Tensor
          &cpu_block_id_tensor, // CPU source block ids (for host2device)
      const int64_t cpu_kv_stride_in_bytes,
      const int64_t cpu_layer_stride_in_bytes,
      const int64_t cpu_block_stride_in_bytes,
      const int64_t cpu_chunk_size_in_bytes,
      const int64_t h2d_cpu_kv_stride_in_bytes,
      const int64_t h2d_cpu_layer_stride_in_bytes,
      const int64_t cpu_tp_stride_in_bytes, const int transfer_cta_num,
      const bool use_ce_transfer, const int num_layers,
      const int layer_granularity, const bool is_mla,
      const int counter_id = 0,
      const torch::Tensor &indexer_gpu_block_id_tensor = torch::Tensor(),
      const torch::Tensor &indexer_cpu_block_id_tensor = torch::Tensor(),
      const int64_t indexer_cpu_block_stride_in_bytes = 0,
      const int64_t indexer_cpu_layer_stride_in_bytes = 0,
      const int64_t indexer_h2d_cpu_kv_stride_in_bytes = 0,
      const int64_t indexer_h2d_cpu_layer_stride_in_bytes = 0,
      const torch::Tensor &indexer_ssd_block_ids = torch::Tensor(),
      const torch::Tensor &indexer_cpu_block_ids_d2h = torch::Tensor(),
      const int64_t indexer_ssd_layer_stride_in_bytes = 0,
      const int64_t indexer_ssd_kv_stride_in_bytes = 0,
      const int64_t indexer_cpu_chunk_size_in_bytes = 0,
      const int indexer_num_blocks_per_file = 0,
      const std::string &mla_d2h_mode = "sharded",
      const std::string &notify_mode = "hostfunc");

  void layerwise_transfer_multi_group(
      const torch::Tensor &ssd_block_ids,
      const torch::Tensor &cpu_block_ids_d2h,
      const int num_blocks_per_file, const int round_robin,
      const int num_threads_per_device,
      const torch::Tensor &gpu_block_id_tensor,
      const torch::Tensor &cpu_block_id_tensor,
      const int transfer_cta_num, const bool use_ce_transfer,
      const bool is_mla, const int counter_id = 0,
      const std::string &notify_mode = "hostfunc");

private:
  int num_gpus_;
  void **gpu_blocks_;
  void *cpu_blocks_;
  int num_tensors_per_gpu_;
  int64_t *gpu_kv_strides_in_bytes_;
  int64_t *gpu_block_strides_in_bytes_;
  int64_t *gpu_layer_strides_in_bytes_;
  int64_t *gpu_chunk_sizes_in_bytes_;

  BackendType backend_type_;
  std::vector<GTensorHandler> gpu_tensor_handlers_;

  std::vector<int> gpu_device_ids_;
  std::vector<cudaStream_t> streams_;
  std::vector<cudaEvent_t> events_;

  // SSD IO context
  bool enable_ssd_;
  std::unique_ptr<SSDIOCTX> ioctx_;

  // Indexer fuse support
  bool enable_indexer_ = false;
  void **indexer_gpu_blocks_ = nullptr;
  void *indexer_cpu_blocks_ = nullptr;
  int indexer_num_tensors_per_gpu_ = 0;
  int64_t *indexer_gpu_kv_strides_in_bytes_ = nullptr;
  int64_t *indexer_gpu_block_strides_in_bytes_ = nullptr;
  int64_t *indexer_gpu_layer_strides_in_bytes_ = nullptr;
  int64_t *indexer_gpu_chunk_sizes_in_bytes_ = nullptr;
  BackendType indexer_backend_type_ = BackendType::SGLANG;
  std::vector<GTensorHandler> indexer_gpu_tensor_handlers_;

  // Indexer SSD IO context
  bool enable_indexer_ssd_ = false;
  std::unique_ptr<SSDIOCTX> indexer_ioctx_;

  // Heterogeneous multi-group state. Legacy single-group/indexer mode keeps
  // these empty and uses the fields above unchanged.
  bool has_multi_group_ = false;
  std::vector<GroupParams> groups_;
  std::vector<std::vector<std::pair<int, int>>> layer_members_;
  int num_original_layers_ = 0;

  // Layer eventfds for notification
  // Shape: [tp_size, num_counters, num_layers]
  bool enable_eventfd_;
  int tp_size_;
  int num_counters_;
  int num_layers_;
  std::vector<int> layer_eventfds_;  // Flat array
  int current_counter_id_;  // Current counter set index for this transfer

  void layer_done_callback(int start_layer, int layers_this_batch,
                           int expected_count,
                           nvtxRangeId_t *current_range_id_ptr,
                           bool is_last_batch,
                           const char *next_range_name,
                           nvtxRangeId_t *next_range_id_ptr,
                           int callbacks_per_gpu = 1);

  // ===== Event polling notification =====
  // Used when notify_mode == "polling".
  // The polling thread queries per-batch CUDA events and writes eventfds
  // as soon as each batch completes on all GPUs.
  enum class NotifyMode { HOSTFUNC, POLLING };
  NotifyMode notify_mode_ = NotifyMode::HOSTFUNC;

  struct PollBatchInfo {
    int start_layer;
    int layers_this_batch;
    std::vector<cudaEvent_t> per_gpu_events;
    bool notified = false;
  };
  std::vector<PollBatchInfo> poll_batches_;
  std::atomic<bool> poll_stop_{false};
  std::atomic<int> poll_next_batch_{0};
  std::thread poll_thread_;

  void notify_layer_batch(int start_layer, int layers_this_batch);
  void event_polling_loop();
};

} // namespace flexkv
