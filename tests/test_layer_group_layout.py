import unittest

import torch

from flexkv.common.config import (
    CacheConfig,
    GLOBAL_CONFIG_FROM_ENV,
    LayerGroupSpec,
    ModelConfig,
    RankInfo,
    UserConfig,
    block_size_in_bytes_for_layer_groups,
    build_layer_member_map,
    recompute_cache_block_counts,
    update_default_config_from_user_config,
)
from flexkv.common.storage import KVCacheLayout, KVCacheLayoutType


def _layer_groups():
    return [
        LayerGroupSpec(
            num_layers=4,
            num_kv_heads=2,
            head_size=8,
            layer_indices=[0, 1, 2, 3],
            dtype=torch.bfloat16,
        ),
        LayerGroupSpec(
            num_layers=2,
            num_kv_heads=1,
            head_size=4,
            layer_indices=[1, 3],
            dtype=torch.uint8,
            compress_ratio=4,
        ),
    ]


class LayerGroupConfigTest(unittest.TestCase):
    def test_member_map_supports_overlap_and_gaps(self):
        member_map = build_layer_member_map(_layer_groups(), 5)

        self.assertEqual(member_map.members_of(0), ((0, 0),))
        self.assertEqual(member_map.members_of(1), ((0, 1), (1, 0)))
        self.assertEqual(member_map.members_of(3), ((0, 3), (1, 1)))
        self.assertEqual(member_map.members_of(4), ())
        self.assertEqual(member_map.total_members, 6)

    def test_model_config_validates_and_invalidates_late_member_map(self):
        model_config = ModelConfig(num_layers=4, layer_groups=_layer_groups())
        model_config.freeze()
        self.assertEqual(model_config.layer_member_map.total_members, 6)

        replacement = [_layer_groups()[0]]
        model_config.layer_groups = replacement
        self.assertEqual(model_config.layer_member_map.total_members, 4)

    def test_invalid_compression_is_rejected(self):
        groups = _layer_groups()
        groups[1].compress_ratio = 0
        with self.assertRaisesRegex(ValueError, "compress_ratio"):
            ModelConfig(num_layers=4, layer_groups=groups).freeze()

    def test_invalid_late_assignment_restores_previous_groups(self):
        original = _layer_groups()
        model_config = ModelConfig(num_layers=4, layer_groups=original)
        model_config.freeze()

        invalid = _layer_groups()
        invalid[0].layer_indices = [0, 0, 2, 3]
        with self.assertRaisesRegex(ValueError, "duplicates"):
            model_config.layer_groups = invalid

        self.assertIs(model_config.layer_groups, original)
        self.assertEqual(model_config.layer_member_map.total_members, 6)


class LayerGroupStorageTest(unittest.TestCase):
    def test_multi_group_layout_uses_exact_byte_shape(self):
        groups = _layer_groups()
        layout = KVCacheLayout(
            type=KVCacheLayoutType.BLOCKFIRST,
            num_layer=4,
            num_block=3,
            tokens_per_block=128,
            num_head=1,
            head_size=1,
            is_mla=False,
            layer_groups=groups,
            tp_size=2,
        )

        main_bytes = 4 * 2 * 128 * 2 * 8 * torch.bfloat16.itemsize
        indexer_bytes = 2 * 2 * (128 // 4) * 1 * 4 * torch.uint8.itemsize
        expected_block_bytes = 2 * (main_bytes + indexer_bytes)

        self.assertEqual(layout.kv_shape, torch.Size([3, expected_block_bytes]))
        self.assertEqual(layout.get_block_stride(), expected_block_bytes)
        self.assertEqual(layout.div_block(3).kv_shape,
                         torch.Size([1, expected_block_bytes]))

        strides = layout.get_group_strides()
        self.assertEqual(strides[0]["offset_bytes"], 0)
        self.assertEqual(strides[1]["offset_bytes"], main_bytes)
        self.assertEqual(strides[1]["chunk_size"], (128 // 4) * 1 * 4)

    def test_layerblock_legacy_layout_is_preserved(self):
        layout = KVCacheLayout(
            type=KVCacheLayoutType.LAYERBLOCK,
            num_layer=2,
            num_block=3,
            tokens_per_block=16,
            num_head=4,
            head_size=8,
            is_mla=False,
        )
        self.assertEqual(layout.kv_shape, torch.Size([2, 3, 2, 16, 4, 8]))

    def test_non_divisible_compression_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "does not divide"):
            KVCacheLayout(
                type=KVCacheLayoutType.BLOCKFIRST,
                num_layer=4,
                num_block=1,
                tokens_per_block=10,
                num_head=1,
                head_size=1,
                is_mla=False,
                layer_groups=_layer_groups(),
                tp_size=2,
            )

    def test_late_group_discovery_recomputes_capacity(self):
        model_config = ModelConfig(
            num_layers=4,
            num_kv_heads=2,
            head_size=8,
            dtype=torch.bfloat16,
            tp_size=2,
        )
        rank_info = RankInfo(model_config=model_config)
        cache_config = CacheConfig(tokens_per_block=128)
        user_config = UserConfig(cpu_cache_gb=1, ssd_cache_gb=0)

        update_default_config_from_user_config(
            rank_info, cache_config, user_config
        )
        uniform_blocks = cache_config.num_cpu_blocks

        model_config.layer_groups = _layer_groups()
        exact_bytes = block_size_in_bytes_for_layer_groups(
            model_config, cache_config, rank_info
        )
        changed = recompute_cache_block_counts(
            model_config, cache_config, rank_info
        )

        self.assertTrue(changed)
        self.assertNotEqual(cache_config.num_cpu_blocks, uniform_blocks)
        self.assertEqual(
            cache_config.num_cpu_blocks,
            int(1024 ** 3 / exact_bytes),
        )

    def test_late_capacity_recompute_preserves_mla_all_write_divisor(self):
        model_config = ModelConfig(
            num_layers=4,
            num_kv_heads=1,
            head_size=8,
            dtype=torch.bfloat16,
            use_mla=True,
            tp_size=2,
            layer_groups=[
                LayerGroupSpec(
                    num_layers=4,
                    num_kv_heads=1,
                    head_size=8,
                    layer_indices=[0, 1, 2, 3],
                    dtype=torch.bfloat16,
                )
            ],
        )
        cache_config = CacheConfig(tokens_per_block=128)
        cache_config._user_cpu_cache_gb = 1
        original_mode = GLOBAL_CONFIG_FROM_ENV.mla_d2h_mode
        try:
            GLOBAL_CONFIG_FROM_ENV.mla_d2h_mode = "all_write"
            exact_bytes = block_size_in_bytes_for_layer_groups(
                model_config, cache_config
            )
            recompute_cache_block_counts(model_config, cache_config)
        finally:
            GLOBAL_CONFIG_FROM_ENV.mla_d2h_mode = original_mode

        self.assertEqual(
            cache_config.num_cpu_blocks,
            int(1024 ** 3 / exact_bytes) // 2,
        )
        single_copy_bytes = (
            4 * 128 * 1 * 8 * torch.bfloat16.itemsize
        )
        self.assertEqual(exact_bytes, single_copy_bytes)


if __name__ == "__main__":
    unittest.main()
