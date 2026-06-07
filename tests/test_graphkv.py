import unittest

import torch

from graphkv import (
    KvQuantConfig,
    QuantizedKvCache,
    get_profile,
    llama_cpp_args,
    llama_cpp_command,
    quantize_kv_layer,
    vllm_cli_args,
    vllm_kwargs,
)
from graphkv.benchmark import run_synthetic_benchmark


class GraphKVPackageTests(unittest.TestCase):
    def test_profiles_return_expected_configs(self):
        int2 = get_profile("graphkv-int2-max")
        int4 = get_profile("graphkv-int4-balanced")

        self.assertEqual(int2.config.bits, 2)
        self.assertEqual(int2.config.quantizer, "affine")
        self.assertEqual(int4.config.bits, 4)
        self.assertEqual(int4.config.quantizer, "symmetric")

    def test_quantized_cache_round_trip(self):
        torch.manual_seed(1)
        past = tuple(
            (
                torch.randn(1, 2, 32, 16),
                torch.randn(1, 2, 32, 16),
            )
            for _ in range(2)
        )
        cfg = KvQuantConfig(bits=4, group_size=16, residual_length=4)
        cache = QuantizedKvCache.from_past_key_values(past, cfg)
        restored = cache.dequantize()

        self.assertGreater(cache.compression_ratio(), 1.0)
        self.assertEqual(restored[0][0].shape, past[0][0].shape)

    def test_channel_group_cap_for_small_head_dim(self):
        keys = torch.randn(1, 4, 64, 8)
        values = torch.randn(1, 4, 64, 8)
        cfg = KvQuantConfig(
            bits=2,
            group_size=128,
            residual_length=0,
            quantizer="affine",
            value_group_axis="channel",
        )
        layer = quantize_kv_layer(keys, values, cfg)

        self.assertEqual(layer.compressed_values.group_size, 8)
        self.assertGreater(layer.compression_ratio(), 4.0)

    def test_engine_recipes(self):
        self.assertEqual(
            vllm_kwargs("vllm-fp8-calibrated"),
            {"kv_cache_dtype": "fp8", "calculate_kv_scales": True},
        )
        self.assertEqual(vllm_cli_args("vllm-fp8"), ["--kv-cache-dtype", "fp8"])
        self.assertEqual(
            llama_cpp_args("llamacpp-q8"),
            ["--cache-type-k", "q8_0", "--cache-type-v", "q8_0"],
        )
        self.assertIn("--cache-type-k q4_0", llama_cpp_command("m.gguf", profile="llamacpp-q4"))

    def test_small_benchmark(self):
        result = run_synthetic_benchmark(
            profile_name="graphkv-int4-balanced",
            tokens=32,
            layers=2,
            heads=2,
            head_dim=16,
            dtype=torch.float32,
            device="cpu",
        )

        self.assertGreater(result.ratio, 1.0)
        self.assertGreater(result.key_cosine, 0.9)


if __name__ == "__main__":
    unittest.main()
