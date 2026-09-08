import concurrent.futures
import time

import numpy as np

from embodied_infer_deploy.plugins import VisionBatchPlugin


def test_native_plugin_batches_requests_and_preserves_order():
    calls = []

    def encoder(batch):
        calls.append(batch.shape)
        return np.arange(batch.shape[0] * 2, dtype=np.float32).reshape(batch.shape[0], 2)

    plugin = VisionBatchPlugin.from_config(
        {"vision_batching": {"enabled": True, "max_batch_size": 4, "batch_wait_ms": 20}},
        encoder=encoder,
    )
    try:
        image = np.zeros((8, 8, 3), dtype=np.uint8)
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(plugin.encode, [image, image, image]) for _ in range(2)]
            outputs = [future.result(timeout=1) for future in futures]
        assert plugin.mode == "native"
        assert calls == [(2, 3, 3, 8, 8)]
        assert outputs[0].shape == (2,)
        assert outputs[1].shape == (2,)
    finally:
        plugin.close()


def test_enabled_plugin_without_split_encoder_is_safe_fallback():
    plugin = VisionBatchPlugin.from_config({"vision_batching": True})
    try:
        assert plugin.mode == "fallback"
        result = plugin.encode([np.zeros((3, 4, 4), dtype=np.uint8)])
        assert result.shape == (1, 3, 4, 4)
    finally:
        plugin.close()
