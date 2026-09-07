import time

import numpy as np

from embodied_infer_deploy.models.pi05.vision_batch import ViTContinuousBatcher


def test_vit_requests_are_coalesced_and_results_preserve_order():
    calls = []

    def encoder(batch):
        calls.append(batch.shape)
        return np.arange(batch.shape[0], dtype=np.float32)[:, None]

    with ViTContinuousBatcher(encoder, max_batch_size=3, batch_wait_ms=20) as batcher:
        futures = [batcher.submit(np.zeros((3, 3, 8, 8), np.uint8)) for _ in range(3)]
        results = [future.result(timeout=1.0) for future in futures]
    assert calls == [(3, 3, 3, 8, 8)]
    assert [float(result[0]) for result in results] == [0.0, 1.0, 2.0]


def test_vit_deadline_is_enforced_before_batch_execution():
    called = []

    with ViTContinuousBatcher(lambda batch: called.append(batch), batch_wait_ms=50) as batcher:
        future = batcher.submit(np.zeros((3, 3, 4, 4), np.float32), deadline=time.monotonic() - 1)
        try:
            future.result(timeout=1.0)
        except TimeoutError:
            pass
        else:
            raise AssertionError("expired ViT request unexpectedly succeeded")
    assert called == []
