import concurrent.futures
import time

import numpy as np

from embodied_infer_deploy.core import (
    BackendResult,
    BatchPolicy,
    InferenceRequest,
    InferenceScheduler,
    ScheduledBatch,
    SequentialModelRunner,
)
from embodied_infer_deploy.models.mock.backend import MockBackend
from embodied_infer_deploy.models.mock.runner import MockModelRunner


def _request(request_id, timeout_ms=300000):
    return {
        "request_id": request_id,
        "timeout_ms": timeout_ms,
        "state": np.zeros(14, dtype=np.float32),
        "images": {
            name: np.zeros((8, 8, 3), dtype=np.uint8)
            for name in ("head", "left_wrist", "right_wrist")
        },
    }


def test_scheduler_coalesces_requests_before_dispatching():
    runner = MockModelRunner({"action_horizon": 3})
    scheduler = InferenceScheduler(
        runner, policy=BatchPolicy(max_batch_size=4, batch_wait_ms=20)
    )
    try:
        futures = [scheduler.submit(_request(i)) for i in range(2)]
        results = [future.result(timeout=1.0) for future in futures]
        assert [result.actions.shape for result in results] == [(3, 14), (3, 14)]
        assert runner.batch_sizes == [2]
    finally:
        scheduler.close()


def test_scheduler_enforces_deadlines():
    runner = MockModelRunner({"action_horizon": 3})
    scheduler = InferenceScheduler(
        runner, policy=BatchPolicy(max_batch_size=4, batch_wait_ms=0)
    )
    try:
        future = scheduler.submit(
            _request(1), deadline=time.monotonic() - 0.001
        )
        with __import__("pytest").raises(TimeoutError):
            future.result(timeout=1.0)
        assert scheduler.stats["expired"] == 1
        assert runner.batch_sizes == []
    finally:
        scheduler.close()


def test_scheduler_reports_queue_overflow():
    runner = MockModelRunner({"action_horizon": 3})
    scheduler = InferenceScheduler(
        runner, policy=BatchPolicy(max_batch_size=1, batch_wait_ms=20, max_queue_size=1)
    )
    try:
        scheduler.submit(_request(1), deadline=time.monotonic() + 5.0)
        with __import__("pytest").raises(OverflowError):
            scheduler.submit(_request(2), deadline=time.monotonic() + 5.0)
    finally:
        scheduler.close()


def test_sequential_runner_wraps_legacy_backend():
    backend = MockBackend({"action_horizon": 4})
    runner = SequentialModelRunner(backend)
    scheduler = InferenceScheduler(
        runner, policy=BatchPolicy(max_batch_size=1, batch_wait_ms=0)
    )
    try:
        result = scheduler.submit(_request(7)).result(timeout=1.0)
        assert result.actions.shape == (4, 14)
    finally:
        scheduler.close()


def test_scheduler_preserves_result_order():
    class OrderRunner:
        def __init__(self):
            self.batch = None

        def execute(self, batch: ScheduledBatch):
            self.batch = batch
            return [
                BackendResult(actions=np.full((1, 1), request.request_id, dtype=np.float32))
                for request in batch.requests
            ]

    runner = OrderRunner()
    scheduler = InferenceScheduler(
        runner, policy=BatchPolicy(max_batch_size=4, batch_wait_ms=20)
    )
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            futures = [pool.submit(scheduler.submit, _request(i)) for i in range(4)]
            results = [future.result(timeout=1.0).result(timeout=1.0) for future in futures]
        assert [int(result.actions[0, 0]) for result in results] == [0, 1, 2, 3]
        assert len(runner.batch.requests) == 4
    finally:
        scheduler.close()
