import unittest

import numpy as np

from embodied_infer_deploy.core import ModelBackend, ModelSpec
from embodied_infer_deploy.models import ModelRegistry, create_model, model_registry
from embodied_infer_deploy.models.turbovla.common import turbovla_spec
import embodied_infer_deploy.models.pi05.remote as pi05_remote
import embodied_infer_deploy.models.pi05.cpp as pi05_cpp
import embodied_infer_deploy.models.pi05.tcp as pi05_tcp
from embodied_infer_deploy.models.pi05.hbm import Pi05HbmBackend


class ModelContractTests(unittest.TestCase):
    def test_builtin_models_are_declared_without_loading_accelerator_modules(self):
        self.assertEqual(
            set(model_registry.names()),
            {
                "mock",
                "turbovla-s600-hbm",
                "turbovla-s600-remote",
                "turbovla-s100-remote",
                "turbovla-tensorrt",
                "pi05-remote",
                "pi05-cpp",
                "pi05-hbm",
                "pi05-tcp",
            },
        )

    def test_mock_implements_model_backend_contract(self):
        backend = create_model("mock", {
            "raw_state_dim": 18,
            "state_dim": 14,
            "raw_action_dim": 18,
            "action_dim": 14,
            "action_horizon": 3,
        })
        self.assertIsInstance(backend, ModelBackend)
        self.assertEqual(backend.spec.model_state_dim, 14)
        self.assertEqual(backend.metadata["raw_state_dim"], 18)
        result = backend.infer({
            "state": np.zeros(14, dtype=np.float32),
            "images": {
                name: np.zeros((8, 8, 3), dtype=np.uint8)
                for name in backend.spec.camera_order
            },
        })
        self.assertEqual(result.actions.shape, (3, 14))

    def test_registry_is_lazy_and_rejects_duplicates(self):
        registry = ModelRegistry()
        registry.register(
            "local-mock",
            "embodied_infer_deploy.models.mock.backend:create_backend",
        )
        with self.assertRaises(ValueError):
            registry.register("local-mock", lambda config: None)
        self.assertEqual(registry.create("local-mock", {}).spec.backend, "mock")

    def test_model_spec_rejects_invalid_dimensions(self):
        with self.assertRaises(ValueError):
            ModelSpec(
                name="bad",
                backend="bad",
                raw_state_dim=0,
                model_state_dim=14,
                raw_action_dim=18,
                model_action_dim=14,
                action_horizon=50,
                camera_order=("head",),
            )

    def test_turbovla_distinguishes_raw_target_and_robot_command_dims(self):
        spec = turbovla_spec("test", {}, default_period_ns=1)
        self.assertEqual(spec.raw_state_dim, 18)
        self.assertEqual(spec.raw_action_dim, 18)
        self.assertEqual(spec.model_action_dim, 14)
        self.assertEqual(spec.metadata()["robot_command_dim"], 16)

    def test_pi05_remote_translates_openpi_observation_and_action_chunk(self):
        class FakeConnection:
            def __init__(self):
                self.sent = None
                self.messages = [pi05_remote._pack({"pi05": "test"})]

            def send(self, payload):
                self.sent = pi05_remote._unpack(payload)
                self.messages.append(pi05_remote._pack({
                    "actions": np.zeros((16, 14), dtype=np.float32),
                    "server_timing": {"infer_ms": 1.5},
                }))

            def recv(self, timeout=None):
                return self.messages.pop(0)

            def close(self):
                pass

        connection = FakeConnection()
        old_connect = pi05_remote.connect
        pi05_remote.connect = lambda *args, **kwargs: connection
        try:
            backend = create_model("pi05-remote", {})
            result = backend.infer({
                "instruction": "pick up the block",
                "state": np.zeros(14, dtype=np.float32),
                "images": {
                    "head": np.zeros((8, 8, 3), dtype=np.uint8),
                    "left_wrist": np.zeros((8, 8, 3), dtype=np.uint8),
                    "right_wrist": np.zeros((8, 8, 3), dtype=np.uint8),
                },
            })
            self.assertEqual(result.actions.shape, (16, 14))
            self.assertEqual(connection.sent["images"]["cam_high"].shape, (3, 8, 8))
            self.assertEqual(connection.sent["prompt"], "pick up the block")
        finally:
            pi05_remote.connect = old_connect

    def test_pi05_cpp_protocol_schema_and_contract(self):
        Image, Request, Response = pi05_cpp._proto_classes()
        req = Request(request_id=7, language_text="test")
        self.assertEqual(req.request_id, 7)
        self.assertEqual(Response().action_dim, 0)
        backend = pi05_cpp.create_backend({"address": "tcp://127.0.0.1:1"})
        self.assertEqual(backend.spec.backend, "pi05-cpp-zmq")
        backend.close()

    def test_pi05_hbm_reports_missing_artifacts(self):
        backend = Pi05HbmBackend({
            "vision_hbm": "/does/not/exist/vision.hbm",
            "llm_hbm": "/does/not/exist/llm.hbm",
            "expert_hbm": "/does/not/exist/expert.hbm",
        })
        self.assertFalse(backend.metadata["hbm_ready"])
        self.assertIn("missing HBM files", backend.metadata["hbm_error"])

    def test_pi05_tcp_backend_contract(self):
        backend = pi05_tcp.create_backend({"host": "127.0.0.1", "port": 1})
        self.assertEqual(backend.spec.backend, "pi05-tcp")
        self.assertEqual(backend.spec.raw_action_dim, 18)
        backend.close()


if __name__ == "__main__":
    unittest.main()
