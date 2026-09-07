import unittest

import numpy as np

from embodied_infer_deploy.core import ModelBackend, ModelSpec
from embodied_infer_deploy.models import ModelRegistry, create_model, model_registry
from embodied_infer_deploy.models.turbovla.common import turbovla_spec


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


if __name__ == "__main__":
    unittest.main()
