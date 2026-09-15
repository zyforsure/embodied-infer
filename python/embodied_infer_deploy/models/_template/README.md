# Model backend template

Create a sibling directory for each model family. Export a backend with:

```python
class MyBackend:
    @property
    def spec(self) -> ModelSpec: ...
    @property
    def metadata(self) -> dict: ...
    def infer(self, request: dict) -> BackendResult: ...
    def reset(self) -> None: ...

def create_backend(config: dict) -> MyBackend: ...
```

Register it lazily with `model_registry.register("name", "module:create_backend")`
or publish an `embodied_infer.models` Python entry point. Model directories
must not contain robot CAN logic or simulator-specific observation parsing.
