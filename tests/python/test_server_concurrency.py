import threading

from embodied_infer_deploy.server import InferenceServer


def test_concurrent_backend_capability_is_opt_in():
    class Backend:
        supports_concurrent_infer = True

    server = InferenceServer(Backend())
    assert server.backend.supports_concurrent_infer is True
    assert isinstance(server._backend_lock, type(threading.Lock()))
