"""Ownership of Genesis' process-global runtime, without importing it eagerly.

All adapter native work belongs to the thread which first acquires the runtime.
Sessions share that runtime; the final session releases adapter-owned state on
its owner thread rather than leaving CUDA teardown to Python's main-thread atexit.
Externally initialized runtimes and external live scenes remain caller-owned.
"""
import importlib
import os
import threading
from contextlib import contextmanager

from unirobosim import LifecycleError

_lock = threading.RLock()
_owner = None
_sessions = set()
_owned = False
_torch_defaults = None
_performance_mode = None


def runtime_options(session, gs):
    """Report public initialization policy separately from actual storage."""
    return {
        "performance_mode_requested": session.config.performance_mode,
        "performance_mode_init": _performance_mode if _owned else None,
        "runtime_ownership": "adapter" if _owned else "external",
        "use_ndarray": bool(gs.use_ndarray),
        "initialization_policy": "cuda-performance-v1",
    }


@contextmanager
def scene_build_options(config, environment_count):
    """Scope the official GS_PARA_LEVEL option to this scene's build.

    A pre-existing environment value always wins. No SDK field is mutated, and
    the temporary value is removed on both success and failure. The same lock
    serializes adapter builds with runtime acquisition/teardown.
    """
    with _lock:
        existing = os.environ.get("GS_PARA_LEVEL")
        selected = config.parallelization_level
        if selected is None and config.device == "cuda":
            selected = 2
        source = "environment" if existing is not None else ("config" if config.parallelization_level is not None else "device-default")
        default = 0 if config.device == "cpu" else (1 if environment_count <= 1 else 2)
        value = existing if existing is not None else (str(selected) if selected is not None else None)
        try:
            effective = default if value is None else int(value)
        except ValueError as exc:
            raise LifecycleError("GS_PARA_LEVEL must be 0, 1 or 2", operation="genesis.scene.build") from exc
        if effective not in (0, 1, 2):
            raise LifecycleError("GS_PARA_LEVEL must be 0, 1 or 2", operation="genesis.scene.build")
        inserted = existing is None and value is not None
        if inserted:
            os.environ["GS_PARA_LEVEL"] = value
        try:
            yield {"parallelization_level": effective, "parallelization_source": source,
                "parallelization_requested": config.parallelization_level,
                "GS_PARA_LEVEL_existing": existing, "environment_count": environment_count}
        finally:
            # Do not overwrite a concurrent external change to the process env.
            if inserted and os.environ.get("GS_PARA_LEVEL") == value:
                os.environ.pop("GS_PARA_LEVEL", None)


def assert_owner(session, operation):
    if session.session_id in _sessions and threading.current_thread() is not _owner:
        raise LifecycleError("Genesis native operations must run on the runtime owner thread",
            operation=operation)


def acquire(session, *, seed=None):
    global _owner, _owned, _torch_defaults, _performance_mode
    with _lock:
        if _sessions:
            assert_owner(session, "genesis.runtime.acquire")
            # A new provider has no lease yet, so check the shared owner as well.
            if threading.current_thread() is not _owner:
                raise LifecycleError("Genesis runtime is owned by another thread",
                    operation="genesis.runtime.acquire")
        gs = importlib.import_module("genesis")
        backend = gs.cpu if session.config.device == "cpu" else gs.cuda
        if gs._initialized:
            if gs.backend != backend or gs.np_float().itemsize * 8 != int(session.config.precision):
                raise LifecycleError("Genesis process already initialized with a different device or precision",
                    operation="genesis.runtime.acquire")
            if _owned and session.config.performance_mode is not None and session.config.performance_mode != _performance_mode:
                raise LifecycleError("Genesis owned runtime already initialized with a different performance mode",
                    operation="genesis.runtime.acquire")
        elif _sessions:
            raise LifecycleError("Genesis runtime was destroyed while adapter sessions remain live",
                operation="genesis.runtime.acquire")
        else:
            import torch
            # Genesis changes the thread's Torch device context and the default
            # dtype. Remember absence of a context, not just its effective CPU
            # device: installing a new CPU context also retains a Python object
            # in C++ TLS, whose destructor can race interpreter shutdown.
            context = getattr(torch._GLOBAL_DEVICE_CONTEXT, "device_context", None)
            previous_dtype = torch.get_default_dtype()
            _torch_defaults = (context.device if context is not None else None,
                previous_dtype, previous_dtype != (torch.float32 if session.config.precision == "32" else torch.float64))
            try:
                performance_mode = session.config.performance_mode
                if performance_mode is None:
                    performance_mode = session.config.device == "cuda"
                gs.init(backend=backend, seed=None if seed is None else seed & 0x7fffffff, precision=session.config.precision,
                    logging_level=session.config.logging_level, performance_mode=performance_mode)
            except BaseException:
                torch.set_default_device(_torch_defaults[0])
                if _torch_defaults[2]:
                    torch.set_default_dtype(_torch_defaults[1])
                _torch_defaults = None
                raise
            _owned = True
            _performance_mode = performance_mode
        if not _sessions:
            _owner = threading.current_thread()
        _sessions.add(session.session_id)
        return gs


def release(session):
    global _owner, _owned, _torch_defaults, _performance_mode
    with _lock:
        if session.session_id not in _sessions:
            return
        assert_owner(session, "genesis.runtime.release")
        _sessions.remove(session.session_id)
        if _sessions:
            return
        gs = importlib.import_module("genesis")
        try:
            # Direct native callers may have added their own scene to an
            # adapter-started runtime. Never destroy that scene implicitly.
            external_scene_alive = any(ref() is not None for ref in gs._scene_registry)
            if _owned and gs._initialized and not external_scene_alive:
                try:
                    gs.destroy()
                finally:
                    import torch
                    torch.set_default_device(_torch_defaults[0])
                    if _torch_defaults[2]:
                        torch.set_default_dtype(_torch_defaults[1])
        finally:
            _owner = None
            _owned = False
            _torch_defaults = None
            _performance_mode = None
