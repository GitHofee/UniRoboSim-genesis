"""Native Genesis lifecycle with lazy optional dependency loading."""
from __future__ import annotations
import importlib.metadata
import importlib.util
import itertools
import os
from dataclasses import replace
from unirobosim import (CapabilityNegotiationError, LifecycleError, ProbeReport,
    ProviderSelectionError, SessionState, ValidationError, WorldBuildError, WorldSpec)
from .config import ENGINE_VERSION, GenesisAdapterConfig
from .descriptor import descriptor_for_config

_IDS = itertools.count(1)

class GenesisProvider:
    def __init__(self, config=None):
        self.config = GenesisAdapterConfig() if config is None else config
        if not isinstance(self.config, GenesisAdapterConfig):
            raise ValidationError("config must be GenesisAdapterConfig", operation="genesis.provider")
        if self.config.asset_profile is None and os.environ.get("UNIROBOSIM_GENESIS_ASSET_PROFILE"):
            self.config = replace(self.config, asset_profile=os.environ["UNIROBOSIM_GENESIS_ASSET_PROFILE"])
        self.descriptor = descriptor_for_config(self.config)
        self._active = None

    def probe(self):
        try:
            version = importlib.metadata.version("genesis-world")
            found = importlib.util.find_spec("genesis") is not None
            available = found and version == ENGINE_VERSION
            reason = None if available else f"requires genesis-world {ENGINE_VERSION}, found {version}"
        except (ImportError, importlib.metadata.PackageNotFoundError) as exc:
            available, reason = False, str(exc)
        return ProbeReport(self.descriptor, available, reason)

    def open(self):
        if self._active is not None and self._active.state is not SessionState.CLOSED:
            raise LifecycleError("provider already has a live session", operation="genesis.provider.open")
        report = self.probe()
        if not report.available:
            raise ProviderSelectionError("Genesis runtime unavailable", operation="genesis.provider.open", details={"reason": report.reason})
        self._active = GenesisSession(self)
        return self._active

class GenesisSession:
    def __init__(self, provider):
        self._provider = provider
        self.config = provider.config
        self.descriptor = provider.descriptor
        self.session_id = f"genesis-session-{next(_IDS)}"
        self.state = SessionState.OPEN
        self._generation = 0
        self._world = None

    def negotiate(self, requirements):
        if self.state is SessionState.CLOSED:
            raise LifecycleError("session closed", operation="genesis.negotiate")
        return self.descriptor.capabilities.negotiate(tuple(requirements))

    def build(self, spec: WorldSpec, *, build_input=None):
        if self.state is not SessionState.OPEN:
            raise LifecycleError("build requires open session", operation="genesis.build")
        if not isinstance(spec, WorldSpec):
            raise ValidationError("build requires WorldSpec", operation="genesis.build")
        negotiation = self.negotiate(spec.requirements)
        if not negotiation.accepted:
            raise CapabilityNegotiationError("world requirements not satisfied", operation="genesis.build", details={"negotiation": negotiation.to_dict()})
        from .build_assets import snapshot_build_input
        lease = snapshot_build_input(spec, build_input)
        self._generation += 1
        runtime_acquired = False
        try:
            from .runtime import acquire
            acquire(self, seed=spec.metadata.to_dict().get("fastsim_initial_generation_seed", self.config.seed))
            runtime_acquired = True
            from .world import GenesisWorld
            self._world = GenesisWorld(self, spec, self._generation, lease)
        except BaseException as exc:
            if lease is not None:
                lease.close()
            if runtime_acquired:
                from .runtime import release
                release(self)
            if not isinstance(exc, Exception):
                raise
            raise WorldBuildError("Genesis world build failed", operation="genesis.build", cause=exc) from exc
        self.state = SessionState.READY
        return self._world

    def _world_closed(self, world):
        if self._world is world:
            self._world = None
            if self.state is not SessionState.CLOSED:
                self.state = SessionState.OPEN

    def close(self):
        if self.state is SessionState.CLOSED:
            return
        from .runtime import assert_owner, release
        assert_owner(self, "genesis.session.close")
        self.state = SessionState.CLOSED
        try:
            if self._world is not None:
                self._world.close()
        finally:
            try:
                release(self)
            finally:
                self._provider._active = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
