"""Import-safe entry point. Genesis is imported only by probe/open/build."""
from .config import GenesisAdapterConfig
from .provider import GenesisProvider

__version__ = "0.1.0"

def create_provider(config: GenesisAdapterConfig | None = None, *, launch_profile: str = "headless") -> GenesisProvider:
    if launch_profile not in {"visible", "headless", "headless-physics"}:
        raise ValueError("unsupported launch profile")
    if config is None:
        config = GenesisAdapterConfig(headless=launch_profile != "visible", enable_cameras=launch_profile != "headless-physics")
    return GenesisProvider(config)

__all__ = ["GenesisAdapterConfig", "GenesisProvider", "create_provider"]
