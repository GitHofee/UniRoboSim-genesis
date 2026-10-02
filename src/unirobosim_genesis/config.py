"""Explicit launch options; physical timestep/gravity come from WorldSpec."""
from dataclasses import dataclass
import math
from unirobosim import ValidationError

ENGINE_VERSION = "1.4.2"

@dataclass(frozen=True)
class GenesisAdapterConfig:
    device: str = "cuda"
    headless: bool = True
    enable_cameras: bool = True
    seed: int | None = None
    precision: str = "32"
    position_stiffness: float = 100.0
    position_damping: float = 8.0
    max_motor_effort: float = 50.0
    max_cached_commands: int = 4096
    logging_level: str = "warning"
    asset_profile: str | None = None
    # None selects the adapter's device default; explicit values opt out.
    performance_mode: bool | None = None
    parallelization_level: int | None = None

    def __post_init__(self):
        if self.asset_profile is not None and (not isinstance(self.asset_profile, str) or not self.asset_profile):
            raise ValidationError("asset_profile must be a nonempty path", operation="genesis.config")
        if self.device not in {"cpu", "cuda"}:
            raise ValidationError("device must be cpu or cuda", operation="genesis.config")
        if self.performance_mode is not None and type(self.performance_mode) is not bool:
            raise ValidationError("performance_mode must be boolean or None", operation="genesis.config")
        if self.parallelization_level is not None and (type(self.parallelization_level) is not int or self.parallelization_level not in (0, 1, 2)):
            raise ValidationError("parallelization_level must be 0, 1, 2 or None", operation="genesis.config")
        if type(self.headless) is not bool or type(self.enable_cameras) is not bool:
            raise ValidationError("launch flags must be boolean", operation="genesis.config")
        if self.seed is not None and (type(self.seed) is not int or not 0 <= self.seed < 2**32):
            raise ValidationError("seed must be a uint32 integer", operation="genesis.config")
        if self.precision not in {"32", "64"}:
            raise ValidationError("precision must be 32 or 64", operation="genesis.config")
        if type(self.max_cached_commands) is not int or self.max_cached_commands < 1:
            raise ValidationError("max_cached_commands must be positive", operation="genesis.config")
        for name in ("position_stiffness", "position_damping", "max_motor_effort"):
            value = getattr(self, name)
            if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                raise ValidationError(f"{name} must be finite and nonnegative", operation="genesis.config")
