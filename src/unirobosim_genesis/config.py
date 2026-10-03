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

    background_color_linear: tuple[float,float,float] = (.04,.08,.12)
    ambient_light_linear: tuple[float,float,float] = (.1,.1,.1)
    directional_light_direction: tuple[float,float,float] = (-1.,-1.,-1.)
    directional_light_color_linear: tuple[float,float,float] = (1.,1.,1.)
    directional_light_intensity: float = 5.0
    shadows: bool = True
    fem_color_linear_rgba: tuple[float,float,float,float] = (.8,.32,.22,1.)
    max_fluid_color_groups: int = 64
    material_roughness: float = 1.0
    fem_mass_policy: str = "strict"
    fem_young_modulus_pa: float | None = None
    fem_poisson_ratio: float | None = None

    def __post_init__(self):
        for field,width in (("background_color_linear",3),("ambient_light_linear",3),("directional_light_color_linear",3),("fem_color_linear_rgba",4)):
            value=getattr(self,field)
            if len(value)!=width or any(type(v) not in (int,float) or not math.isfinite(v) or not 0<=v<=1 for v in value):
                raise ValidationError(f"{field} must be finite values in [0,1]",operation="genesis.config")
        if len(self.directional_light_direction)!=3 or not any(self.directional_light_direction) or any(not math.isfinite(v) for v in self.directional_light_direction):
            raise ValidationError("directional light direction must be finite and nonzero",operation="genesis.config")
        if type(self.directional_light_intensity) not in (int,float) or not math.isfinite(self.directional_light_intensity) or self.directional_light_intensity<0 or type(self.shadows) is not bool:
            raise ValidationError("invalid light intensity or shadow flag",operation="genesis.config")
        if type(self.max_fluid_color_groups) is not int or not 1 <= self.max_fluid_color_groups <= 256:
            raise ValidationError("max_fluid_color_groups must be between 1 and 256", operation="genesis.config")
        if type(self.material_roughness) not in (int,float) or not 0 <= self.material_roughness <= 1:
            raise ValidationError("material_roughness must be in [0,1]", operation="genesis.config")
        if self.fem_mass_policy not in {"strict", "preserve_total_mass"}:
            raise ValidationError("unsupported FEM mass policy", operation="genesis.config")
        if self.fem_young_modulus_pa is not None and (type(self.fem_young_modulus_pa) not in (int,float) or not math.isfinite(self.fem_young_modulus_pa) or self.fem_young_modulus_pa <= 0):
            raise ValidationError("FEM Young modulus must be positive", operation="genesis.config")
        if self.fem_poisson_ratio is not None and (type(self.fem_poisson_ratio) not in (int,float) or not math.isfinite(self.fem_poisson_ratio) or not -1 < self.fem_poisson_ratio < .5):
            raise ValidationError("FEM Poisson ratio must be between -1 and .5", operation="genesis.config")
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
