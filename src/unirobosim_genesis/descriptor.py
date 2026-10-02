"""Capabilities implemented by the native Genesis adapter."""
from unirobosim import (
    CapabilityDeclaration, CapabilityId, CapabilitySet, FrozenMap, ProviderDescriptor,
    WORLD_SCHEMA_VERSION, PHYSICAL_WORLD_SCHEMA_VERSION, COMPOSITE_WORLD_SCHEMA_VERSION,
)
from .config import ENGINE_VERSION

_BASIC = (
    "state.rigid_body@1", "control.rigid_body.wrench@1", "contact.binary@1",
    "contact.net_normal_force@1", "state.articulation@1", "state.articulation.axis-units@1",
    "control.articulation.position@1", "control.articulation.position.axis-units@1",
    "control.articulation.velocity@1", "control.articulation.effort@1",
    "state.kinematics.selected@1", "world.multi-environment@1", "scene.snapshot@1",
    "scene.delta@1", "scene.command.pose@1", "scene.command.attachment@1",
    "scene.composite@1", "scene.composite.unbound-rigid-mode@1", "render.state.apply@1",
)

def descriptor_for_config(config):
    declarations = [CapabilityDeclaration(CapabilityId(name)) for name in _BASIC]
    declarations.extend((
        CapabilityDeclaration(CapabilityId("profile.core-robotics@1"), FrozenMap({
            "coordinate_system": "right-handed-z-up", "quaternion_order": "xyzw", "array_layout": "batch-first"})),
        CapabilityDeclaration(CapabilityId("asset.formats@1"), FrozenMap({
            "rigid_body": ["model/vnd.usd", "model/vnd.usda", "model/vnd.usdc", "model/vnd.urdf+xml"],
            "articulation": ["model/vnd.usd", "model/vnd.usda", "model/vnd.usdc", "model/vnd.urdf+xml", "model/vnd.mujoco.mjcf+xml"],
            "static_scene": ["model/vnd.usd", "model/vnd.usda", "model/vnd.usdc"]})),
        CapabilityDeclaration(CapabilityId("planning.scene@2"), FrozenMap({
            "authority_thread": "synchronous", "axis_convention": "right_handed_z_up",
            "geometry_read_limit_bytes": 64 * 1024 * 1024, "resource_layout": "catalog-pinned-v1",
            "single_representation_per_geometry": True, "representation_fallback": False})),
        CapabilityDeclaration(CapabilityId("scene.point_closures.read@1"), FrozenMap({
            "type":"native-connect-or-spherical-plus-public-weld", "anchors":"link-local-SI", "motion_solving":False})),
    ))
    declarations.append(CapabilityDeclaration(CapabilityId("render.quality@1"), FrozenMap({
        "renderer":"official-genesis-rasterizer", "supported_flags":[[False,False]],
    })))
    if config.enable_cameras:
        declarations.extend(CapabilityDeclaration(CapabilityId(name), FrozenMap({
            "profile_usd_worlds": True,
            "input_scope": "USD and primitives; uncalibrated URDF/MJCF cameras retain the native scene path",
            "projection": "rational8 inverse rays with bilinear RGB resampling",
            "exclusions": "shared kinematic USD scene with camera-scoped public visual-vertex exclusions",
        })) for name in (
            "sensor.camera@1", "sensor.camera.rgb@1", "sensor.camera.depth@1",
            "sensor.camera.calibrated@1", "sensor.camera.render-exclusions@1"))
    return ProviderDescriptor("genesis-world.genesis", "UniRoboSim Genesis", "0.1.0", "v0alpha6",
        CapabilitySet(tuple(declarations)),
        (WORLD_SCHEMA_VERSION, PHYSICAL_WORLD_SCHEMA_VERSION, COMPOSITE_WORLD_SCHEMA_VERSION),
        FrozenMap({
            "genesis-world": ENGINE_VERSION,
            "engine_distribution": "unmodified-official",
            "dynamics": "native-genesis",
            "usd_loading": "native-usd-with-explicit-hash-pinned-asset-profile",
            "usd_acceleration_drives": "articulated_mass_normalized_pd",
            "acceleration_gain_update": "outer_control_tick",
            "acceleration_response": "public_cached_mass_matrix_minus_implicit_damping",
            "limitations": [
                "Profile USD interactive viewer remains unsupported; headless image cameras use a shared kinematic visual scene.",
                "Rasterizer supports GI=false/AO=false only; calibrated RGB uses an explicit bilinear resampling stage.",
                "The same link pair cannot be welded in multiple environments concurrently.",
                "Authored joint maximum velocity is recorded but not natively enforced.",
                "Acceleration drive response uses the previous internal-step mass matrix and is frozen over outer ticks.",
                "Static surface friction and restitution use declared native-supported mappings.",
            ],
        }))
