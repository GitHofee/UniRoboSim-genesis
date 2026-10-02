# UniRoboSim Genesis adapter

Provider `genesis-world.genesis` 0.1.0 implements UniRoboSim 0.10.9 contract
v0alpha6 using **unmodified official genesis-world==1.4.2**. Import is optional
dependency safe. Install `unirobosim-genesis[engine]`; trimesh is pinned to 4.12.2.
Experimental Genesis forks and the former USD-to-MJCF route are not supported.

`create_provider(config=None, *, launch_profile="headless")` accepts `visible`,
`headless`, and `headless-physics`. WorldSpec owns gravity, outer timestep, internal
substeps and environments. FastSim's Genesis balanced profile uses eight internal
substeps at the requested outer frequency. Arrays are batch first, SI, XYZW.
RigidBodyState is the physical root-link pose/twist; scene/planning entity poses
retain the separate asset-root offset.

For adapter-owned CUDA runtimes, `performance_mode=None` selects the official
`gs.init(performance_mode=True)` storage option; CPU keeps the native default.
Set the config field to `False` to opt out. It only applies when initializing a
runtime: external runtimes retain their caller's storage, and concurrent adapter
sessions cannot request conflicting explicit initialization modes.
`parallelization_level=None` selects the official `GS_PARA_LEVEL=2` build option
for CUDA, including a single environment; CPU retains the native default.
Explicit config values 0, 1, or 2 override this device policy. An existing
`GS_PARA_LEVEL` environment value always takes precedence and is never modified.
The adapter scopes any temporary value to `Scene.build` and removes it even on
failure. This environment-based official option is process-wide while building;
external concurrent scene builders must coordinate their process environment.
Effective storage, ownership and parallelization provenance are recorded under
`asset_provenance.physics.execution`. These options change execution/storage,
not timestep, substeps, gains, geometry or environment count. Floating-point
reduction order can differ. These options do not change `device` (default `cuda`);
callers can explicitly select `cpu` in config.

Planning joint limits are read back from the native runtime and use the same
absolute-reference conversion as joint state. This retains native representable
endpoints, which can differ from source doubles by one float32 rounding step.
Authoring bounds, native bounds and reference offsets remain separately recorded
in `asset_provenance.physics.planning_joint_limit_readback`.

## Native USD asset profiles

USD inputs load only through public `gs.morphs.USD` and `Scene.add_entity` or
`Scene.add_stage`. Runtime performs no format conversion. A public
`GenesisAdapterConfig(asset_profile="/absolute/profile.json")` selects explicit
independently repaired USD assets. `UNIROBOSIM_GENESIS_ASSET_PROFILE` is an optional
fallback when the config field is absent. Original cloud/Isaac assets are never
modified. The profile is an installed asset selection, not an automatic repair
system for arbitrary USD files.

Schema `unirobosim-genesis-asset-profile/v1` has `entries`, keyed uniquely by the
original `source_sha256`. Each entry pins `source_files` (all composed source USD
layers), `pinned_files` (all prepared USD layers and repair manifests), and `loads`.
A load specifies `file`, `sha256`, `mode` (`add_entity` or `add_stage`),
`morph_kwargs`, and `material_kwargs`. Profile settings cannot override entity
position, orientation or scale. A changed dependency fails before import.
Unknown hashes never inherit another asset's repair. Unprofiled articulated USD
is rejected because native reference/drive/closure semantics need validation.
`world.asset_provenance` exposes profile digest, original-to-prepared mappings,
source/layer identities and effective physics choices. Core BuildReport remains
unchanged.

The production G2 profile uses independent USD repairs: parallel rigid-body
organization retaining world frames, explicit collision/visual roles, constant
material interface resolution, and two documented physically realizable inertia
repairs. Eight original spherical loop constraints remain represented by eight
auxiliary spherical joints plus public welds. Each auxiliary shares exactly half
its partner's original mass/inertia at the same COM/frame; welded composite mass
properties are unchanged apart from the two explicit inertia repairs. A fixed
zero-mass support preserves the original base frame. Auxiliary bodies/DOFs are
internal; the original 43 controlled axes, 65 logical bodies, 78 recording frames
and eight planning closures remain public. Reset restores permanent welds.

Author angular drive gains are converted to SI; armature, passive damping,
effort limits and equal static/dynamic joint friction effort are preserved.
Acceleration drives use explicit `articulated_mass_normalized_pd`: public native
cached mass matrices, with the known implicit damping terms removed, determine
force gains via `(M^-1)[j,j]`. This preserves authored gains/type and force caps,
but is an approximation: the mass matrix comes from the previous last internal
substep, and effective gains are held over an outer tick. One force-mode warmup
is discarded with public reset before tick zero; initial q/v and clock are
verified. This is not PhysX constraint-iteration drive equivalence or a custom
integrator. Explicit force commands bypass native PD.

Static USD keeps independent authored collider owners. Only a profile-marked
static convex group receives a smaller SDF grid, after public native inspection
proves every movable collider convex; those static grids are unused by eligible
rigid contact pairs. Nonconvex grids retain native defaults. Dense SDF allocation
is checked against integer and minimum CUDA memory limits before scene build.

## Lifecycle and limits

Attachments use public dynamic weld registration with capacity checks. Public
constraint parameter setters harden welds and restore all non-weld geometry,
joint and equality parameters exactly. Native calls remain on their owning
thread, and the final owned runtime teardown restores Torch defaults changed by
Genesis. Borrowed runtimes stay caller-owned. Original uint32 seeds are retained;
native initialization uses documented `seed & 0x7fffffff`, host RNGs retain uint32.

Implemented interfaces include state/control, selected kinematics, contacts,
reset, scene commands, planning geometry leases and closure facts. Headless USD
cameras use the explicit native rendering path described below; the profile USD
interactive viewer remains unsupported. Multi-environment closed USD robots
and the same weld link pair in multiple environments are rejected because of an
official runtime limitation. Authored maximum joint velocity is not natively
enforced. Native material friction/cooking differ from PhysX; asset provenance
records the mapping. Unsupported compliance, embedded bindings and nonuniform
articulation scaling fail explicitly.

Current USD-only evidence is in `20260930-genesis-native-usd`: G2 CUDA 600-step
hold and moving-target tests, public mass-query energy verification, and seven
movable entities with 78-frame/43-axis/planning/reset checks. Full production
success requires the official FastSim receipt and finalized recording; isolated
robot acceptance does not establish task success. Older MJCF/fork tests are
historical evidence, not acceptance of this route.

## Render-state replay and cameras

`render.state.apply@1` writes selected joint positions/velocities and physical
root poses/twists through public Genesis entity setters. It validates the entire
frame first, restores affected native state if a later setter fails, advances a
render revision and leaves the physics tick unchanged. A failed rollback closes
the world. Original absolute joint reference offsets are applied once, and the
G2 auxiliary spherical orientations follow their original partner links without
stepping physics. Fixed roots cannot carry a nonzero twist. Multi-environment
articulations use the public `batch_fixed_verts=True` option so their fixed bases
can be positioned per environment; static scene geometry is unchanged.

Headless USD cameras share one public `gs.materials.Kinematic` visual scene,
so house geometry, materials and GPU textures are loaded once. This scene has no
rigid dynamics or SDF allocation. It reads the original/prepared visual USD.
For an exact camera exclusion, the public visual-vertex API temporarily makes
only that prim's native visual triangles zero-area at an existing vertex. A
`finally` block restores every vertex exactly before another camera can render;
restoration failure closes the world. Public visual-state refresh followed by
`set_vverts(None)` keeps the custom buffer at current FK after each state update.
Invisible/non-render-purpose geometry and generated collision-only copies are
omitted from the visual USD to prevent native collision-to-visual fallback.
An asset profile may provide `visual_usd: {file, sha256}`; that file and its USD
composition layers must also occur in `pinned_files`. Original source assets and
physical loads are not changed. Source materials unsupported by the official
USD renderer still require explicit asset compatibility preparation; pixel
appearance parity with another engine is not claimed.

The camera pose convention is OpenGL (-Z forward, +Y up). Calibrated pinhole
cameras invert all requested OpenCV rational8 pixel rays, render a centered
native pinhole with enough overscan, and resample RGB bilinearly to the exact
requested resolution. Depth uses nearest sampling. This includes asymmetric
fx/fy and principal-point offsets. Noninvertible distortion or overscan beyond
8192 pixels per dimension is rejected. The image resampling stage is recorded
in provenance and is not claimed to be pixel-identical to Isaac rendering.

The public rasterizer supports only GI=false/AO=false. `configure_render_quality`
rejects either flag set to true, without silently changing a request. On a
headless Linux host set `PYOPENGL_PLATFORM=egl` in the process environment before
OpenGL is imported. No system graphics library or Genesis package patch is used.
Original three-camera replay acceptance, including original visual surfaces,
source-frame correspondence and saved images, remains a separate full-run gate.
