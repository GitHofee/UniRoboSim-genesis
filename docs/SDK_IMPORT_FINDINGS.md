# Genesis 1.4.2 USD import findings

Scope: the unchanged prepared production 1a USD environment, G2 robot, and objects.
These are import correctness findings, not a claim of production acceptance.
Source USD files have remained unchanged. Logs are retained in the workspace
integration case `20260929-genesis-backend`.

1. `usd_geometry.parse_prim_geoms` decides geometry roles using name patterns and
   render visibility. It does not derive collider participation from CollisionAPI
   and `physics:collisionEnabled`. A visually hidden physical collider can disappear,
   while an unmarked visual/semantic helper can become a collider. The production
   source must be interpreted using its authored physics API; legacy name fallback
   should apply only when physics roles are not authored.
2. The same function splits a mesh into material subsets, then uses those visual
   pieces as collision meshes. An authored convexHull on a mesh with a one-triangle
   material subset fails QHull's minimum-vertex requirement. Collision cooking must
   operate on the complete authored mesh independently from visual subsets.
3. `find_rigid_bodies_in_range` prunes descendants of a rigid body. G2 authors nested
   rigid-body prims, so only the top body is discovered. The importer must collect
   all authored physical links while assigning child collider prims to their owning
   body; a collider child must not be counted as a separate physical link.
4. `resolve_rigid_body_link_path` treats a non-rigid wrapper as its sole descendant
   rigid body. G2's `world_anchor` uses the non-rigid root as an immobile frame, so
   this fallback produces a self-parent. Resolve an actual physical body/ancestor,
   or preserve the target's world-frame anchor and rotation.
5. G2 contains eight spherical closure joints with
   `physics:excludeFromArticulation=True`. The current importer puts them into the
   articulated tree. These must remain physical constraints: native Genesis CONNECT
   equalities can express both authored local anchor positions without adding tree
   DOFs. `physics:jointEnabled=False` also needs to be honored.
6. The generic `_parse_scene` cleanup removes a geometry-free fixed root as if it
   were a synthetic URDF/MJCF world link. USD parser links refer to authored prims;
   preserve their identity even when they have no geometry.
7. The no-joint branch uses the selected asset wrapper as its physical link even
   when a unique authored RigidBodyAPI child exists. That loses the body's authored
   mass, inertia, dynamic state and link frame. Select the physical body; reject
   several independent bodies in add_entity and direct callers to add_stage.


8. One-DOF USD joints retain authored, already posed link transforms but upstream
   sets their reference qpos to zero. Genesis exposes `get_qpos` as the absolute
   joint coordinate, while `get_dofs_position` and its PD targets are deflections
   from `qpos0`. The importer must derive the reference from the two authored
   local joint frames and body poses, and cross-check authored joint state where
   present. Importing a nonzero reference as zero repeats the angle when the
   portable initial absolute position is applied. The SDK patch preserves that
   reference; the adapter uses absolute qpos and converts only PD targets to
   deflection. Tests cover angular and linear references, explicit state or
   derived state, stage units and morph scaling, and reversed body-to-world joints.

The installed `1.4.2+fastsim.3` G2 CUDA diagnostic preserves all 65 physical links,
43 moving DOFs, eight native CONNECT constraints and 78 requested frames. Public
initial q matches the input within 7.8e-8 radians. With hard equality parameters,
the maximum closure-anchor residual after three steps is 2.21 micrometres. This is
an isolated robot diagnostic; full production acceptance is a separate gate.

The original G2 has 65 positive-definite authored tensors. Two (body_link4 and
head_link1) violate the principal-inertia triangle inequality. Original Isaac/
PhysX native readback retains those values; body_link4's native eigenvalues agree
within 1.6e-9 and remain unchanged after a physics step. The patched USD morph has
an explicit `enforce_inertia_triangle_inequality` boolean, default True. The adapter
sets False to preserve original simulation parameters and records warnings and
provenance. Numerical validation remains enabled. Neither source tensors nor
source USD are repaired or recomputed.

A temporary adapter-owned USD rewrite established the collision-role and
world-anchor diagnoses. It is retained only as diagnostic evidence; the adapter
now uses a versioned SDK importer correction. Static composite
intent remains an adapter responsibility because it comes from WorldSpec rather
than source USD semantics. No process-global monkeypatch or untracked installed
SDK edit is appropriate.

The original environment also reached a `trimesh 4.11.1` out-of-bounds face-index
failure during authored mesh simplification. Installing 4.12.2 solely in the
isolated Genesis environment cleared that failure. The adapter engine extra pins
4.12.2. This verifies the mitigation, not an exhaustive upstream root-cause analysis.

The generic Genesis nonuniform-scale warning needs per-type interpretation:
its Cube branch actually multiplies all three scale components. The warning alone
is not evidence of cube collider distortion. Exact shape/transform assertions
remain necessary for other primitive types and for source-vs-native comparison.

## Static collision cooking boundary

An exhaustive source-to-staging audit (including instance proxies) preserves all
3,312 enabled house colliders, geometry attributes, world transforms, units and
approximation tokens. All 3,477 source mesh prims have finite points/transforms and
valid face indices. The collision envelope is approximately 16.1 × 18.5 × 5.0 m.
Those checks rule out source numeric corruption and adapter scale inflation; they
do not by themselves prove native collision equivalence.

The adapter's declared-static composition removes private dynamics. With the
result represented by one native link, the generic Genesis postprocessor groups
same-option colliders and may fuse them before convex decomposition. Instrumented
native import shows that 479 separately authored convexDecomposition meshes are
cooked jointly into 33 hulls, spanning the house. This explains the entire count
reduction from 3,312 input collision prims to 2,866 house geoms. It is an interaction
between static normalization and SDK cooking, not evidence that the asset itself
is invalid. Pre-cook USD signatures cannot catch the resulting shape change.

The source correction makes USD cooking respect each complete authored collision
prim's MeshCollisionAPI recipe. A native regression verifies separate cooking,
source identity and an empty gap between distant concave meshes. Native URDF and
MJCF grouping remains unchanged. This correction is installed in SDK fastsim.4;
the full original eight-entity scene built and stepped three times successfully,
while official Mission/recording acceptance remains pending. Separate per-prim
cooking produces 2,946 hulls from the 479 decomposition prims. The 33 old witness
points lie outside that corrected decomposition hull union; this does not assert
that they are outside every other collider in the scene.

For the original scene, 33 guaranteed-inside output hull samples lie outside every
input AABB in the 479-prim decomposition group. All samples do overlap another
source collider AABB when checking the full 3,312-collider scene. Thus the evidence
proves cross-prim cooking expansion, but those samples do not alone prove globally
empty room volume. The separate two-prim fixture proves an empty-gap regression.

Separately, SDK3 eagerly allocates SDF grids for every geometry, including 2,258
house boxes already represented as primitives. Whole-scene logical storage exceeds
the gradient array's 32-bit indexing limit. The source SDF correction evaluates BOX
queries analytically, preserves virtual grid metadata used by contact margins,
and shares exactly identical mesh grids at their original resolution. Native CPU
and CUDA query/contact regressions pass; this storage correction does not repair
the collision-cooking boundary and is not production acceptance by itself.
