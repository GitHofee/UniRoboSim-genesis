"""A shared public Kinematic visual scene with camera-scoped mesh exclusions.

Only the selected visual geometry is temporarily made zero-area through the
public visual-vertex API. Its exact vertices are restored before returning.
"""
from pathlib import Path
import hashlib
import numpy as np
from unirobosim import EntityKind, LifecycleError, ValidationError
from .math import compose, numpy, rotation, wxyz
from .camera_projection import prepare_projection


def _visual_usd(world, entity, directory):
    from pxr import Sdf, Usd, UsdGeom, UsdPhysics, UsdUtils
    selection = world._asset_selections.get(entity.path)
    static = world._static_intent(entity)
    if selection and selection.get('visual_usd'):
        source = Path(selection['visual_usd']['file'])
    elif selection and not static:
        source = Path(selection['loads'][0]['file'])
    else:
        source = Path(world._usd_materialized[entity.path][2]['source'])
    source_stage = Usd.Stage.Open(str(source))
    layer = source_stage.Flatten()
    UsdUtils.ModifyAssetPaths(layer, lambda value: Sdf.ComputeAssetPathRelativeToLayer(source_stage.GetRootLayer(), value))
    stage = Usd.Stage.Open(layer)
    if static:
        # Removing a parent expires every descendant prim handle. Collect paths
        # before mutating the stage, then traverse its surviving prims afresh.
        joints = [prim.GetPath() for prim in stage.Traverse() if prim.IsA(UsdPhysics.Joint)]
        for path in sorted(joints, key=lambda value: str(value).count('/'), reverse=True):
            stage.RemovePrim(path)
        for prim in stage.Traverse():
            for api in (UsdPhysics.RigidBodyAPI, UsdPhysics.ArticulationRootAPI, UsdPhysics.MassAPI):
                if prim.HasAPI(api):
                    prim.RemoveAPI(api)
    nonvisual_removed = []
    for prim in stage.Traverse():
        if not prim.IsA(UsdGeom.Gprim):
            continue
        imageable = UsdGeom.Imageable(prim)
        generated_collision = entity.path in world._usd_robots and prim.GetName().startswith('collision_genesis_')
        if generated_collision or imageable.ComputeVisibility() == UsdGeom.Tokens.invisible or imageable.ComputePurpose() not in (UsdGeom.Tokens.default_, UsdGeom.Tokens.render):
            nonvisual_removed.append(str(prim.GetPath()))
    for path in sorted(nonvisual_removed, key=lambda value: value.count('/'), reverse=True):
        stage.RemovePrim(path)
    visible_prims = [str(p.GetPath()) for p in stage.Traverse() if p.IsA(UsdGeom.Gprim)]
    filename = directory / (hashlib.sha256(entity.path.value.encode()).hexdigest()[:12]+'.usdc')
    layer.Export(str(filename))
    opts = dict(selection['loads'][0].get('morph_kwargs', {})) if selection and not static else {}
    for name in ('coacd_options', 'prim_path'):
        opts.pop(name, None)
    opts.update(file=str(filename), collision=False, visualization=True, convexify=False,
        decimate=False, watertighten=None, align=False, default_armature=None,
        collision_mesh_prim_patterns=['(?!)'], visual_mesh_prim_patterns=['.*'])
    if static:
        opts['fixed'] = True
    return opts, {'source': str(source), 'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
        'source_default_prim': str(source_stage.GetDefaultPrim().GetPath()),
        'visual_usd': str(filename), 'visual_usd_sha256': hashlib.sha256(filename.read_bytes()).hexdigest(),
        'static_physics_removed': static, 'nonvisual_removed': nonvisual_removed, 'visible_prims': visible_prims}


def _exclusion_path(world, entity, relative_path, provenance):
    requested = provenance['source_default_prim'] + '/' + relative_path
    resolved = requested
    metadata = world._usd_robots.get(entity.path, {})
    for old, new in sorted(metadata.get('body_namespace_map', {}).items(), key=lambda pair: -len(pair[0])):
        if requested == old or requested.startswith(old + '/'):
            resolved = new + requested[len(old):]
            break
    if resolved not in provenance['visible_prims']:
        raise ValidationError('camera exclusion does not resolve to an exact visible geometry prim',
            operation='genesis.camera', details={'requested': requested, 'resolved': resolved})
    return requested, resolved


class _SharedVisualScene:
    def __init__(self, world):
        self.world = world
        self.scene = None
        self.bindings = []
        self.custom_entities = []
        self.cameras = {}
        self.projections = {}
        self.exclusions = {}
        self.provenance = {}
        camera_entities = [e for e in world._spec.entities if e.kind is EntityKind.CAMERA_SENSOR and world._cameras[e.path] is None]
        custom_paths = {x.entity_path for e in camera_entities for x in e.camera.render_exclusions}
        directory = world._derived_root / 'camera-shared'
        directory.mkdir()
        gs = world._gs
        native_entities, assets = {}, {}
        try:
            self.scene = gs.Scene(show_viewer=False, renderer=gs.renderers.Rasterizer())
            for entity in world._spec.entities:
                if entity.kind is EntityKind.CAMERA_SENSOR:
                    continue
                placement = dict(pos=entity.pose.position, quat=wxyz(entity.pose.orientation_xyzw))
                custom = entity.path in custom_paths
                if entity.box is not None:
                    if custom:
                        world._unsupported('mesh exclusions require a USD geometry prim', 'genesis.camera')
                    native = self.scene.add_entity(gs.morphs.Box(size=tuple(np.asarray(entity.box.dimensions_m)*entity.scale_xyz), collision=False, **placement),
                        material=gs.materials.Kinematic(), surface=gs.surfaces.Default(color=entity.box.color_rgba))
                elif entity.path in world._usd_materialized:
                    opts, provenance = _visual_usd(world, entity, directory)
                    opts.update(placement, scale=entity.scale_xyz[0], enable_custom_vverts=custom)
                    if world._spec.environments.count > 1 and (custom or not world._static_intent(entity)):
                        opts['batch_fixed_verts'] = True
                    native = self.scene.add_entity(gs.morphs.USD(**opts), material=gs.materials.Kinematic())
                    assets[entity.path] = provenance
                else:
                    world._unsupported('camera visual scenes currently require USD or explicit primitives', 'genesis.camera')
                native_entities[entity.path] = native
                if custom:
                    self.custom_entities.append(native)
                if not world._static_intent(entity):
                    if len(world._bodies[entity.path]) != 1:
                        world._unsupported('dynamic visual synchronization requires one native articulation/body', 'genesis.camera')
                    source = world._root(entity.path)
                    source_joints = {j.name: j for j in source.joints}
                    qs_source, qs_visual = [], []
                    for joint in native.joints:
                        other = source_joints.get(joint.name)
                        if other is None or joint.n_qs != other.n_qs:
                            raise ValidationError('visual and physical joint topology differ', operation='genesis.camera', details={'joint': joint.name})
                        qs_source.extend(range(other.q_start-source.q_start, other.q_end-source.q_start))
                        qs_visual.extend(range(joint.q_start-native.q_start, joint.q_end-native.q_start))
                    self.bindings.append((source, native, tuple(qs_source), tuple(qs_visual)))
            for entity in camera_entities:
                camera = entity.camera
                projection = prepare_projection(camera)
                self.projections[entity.path] = projection
                self.cameras[entity.path] = [self.scene.add_camera(res=projection.native_resolution,
                    fov=projection.vertical_fov_degrees, near=camera.near_plane_m, far=camera.far_plane_m,
                    GUI=False, env_idx=env) for env in range(world._spec.environments.count)]
                selected, exclusions = [], []
                for exclusion in camera.render_exclusions:
                    target = world._entities[exclusion.entity_path]
                    requested, resolved = _exclusion_path(world, target, exclusion.relative_prim_path, assets[target.path])
                    geoms = [g for g in native_entities[target.path].vgeoms if g.metadata.get('name') == resolved or g.metadata.get('name', '').startswith(resolved + '/')]
                    if not geoms:
                        raise ValidationError('camera exclusion has no native visual geometry', operation='genesis.camera', details={'prim': resolved})
                    selected.extend(geoms)
                    exclusions.append({'requested': requested, 'resolved': resolved, 'native_visual_geometries': [g.metadata['name'] for g in geoms]})
                self.exclusions[entity.path] = list(dict.fromkeys(selected))
                self.provenance[entity.path] = {'camera': entity.path.value, 'method': 'shared-public-kinematic-usd-scene',
                    'assets': list(assets.values()), 'exclusions': exclusions,
                    'exclusion_method': 'public-visual-vertex-zero-area-with-finally-restoration',
                    'native_resolution': projection.native_resolution, 'output_resolution': (camera.width_px, camera.height_px),
                    'projection_roundtrip_error_px': projection.max_roundtrip_error_px,
                    'rgb_sampling': 'bilinear' if projection.map_x is not None else 'native'}
            from .runtime import scene_build_options
            with scene_build_options(world._session.config, world._spec.environments.count):
                self.scene.build(n_envs=world._spec.environments.count)
            if self.scene.rigid_solver.n_entities:
                raise ValidationError('camera visual scene unexpectedly contains rigid dynamics', operation='genesis.camera')
            self.sync()
            for provenance in self.provenance.values():
                provenance.update(physical_entities=0, visual_geometries=sum(b.n_vgeoms for b in self.scene.entities),
                    shared_camera_count=len(camera_entities))
        except BaseException:
            self.close()
            raise

    def sync(self):
        for source, visual, source_qs, visual_qs in self.bindings:
            visual.set_pos(numpy(source.get_pos()), zero_velocity=False)
            visual.set_quat(numpy(source.get_quat()), zero_velocity=False)
            if source_qs:
                visual.set_qpos(numpy(source.get_qpos(source_qs)), qs_idx_local=visual_qs, zero_velocity=False)
        # The public custom-vertex FK API reads visual geom transforms. Refresh
        # those through the public visualizer before rebuilding custom vertices.
        self.scene.visualizer.update_visual_states(force_render=True)
        for entity in self.custom_entities:
            entity.set_vverts(None)

    def close(self):
        if self.scene is not None:
            self.scene.destroy()
            self.scene = None


class CameraScene:
    """One camera view of the world's shared visual scene."""
    def __init__(self, world, camera_entity):
        self.world, self.entity = world, camera_entity
        shared = getattr(world, '_shared_visual_scene', None)
        if shared is None:
            shared = _SharedVisualScene(world)
            world._shared_visual_scene = shared
        self.shared = shared
        self.bindings = shared.bindings
        self.projection = shared.projections[camera_entity.path]
        self.provenance = shared.provenance[camera_entity.path]
        self.cameras = shared.cameras[camera_entity.path]
        self.exclusions = shared.exclusions[camera_entity.path]

    @property
    def scene(self):
        return self.shared.scene

    def _render_native(self, camera, *, rgb, depth):
        return camera.render(rgb=rgb, depth=depth, force_render=True)

    def render(self, env, *, rgb, depth):
        self.shared.sync()
        entity = self.entity
        pose = entity.pose
        if entity.mount is not None:
            parent = self.world._link_state(entity.mount.parent_path, entity.mount.parent_link_name, env)[0]
            pose = compose(parent, pose)
        r = rotation(pose.orientation_xyzw)
        camera = self.cameras[env]
        camera.set_pose(pos=pose.position, lookat=tuple(np.asarray(pose.position)-r[:, 2]), up=tuple(r[:, 1]))
        saved = []
        try:
            for geom in self.exclusions:
                vertices = numpy(geom.get_vverts(envs_idx=[env])).copy()
                saved.append((geom, vertices))
                if vertices.shape[-2]:
                    geom.set_vverts(np.broadcast_to(vertices[:, :1, :], vertices.shape).copy(), envs_idx=[env])
            rgb_image, depth_image, *_ = self._render_native(camera, rgb=rgb, depth=depth)
            return (self.projection.sample(rgb_image) if rgb else None,
                self.projection.sample(depth_image, depth=True) if depth else None)
        finally:
            try:
                for geom, vertices in reversed(saved):
                    geom.set_vverts(vertices, envs_idx=[env])
            except Exception as error:
                self.world.close()
                raise LifecycleError('camera visual restoration failed; world closed', operation='genesis.camera') from error

    def close(self):
        self.shared.close()
