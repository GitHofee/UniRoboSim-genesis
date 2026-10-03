"""UniRoboSim values backed by official Genesis rigid, FEM and SPH dynamics."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import hashlib
import importlib
import math
import shutil
import tempfile
from urllib.parse import unquote, urlparse
import numpy as np
from unirobosim import (
    ArrayValue, ArticulationState, BuildFingerprint, BuildReport, CameraModality,
    CommandMode, ContactState, DebugPublishReport, EntityHandle, EntityKind, EntityPath,
    EntityNotFoundError, KinematicState, LifecycleError, Pose, ResetResult,
    RigidBodyState, SceneCommandKind, SceneCommandResult, SceneCommandStatus,
    SceneDelta, SceneEntityState, SceneSnapshot, SensorChannel, SensorSample,
    StaleHandleError, Tick, UnsupportedCapabilityError, ValidationError, WorldState,
)
from .math import array, numpy, wxyz, xyzw, rotation, compose, inverse
from .planning import PlanningMixin
from .render_state import RenderStateMixin
from .soft import SoftMixin
from .appearance import AppearanceMixin


@dataclass
class Attachment:
    attachment_id: str
    environment_index: int
    parent_path: EntityPath
    child_path: EntityPath
    parent_link: object
    child_link: object
    parent_T_child: Pose


class GenesisWorld(AppearanceMixin, SoftMixin, RenderStateMixin, PlanningMixin):
    def __init__(self, session, spec, generation, asset_lease=None):
        self._session, self._spec, self._generation = session, spec, generation
        self._asset_lease = asset_lease
        self._state = WorldState.READY
        self._step_index = 0
        self._scene_sequence = 0
        self._reset_count = 0
        self._entities = {entity.path: entity for entity in spec.entities}
        self._bodies = {}
        self._soft_entities = {}
        self._fluid_colors = {}
        self._dofs = {}
        self._q_indices = {}
        self._position_references = {}
        self._links = {}
        self._named_frames = {}
        self._fixed_joint_names = {}
        self._attachments = {}
        self._scene_results = {}
        self._wrenches = {}
        self._cameras = {}
        self._provenance = []
        self._usd_materialized = {}
        self._usd_robots = {}
        self._asset_selections = {}
        self._mass_matrices = {}
        self._drive_modes = {}
        self._q_read_offsets = {}
        self._permanent_welds = []
        from .asset_profile import AssetProfile
        self._asset_profile = AssetProfile(session.config.asset_profile)
        self._derived_root = Path(tempfile.mkdtemp(prefix="unirobosim-genesis-derived-"))
        self._gs = importlib.import_module("genesis")
        self._scene = None
        self._planning_catalogs = {}
        self._planning_resources = {}
        self._planning_cache = {}
        self._attachment_revision = 1
        try:
            self._initialize_native()
            self._build_native()
        except BaseException:
            for camera in self._cameras.values():
                if hasattr(camera, "close"):camera.close()
            if self._scene is not None:
                self._scene.destroy()
            shutil.rmtree(self._derived_root, ignore_errors=True)
            raise
        descriptor = session.descriptor
        self._build_report = BuildReport(BuildFingerprint(descriptor.provider_id,
            descriptor.version, descriptor.contract_version, spec.digest,
            descriptor.capabilities.digest), spec.world_id, generation,
            spec.environments.count, len(spec.entities))

    def _initialize_native(self):
        gs = self._gs
        config = self._session.config
        seed = self._spec.metadata.to_dict().get("fastsim_initial_generation_seed", config.seed)
        # Genesis runtime is process-global; the same explicit seed is applied to
        # each world before asset processing as well as the first initialization.
        if seed is not None:
            import random
            import torch
            if type(seed) is not int or not 0 <= seed < 2**32:
                raise ValidationError("world seed must be uint32",operation="genesis.init")
            gs.set_random_seed(seed)
        policies=[]
        for entity in self._spec.entities:
            if entity.asset_uri is None:continue
            path=self._local_path(entity)
            provenance=None
            if path.suffix.lower() in {".usd",".usda",".usdc",".usdz"}:
                from .usd_assets import materialize_usd
                selected=self._asset_profile.select(path)
                if selected:
                    self._asset_selections[entity.path]=selected
                    if "robot_metadata" in selected:self._usd_robots[entity.path]=selected["robot_metadata"]
                prepared=selected["loads"][0]["file"] if selected else None
                self._usd_materialized[entity.path]=materialize_usd(entity,self._asset_lease,self._derived_root,source_override=prepared)
                provenance=self._usd_materialized[entity.path][2]
                if selected:provenance["asset_profile_selection"]={k:v for k,v in selected.items() if k!="robot_metadata"}
            static=entity.kind is EntityKind.STATIC_SCENE or (entity.kind is EntityKind.COMPOSITE_SCENE and
                entity.metadata.to_dict().get("composite_unbound_rigid_mode")=="static")
            if static:continue
            if entity.kind is EntityKind.ARTICULATION or (provenance and provenance["has_authored_articulation"]):
                authors=provenance["self_collision_authoring"] if provenance else []
                values={item["value"] for item in authors}
                if len(values)>1:
                    raise UnsupportedCapabilityError("conflicting authored articulation self-collision policies",
                        operation="genesis.physics",details={"entity":entity.path.value,"authoring":authors})
                policy={"entity":entity.path.value,"value":next(iter(values)) if values else True,
                    "source":"authored-consensus" if values else "genesis-default","authoring":authors}
                policies.append(policy)
                if provenance is not None:provenance["self_collision_policy"]=policy
        if len({p["value"] for p in policies})>1:
            raise UnsupportedCapabilityError("Genesis global self-collision option cannot preserve mixed articulation policies",
                operation="genesis.physics",details={"policies":policies})
        self._physics_provenance={"articulation_self_collision_policies":policies,
            "enable_self_collision":policies[0]["value"] if policies else True,"enable_neutral_collision":True,
            "seed":{"original_uint32":seed,"native_seed":getattr(gs,"SEED",None),"mapping":"uint32-low31-v1","host_seed":seed}}
        from .particle_colors import linear_to_srgb
        self._scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=self._spec.physics.time_step_seconds,
                substeps=self._spec.physics.substeps, gravity=self._spec.physics.gravity_m_s2),
            show_viewer=not config.headless,
            vis_options=gs.options.VisOptions(background_color=tuple(v**(1/2.2) for v in config.background_color_linear),
                ambient_light=config.ambient_light_linear,shadow=config.shadows,
                lights=[dict(type="directional",dir=config.directional_light_direction,
                    color=config.directional_light_color_linear,intensity=config.directional_light_intensity)]),
            **self._soft_options(),
            rigid_options=gs.options.RigidOptions(enable_self_collision=self._physics_provenance["enable_self_collision"],
                enable_neutral_collision=True,batch_dofs_info=True,
                max_dynamic_constraints=8+sum(len(m.get("closures",())) for m in self._usd_robots.values())),
        )

    def _local_path(self, entity):
        if self._asset_lease is not None:
            return self._asset_lease.selected_path(
                entity_id=entity.metadata.to_dict().get("fastsim_entity_id",entity.path.value),
                asset_uri=entity.asset_uri)
        parsed = urlparse(entity.asset_uri)
        if parsed.scheme not in ("", "file"):
            self._unsupported("only pinned local assets are accepted", "genesis.assets")
        return Path(unquote(parsed.path)).resolve()

    @staticmethod
    def _static_intent(entity):
        return entity.kind is EntityKind.STATIC_SCENE or (entity.kind is EntityKind.COMPOSITE_SCENE and
            entity.metadata.to_dict().get("composite_unbound_rigid_mode")=="static")

    def _build_native(self):
        from .particle_colors import linear_to_srgb
        gs = self._gs
        # Static materials may reduce grids only after every potentially moving
        # collision geometry has been inspected through the native public API.
        ordered=sorted(self._spec.entities,key=self._static_intent)
        for entity in ordered:
            if entity.kind is EntityKind.CAMERA_SENSOR:
                self._add_camera(entity)
                continue
            if entity.kind is EntityKind.PARTICLE_FLUID:
                self._add_fluid(entity)
                continue
            if entity.kind in {EntityKind.SURFACE_DEFORMABLE,EntityKind.VOLUME_DEFORMABLE}:
                self._add_deformable(entity)
                continue
            if entity.contact_compliance is not None:
                self._unsupported("explicit contact compliance is not implemented", "genesis.build")
            if entity.embedded_binding is not None:
                self._unsupported("embedded composite bindings are not implemented", "genesis.build")
            if entity.kind not in {EntityKind.RIGID_BODY, EntityKind.ARTICULATION,
                    EntityKind.STATIC_SCENE, EntityKind.COMPOSITE_SCENE}:
                self._unsupported(f"entity kind {entity.kind} is not implemented", "genesis.build")
            kwargs = dict(pos=entity.pose.position, quat=wxyz(entity.pose.orientation_xyzw))
            named = {}
            if entity.box is not None:
                box = entity.box
                if box.dynamic_friction != box.static_friction or box.restitution != 0:
                    self._unsupported("Genesis rigid contact does not preserve distinct static/dynamic friction or restitution", "genesis.build")
                morph = gs.morphs.Box(size=tuple(np.asarray(box.dimensions_m)*entity.scale_xyz), **kwargs)
                volume = float(np.prod(np.asarray(box.dimensions_m)*entity.scale_xyz))
                native = self._scene.add_entity(morph,
                    material=gs.materials.Rigid(rho=box.mass_kg/volume, friction=box.dynamic_friction),
                    surface=gs.surfaces.Default(color=linear_to_srgb(box.color_rgba),roughness=self._session.config.material_roughness,metallic=0.0))
                bodies = [native]
            elif entity.asset_uri is not None:
                path = self._local_path(entity)
                scale = entity.scale_xyz
                if len(set(scale)) != 1:
                    self._unsupported("non-uniform articulated/USD scaling requires a lossless asset conversion", "genesis.build")
                kwargs.update(file=str(path), scale=scale[0], convexify=False, decimate=False,
                    watertighten=None, align=False)
                if entity.kind is EntityKind.ARTICULATION and self._spec.environments.count>1:
                    kwargs["batch_fixed_verts"]=True
                if path.suffix.lower() in {".usd", ".usda", ".usdc", ".usdz"}:
                    path,named,provenance = self._usd_materialized[entity.path]
                    self._provenance.append(provenance)
                    self._fixed_joint_names[entity.path]=provenance["fixed_joint_names_by_link"]
                    bodies=self._load_native_usd(entity,path,kwargs,provenance)
                elif path.suffix.lower()==".urdf":
                    import xml.etree.ElementTree as ET
                    self._fixed_joint_names[entity.path]={
                        joint.find("child").attrib["link"]:joint.attrib["name"]
                        for joint in ET.parse(path).getroot().findall("joint") if joint.attrib.get("type")=="fixed"}
                    morph=gs.morphs.URDF(merge_fixed_links=False,
                        fixed=bool(entity.metadata.to_dict().get("fixed_base",True)), **kwargs)
                    bodies=[self._scene.add_entity(morph,surface=gs.surfaces.Default(roughness=self._session.config.material_roughness,metallic=0.0))]
                elif path.suffix.lower() in {".xml",".mjcf"}:
                    bodies=[self._scene.add_entity(gs.morphs.MJCF(**kwargs))]
                else:
                    self._unsupported(f"asset format {path.suffix}","genesis.build")
            else:
                self._unsupported("rigid/articulation requires an asset or explicit box geometry", "genesis.build")
            if not bodies:
                raise ValidationError("asset contains no native physical entity",operation="genesis.build",entity_path=entity.path.value)
            self._bodies[entity.path] = bodies
            links={}
            for body in bodies:
                for link in body.links:
                    # Names are authored names, never Genesis allocation indices.
                    name=link.name.rsplit("/",1)[-1]
                    if name in links:
                        if entity.kind not in {EntityKind.COMPOSITE_SCENE,EntityKind.STATIC_SCENE}:
                            raise ValidationError("ambiguous authored link name", operation="genesis.build",details={"name":name})
                    else:
                        links[name]=link
            self._links[entity.path]=links
            self._named_frames[entity.path]=named
            if entity.kind is EntityKind.ARTICULATION:
                body=bodies[0]
                joint_by_name={joint.name.rsplit("/",1)[-1]:joint for joint in body.joints}
                missing=set(entity.joint_names)-joint_by_name.keys()
                if missing:
                    raise ValidationError("authored joints missing from Genesis",operation="genesis.build",details={"missing":sorted(missing),"actual":list(joint_by_name)})
                native_joints=[joint_by_name[name] for name in entity.joint_names]
                if any(joint.n_dofs != 1 for joint in native_joints):
                    self._unsupported("declared articulation joints must each have one DOF", "genesis.build")
                self._dofs[entity.path]=tuple(joint.dofs_idx_local[0] for joint in native_joints)
                self._q_indices[entity.path]=tuple(joint.q_start-body.q_start for joint in native_joints)
                metadata=self._usd_robots.get(entity.path)
                self._position_references[entity.path]=np.asarray([metadata["joints"][name]["reference"] for name in entity.joint_names]) if metadata else np.asarray([joint.init_qpos[0] for joint in native_joints])
                self._q_read_offsets[entity.path]=self._position_references[entity.path] if metadata else np.zeros(len(native_joints))
        self._sdf_preflight()
        from .runtime import runtime_options, scene_build_options
        with scene_build_options(self._session.config, self._spec.environments.count) as build_options:
            self._scene.build(n_envs=self._spec.environments.count)
        self._physics_provenance["execution"] = {**runtime_options(self._session, self._gs), **build_options}
        for entity in self._spec.entities:
            if entity.kind is EntityKind.ARTICULATION:
                self._initialize_articulation(entity)
        if self._usd_robots:self._initialize_usd_mass_queries()
        self._initialize_soft()
        self._initial_state=self._scene.get_state()
        self._initial_mass_matrices={p:m.copy() for p,m in self._mass_matrices.items()}
        self._init_root_offsets={}
        for entity in self._spec.entities:
            if entity.path in self._bodies and len(self._bodies[entity.path])==1:
                self._init_root_offsets[entity.path]=compose(inverse(self._native_pose(self._root(entity.path),0)),entity.pose)
        from .camera_rendering import CameraScene
        for entity in self._spec.entities:
            if entity.kind is EntityKind.CAMERA_SENSOR and self._cameras[entity.path] is None:
                self._cameras[entity.path]=CameraScene(self,entity)
        self._physics_provenance["camera_rendering"]=[camera.provenance for camera in self._cameras.values() if isinstance(camera,CameraScene)]

    def _sdf_preflight(self):
        # Official Genesis 1.4.2 allocates one global packed dense SDF and a
        # three-component gradient when rigid nonconvex contacts activate it.
        geoms=[g for bodies in self._bodies.values() for body in bodies for link in body.links for g in link.geoms]
        active=any(not g.is_convex and g.type is not self._gs.GEOM_TYPE.TERRAIN for g in geoms)
        cells=sum(g.n_cells for g in geoms) if active else 0
        bytes_per_cell=4*(int(self._session.config.precision)//8)+4
        minimum_bytes=cells*bytes_per_cell
        info={"active":active,"geometry_count":len(geoms),"packed_cells":cells,
            "minimum_grid_bytes":minimum_bytes,"integer_gradient_limit":(2**31-1)//3}
        self._physics_provenance["sdf_preflight"]=info
        if cells>(2**31-1)//3:
            raise UnsupportedCapabilityError("native dense SDF gradient would exceed the int32 allocation limit",operation="genesis.sdf_preflight",details=info)
        if cells and self._session.config.device=="cuda":
            import torch
            free,total=torch.cuda.mem_get_info()
            info.update(cuda_free_bytes=free,cuda_total_bytes=total)
            if minimum_bytes>free*.85:
                raise UnsupportedCapabilityError("native dense SDF minimum allocation exceeds available CUDA budget",operation="genesis.sdf_preflight",details=info)

    def _load_native_usd(self,entity,path,kwargs,provenance):
        gs=self._gs;selection=self._asset_selections.get(entity.path)
        if not self._session.config.headless:
            self._unsupported("profile USD interactive viewer is not implemented; use headless camera rendering", "genesis.visuals")
        if not selection:
            if entity.kind is EntityKind.ARTICULATION:
                self._unsupported("USD articulation requires an explicit validated native USD asset profile", "genesis.usd")
            loads=[{"file":str(path),"mode":"add_stage" if entity.kind in {EntityKind.STATIC_SCENE,EntityKind.COMPOSITE_SCENE} else "add_entity",
                "morph_kwargs":{"fixed":True if self._static_intent(entity) else None,"visualization":False},"material_kwargs":{}}]
        else:loads=selection["loads"]
        bodies=[]
        moving=[g for p,items in self._bodies.items() if not self._static_intent(self._entities[p]) for b in items for g in b.geoms if not g.is_fixed]
        for load in loads:
            native_kwargs=dict(kwargs);native_kwargs.update(load.get("morph_kwargs",{}));native_kwargs["file"]=load["file"]
            materials=dict(load.get("material_kwargs",{}))
            if load.get("conditional_static_convex_cap") is not None:
                if not self._static_intent(entity):
                    raise ValidationError("static SDF profile applied to movable entity",operation="genesis.usd")
                all_convex=all(g.is_convex for g in moving)
                cap=load["conditional_static_convex_cap"]
                if all_convex:materials.update(sdf_min_res=cap,sdf_max_res=cap)
                provenance["static_grid_guard"]={"all_movable_convex":all_convex,"movable_geometries":len(moving),"selected_cap":cap if all_convex else None}
            morph=gs.morphs.USD(**native_kwargs);material=gs.materials.Rigid(**materials)
            added=list(self._scene.add_stage(morph,material=material)) if load["mode"]=="add_stage" else [self._scene.add_entity(morph,material=material)]
            if self._static_intent(entity) and not all(l.is_fixed for b in added for l in b.links):
                raise ValidationError("native USD static profile produced movable links",operation="genesis.usd")
            if (load.get("expected_all_convex") or (load.get("conditional_static_convex_cap") is not None and all_convex)) and not all(g.is_convex for b in added for g in b.geoms):
                raise ValidationError("native convex flags differ from pinned USD profile",operation="genesis.usd")
            bodies.extend(added)
        return bodies

    def _initialize_articulation(self, entity):
        body=self._root(entity.path);dofs=self._dofs[entity.path];metadata=self._usd_robots.get(entity.path)
        initial=np.tile(entity.initial_joint_positions,(self._spec.environments.count,1))
        body.set_qpos(initial-self._q_read_offsets[entity.path],qs_idx_local=self._q_indices[entity.path],zero_velocity=True)
        if metadata:
            if metadata.get("closures") and self._spec.environments.count!=1:
                self._unsupported("official Genesis repeated-link-pair welds do not support multi-environment closed USD robots", "genesis.usd")
            properties=[metadata["joints"][name] for name in entity.joint_names]
            for method,key in (("set_dofs_armature","armature"),("set_dofs_frictionloss","frictionloss"),("set_dofs_damping","viscous_damping")):
                getattr(body,method)(np.tile([p[key] for p in properties],(self._spec.environments.count,1)),dofs_idx_local=dofs)
            cap=np.tile([p["max_force"] for p in properties],(self._spec.environments.count,1));body.set_dofs_force_range(-cap,cap,dofs_idx_local=dofs)
            limits=np.asarray([p["limits"] for p in properties]);ref=self._q_read_offsets[entity.path]
            body.set_dofs_limit(np.tile(limits[:,0]-ref,(self._spec.environments.count,1)),np.tile(limits[:,1]-ref,(self._spec.environments.count,1)),dofs_idx_local=dofs)
            from scipy.spatial.transform import Rotation
            joints={j.name.rsplit("/",1)[-1]:j for j in body.joints}
            for closure in metadata.get("closures",()):
                a=self._links[entity.path][closure["auxiliary_name"]];b=self._links[entity.path][closure["original_body_name"]]
                quats=xyzw(body.get_links_quat())[0]
                delta=(Rotation.from_quat(quats[a.idx_local]).inv()*Rotation.from_quat(quats[b.idx_local])).as_quat()
                j=joints[closure["name"]];body.set_qpos(np.asarray([wxyz(delta)]),qs_idx_local=range(j.q_start-body.q_start,j.q_start-body.q_start+4))
                self._permanent_welds.append((a.idx,b.idx))
            self._restore_permanent_welds()
            self._drive_modes[entity.path]=np.full((self._spec.environments.count,body.n_dofs),False)
            self._drive_modes[entity.path][:,dofs]=True
            metadata["acceleration_drive_lowering"]={"method":"articulated_mass_normalized_pd","cadence":"outer_control_tick","approximation":True,
                "response_query":"public_cached_mass_matrix_minus_implicit_damping","matrix_time":"previous last internal substep before integration",
                "limitation":"mass response is lagged by one internal substep; effective gains held over outer tick; not PhysX drive equivalence"}
        else:
            body.set_dofs_kp(self._session.config.position_stiffness,dofs_idx_local=dofs)
            body.set_dofs_kv(self._session.config.position_damping,dofs_idx_local=dofs)
        if entity.joint_effort_limits:
            limits=np.asarray(entity.joint_effort_limits);body.set_dofs_force_range(-limits,limits,dofs_idx_local=dofs)
        body.control_dofs_position(self._position_targets(entity.path,initial),dofs_idx_local=dofs)

    def _restore_permanent_welds(self):
        if not self._permanent_welds:return
        solver=self._scene.sim.rigid_solver;current=solver.get_weld_constraints()
        pairs=set(zip(numpy(current["link_a"]).reshape(-1).tolist(),numpy(current["link_b"]).reshape(-1).tolist()))
        for a,b in self._permanent_welds:
            if (a,b) not in pairs and (b,a) not in pairs:solver.add_weld_constraint(a,b)
        actual=solver.get_weld_constraints();pairs=set(zip(numpy(actual["link_a"]).reshape(-1).tolist(),numpy(actual["link_b"]).reshape(-1).tolist()))
        if any((a,b) not in pairs and (b,a) not in pairs for a,b in self._permanent_welds):
            raise ValidationError("native permanent weld registration failed",operation="genesis.usd")
        from .constraints import harden_dynamic_welds
        harden_dynamic_welds(solver,self._spec.physics.time_step_seconds/self._spec.physics.substeps)

    def _capture_mass_matrix(self,path,active_modes):
        body=self._root(path);matrix=numpy(body.get_mass_mat()).astype(np.float64)
        damping=numpy(body.get_dofs_damping())+numpy(body.get_dofs_kv())*active_modes
        idx=np.arange(body.n_dofs);matrix[:,idx,idx]-=self._spec.physics.time_step_seconds/self._spec.physics.substeps*damping
        if not np.isfinite(matrix).all() or not np.allclose(matrix,matrix.swapaxes(-1,-2),atol=1e-7) or np.linalg.eigvalsh(matrix).min()<=0:
            raise ValidationError("native physical mass response is not finite symmetric positive definite",operation="genesis.drives")
        self._mass_matrices[path]=matrix

    def _initialize_usd_mass_queries(self):
        state=self._scene.get_state();before={p:(numpy(self._root(p).get_qpos()).copy(),numpy(self._root(p).get_dofs_velocity()).copy()) for p in self._usd_robots}
        for p in self._usd_robots:self._root(p).control_dofs_force(np.zeros((self._spec.environments.count,self._root(p).n_dofs)))
        self._scene.step()
        for p in self._usd_robots:self._capture_mass_matrix(p,False)
        self._scene.reset(state)
        for p,(q,v) in before.items():
            if not np.array_equal(q,numpy(self._root(p).get_qpos())) or not np.array_equal(v,numpy(self._root(p).get_dofs_velocity())):
                raise ValidationError("native initialization reset changed initial robot state",operation="genesis.drives")
        if self._scene.sim.cur_step_global!=0:raise ValidationError("native initialization reset did not clear clock",operation="genesis.drives")
        self._restore_permanent_welds()
        for p in self._usd_robots:
            self._update_drive_mapping(p);e=self._entities[p]
            self._root(p).control_dofs_position(self._position_targets(p,np.tile(e.initial_joint_positions,(self._spec.environments.count,1))),dofs_idx_local=self._dofs[p])
        self._physics_provenance["mass_query_initialization"]={"warmup_steps":1,"discarded_with_public_reset":True,"zero_logical_time":True,"permanent_welds":len(self._permanent_welds)}

    def _update_drive_mapping(self,path):
        from .drive_mapping import normalized_gains
        metadata=self._usd_robots[path];body=self._root(path)
        props=[metadata["joints"][name] for name in self._entities[path].joint_names]
        kp,kv=normalized_gains(self._mass_matrices[path],self._dofs[path],props)
        body.set_dofs_kp(kp,dofs_idx_local=self._dofs[path]);body.set_dofs_kv(kv,dofs_idx_local=self._dofs[path])
        metadata["acceleration_drive_lowering"].update(effective_force_kp=kp.tolist(),effective_force_kv=kv.tolist(),tick=self._step_index)

    @property
    def asset_provenance(self):
        import copy
        return copy.deepcopy({"profile_path":str(self._asset_profile.path) if self._asset_profile.path else None,
            "profile_sha256":self._asset_profile.digest,"assets":self._provenance,"physics":self._physics_provenance,
            "planning_geometry_exports":getattr(self,"_planning_mesh_exports",{}),
            "robot_runtime":{p.value:m.get("acceleration_drive_lowering") for p,m in self._usd_robots.items()}})

    def _absolute_positions(self,path,native,indices=None):
        reference=self._q_read_offsets[path]
        if indices is not None:reference=reference[list(indices)]
        return np.asarray(native)+reference

    def _position_targets(self,path,absolute,indices=None):
        reference=self._position_references[path]
        if indices is not None:reference=reference[list(indices)]
        return np.asarray(absolute)-reference

    @property
    def world_id(self): return self._spec.world_id
    @property
    def generation(self): return self._generation
    @property
    def state(self): return self._state
    @property
    def tick(self): return Tick(self._step_index,self._step_index*self._spec.physics.time_step_seconds)
    @property
    def build_report(self): return self._build_report

    def _ensure(self,operation):
        if self._state is WorldState.CLOSED:
            raise LifecycleError("world closed",operation=operation)
        from .runtime import assert_owner
        assert_owner(self._session, operation)

    @staticmethod
    def _unsupported(message,operation):
        raise UnsupportedCapabilityError(message,operation=operation)

    def _indices(self,values,size,operation):
        result=tuple(range(size)) if values is None else tuple(values)
        if not result or len(set(result))!=len(result) or any(type(v) is not int or not 0<=v<size for v in result):
            raise ValidationError("selection index out of range or duplicate",operation=operation)
        return result

    def _root(self,path): return self._bodies[path][0]

    def resolve(self,path):
        self._ensure("genesis.resolve")
        if path not in self._entities:
            raise EntityNotFoundError("entity does not exist",operation="genesis.resolve")
        return EntityHandle(self._session.descriptor.provider_id,self._session.session_id,
            self.world_id,self.generation,path,self._entities[path].kind,
            hashlib.sha256(f"{self._session.session_id}:{self.generation}:{path.value}".encode()).hexdigest())

    def _entity(self,handle,operation):
        self._ensure(operation)
        if not isinstance(handle,EntityHandle) or handle.path not in self._entities:
            raise StaleHandleError("invalid entity handle",operation=operation)
        if handle != self.resolve(handle.path):
            raise StaleHandleError("entity handle belongs to another generation/session",operation=operation)
        return self._entities[handle.path]

    def read_articulation(self,handle):
        entity=self._entity(handle,"genesis.read_articulation")
        if entity.kind is not EntityKind.ARTICULATION:
            raise ValidationError("target is not an articulation",operation="genesis.read_articulation")
        body=self._root(entity.path); dofs=self._dofs[entity.path]
        units=entity.joint_position_units
        return ArticulationState(entity.path.value,self.generation,self.tick,entity.joint_names,
            array(self._absolute_positions(entity.path,numpy(body.get_qpos(self._q_indices[entity.path])))),array(body.get_dofs_velocity(dofs)),units,
            tuple("rad/s" if u=="rad" else "m/s" for u in units))

    def apply_articulation_command(self,command):
        entity=self._entity(command.handle,"genesis.control")
        if entity.kind is not EntityKind.ARTICULATION:
            raise ValidationError("target is not an articulation",operation="genesis.control")
        envs=self._indices(command.environment_indices,self._spec.environments.count,"genesis.control")
        indices=self._indices(command.degree_of_freedom_indices,len(entity.joint_names),"genesis.control")
        if command.targets.shape!=(len(envs),len(indices)):
            raise ValidationError("command targets do not match selected axes/environments",operation="genesis.control")
        units=tuple(entity.joint_position_units[i] for i in indices)
        if command.mode is CommandMode.VELOCITY:
            units=tuple("rad/s" if u=="rad" else "m/s" for u in units)
        elif command.mode is CommandMode.EFFORT:
            units=tuple("N*m" if u=="rad" else "N" for u in units)
        if command.target_units and command.target_units!=units:
            raise ValidationError("command axis units mismatch",operation="genesis.control")
        method={CommandMode.POSITION:"control_dofs_position",CommandMode.VELOCITY:"control_dofs_velocity",
            CommandMode.EFFORT:"control_dofs_force"}[command.mode]
        values=np.asarray(command.targets.rows(),dtype=np.float64)
        if command.mode is CommandMode.POSITION:values=self._position_targets(entity.path,values,indices)
        if entity.path in self._drive_modes:
            native_indices=[self._dofs[entity.path][i] for i in indices]
            self._drive_modes[entity.path][np.ix_(envs,native_indices)]=command.mode is not CommandMode.EFFORT
        getattr(self._root(entity.path),method)(values,
            dofs_idx_local=[self._dofs[entity.path][i] for i in indices],envs_idx=envs)

    def read_rigid_body(self,handle):
        entity=self._entity(handle,"genesis.read_rigid_body")
        if entity.path not in self._bodies or len(self._bodies[entity.path])!=1:
            raise ValidationError("target does not have one root physical body",operation="genesis.read_rigid_body")
        link=self._logical_root_link(entity.path);body=link.entity;indices=[link.idx_local]
        return RigidBodyState(array(numpy(body.get_links_pos(indices))[:,0]),array(xyzw(body.get_links_quat(indices))[:,0]),
            array(numpy(body.get_links_vel(indices))[:,0]),array(numpy(body.get_links_ang(indices))[:,0]),self.tick)

    def _logical_root_link(self,path):
        body=self._root(path);metadata=self._usd_robots.get(path)
        if metadata:
            for link in body.links:
                data=metadata["bodies"].get(link.name.rsplit("/",1)[-1],{})
                if not data.get("auxiliary") and not data.get("world_support"):return link
        return body.base_link

    def _native_pose(self,body,environment):
        return Pose(tuple(float(v) for v in numpy(body.get_pos())[environment]),
            tuple(float(v) for v in xyzw(body.get_quat())[environment]))

    def _link_state(self,path,name,environment):
        if name is None:
            link=self._logical_root_link(path); offset=Pose()
        elif name in self._named_frames[path]:
            owner,offset=self._named_frames[path][name]; link=self._links[path][owner]
        else:
            link=self._links[path].get(name); offset=Pose()
            if link is None:
                raise EntityNotFoundError("authored link/frame not found",operation="genesis.kinematics",details={"link_name":name})
        body=link.entity
        pos=numpy(body.get_links_pos([link.idx_local]))[environment,0]
        quat=xyzw(body.get_links_quat([link.idx_local]))[environment,0]
        vel=numpy(body.get_links_vel([link.idx_local]))[environment,0]
        ang=numpy(body.get_links_ang([link.idx_local]))[environment,0]
        pose=Pose(tuple(float(v) for v in pos),tuple(float(v) for v in quat))
        result=compose(pose,offset)
        vel=vel+np.cross(ang,rotation(pose.orientation_xyzw)@offset.position)
        return result,tuple(float(v) for v in vel),tuple(float(v) for v in ang)

    def read_selected_kinematics(self,targets,environment_index=0):
        self._ensure("genesis.kinematics")
        self._indices((environment_index,),self._spec.environments.count,"genesis.kinematics")
        targets=tuple(targets)
        if len({t.target_id for t in targets})!=len(targets):
            raise ValidationError("duplicate kinematic target IDs",operation="genesis.kinematics")
        result=[];resolved=[];groups={}
        for target in targets:
            path=target.entity_path;name=target.link_name
            if path not in self._bodies:
                raise EntityNotFoundError("kinematic entity missing",operation="genesis.kinematics")
            offset=Pose()
            if name is None: link=self._logical_root_link(path)
            else:
                owner,offset=self._named_frames[path].get(name,(name,Pose()))
                link=self._links[path].get(owner)
                if link is None:
                    raise EntityNotFoundError("kinematic link missing",operation="genesis.kinematics",details={"name":name})
            resolved.append((target,link,offset))
            groups.setdefault(link.entity.idx,(link.entity,set()))[1].add(link.idx_local)
        samples={}
        for body,indices in groups.values():
            indices=sorted(indices)
            p=numpy(body.get_links_pos(indices,envs_idx=[environment_index]))[0]
            q=xyzw(body.get_links_quat(indices,envs_idx=[environment_index]))[0]
            v=numpy(body.get_links_vel(indices,envs_idx=[environment_index]))[0]
            a=numpy(body.get_links_ang(indices,envs_idx=[environment_index]))[0]
            for i,index in enumerate(indices): samples[(body.idx,index)]=(p[i],q[i],v[i],a[i])
        for target,link,offset in resolved:
            p,q,v,a=samples[(link.entity.idx,link.idx_local)]
            pose=Pose(tuple(map(float,p)),tuple(map(float,q)))
            v=v+np.cross(a,rotation(q)@offset.position)
            result.append(KinematicState(target.target_id,target.entity_path,target.link_name,self.tick,
                compose(pose,offset),tuple(map(float,v)),tuple(map(float,a))))
        return tuple(result)

    def apply_rigid_body_command(self,command):
        entity=self._entity(command.handle,"genesis.wrench")
        if entity.kind is not EntityKind.RIGID_BODY:
            raise ValidationError("wrench target is not a rigid body",operation="genesis.wrench")
        envs=self._indices(command.environment_indices,self._spec.environments.count,"genesis.wrench")
        if command.forces_n.shape!=(len(envs),3):
            raise ValidationError("wrench shape mismatch",operation="genesis.wrench")
        for env,force,torque in zip(envs,command.forces_n.rows(),command.torques_n_m.rows(),strict=True):
            self._wrenches[(entity.path,env)]=(force,torque)

    def read_contact(self,handle,force_threshold_n=1e-6):
        entity=self._entity(handle,"genesis.contact")
        if not math.isfinite(force_threshold_n) or force_threshold_n<0:
            raise ValidationError("invalid contact threshold",operation="genesis.contact")
        body=self._root(entity.path)
        contacts=body.get_contacts()
        count=self._spec.environments.count
        forces=np.zeros((count,3));flags=np.zeros(count,dtype=bool)
        mask=numpy(contacts.get("valid_mask",np.ones(numpy(contacts["link_a"]).shape,dtype=bool)))
        normal=contacts.get("normal")
        if normal is None and bool(mask.any()):
            self._unsupported("Genesis contact API omitted normal directions", "genesis.contact")
        if normal is not None:
            normal=numpy(normal); fa=numpy(contacts["force_a"]); fb=numpy(contacts["force_b"])
            own_a=(numpy(contacts["link_a"])>=body.link_start)&(numpy(contacts["link_a"])<body.link_end)
            own_b=(numpy(contacts["link_b"])>=body.link_start)&(numpy(contacts["link_b"])<body.link_end)
            f=np.where(own_a[...,None],fa,0)+np.where(own_b[...,None],fb,0)
            f=np.sum(f*normal,axis=-1)[...,None]*normal
            f=np.where(mask[...,None],f,0)
            forces=f.sum(axis=1)
            flags=(np.linalg.norm(f,axis=-1)>force_threshold_n).any(axis=1)
        return ContactState(array(forces),array(flags,"bool"),self.tick)

    def step(self,count=1):
        self._ensure("genesis.step")
        if type(count) is not int or count<1:
            raise ValidationError("step count must be positive",operation="genesis.step")
        for _ in range(count):
            for path in self._usd_robots:self._update_drive_mapping(path)
            for (path,env),(force,torque) in self._wrenches.items():
                self._scene.rigid_solver.apply_links_external_wrench(
                    force=np.asarray(force).reshape(1,1,3),torque=np.asarray(torque).reshape(1,1,3),
                    links_idx=[self._root(path).base_link.idx],envs_idx=[env])
            self._scene.step(update_visualizer=bool(self._cameras) or not self._session.config.headless)
            for path in self._usd_robots:self._capture_mass_matrix(path,self._drive_modes[path])
            self._step_index+=1
        self._scene_sequence+=count
        self._planning_cache.clear()
        return self.tick

    def reset(self,environment_indices=None):
        self._ensure("genesis.reset")
        envs=self._indices(environment_indices,self._spec.environments.count,"genesis.reset")
        for key,attachment in tuple(self._attachments.items()):
            if attachment.environment_index in envs:
                self._scene.rigid_solver.delete_weld_constraint(attachment.parent_link.idx,
                    attachment.child_link.idx,envs_idx=[attachment.environment_index])
                del self._attachments[key]
        self._scene.reset(self._initial_state,envs_idx=envs)
        for entity in self._spec.entities:
            if entity.kind is EntityKind.ARTICULATION:
                self._root(entity.path).control_dofs_position(
                    self._position_targets(entity.path,np.tile(entity.initial_joint_positions,(len(envs),1))),
                    dofs_idx_local=self._dofs[entity.path],envs_idx=envs)
        self._restore_permanent_welds()
        for path in self._usd_robots:
            self._mass_matrices[path][list(envs)]=self._initial_mass_matrices[path][list(envs)]
            self._drive_modes[path][np.ix_(envs,self._dofs[path])]=True
            self._update_drive_mapping(path)
        self._wrenches={k:v for k,v in self._wrenches.items() if k[1] not in envs}
        self._reset_count+=1;self._scene_sequence+=1;self._attachment_revision+=1
        self._planning_cache.clear()
        return ResetResult(envs,self._reset_count,self.tick)

    def _set_pose(self,path,environment,pose):
        body=self._root(path)
        root_pose=compose(pose,inverse(self._init_root_offsets.get(path,Pose())))
        body.set_pos(np.asarray([root_pose.position]),envs_idx=[environment])
        body.set_quat(np.asarray([wxyz(root_pose.orientation_xyzw)]),envs_idx=[environment])

    def _attachment_link(self,path,name):
        if name is None: return self._logical_root_link(path)
        owner=self._named_frames[path].get(name,(name,None))[0]
        return self._links[path][owner]

    def apply_scene_command(self,command):
        self._ensure("genesis.scene_command")
        previous=self._scene_results.get(command.command_id)
        if previous is not None:
            return SceneCommandResult(command.command_id,SceneCommandStatus.DUPLICATE,self.generation,
                self._scene_sequence,self.tick,attachment_id=previous.attachment_id)
        status=SceneCommandStatus.APPLIED;code=message=None
        path=command.entity_path
        if command.expected_generation!=self.generation or path not in self._entities:
            status=SceneCommandStatus.REJECTED;code="stale_or_missing_entity";message="generation or entity does not match"
        else:
            env=command.environment_index
            self._indices((env,),self._spec.environments.count,"genesis.scene_command")
            if command.kind is SceneCommandKind.SET_POSE:
                self._set_pose(path,env,command.target_pose)
            elif command.kind is SceneCommandKind.ATTACH:
                key=(env,command.attachment_id)
                if key in self._attachments or any(a.child_path==path and a.environment_index==env for a in self._attachments.values()):
                    status=SceneCommandStatus.REJECTED;code="attachment_exists";message="child is already attached"
                else:
                    parent=self._attachment_link(command.parent_entity_path,command.parent_link_name)
                    child=self._attachment_link(path,command.child_link_name)
                    pp=self._link_state(command.parent_entity_path,parent.name.rsplit("/",1)[-1],env)[0]
                    cp=self._link_state(path,child.name.rsplit("/",1)[-1],env)[0]
                    solver=self._scene.rigid_solver
                    live=solver.get_equality_constraints(as_tensor=True,to_torch=False)
                    used=int(np.count_nonzero(numpy(live["type"])[env]>=0))
                    pair_in_other_environment=any(a.environment_index!=env and
                        {a.parent_link.idx,a.child_link.idx}=={parent.idx,child.idx} for a in self._attachments.values())
                    if pair_in_other_environment:
                        status=SceneCommandStatus.REJECTED;code="native_cross_environment_pair_unsupported"
                        message="official Genesis 1.4.2 cannot weld the same link pair independently in multiple environments"
                    elif used>=solver.n_candidate_equalities_:
                        status=SceneCommandStatus.REJECTED;code="native_constraint_capacity"
                        message="Genesis equality constraint capacity is exhausted"
                    else:
                        original_cp=cp
                        original_velocity=numpy(child.entity.get_dofs_velocity())[env].copy()
                        registered=False
                        try:
                            if command.parent_T_child is not None:
                                if child is not self._root(path).base_link:
                                    self._unsupported("explicit attachment transform requires the child root link", "genesis.attach")
                                parent_anchor=self._link_state(command.parent_entity_path,command.parent_link_name,env)[0]
                                cp=compose(parent_anchor,command.parent_T_child)
                                body=self._root(path)
                                body.set_pos(np.asarray([cp.position]),envs_idx=[env])
                                body.set_quat(np.asarray([wxyz(cp.orientation_xyzw)]),envs_idx=[env])
                            relative=compose(inverse(pp),cp)
                            solver.add_weld_constraint(parent.idx,child.idx,envs_idx=[env])
                            welds=solver.get_weld_constraints(as_tensor=True,to_torch=False)
                            a=numpy(welds["link_a"])[env];b=numpy(welds["link_b"])[env]
                            registered=bool((((a==parent.idx)&(b==child.idx))|((b==parent.idx)&(a==child.idx))).any())
                            if not registered:
                                raise RuntimeError("Genesis did not register the weld constraint")
                            from .constraints import harden_dynamic_welds
                            harden_dynamic_welds(solver,self._spec.physics.time_step_seconds/self._spec.physics.substeps)
                            self._physics_provenance["dynamic_weld_parameters"]="public_global_setter_with_exact_nonweld_restore"
                        except Exception as exc:
                            from .constraints import ConstraintRestoreError
                            if isinstance(exc,ConstraintRestoreError):
                                self.close()
                                raise LifecycleError("native constraint parameter restoration failed; world closed",operation="genesis.attach",cause=exc) from exc
                            # A failed native transaction must not move or stop the child.
                            if registered:
                                solver.delete_weld_constraint(parent.idx,child.idx,envs_idx=[env])
                            if command.parent_T_child is not None:
                                body=self._root(path)
                                body.set_pos(np.asarray([original_cp.position]),envs_idx=[env])
                                body.set_quat(np.asarray([wxyz(original_cp.orientation_xyzw)]),envs_idx=[env])
                                body.set_dofs_velocity(original_velocity.reshape(1,-1),envs_idx=[env])
                            status=SceneCommandStatus.REJECTED;code="native_constraint_error";message=str(exc)
                        else:
                            self._attachments[key]=Attachment(command.attachment_id,env,command.parent_entity_path,path,parent,child,relative)
                            self._attachment_revision+=1
            elif command.kind is SceneCommandKind.DETACH:
                key=(env,command.attachment_id);attachment=self._attachments.get(key)
                if attachment is None or attachment.child_path!=path:
                    status=SceneCommandStatus.REJECTED;code="attachment_not_active";message="attachment not found"
                else:
                    self._scene.rigid_solver.delete_weld_constraint(attachment.parent_link.idx,attachment.child_link.idx,envs_idx=[env])
                    del self._attachments[key];self._attachment_revision+=1
            else:
                status=SceneCommandStatus.REJECTED;code="unsupported_command";message="scene command is unsupported"
        if status is SceneCommandStatus.APPLIED:
            self._scene_sequence+=1;self._planning_cache.clear()
        result=SceneCommandResult(command.command_id,status,self.generation,self._scene_sequence,
            self.tick,code,message,attachment_id=command.attachment_id)
        self._scene_results[command.command_id]=result
        if len(self._scene_results)>self._session.config.max_cached_commands:
            self._scene_results.pop(next(iter(self._scene_results)))
        return result

    def _scene_entities(self):
        states=[]
        for entity in self._spec.entities:
            for env in range(self._spec.environments.count):
                pose=entity.pose;vel=ang=(0.,0.,0.);joints=()
                if entity.path in self._bodies and len(self._bodies[entity.path])==1:
                    body=self._root(entity.path)
                    pose=self._native_pose(body,env)
                    vel=numpy(body.get_vel())[env];ang=numpy(body.get_ang())[env]
                    root_pose=pose
                    pose=compose(pose,self._init_root_offsets.get(entity.path,Pose()))
                    vel=tuple(float(v) for v in np.asarray(vel)+np.cross(ang,np.asarray(pose.position)-root_pose.position))
                if entity.kind is EntityKind.ARTICULATION:
                    joints=tuple(float(x) for x in self._absolute_positions(entity.path,numpy(self._root(entity.path).get_qpos(self._q_indices[entity.path]))[env]))
                states.append(SceneEntityState(entity.path,entity.kind,env,pose,vel,ang,entity.joint_names,joints))
        return tuple(states)

    def scene_snapshot(self):
        self._ensure("genesis.scene_snapshot")
        return SceneSnapshot(self._session.descriptor.provider_id,self.world_id,self.generation,
            self._scene_sequence,self.tick,self._scene_entities())

    def scene_delta(self,base_sequence):
        self._ensure("genesis.scene_delta")
        if type(base_sequence) is not int or not 0<=base_sequence<=self._scene_sequence:
            raise ValidationError("invalid scene base sequence",operation="genesis.scene_delta")
        return SceneDelta(self.world_id,self.generation,base_sequence,self._scene_sequence,self.tick,
            () if base_sequence==self._scene_sequence else self._scene_entities())

    def _add_camera(self,entity):
        if not self._session.config.enable_cameras:
            self._unsupported("cameras disabled", "genesis.camera")
        camera=entity.camera
        if any(m not in {CameraModality.RGB,CameraModality.DEPTH} for m in camera.modalities):
            self._unsupported("camera modality is not implemented", "genesis.camera")
        if camera.calibration is not None or camera.render_exclusions or self._usd_materialized:
            if any(e.kind in {EntityKind.PARTICLE_FLUID,EntityKind.SURFACE_DEFORMABLE,EntityKind.VOLUME_DEFORMABLE} for e in self._spec.entities):
                self._unsupported("calibrated/USD visual proxy cannot render soft matter; use native camera", "genesis.camera")
            self._cameras[entity.path]=None
            return
        fov=math.degrees(2*math.atan(math.tan(math.radians(camera.horizontal_fov_degrees)/2)*camera.height_px/camera.width_px))
        r=rotation(entity.pose.orientation_xyzw)
        # Public camera convention follows OpenGL: -Z forward, +Y up.
        pos=entity.pose.position
        self._cameras[entity.path]=[self._scene.add_camera(res=(camera.width_px,camera.height_px),
            pos=pos,lookat=tuple(np.asarray(pos)-r[:,2]),up=tuple(r[:,1]),fov=fov,
            near=camera.near_plane_m,far=camera.far_plane_m,GUI=False,env_idx=env)
            for env in range(self._spec.environments.count)]

    def read_sensor(self,handle):
        entity=self._entity(handle,"genesis.sensor")
        data={modality:[] for modality in entity.camera.modalities}
        from .camera_rendering import CameraScene
        selected=self._cameras[entity.path]
        if isinstance(selected,CameraScene):
            for env in range(self._spec.environments.count):
                rgb,depth=selected.render(env,rgb=CameraModality.RGB in data,depth=CameraModality.DEPTH in data)
                if CameraModality.RGB in data:data[CameraModality.RGB].append(rgb)
                if CameraModality.DEPTH in data:data[CameraModality.DEPTH].append(depth)
            native_cameras=()
        else:native_cameras=selected
        for env,camera in enumerate(native_cameras):
            if entity.mount is not None:
                parent=self._link_state(entity.mount.parent_path,entity.mount.parent_link_name,env)[0]
                pose=compose(parent,entity.pose);r=rotation(pose.orientation_xyzw)
                camera.set_pose(pos=pose.position,lookat=tuple(np.asarray(pose.position)-r[:,2]),up=tuple(r[:,1]))
            rgb,depth,*_=camera.render(rgb=CameraModality.RGB in data,depth=CameraModality.DEPTH in data)
            if CameraModality.RGB in data: data[CameraModality.RGB].append(rgb)
            if CameraModality.DEPTH in data: data[CameraModality.DEPTH].append(depth)
        channels=[]
        for modality,frames in data.items():
            stacked=np.asarray(frames)
            value=ArrayValue.from_uint8_bytes(tuple(stacked.shape),stacked.astype(np.uint8).tobytes()) if modality is CameraModality.RGB else array(stacked,"float32")
            channels.append(SensorChannel(modality,value))
        return SensorSample(handle,tuple(channels),self.tick)

    def publish_debug(self,*args,**kwargs): self._unsupported("native debug overlay is not implemented","genesis.debug")
    def clear_debug(self,*args,**kwargs): return 0
    def apply_deformable_command(self,*args,**kwargs): self._unsupported("deformable API is not implemented","genesis.deformable")

    def close(self):
        if self._state is WorldState.CLOSED: return
        self._ensure("genesis.world.close")
        self._state=WorldState.CLOSED
        try:
            for camera in self._cameras.values():
                if hasattr(camera,"close"):camera.close()
            self._scene.destroy()
        finally:
            self._planning_resources.clear()
            if self._asset_lease is not None: self._asset_lease.close()
            shutil.rmtree(self._derived_root,ignore_errors=True)
            self._session._world_closed(self)

    def __enter__(self): return self
    def __exit__(self,*args): self.close()
