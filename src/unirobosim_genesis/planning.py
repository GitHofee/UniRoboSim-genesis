"""Planning catalogs and immutable geometry read from the built Genesis scene."""
from __future__ import annotations
import hashlib
import json
import numpy as np
from unirobosim import (
    EntityKind, Pose, WorldState, PlanningEntityDescriptor, PlanningEntityKind,
    PlanningLinkDescriptor, PlanningJointDescriptor, PlanningJointType,
    PlanningFrameDescriptor, PlanningFrameKind, PlanningGeometryDescriptor,
    PlanningGeometryPurpose, PlanningGeometryRepresentation, PlanningGeometryLocalPose,
    PlanningGeometryMotionClass, PlanningGeometryContentProfile, PlanningGeometryResourceLayout,
    PlanningGeometryDType, PlanningGeometryResourceDescriptor, PlanningGeometryStorageKind,
    PlanningGeometryAxisConvention, PlanningSceneCatalog, PlanningSceneState,
    PlanningEntityState, PlanningLinkState, PlanningFrameState, PlanningArticulationState,
    PlanningGeometryTransform, PlanningPose, PlanningTwist, PlanningAttachment,
    PlanningSceneContractError, PlanningSceneRepresentationError, PlanningSceneNotFoundError,
    PlanningGeometryResourceRevokedError, PlanningSceneDelta, PlanningSceneDeltaKind,
    PlanningSceneDeltaContinuityError, PlanningPointClosureDescriptor,
)
from .math import numpy, xyzw, compose, inverse, rotation
from .config import ENGINE_VERSION

WORLD_FRAME="frame.world"

def _vector(x): return tuple(float(v) for v in x)
def _pose(p): return PlanningPose(WORLD_FRAME,p.position,p.orientation_xyzw)
def _eid(i): return f"entity.{i:04d}"
def _efid(i): return f"frame.entity.{i:04d}"
def _lid(i): return f"link.{i:06d}"
def _lfid(i): return f"frame.link.{i:06d}"
def _jid(i): return f"joint.{i:06d}"

def _encode_mesh_resource(vertices, faces, *, provenance=False):
    """Encode a native indexed mesh using the portable unsigned-index profile."""
    vertices=np.asarray(vertices)
    faces=np.asarray(faces)
    if (vertices.ndim!=2 or vertices.shape[1]!=3 or not len(vertices)
            or vertices.dtype.kind not in "fiu" or not np.isfinite(vertices).all()):
        raise PlanningSceneRepresentationError("native collider vertices must be finite XYZ rows",operation="genesis.planning")
    if (faces.ndim!=2 or faces.shape[1]!=3 or not len(faces)
            or faces.dtype.kind not in "iu" or np.any(faces<0)
            or np.any(faces>=len(vertices)) or np.any(faces>np.iinfo(np.uint32).max)):
        raise PlanningSceneRepresentationError("native collider indices must be integer triangle rows within the vertex array and uint32 range",operation="genesis.planning")
    native_vertices=np.ascontiguousarray(vertices)
    native_faces=np.ascontiguousarray(faces)
    triangles=np.asarray(faces,dtype=np.int64)
    precise=np.asarray(vertices,dtype=np.float64)
    cross=np.cross(precise[triangles[:,1]]-precise[triangles[:,0]],
        precise[triangles[:,2]]-precise[triangles[:,0]])
    # Exact equality only. A triangle with nonzero area, however thin, survives.
    keep=np.any(cross!=0.,axis=1)
    if not np.any(keep):
        raise PlanningSceneRepresentationError("native collider has no nondegenerate surface triangles",operation="genesis.planning")
    vertices=np.ascontiguousarray(vertices,dtype="<f4")
    if not np.isfinite(vertices).all():
        raise PlanningSceneRepresentationError("native collider vertices exceed float32 range",operation="genesis.planning")
    faces=np.ascontiguousarray(faces[keep],dtype="<u4")
    canonical=np.asarray(vertices,dtype=np.float64)
    collapsed=~np.any(np.cross(canonical[faces[:,1]]-canonical[faces[:,0]],
        canonical[faces[:,2]]-canonical[faces[:,0]])!=0.,axis=1)
    vertex_dtype=PlanningGeometryDType.FLOAT32
    if np.any(collapsed):
        # Do not delete a real thin triangle merely because float32 rounds its
        # vertices together. The portable profile also supports float64.
        vertices=np.ascontiguousarray(precise,dtype="<f8")
        vertex_dtype=PlanningGeometryDType.FLOAT64
    layout=PlanningGeometryResourceLayout(PlanningGeometryRepresentation.TRIANGLE_MESH,
        PlanningGeometryContentProfile.MESH_TRIANGLES_RAW_LE_V1,
        vertex_dtype=vertex_dtype,vertex_shape=tuple(vertices.shape),
        index_dtype=PlanningGeometryDType.UINT32,index_shape=tuple(faces.shape))
    payload=vertices.tobytes()+faces.tobytes()
    if not provenance:return payload,layout
    evidence={"policy":"native-exact-zero-area-only-v1","native_vertex_dtype":str(native_vertices.dtype),
        "native_vertices_sha256":hashlib.sha256(native_vertices.tobytes()).hexdigest(),
        "native_index_dtype":str(native_faces.dtype),"native_faces_sha256":hashlib.sha256(native_faces.tobytes()).hexdigest(),
        "native_triangle_count":len(native_faces),"exported_triangle_count":len(faces),
        "removed_triangle_indices":np.flatnonzero(~keep).tolist(),"removed_cross_product_exactly_zero":True,
        "float32_collapse_triangle_indices":np.flatnonzero(keep)[collapsed].tolist(),
        "vertex_encoding":vertex_dtype.value,"physics_geometry_modified":False}
    return payload,layout,evidence

class GeometryLease:
    def __init__(self,world,descriptor,payload):
        self._world=world;self.descriptor=descriptor;self._payload=payload;self._closed=False
    @property
    def closed(self): return self._closed or self._world.state is WorldState.CLOSED
    def read(self,offset=0,length=None):
        if self.closed:
            raise PlanningGeometryResourceRevokedError("geometry lease revoked",operation="genesis.geometry")
        if type(offset) is not int or offset<0 or offset>len(self._payload) or (length is not None and (type(length) is not int or length<0)):
            raise PlanningSceneContractError("invalid geometry read range",operation="genesis.geometry")
        stop=len(self._payload) if length is None else min(len(self._payload),offset+length)
        if stop-offset>64*1024*1024:
            raise PlanningSceneContractError("geometry read exceeds 64 MiB limit",operation="genesis.geometry")
        return self._payload[offset:stop]
    def close(self): self._closed=True;self._payload=b""

class PlanningMixin:
    def _planning_environment(self,environment):
        self._ensure("genesis.planning")
        self._indices((environment,),self._spec.environments.count,"genesis.planning")
        return environment

    def _build_planning_catalog(self,environment):
        entities=[];links=[];joints=[];geometries=[];closures=[]
        frames=[PlanningFrameDescriptor(WORLD_FRAME,PlanningFrameKind.WORLD,None,None,None)]
        self._planning_geoms={};self._planning_native_links={};self._planning_joint_bindings={}
        self._planning_named={};self._planning_entity_indices={};self._planning_joint_frames={}
        self._planning_mesh_exports={}
        limit_provenance={}
        for index,entity in enumerate(self._spec.entities):
            eid=_eid(index);efid=_efid(index);self._planning_entity_indices[entity.path]=index
            frame_ids=[efid];link_ids=[];joint_ids=[];geom_ids=[]
            frames.append(PlanningFrameDescriptor(efid,PlanningFrameKind.ENTITY,WORLD_FRAME,eid,None))
            native_links=[link for body in self._bodies.get(entity.path,()) for link in body.links]
            metadata=self._usd_robots.get(entity.path)
            native_limits={}
            for body in self._bodies.get(entity.path,()):
                if body.n_dofs:
                    lower,upper=body.get_dofs_limit(envs_idx=[environment])
                    native_limits[body.idx]=np.stack((numpy(lower),numpy(upper)),axis=-1).reshape(-1,body.n_dofs,2)[0].astype(np.float64)
            if metadata:
                native_links=[link for link in native_links if not any(metadata["bodies"].get(link.name.rsplit("/",1)[-1],{}).get(k) for k in ("auxiliary","world_support"))]
            allowed={link.idx for link in native_links}
            for link in native_links:
                lid=_lid(link.idx);lfid=_lfid(link.idx)
                self._planning_native_links[lid]=link
                parent=link.parent_idx if link.parent_idx in allowed else None
                pframe=_lfid(parent) if parent is not None else efid
                local_geom_ids=[]
                for geom in link.geoms:
                    if geom.contype==0 and geom.conaffinity==0:
                        continue
                    gid=f"geometry.{geom.idx:06d}";rid=f"resource.{geom.idx:06d}"
                    payload,layout,evidence=_encode_mesh_resource(geom.init_verts,geom.init_faces,provenance=True)
                    self._planning_mesh_exports[gid]=dict(evidence,native_prim=geom.mesh.metadata.get("name"))
                    digest=hashlib.sha256(payload).hexdigest()
                    representation=PlanningGeometryRepresentation.TRIANGLE_MESH
                    profile=PlanningGeometryContentProfile.MESH_TRIANGLES_RAW_LE_V1
                    local=PlanningGeometryLocalPose(_vector(geom.init_pos),_vector(xyzw(geom.init_quat)))
                    geometries.append(PlanningGeometryDescriptor(gid,eid,lid,lfid,
                        PlanningGeometryPurpose.COLLISION,representation,local,(1.,1.,1.),
                        PlanningGeometryMotionClass.STATIC if link.is_fixed else PlanningGeometryMotionClass.DYNAMIC,
                        int(geom.contype)&0xffffffff,int(geom.conaffinity)&0xffffffff,
                        hashlib.sha256(f"genesis{ENGINE_VERSION}:{self._spec.digest}:{geom.idx}:{digest}".encode()).hexdigest(),
                        resource_id=rid,sha256=digest,content_profile=profile,resource_layout=layout))
                    self._planning_resources[gid]=payload
                    self._planning_geoms[gid]=geom
                    local_geom_ids.append(gid);geom_ids.append(gid)
                links.append(PlanningLinkDescriptor(lid,eid,link.name.rsplit("/",1)[-1],lfid,
                    None if parent is None else _lid(parent),tuple(sorted(local_geom_ids))))
                link_frame_index=len(frames)
                frames.append(PlanningFrameDescriptor(lfid,PlanningFrameKind.LINK,pframe,eid,lid))
                link_ids.append(lid);frame_ids.append(lfid)
                if parent is not None:
                    native_joints=list(link.joints)
                    if len(native_joints)>1:
                        raise PlanningSceneRepresentationError("planning contract requires one joint per link edge",operation="genesis.planning",
                            details={"link":link.name,"joints":[{"name":j.name,"dofs":j.n_dofs,"type":j.type.name} for j in native_joints]})
                    joint=native_joints[0] if native_joints else None
                    if joint is not None and joint.n_dofs>1:
                        raise PlanningSceneRepresentationError("spherical/free internal joints need expanded topology",operation="genesis.planning")
                    jt=PlanningJointType.FIXED
                    axis=(0.,0.,1.);unit="rad";lo=hi=effort=None
                    if joint is not None and joint.n_dofs:
                        prismatic=joint.type==self._gs.JOINT_TYPE.PRISMATIC
                        jt=PlanningJointType.PRISMATIC if prismatic else PlanningJointType.REVOLUTE
                        unit="m" if prismatic else "rad"
                        raw=joint.desc.dofs_motion_vel[0] if prismatic else joint.desc.dofs_motion_ang[0]
                        axis_world=numpy(joint.get_anchor_axis())[environment]
                        child_quat=xyzw(link.entity.get_links_quat([link.idx_local]))[environment,0]
                        raw=rotation(child_quat).T@axis_world
                        axis=_vector(raw/np.linalg.norm(raw))
                        # Use the same effective native precision and reference
                        # mapping as public joint state. Authoring doubles can
                        # exclude a valid float32 boundary state by one ULP.
                        name=joint.name.rsplit("/",1)[-1]
                        raw_limits=native_limits[joint.entity.idx][joint.dofs_idx_local[0]]
                        limits=raw_limits.copy();offset=0.
                        if name in entity.joint_names:
                            axis_index=entity.joint_names.index(name)
                            limits=self._absolute_positions(entity.path,limits[:,None],[axis_index])[:,0]
                            offset=float(self._q_read_offsets[entity.path][axis_index])
                        props=metadata["joints"].get(name) if metadata else None
                        authored=np.asarray(props["limits"] if props else joint.desc.dofs_limit[0])
                        encode=lambda values:[float(v) if np.isfinite(v) else None for v in values]
                        limit_provenance.setdefault(entity.path.value,{})[name]={
                            "authoring_or_import_description_limits":encode(authored),
                            "native_limits":encode(raw_limits),"position_read_offset":offset,
                            "effective_absolute_limits":encode(limits)}
                        if np.all(np.isfinite(limits)): lo,hi=map(float,limits)
                        elif not prismatic: jt=PlanningJointType.CONTINUOUS
                        force=np.asarray(joint.desc.dofs_force_range)[0]
                        if np.all(np.isfinite(force)) and np.max(np.abs(force))>0: effort=float(np.max(np.abs(force)))
                    jid=_jid(joint.idx) if joint is not None else f"joint.fixed.{link.idx:06d}"
                    jfid=f"frame.{jid}"
                    name=(joint.name.rsplit("/",1)[-1] if joint is not None else
                        self._fixed_joint_names.get(entity.path,{}).get(link.name,f"fixed:{link.name.rsplit('/',1)[-1]}"))
                    frames[link_frame_index]=PlanningFrameDescriptor(lfid,PlanningFrameKind.LINK,jfid,eid,lid)
                    frames.append(PlanningFrameDescriptor(jfid,PlanningFrameKind.JOINT,pframe,eid,lid))
                    frame_ids.append(jfid);self._planning_joint_frames[jfid]=(link,joint)
                    joints.append(PlanningJointDescriptor(jid,eid,name,
                        _lid(parent),lid,jt,jfid,axis,unit,lo,hi,max_effort=effort))
                    self._planning_joint_bindings[jid]=joint
                    joint_ids.append(jid)
            declarations=entity.metadata.to_dict().get("planning_frame_declarations",{}).get("entries",())
            for ni,declaration in enumerate(declarations):
                name=declaration["name"];owner_name=declaration["owner_link"]
                owner=self._links[entity.path].get(owner_name)
                if owner is None:
                    raise PlanningSceneNotFoundError("declared frame owner not present in native links",operation="genesis.planning",details={"owner":owner_name})
                fid=f"frame.named.{index:04d}.{ni:04d}"
                frames.append(PlanningFrameDescriptor(fid,PlanningFrameKind.NAMED,_lfid(owner.idx),eid,_lid(owner.idx),name))
                self._planning_named[fid]=(entity.path,name)
                frame_ids.append(fid)
            kind=PlanningEntityKind(entity.metadata.to_dict().get("planning_entity_kind",
                "robot" if entity.kind is EntityKind.ARTICULATION else "rigid_object" if entity.kind is EntityKind.RIGID_BODY else "other"))
            entities.append(PlanningEntityDescriptor(eid,entity.path.value,kind,True,efid,
                tuple(sorted(link_ids)),tuple(sorted(frame_ids)),tuple(sorted(geom_ids)),tuple(joint_ids)))
            adjacency={}
            for joint in joints:
                if joint.entity_id!=eid:continue
                adjacency.setdefault(joint.parent_link_id,[]).append((joint.child_link_id,joint))
                adjacency.setdefault(joint.child_link_id,[]).append((joint.parent_link_id,joint))
            for body in self._bodies.get(entity.path,()):
                for equality in body.equalities:
                    if equality.type!=self._gs.EQUALITY_TYPE.CONNECT:continue
                    a,b=_lid(equality.eq_obj1id),_lid(equality.eq_obj2id)
                    pending=[(a,())];visited=set();coupled=None
                    while pending:
                        current,path=pending.pop()
                        if current==b:
                            coupled=tuple(sorted(j.joint_id for j in path if j.joint_type is not PlanningJointType.FIXED))
                            break
                        if current in visited:continue
                        visited.add(current)
                        pending.extend((neighbor,path+(j,)) for neighbor,j in adjacency.get(current,()) if neighbor not in visited)
                    if coupled is None:
                        raise PlanningSceneRepresentationError("native point closure crosses disconnected trees",operation="genesis.planning")
                    anchors=np.asarray(equality.eq_data)[:6].reshape((2,3))
                    fact={"engine":ENGINE_VERSION,"entity":entity.path.value,"name":equality.name,
                        "type":"CONNECT","endpoints":[a,b],"anchors_m":anchors.tolist()}
                    digest=hashlib.sha256(json.dumps(fact,sort_keys=True,separators=(",",":")).encode()).hexdigest()
                    closures.append(PlanningPointClosureDescriptor(f"closure.{equality.idx:06d}",eid,
                        equality.name.rsplit("/",1)[-1],a,b,_vector(anchors[0]),_vector(anchors[1]),digest,coupled))
            if metadata:
                for ci,closure in enumerate(metadata.get("closures",())):
                    a=_lid(self._links[entity.path][closure["a"].rsplit("/",1)[-1]].idx)
                    b=_lid(self._links[entity.path][closure["b"].rsplit("/",1)[-1]].idx)
                    pending=[(a,())];visited=set();coupled=None
                    while pending:
                        current,path=pending.pop()
                        if current==b:
                            coupled=tuple(sorted(j.joint_id for j in path if j.joint_type is not PlanningJointType.FIXED));break
                        if current in visited:continue
                        visited.add(current)
                        pending.extend((neighbor,path+(j,)) for neighbor,j in adjacency.get(current,()) if neighbor not in visited)
                    if coupled is None:raise PlanningSceneRepresentationError("USD closure crosses disconnected logical trees",operation="genesis.planning")
                    anchors=[np.asarray(closure[k])[:3,3] for k in ("j0","j1")]
                    digest=hashlib.sha256(json.dumps({"asset_sha256":metadata["output_sha256"],"closure":closure},sort_keys=True).encode()).hexdigest()
                    closures.append(PlanningPointClosureDescriptor(f"closure.usd.{index:04d}.{ci:03d}",eid,
                        closure["name"],a,b,_vector(anchors[0]),_vector(anchors[1]),digest,coupled))
        self._physics_provenance.setdefault("planning_joint_limit_readback",{})[str(environment)]={
            "method":"public_get_dofs_limit_with_position_read_reference",
            "unbounded_endpoint_encoding":None,"entities":limit_provenance}
        return PlanningSceneCatalog.build(self._session.descriptor.provider_id,self.world_id,
            self.generation,environment,1,1,tuple(sorted(entities,key=lambda x:x.entity_id)),
            tuple(sorted(links,key=lambda x:x.link_id)),tuple(sorted(joints,key=lambda x:x.joint_id)),
            tuple(sorted(frames,key=lambda x:x.frame_id)),tuple(sorted(geometries,key=lambda x:x.geometry_id)),
            tuple(sorted(closures,key=lambda x:x.closure_id)))

    def planning_scene_catalog(self,environment_index=0):
        env=self._planning_environment(environment_index)
        if env not in self._planning_catalogs:
            self._planning_catalogs[env]=self._build_planning_catalog(env)
        return self._planning_catalogs[env]

    def _read_planning_native_states(self,environment):
        # Entity getters delegate to these same public solver getters. Preserve
        # their relative=True authored-origin convention (solver defaults False),
        # and transfer each field once rather than once per fixed USD body.
        solver=self._scene.rigid_solver
        selection=[environment]
        positions=numpy(solver.get_links_pos(envs_idx=selection,relative=True))[0]
        quaternions=xyzw(solver.get_links_quat(envs_idx=selection,relative=True))[0]
        velocities=numpy(solver.get_links_vel(envs_idx=selection,relative=True))[0]
        angular=numpy(solver.get_links_ang(envs_idx=selection))[0]
        native_states={}
        for bodies in self._bodies.values():
            for body in bodies:
                indices=[link.idx for link in body.links]
                native_states[body.idx]=(positions[indices],quaternions[indices],
                    velocities[indices],angular[indices],
                    numpy(body.get_qpos())[environment] if body.n_dofs else (),
                    numpy(body.get_dofs_velocity())[environment] if body.n_dofs else ())
        return native_states

    def planning_scene_state(self,environment_index=0):
        env=self._planning_environment(environment_index)
        if env in self._planning_cache: return self._planning_cache[env]
        catalog=self.planning_scene_catalog(env)
        entities=[];links=[];frames=[PlanningFrameState(WORLD_FRAME,_pose(Pose()))]
        articulations=[];transforms=[];attachments=[];link_poses={}
        native_states=self._read_planning_native_states(env)
        for link_id,link in self._planning_native_links.items():
            pos,q,vel,ang,_,_=native_states[link.entity.idx];i=link.idx_local
            p=Pose(_vector(pos[i]),_vector(q[i]));link_poses[link.idx]=p
            links.append(PlanningLinkState(link_id,_pose(p),PlanningTwist(WORLD_FRAME,_vector(vel[i]),_vector(ang[i]))))
            frames.append(PlanningFrameState(_lfid(link.idx),_pose(p)))
        for index,entity in enumerate(self._spec.entities):
            p=entity.pose;twist=PlanningTwist(WORLD_FRAME)
            if entity.path in self._bodies and len(self._bodies[entity.path])==1:
                body=self._root(entity.path);native_root=self._native_pose(body,env);p=compose(native_root,self._init_root_offsets.get(entity.path,Pose()))
                _,_,v,w,_,_=native_states[body.idx];i=body.base_link.idx_local
                offset=np.asarray(p.position)-native_root.position
                twist=PlanningTwist(WORLD_FRAME,_vector(v[i]+np.cross(w[i],offset)),_vector(w[i]))
            entities.append(PlanningEntityState(_eid(index),_pose(p),twist))
            frames.append(PlanningFrameState(_efid(index),_pose(p)))
            descriptor=next(x for x in catalog.entities if x.entity_id==_eid(index))
            if descriptor.joint_ids:
                positions=[];velocities=[];units=[]
                for jid in descriptor.joint_ids:
                    joint=self._planning_joint_bindings[jid]
                    if joint is not None and joint.n_dofs:
                        _,_,_,_,q,v=native_states[joint.entity.idx]
                        value=float(q[joint.q_start-joint.entity.q_start])
                        name=joint.name.rsplit("/",1)[-1]
                        if name in entity.joint_names:
                            axis_index=entity.joint_names.index(name)
                            value=float(self._absolute_positions(entity.path,[value],[axis_index])[0])
                        positions.append(value);velocities.append(float(v[joint.dofs_idx_local[0]]))
                    else: positions.append(0.);velocities.append(0.)
                    units.append("m" if joint is not None and joint.type==self._gs.JOINT_TYPE.PRISMATIC else "rad")
                articulations.append(PlanningArticulationState(_eid(index),descriptor.joint_ids,tuple(positions),tuple(velocities),tuple(units)))
        for fid,(link,joint) in self._planning_joint_frames.items():
            child_pose=link_poses[link.idx]
            pos=_vector(numpy(joint.get_anchor_pos())[env]) if joint is not None and joint.n_dofs else child_pose.position
            frames.append(PlanningFrameState(fid,PlanningPose(WORLD_FRAME,pos,child_pose.orientation_xyzw)))
        for fid,(path,name) in self._planning_named.items():
            if name in self._named_frames[path]:
                owner,offset=self._named_frames[path][name];p=compose(link_poses[self._links[path][owner].idx],offset)
            else:
                p=link_poses[self._links[path][name].idx]
            frames.append(PlanningFrameState(fid,_pose(p)))
        for gid,geom in self._planning_geoms.items():
            local=Pose(_vector(geom.init_pos),_vector(xyzw(geom.init_quat)))
            transforms.append(PlanningGeometryTransform(gid,_pose(compose(link_poses[geom.link.idx],local))))
        for (aenv,aid),a in self._attachments.items():
            if aenv!=env: continue
            parent_index=self._planning_entity_indices[a.parent_path];child_index=self._planning_entity_indices[a.child_path]
            child_geoms=tuple(sorted(gid for gid,g in self._planning_geoms.items() if g.link.idx==a.child_link.idx))
            pf=_lfid(a.parent_link.idx);cf=_lfid(a.child_link.idx)
            relative=compose(inverse(link_poses[a.parent_link.idx]),link_poses[a.child_link.idx])
            attachments.append(PlanningAttachment(aid,_eid(parent_index),_eid(child_index),pf,cf,
                PlanningPose(pf,relative.position,relative.orientation_xyzw),
                child_geoms,_lid(a.parent_link.idx),_lid(a.child_link.idx)))
        revision=self._scene_sequence+1
        result=PlanningSceneState(self._session.descriptor.provider_id,self.world_id,self.generation,env,
            self.tick,revision,revision,1,1,catalog.content_sha256,revision,self._attachment_revision,
            WORLD_FRAME,tuple(sorted(entities,key=lambda x:x.entity_id)),tuple(sorted(links,key=lambda x:x.link_id)),
            tuple(sorted(frames,key=lambda x:x.frame_id)),tuple(sorted(articulations,key=lambda x:x.entity_id)),
            tuple(sorted(transforms,key=lambda x:x.geometry_id)),tuple(sorted(attachments,key=lambda x:x.attachment_id)))
        result.validate_against(catalog)
        self._planning_cache[env]=result
        return result

    def planning_scene_delta(self,base_sequence,environment_index=0):
        current=self.planning_scene_state(environment_index)
        if type(base_sequence) is not int or base_sequence<1 or base_sequence>=current.sequence:
            raise PlanningSceneDeltaContinuityError("no committed planning delta after base",operation="genesis.planning.delta")
        # Full resync is explicit; no history is fabricated for a base no longer retained.
        return PlanningSceneDelta(provider_id=current.provider_id,world_id=self.world_id,generation=self.generation,
            environment_index=environment_index,tick=self.tick,base_sequence=base_sequence,sequence=current.sequence,
            previous_world_revision=base_sequence,world_revision=current.world_revision,
            previous_catalog_revision=1,catalog_revision=1,previous_catalog_content_sha256=None,catalog_content_sha256=None,
            previous_geometry_revision=1,geometry_revision=1,previous_transform_revision=base_sequence,
            transform_revision=current.transform_revision,previous_attachment_revision=1,
            attachment_revision=current.attachment_revision,kind=PlanningSceneDeltaKind.RESYNC,resync_required=True)

    def resolve_planning_geometry(self,geometry_id,representation=None,environment_index=0):
        catalog=self.planning_scene_catalog(environment_index)
        geometry=next((g for g in catalog.geometries if g.geometry_id==geometry_id),None)
        if geometry is None:
            raise PlanningSceneNotFoundError("geometry absent",operation="genesis.geometry")
        if representation is not None and representation is not geometry.representation:
            raise PlanningSceneRepresentationError("representation differs from catalog",operation="genesis.geometry")
        payload=self._planning_resources[geometry_id]
        token=hashlib.sha256(f"{self._session.session_id}:{self.generation}:{environment_index}:{geometry_id}".encode()).hexdigest()
        descriptor=PlanningGeometryResourceDescriptor(self._session.descriptor.provider_id,self.world_id,
            self.generation,environment_index,1,1,catalog.content_sha256,token,geometry.resource_id,geometry_id,
            geometry.representation,PlanningGeometryStorageKind.IMMUTABLE_MEMORY,f"memory.{token}",
            geometry.content_profile,"m",PlanningGeometryAxisConvention.RIGHT_HANDED_Z_UP,
            geometry.resource_layout,len(payload),geometry.sha256)
        descriptor.validate_against(catalog)
        return GeometryLease(self,descriptor,payload)
