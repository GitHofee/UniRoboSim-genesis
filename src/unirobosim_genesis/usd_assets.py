"""Lossless USD composition in adapter-owned files; source assets stay untouched."""
from pathlib import Path
import hashlib
import json
import math
from urllib.parse import unquote, urlparse
from unirobosim import EntityKind, Pose, ValidationError
from .config import ENGINE_VERSION


def _pose(matrix):
    from pxr import Gf
    tf = Gf.Transform(matrix)
    q = tf.GetRotation().GetQuat()
    return Pose(tuple(float(v) for v in tf.GetTranslation()),
        tuple(float(v) for v in q.GetImaginary())+(float(q.GetReal()),))


def _collision_signature(stage):
    """Identity of collider geometry and authored world transforms before import."""
    from pxr import Usd, UsdGeom, UsdPhysics
    cache=UsdGeom.XformCache()
    entries=[]
    geometry_attributes={"points","faceVertexCounts","faceVertexIndices","orientation",
        "subdivisionScheme","holeIndices","size","radius","height","axis","extent",
        "physics:collisionEnabled","physics:approximation"}
    for prim in Usd.PrimRange(stage.GetPseudoRoot(), Usd.TraverseInstanceProxies()):
        if prim.HasAPI(UsdPhysics.CollisionAPI) and UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get() is not False:
            path=str(prim.GetPath())
            entries.append((path,prim.GetTypeName(),
                str(cache.GetLocalToWorldTransform(prim)),
                tuple((a.GetName(),str(a.Get())) for a in prim.GetAttributes()
                    if a.GetName() in geometry_attributes)))
    payload=json.dumps(sorted(entries),sort_keys=True).encode()
    return {"collider_count":len(entries),"geometry_and_transform_sha256":hashlib.sha256(payload).hexdigest(),
        "traversal":"composed-active-prims-including-instance-proxies"}


def materialize_usd(entity, lease, derived_root, *, source_override=None):
    """Inspect a USD without converting formats or rewriting physics at runtime."""
    from pxr import Usd, UsdGeom, UsdPhysics
    original = Path(unquote(urlparse(entity.asset_uri).path)).resolve()
    if lease is not None:
        snapshot=lease.selected_path(entity_id=entity.metadata.to_dict().get("fastsim_entity_id",entity.path.value), asset_uri=entity.asset_uri)
        if hashlib.sha256(original.read_bytes()).hexdigest()!=lease.entry_for_path(snapshot).sha256:
            raise ValidationError("USD root source changed after BuildInput snapshot",operation="genesis.usd")
    source=Path(source_override).resolve() if source_override else original
    stage=Usd.Stage.Open(str(source))
    if stage is None:
        raise ValidationError("cannot open USD stage",operation="genesis.usd")
    unit=UsdGeom.GetStageMetersPerUnit(stage)
    up_axis=str(UsdGeom.GetStageUpAxis(stage))
    layers=[]
    for layer in stage.GetUsedLayers():
        if not layer.anonymous:
            path=Path(layer.realPath).resolve()
            layers.append({"path":str(path),"sha256":hashlib.sha256(path.read_bytes()).hexdigest()})
    provenance={"source":str(original),"source_sha256":hashlib.sha256(original.read_bytes()).hexdigest(),
        "native_usd":str(source),"native_usd_sha256":hashlib.sha256(source.read_bytes()).hexdigest(),
        "used_layers":layers,"meters_per_unit":unit,"up_axis":up_axis,
        "physics_importer":f"official genesis-world=={ENGINE_VERSION}","runtime_format_conversion":False}
    fixed_names={}
    for prim in stage.Traverse():
        if not prim.IsA(UsdPhysics.FixedJoint):continue
        joint=UsdPhysics.Joint(prim)
        if joint.GetJointEnabledAttr().Get() is False or joint.GetExcludeFromArticulationAttr().Get():continue
        targets=joint.GetBody1Rel().GetTargets()
        if not targets:continue
        owner=stage.GetPrimAtPath(targets[0])
        while owner and not owner.IsPseudoRoot() and not owner.HasAPI(UsdPhysics.RigidBodyAPI):owner=owner.GetParent()
        if owner and not owner.IsPseudoRoot():fixed_names[str(owner.GetPath())]=prim.GetName()
    provenance["fixed_joint_names_by_link"]=fixed_names
    provenance["has_authored_articulation"]=any(p.HasAPI(UsdPhysics.ArticulationRootAPI) for p in stage.Traverse())
    provenance["self_collision_authoring"]=[]
    for prim in stage.Traverse():
        for name in ("physxArticulation:enabledSelfCollisions","newton:selfCollisionEnabled"):
            attribute=prim.GetAttribute(name)
            if attribute and attribute.HasAuthoredValueOpinion():
                value=attribute.Get()
                if type(value) is not bool:
                    raise ValidationError("authored self-collision policy must be boolean",operation="genesis.usd")
                provenance["self_collision_authoring"].append({"path":str(prim.GetPath()),"attribute":name,"value":value})
    # Link and native-named frame transforms remain authored USD values. Named
    # frames have no physics body; evaluate their fixed offset from their owner.
    by_name={}
    for prim in stage.Traverse():
        if prim.IsA(UsdGeom.Xformable):
            by_name.setdefault(prim.GetName(),[]).append(prim)
    cache=UsdGeom.XformCache()
    named={}
    declarations=entity.metadata.to_dict().get("planning_frame_declarations",{}).get("entries",[])
    for declaration in declarations:
        source_info=declaration["source"]
        if source_info["kind"] == "link":
            continue
        owner_name=declaration["owner_link"]
        frame_name=source_info["name"]
        owners=by_name.get(owner_name,[]); frames=by_name.get(frame_name,[])
        if len(owners)!=1 or len(frames)!=1:
            raise ValidationError("named USD frame is missing or ambiguous",operation="genesis.usd.frames",
                details={"name":frame_name,"owner":owner_name})
        relative=cache.GetLocalToWorldTransform(frames[0])*cache.GetLocalToWorldTransform(owners[0]).GetInverse()
        offset=_pose(relative)
        offset=Pose(tuple(float(v*unit*entity.scale_xyz[i]) for i,v in enumerate(offset.position)),offset.orientation_xyzw)
        named[declaration["name"]]=(owner_name,offset)
    return source,named,provenance
