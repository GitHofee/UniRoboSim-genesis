"""Adapter-owned USD Physics → MJCF conversion through public asset formats.

No Genesis importer or solver implementation is imported here. Derived files retain
one geometry per authored collider, the authored body frames, and explicit loop
anchors. Numerically positive authored inertias are restored with public setters
before the first simulation step when MJCF's stronger realizability check rejects
an otherwise positive definite tensor.
"""
from pathlib import Path
import hashlib
import json
import math
import xml.etree.ElementTree as ET
import numpy as np
from unirobosim import EntityKind, UnsupportedCapabilityError, ValidationError


def _fail(message, **details):
    raise UnsupportedCapabilityError(message, operation="genesis.usd_mjcf", details=details)


def _fmt(v):
    return " ".join(format(float(x), ".17g") for x in np.asarray(v).reshape(-1))


def _quat(q):
    return np.array([*q.GetImaginary(), q.GetReal()], dtype=float)


def _matrix(pos, quat):
    from scipy.spatial.transform import Rotation
    out=np.eye(4);out[:3,:3]=Rotation.from_quat(quat).as_matrix();out[:3,3]=pos
    return out


def _attrs(t):
    from scipy.spatial.transform import Rotation
    q=Rotation.from_matrix(t[:3,:3]).as_quat()
    return dict(pos=_fmt(t[:3,3]),quat=_fmt(q[[3,0,1,2]]))


def convert_articulation(source, entity, directory, provenance, substep_dt):
    from pxr import Usd, UsdGeom, UsdPhysics, UsdShade
    from scipy.spatial.transform import Rotation
    from .constraints import rigid_equality_params
    stage=Usd.Stage.Open(str(source));cache=UsdGeom.XformCache()
    unit=float(provenance["meters_per_unit"])
    if provenance["up_axis"]!="Z":_fail("articulated MJCF conversion currently requires authored Z up")
    if len(set(entity.scale_xyz))!=1:_fail("articulated MJCF conversion requires uniform entity scale")
    length=unit*entity.scale_xyz[0]
    prims=list(Usd.PrimRange(stage.GetPseudoRoot(),Usd.TraverseInstanceProxies()))
    bodies={str(p.GetPath()):p for p in prims if p.HasAPI(UsdPhysics.RigidBodyAPI)}
    if not bodies:_fail("USD articulation has no authored rigid bodies")
    def world(p):
        t=np.array(cache.GetLocalToWorldTransform(p),dtype=float).T
        t[:3,3]*=length
        if not np.allclose(t[:3,:3].T@t[:3,:3],np.eye(3),atol=1e-5):
            _fail("scaled rigid-body frame requires explicit inertia-frame conversion",path=str(p.GetPath()))
        return t
    transforms={k:world(p) for k,p in bodies.items()}
    def owner(path):
        p=stage.GetPrimAtPath(path)
        while p and not p.IsPseudoRoot():
            if str(p.GetPath()) in bodies:return str(p.GetPath())
            p=p.GetParent()
        return None
    def joint_side(j,i):
        targets=(j.GetBody0Rel() if i==0 else j.GetBody1Rel()).GetTargets()
        pos=(j.GetLocalPos0Attr() if i==0 else j.GetLocalPos1Attr()).Get()
        q=(j.GetLocalRot0Attr() if i==0 else j.GetLocalRot1Attr()).Get()
        local=_matrix(np.asarray(pos)*length,_quat(q))
        if not targets:return None,local,local
        if len(targets)!=1:_fail("joint has multiple body targets",joint=str(j.GetPath()))
        target=stage.GetPrimAtPath(targets[0]);o=owner(targets[0]);tw=world(target)
        anchored=tw@local
        return o,np.linalg.inv(transforms[o])@anchored if o else anchored,anchored
    edges={};closures=[];joints={}
    for p in prims:
        if not p.IsA(UsdPhysics.Joint):continue
        j=UsdPhysics.Joint(p)
        if j.GetJointEnabledAttr().Get() is False:continue
        a,j0,w0=joint_side(j,0);b,j1,w1=joint_side(j,1)
        item=dict(prim=p,a=a,b=b,j0=j0,j1=j1,w0=w0,w1=w1)
        if j.GetExcludeFromArticulationAttr().Get():
            if not p.IsA(UsdPhysics.SphericalJoint):_fail("excluded joint requires unsupported equality",joint=p.GetName())
            if not a or not b:_fail("closure must join two authored bodies",joint=p.GetName())
            closures.append(item);continue
        if not b:_fail("tree joint body1 has no authored rigid body",joint=p.GetName())
        if b in edges:_fail("rigid body has multiple tree parents",body=b)
        edges[b]=item
    root=ET.Element("mujoco",model=stage.GetDefaultPrim().GetName())
    ET.SubElement(root,"compiler",angle="radian",inertiafromgeom="false",fusestatic="false",balanceinertia="false")
    assets=ET.SubElement(root,"asset");wb=ET.SubElement(root,"worldbody");eq=ET.SubElement(root,"equality")
    elements={};inertias={};geometry=[];fixed_names={}
    outdir=Path(directory)/("mjcf-"+hashlib.sha256(str(source).encode()).hexdigest()[:16]);outdir.mkdir(exist_ok=True)
    def add_body(path,seen=()):
        if path in elements:return elements[path]
        if path in seen:_fail("joint tree is cyclic",body=path)
        p=bodies[path];edge=edges.get(path);parent=edge["a"] if edge else None
        parent_element=add_body(parent,seen+(path,)) if parent else wb
        relative=np.linalg.inv(transforms[parent])@transforms[path] if parent else transforms[path]
        el=ET.SubElement(parent_element,"body",name=p.GetName(),**_attrs(relative));elements[path]=el
        massapi=UsdPhysics.MassAPI(p);mass=massapi.GetMassAttr().Get();com=massapi.GetCenterOfMassAttr().Get();diag=massapi.GetDiagonalInertiaAttr().Get();axes=massapi.GetPrincipalAxesAttr().Get()
        if mass is None or com is None or diag is None or axes is None:_fail("body lacks explicit mass/COM/inertia",body=path)
        d=np.asarray(diag,dtype=float)*length**2;c=np.asarray(com,dtype=float)*length;q=_quat(axes)
        # USD's unauthored COM/axes sentinels request automatic inference. The
        # mass-only coordinate links have no geometry and use origin/identity.
        if not massapi.GetCenterOfMassAttr().HasAuthoredValueOpinion():c=np.zeros(3)
        if not massapi.GetPrincipalAxesAttr().HasAuthoredValueOpinion():q=np.array([0.,0.,0.,1.])
        automatic=not massapi.GetDiagonalInertiaAttr().HasAuthoredValueOpinion()
        if automatic:
            if entity.kind is not EntityKind.RIGID_BODY or len(bodies)!=1:
                _fail("automatic articulation inertia requires separate validation",body=path)
            d=np.ones(3)*1e-3  # Compile-only; official recompute_inertia replaces it.
        if not math.isfinite(mass) or mass<=0 or not np.isfinite(c).all() or not np.isfinite(d).all() or np.min(d)<=0:
            _fail("authored inertia must have finite positive mass and eigenvalues",body=path,mass=mass,inertia=d.tolist())
        temporary=bool(2*np.max(d)>np.sum(d));compile_d=np.repeat(np.mean(d),3) if temporary else d
        ET.SubElement(el,"inertial",mass=str(mass),pos=_fmt(c),quat=_fmt(q[[3,0,1,2]]),diaginertia=_fmt(compile_d))
        inertias[p.GetName()]={"mass":mass,"com":c.tolist(),"principal_inertia":d.tolist(),"principal_axes_xyzw":q.tolist(),"compile_placeholder":temporary or automatic,"automatic_inertia":automatic}
        if edge and not edge["prim"].IsA(UsdPhysics.FixedJoint):
            jp=edge["prim"];angular=jp.IsA(UsdPhysics.RevoluteJoint)
            if not angular and not jp.IsA(UsdPhysics.PrismaticJoint):_fail("unsupported articulation joint",joint=jp.GetName())
            schema=UsdPhysics.RevoluteJoint(jp) if angular else UsdPhysics.PrismaticJoint(jp)
            axis=np.eye(3)[{"X":0,"Y":1,"Z":2}[str(schema.GetAxisAttr().Get())]]
            relative_joint=np.linalg.inv(edge["w0"])@edge["w1"]
            reference=float(np.dot(Rotation.from_matrix(relative_joint[:3,:3]).as_rotvec(),axis)) if angular else float(relative_joint[:3,3]@axis)
            axis_body=edge["j1"][:3,:3]@axis
            factor=math.pi/180 if angular else length
            lo=float(schema.GetLowerLimitAttr().Get())*factor;hi=float(schema.GetUpperLimitAttr().Get())*factor
            axis_token="angular" if angular else "linear";drive=UsdPhysics.DriveAPI(jp,axis_token)
            def value(name,default):
                a=jp.GetAttribute(name);v=a.Get() if a else None
                return default if v is None else v
            arm=float(value(f"physxJointAxis:{axis_token}:armature",value("physxJoint:armature",0.)))
            static_friction=float(value(f"physxJointAxis:{axis_token}:staticFrictionEffort",value("urdf:dynamics:friction",0.)))
            dynamic_friction=float(value(f"physxJointAxis:{axis_token}:dynamicFrictionEffort",static_friction))
            viscous=float(value(f"physxJointAxis:{axis_token}:viscousFrictionCoefficient",0.))
            if static_friction!=dynamic_friction:
                _fail("distinct static/dynamic joint friction is unsupported",joint=jp.GetName())
            attr=dict(name=jp.GetName(),type="hinge" if angular else "slide",pos=_fmt(edge["j1"][:3,3]),axis=_fmt(axis_body),ref=str(reference),armature=str(arm),frictionloss=str(dynamic_friction),damping=str(viscous))
            if np.isfinite([lo,hi]).all():attr.update(limited="true",range=_fmt([lo,hi]))
            else:attr["limited"]="false"
            ET.SubElement(el,"joint",**attr)
            gain_scale=180/math.pi if angular else 1.
            stiffness=float(drive.GetStiffnessAttr().Get() or 0.)*gain_scale;damping=float(drive.GetDampingAttr().Get() or 0.)*gain_scale
            cap=drive.GetMaxForceAttr().Get();cap=float(cap) if cap is not None else math.inf
            joints[jp.GetName()]={"reference":reference,"stiffness":stiffness,"damping":damping,"drive_type":str(drive.GetTypeAttr().Get() or "force"),"max_force":cap,"armature":arm,"frictionloss":dynamic_friction,"viscous_damping":viscous,
                "limits":[lo,hi],"authored_velocity_limit":float(value(f"physxJointAxis:{axis_token}:maxJointVelocity",value("physxJoint:maxJointVelocity",math.inf)))*factor,
                "velocity_limit_enforcement":"not available as a public native Genesis 1.4.2 property",
                "authored_target_position":float(drive.GetTargetPositionAttr().Get() or 0.)*factor,
                "authored_target_velocity":float(drive.GetTargetVelocityAttr().Get() or 0.)*factor,
                "initial_state_policy":"WorldSpec absolute positions and zero velocities override USD state/targets",
                "authored_state_position":float(value(f"state:{axis_token}:physics:position",reference/factor))*factor,
                "authored_state_velocity":float(value(f"state:{axis_token}:physics:velocity",0.))*factor,
                "axis_units":"radians" if angular else "meters","source":str(jp.GetPath())}
        elif edge:fixed_names[p.GetName()]=edge["prim"].GetName()
        elif entity.kind is EntityKind.RIGID_BODY or not entity.metadata.to_dict().get("fixed_base",True):
            ET.SubElement(el,"freejoint",name=p.GetName()+"_free")
        return el
    for path in bodies:add_body(path)
    for item in closures:
        name=item["prim"].GetName();sites=[]
        for i,body in enumerate((item["a"],item["b"])):
            sid=f"{name}_anchor{i}";sites.append(sid)
            ET.SubElement(elements[body],"site",name=sid,pos=_fmt(item[f"j{i}"][:3,3]),size="0.0001",rgba="0 0 0 0")
        params=rigid_equality_params(substep_dt)
        ET.SubElement(eq,"connect",name=name,site1=sites[0],site2=sites[1],solref=_fmt(params[:2]),solimp=_fmt(params[2:]))
    for p in prims:
        if not p.HasAPI(UsdPhysics.CollisionAPI) or UsdPhysics.CollisionAPI(p).GetCollisionEnabledAttr().Get() is False:continue
        path=str(p.GetPath());o=owner(p.GetPath())
        if not o:_fail("articulation collider is outside authored rigid bodies",path=path)
        geometry_world=np.array(cache.GetLocalToWorldTransform(p),dtype=float).T
        geometry_world[:3,3]*=length
        transform=np.linalg.inv(transforms[o])@geometry_world
        linear=transform[:3,:3];scale=np.linalg.norm(linear,axis=0);rot=linear/scale
        if not np.allclose(rot.T@rot,np.eye(3),atol=1e-5):_fail("sheared collider is unsupported",path=path)
        rigid=transform.copy();rigid[:3,:3]=rot
        name="collision_"+hashlib.sha256(path.encode()).hexdigest()[:16]
        attrs=dict(name=name,group="3",**_attrs(rigid))
        material_mapping=None
        bound,_=UsdShade.MaterialBindingAPI(p).ComputeBoundMaterial("physics")
        if bound and bound.GetPrim().HasAPI(UsdPhysics.MaterialAPI):
            material=UsdPhysics.MaterialAPI(bound.GetPrim())
            sf=float(material.GetStaticFrictionAttr().Get());df=float(material.GetDynamicFrictionAttr().Get());rest=float(material.GetRestitutionAttr().Get())
            if sf!=df or rest!=0:
                _fail("movable rigid contact requires equal static/dynamic friction and zero restitution",path=path)
            attrs["friction"]=_fmt([df,0.,0.])
            material_mapping={"source":str(bound.GetPath()),"static_friction":sf,"dynamic_friction":df,"restitution":rest,"effective_friction":df}
        cooking={}
        for token,key in (("physxConvexDecompositionCollision:maxConvexHulls","max_convex_hull"),("physxConvexDecompositionCollision:hullVertexLimit","max_ch_vertex")):
            attribute=p.GetAttribute(token)
            if attribute and attribute.HasAuthoredValueOpinion():cooking[key]=int(attribute.Get())
        approximation=p.GetAttribute("physics:approximation").Get()
        if p.IsA(UsdGeom.Mesh):
            if approximation not in ("convexHull","convexDecomposition"):_fail("robot mesh approximation not supported by this converter",path=path,approximation=approximation)
            mesh=UsdGeom.Mesh(p);vertices=np.asarray(mesh.GetPointsAttr().Get(),dtype=float)*scale*length
            counts=np.asarray(mesh.GetFaceVertexCountsAttr().Get());indices=np.asarray(mesh.GetFaceVertexIndicesAttr().Get());faces=[];offset=0
            for count in counts:
                face=indices[offset:offset+count];offset+=count
                for i in range(1,int(count)-1):faces.append([face[0],face[i],face[i+1]])
            if not np.isfinite(vertices).all() or offset!=len(indices) or np.min(indices)<0 or np.max(indices)>=len(vertices):_fail("invalid collider mesh",path=path)
            if mesh.GetOrientationAttr().Get()==UsdGeom.Tokens.leftHanded:faces=np.asarray(faces)[:,::-1]
            meshfile=outdir/(name+".obj")
            meshfile.write_text("".join("v "+_fmt(v)+"\n" for v in vertices)+"".join("f "+" ".join(str(int(i)+1) for i in f)+"\n" for f in faces))
            ET.SubElement(assets,"mesh",name=name,file=str(meshfile));attrs.update(type="mesh",mesh=name)
        elif p.IsA(UsdGeom.Cube):attrs.update(type="box",size=_fmt(scale*float(UsdGeom.Cube(p).GetSizeAttr().Get())*length/2))
        elif p.IsA(UsdGeom.Cylinder):
            cylinder=UsdGeom.Cylinder(p);axis=str(cylinder.GetAxisAttr().Get());idx={"X":0,"Y":1,"Z":2}[axis];radial=np.delete(scale,idx)
            if not np.isclose(*radial):_fail("elliptical cylinder requires mesh conversion",path=path)
            orient=Rotation.from_euler("y",math.pi/2).as_matrix() if axis=="X" else Rotation.from_euler("x",-math.pi/2).as_matrix() if axis=="Y" else np.eye(3)
            rigid[:3,:3]=rot@orient;attrs.update(_attrs(rigid));attrs.update(type="cylinder",size=_fmt([float(cylinder.GetRadiusAttr().Get())*radial[0]*length,float(cylinder.GetHeightAttr().Get())*scale[idx]*length/2]))
        else:_fail("unsupported articulation collider type",path=path,type=p.GetTypeName())
        ET.SubElement(elements[o],"geom",**attrs)
        geometry.append({"source":path,"native_name":name,"owner":bodies[o].GetName(),"approximation":approximation,"type":attrs["type"],"material_mapping":material_mapping,"cooking_options":cooking})
    if any(g["approximation"]=="convexDecomposition" for g in geometry):
        if len(bodies)!=1 or len(geometry)!=1:
            _fail("convex decomposition requires independently cooked colliders",colliders=len(geometry))
    destination=outdir/"model.xml";ET.indent(root);ET.ElementTree(root).write(destination,encoding="unicode")
    conversion={"format":"MJCF","source_usd_sha256":hashlib.sha256(Path(source).read_bytes()).hexdigest(),"mjcf_sha256":hashlib.sha256(destination.read_bytes()).hexdigest(),"bodies":inertias,"joints":joints,"closures":[i["prim"].GetName() for i in closures],"geometry":geometry,"fixed_joint_names":fixed_names,"visuals":"source USD retained; conversion contains collision geometry, visual material conversion pending","visual_source_usd":str(source)}
    (outdir/"conversion.json").write_text(json.dumps(conversion,indent=2)+"\n")
    return destination,conversion
