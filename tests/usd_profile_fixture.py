"""Explicit USD-only asset copies for small native tests, never runtime conversion.

The fixture records every source/copy digest and authors compatibility repairs
before selecting the copy. It does not alter the Genesis package or source USD.
"""
from dataclasses import replace
from pathlib import Path
import hashlib
import json
import math
import numpy as np
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics
from unirobosim_genesis import create_provider


def _pin(path):
    return {'file':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}


def usd_profile_provider(source, provider, *, articulation=True, prim_path=None, static=False):
    source=Path(source);before=source.read_bytes()
    prepared=source.with_name(source.stem+'-native.usda')
    Usd.Stage.Open(str(source)).Flatten().Export(str(prepared))
    stage=Usd.Stage.Open(str(prepared));root=stage.GetDefaultPrim()
    cache=UsdGeom.XformCache();bodies={};joints={};closures=[]
    collision_names=[]
    for prim in stage.Traverse():
        if prim.HasAPI(UsdPhysics.CollisionAPI):
            if UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get() is False:continue
            collision_names.append(prim.GetName())
            UsdGeom.Imageable(prim).CreateVisibilityAttr('inherited')
            UsdGeom.Imageable(prim).CreatePurposeAttr('guide')
        if articulation and prim.HasAPI(UsdPhysics.RigidBodyAPI):
            mass=UsdPhysics.MassAPI(prim)
            axes=mass.GetPrincipalAxesAttr().Get() if mass.GetPrincipalAxesAttr().HasAuthoredValueOpinion() else Gf.Quatf(1)
            bodies[prim.GetName()]={'source_path':str(prim.GetPath()),'mass':float(mass.GetMassAttr().Get()),
                'com':list(mass.GetCenterOfMassAttr().Get()) if mass.GetCenterOfMassAttr().HasAuthoredValueOpinion() else [0.,0.,0.],
                'principal_inertia':list(mass.GetDiagonalInertiaAttr().Get()),
                'principal_axes_xyzw':list(axes.GetImaginary())+[axes.GetReal()]}
        if prim.IsA(UsdPhysics.RevoluteJoint) or prim.IsA(UsdPhysics.PrismaticJoint):
            angular=prim.IsA(UsdPhysics.RevoluteJoint);unit='angular' if angular else 'linear'
            joint=UsdPhysics.RevoluteJoint(prim) if angular else UsdPhysics.PrismaticJoint(prim)
            drive=UsdPhysics.DriveAPI(prim,unit);factor=180/math.pi if angular else 1.
            state=prim.GetAttribute(f'state:{unit}:physics:position')
            armature=prim.GetAttribute(f'physxJointAxis:{unit}:armature')
            joints[prim.GetName()]={'source_path':str(prim.GetPath()),'reference':float(state.Get())/factor if state and state.HasAuthoredValueOpinion() else 0.,
                'limits':[float(joint.GetLowerLimitAttr().Get())/factor,float(joint.GetUpperLimitAttr().Get())/factor],
                'stiffness':float(drive.GetStiffnessAttr().Get() or 0.)*factor,
                'damping':float(drive.GetDampingAttr().Get() or 0.)*factor,
                'max_force':float(drive.GetMaxForceAttr().Get()) if drive else 0.,
                'drive_type':str(drive.GetTypeAttr().Get() or 'force'),
                'armature':float(armature.Get()) if armature and armature.HasAuthoredValueOpinion() else 0.,
                'frictionloss':0.,'viscous_damping':0.}
    if articulation:
        # Keep the original physical base as a named link even when the importer
        # simplifies its fixed world edge. The added support carries no physics.
        support=UsdGeom.Xform.Define(stage,root.GetPath().AppendChild('world_support'))
        support.AddTranslateOp().Set((0.,0.,-.125))
        UsdPhysics.RigidBodyAPI.Apply(support.GetPrim())
        mass=UsdPhysics.MassAPI.Apply(support.GetPrim());mass.CreateMassAttr(0.);mass.CreateDiagonalInertiaAttr((0.,0.,0.))
        bodies['world_support']={'world_support':True,'source_path':str(support.GetPath()),'mass':0.,'com':[0.,0.,0.],'principal_inertia':[0.,0.,0.],'principal_axes_xyzw':[0.,0.,0.,1.]}
        for prim in list(stage.Traverse()):
            if prim.IsA(UsdPhysics.FixedJoint):
                joint=UsdPhysics.FixedJoint(prim)
                if not joint.GetBody0Rel().GetTargets():
                    joint.CreateBody0Rel().SetTargets([support.GetPath()])
                    joint.CreateLocalPos0Attr(tuple(np.asarray(joint.GetLocalPos0Attr().Get())+(0.,0.,.125)))
            if not prim.IsA(UsdPhysics.SphericalJoint):continue
            joint=UsdPhysics.SphericalJoint(prim)
            if not joint.GetExcludeFromArticulationAttr().Get():continue
            a,b=(str(r.GetTargets()[0]) for r in (joint.GetBody0Rel(),joint.GetBody1Rel()))
            original=stage.GetPrimAtPath(b);name='aux_'+prim.GetName();aux_path=root.GetPath().AppendChild(name)
            aux=UsdGeom.Xform.Define(stage,aux_path)
            aux.AddTransformOp().Set(cache.GetLocalToWorldTransform(original)*cache.GetLocalToWorldTransform(root).GetInverse())
            UsdPhysics.RigidBodyAPI.Apply(aux.GetPrim())
            props=dict(bodies[original.GetName()]);props['mass']/=2;props['principal_inertia']=[v/2 for v in props['principal_inertia']]
            for p in (original,aux.GetPrim()):
                m=UsdPhysics.MassAPI.Apply(p);m.CreateMassAttr(props['mass']);m.CreateDiagonalInertiaAttr(tuple(props['principal_inertia']))
                m.CreateCenterOfMassAttr(tuple(props['com']));q=props['principal_axes_xyzw'];m.CreatePrincipalAxesAttr(Gf.Quatf(q[3],Gf.Vec3f(*q[:3])))
            bodies[original.GetName()]=props;bodies[name]=dict(props,auxiliary=True,source_path=str(aux_path))
            j0=np.eye(4);j1=np.eye(4);j0[:3,3]=joint.GetLocalPos0Attr().Get();j1[:3,3]=joint.GetLocalPos1Attr().Get()
            closures.append({'name':prim.GetName(),'a':a,'b':b,'j0':j0.tolist(),'j1':j1.tolist(),'auxiliary_name':name,'original_body_name':original.GetName()})
            joint.CreateBody1Rel().SetTargets([aux_path]);joint.CreateExcludeFromArticulationAttr(False)
    stage.GetRootLayer().Save()
    metadata=prepared.with_suffix('.manifest.json')
    metadata.write_text(json.dumps({'output':str(prepared),'output_sha256':_pin(prepared)['sha256'],
        'bodies':bodies,'joints':joints,'closures':closures},indent=2))
    kwargs={'align':False,'fixed':bool(articulation or static),'visualization':False,'convexify':False,'decimate':False,'watertighten':None,
        'default_armature':0.,'collision_mesh_prim_patterns':['^'+n+'$' for n in sorted(set(collision_names))],
        'visual_mesh_prim_patterns':['^never_visual$']}
    if prim_path:kwargs['prim_path']=prim_path
    entry={'source_sha256':_pin(source)['sha256'],'source_files':[_pin(source)],'pinned_files':[_pin(prepared),_pin(metadata)],
        'loads':[dict(_pin(prepared),mode='add_stage' if static else 'add_entity',morph_kwargs=kwargs,material_kwargs={})]}
    if articulation:entry['robot_manifest']=str(metadata)
    profile=prepared.with_suffix('.profile.json');profile.write_text(json.dumps({'schema':'unirobosim-genesis-asset-profile/v1','entries':[entry]},indent=2))
    assert source.read_bytes()==before
    return create_provider(replace(provider.config,asset_profile=str(profile)))
