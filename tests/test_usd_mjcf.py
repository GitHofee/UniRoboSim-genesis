"""Asset conversion boundaries independent of the Genesis runtime."""
import hashlib
import math
import numpy as np
import pytest
from pxr import Gf, Usd, UsdGeom, UsdPhysics
from unirobosim import EntityKind, EntityPath, EntitySpec
from unirobosim_genesis.usd_assets import materialize_usd
from unirobosim_genesis.usd_mjcf import convert_articulation


def test_reference_frames_inertia_closure_and_invisible_collider(tmp_path):
    mujoco=pytest.importorskip('mujoco')
    source=tmp_path/'robot.usda';s=Usd.Stage.CreateNew(str(source))
    r=UsdGeom.Xform.Define(s,'/robot');s.SetDefaultPrim(r.GetPrim())
    UsdGeom.SetStageMetersPerUnit(s,1.);UsdGeom.SetStageUpAxis(s,'Z')
    for name,position,angle in [('base',(0,0,0),0),('tip',(0,0,1),30)]:
        b=UsdGeom.Xform.Define(s,'/robot/'+name);b.AddTranslateOp().Set(position);b.AddRotateZOp().Set(angle)
        UsdPhysics.RigidBodyAPI.Apply(b.GetPrim());m=UsdPhysics.MassAPI.Apply(b.GetPrim())
        m.CreateMassAttr(2.);m.CreateCenterOfMassAttr((.1,0,0));m.CreateDiagonalInertiaAttr((.1,.2,.4));m.CreatePrincipalAxesAttr(Gf.Quatf(1))
        g=UsdGeom.Cube.Define(s,b.GetPath().AppendChild('hidden_collider'));g.CreateSizeAttr(.2)
        g.CreateVisibilityAttr('invisible');UsdPhysics.CollisionAPI.Apply(g.GetPrim())
    fixed=UsdPhysics.FixedJoint.Define(s,'/robot/anchor');fixed.CreateBody1Rel().SetTargets(['/robot/base'])
    j=UsdPhysics.RevoluteJoint.Define(s,'/robot/hinge');j.CreateBody0Rel().SetTargets(['/robot/base']);j.CreateBody1Rel().SetTargets(['/robot/tip']);j.CreateAxisAttr('Z');j.CreateLocalPos0Attr((0,0,1));j.CreateLowerLimitAttr(-90);j.CreateUpperLimitAttr(90)
    d=UsdPhysics.DriveAPI.Apply(j.GetPrim(),'angular');d.CreateTypeAttr('acceleration');d.CreateStiffnessAttr(2.);d.CreateDampingAttr(.2);d.CreateMaxForceAttr(0.)
    from pxr import Sdf
    j.GetPrim().CreateAttribute('physxJointAxis:angular:armature',Sdf.ValueTypeNames.Float).Set(.3)
    closure=UsdPhysics.SphericalJoint.Define(s,'/robot/loop');closure.CreateBody0Rel().SetTargets(['/robot/base']);closure.CreateBody1Rel().SetTargets(['/robot/tip']);closure.CreateLocalPos0Attr((0,0,1));closure.CreateExcludeFromArticulationAttr(True)
    s.GetRootLayer().Save();digest=hashlib.sha256(source.read_bytes()).hexdigest()
    e=EntitySpec(EntityPath('/robot'),EntityKind.ARTICULATION,asset_uri=source.as_uri(),joint_names=('hinge',),initial_joint_positions=(math.pi/6,))
    derived=tmp_path/'derived';derived.mkdir();usd,_,p=materialize_usd(e,None,derived)
    xml,c=convert_articulation(usd,e,derived,p,.001);m=mujoco.MjModel.from_xml_path(str(xml))
    assert m.nbody==3 and m.nv==1 and m.neq==1 and m.ngeom==2
    assert m.qpos0[0]==pytest.approx(math.pi/6)
    assert c['joints']['hinge']['stiffness']==pytest.approx(2*180/math.pi)
    assert c['joints']['hinge']['max_force']==0
    assert m.dof_armature[0]==pytest.approx(.3)
    assert np.allclose(m.site_pos[m.eq_obj1id[0]],(0,0,1))
    assert np.allclose(m.site_pos[m.eq_obj2id[0]],(0,0,0))
    assert c['bodies']['tip']['compile_placeholder'] is True
    assert c['bodies']['tip']['principal_inertia']==pytest.approx((.1,.2,.4))
    assert hashlib.sha256(source.read_bytes()).hexdigest()==digest
