"""Real official Genesis API checks for adapter-owned acceleration lowering."""
import os
import numpy as np
import pytest
from pxr import Gf, Usd, UsdGeom, UsdPhysics, Sdf
from unirobosim import (ArrayValue, ArticulationCommand, CommandMode, EntityKind,
    EntityPath, EntitySpec, PhysicsSpec, WorldSpec)
from unirobosim_genesis import create_provider, GenesisAdapterConfig
from unirobosim_genesis.drive_mapping import articulated_mass_matrix
from unirobosim_genesis.math import numpy
from test_native import build_input_for
from usd_profile_fixture import usd_profile_provider

pytestmark=pytest.mark.engine


def make_slider(path,mass,cap=1000.):
    stage=Usd.Stage.CreateNew(str(path));root=UsdGeom.Xform.Define(stage,'/slider');stage.SetDefaultPrim(root.GetPrim())
    UsdGeom.SetStageMetersPerUnit(stage,1.);UsdGeom.SetStageUpAxis(stage,'Z')
    for name,m in [('base',1.),('tip',mass)]:
        b=UsdGeom.Xform.Define(stage,'/slider/'+name);UsdPhysics.RigidBodyAPI.Apply(b.GetPrim())
        ma=UsdPhysics.MassAPI.Apply(b.GetPrim());ma.CreateMassAttr(m);ma.CreateCenterOfMassAttr((0,0,0));ma.CreateDiagonalInertiaAttr((.01,.01,.01));ma.CreatePrincipalAxesAttr(Gf.Quatf(1))
        g=UsdGeom.Cube.Define(stage,b.GetPath().AppendChild('collision'));g.CreateSizeAttr(.1);UsdPhysics.CollisionAPI.Apply(g.GetPrim())
    fixed=UsdPhysics.FixedJoint.Define(stage,'/slider/anchor');fixed.CreateBody1Rel().SetTargets(['/slider/base'])
    j=UsdPhysics.PrismaticJoint.Define(stage,'/slider/slide');j.CreateBody0Rel().SetTargets(['/slider/base']);j.CreateBody1Rel().SetTargets(['/slider/tip']);j.CreateAxisAttr('X');j.CreateLowerLimitAttr(-1);j.CreateUpperLimitAttr(1)
    j.GetPrim().CreateAttribute('physxJointAxis:linear:armature',Sdf.ValueTypeNames.Float).Set(.5)
    drive=UsdPhysics.DriveAPI.Apply(j.GetPrim(),'linear');drive.CreateTypeAttr('acceleration');drive.CreateStiffnessAttr(100.);drive.CreateDampingAttr(20.);drive.CreateMaxForceAttr(cap)
    root.GetPrim().CreateAttribute('newton:selfCollisionEnabled',Sdf.ValueTypeNames.Bool).Set(False)
    stage.GetRootLayer().Save()


def build_world(tmp_path,mass,cap=1000.):
    p=tmp_path/f'slider{mass}-{cap}.usda';make_slider(p,mass,cap);build_input=build_input_for(p)
    entity=EntitySpec(EntityPath('/robot'),EntityKind.ARTICULATION,asset_uri=p.as_uri(),joint_names=('slide',),initial_joint_positions=(.1,),joint_position_units=('m',))
    spec=WorldSpec('drive',(entity,),physics=PhysicsSpec(time_step_seconds=1/60,substeps=8,gravity_m_s2=(0.,0.,0.)),schema_version='unirobosim.world/v0alpha6',build_resource_manifest_sha256=build_input.manifest.sha256)
    provider=create_provider(GenesisAdapterConfig(device=os.environ.get('GENESIS_TEST_DEVICE','cpu')))
    provider=usd_profile_provider(p,provider)
    session=provider.open();world=session.build(spec,build_input=build_input)
    return session,world,entity


def test_mass_normalized_native_response_and_force_bypass(tmp_path):
    positions=[]
    for mass in (1.,10.):
        session,w,e=build_world(tmp_path,mass)
        try:
            body=w._root(e.path)
            properties={link.name:w._usd_robots[e.path]['bodies'][link.name.rsplit('/',1)[-1]] for link in body.links}
            assert numpy(body.get_dofs_armature())[0,0]==pytest.approx(.5)
            assert articulated_mass_matrix(body,properties)[0,0,0]==pytest.approx(mass+.5,rel=1e-6)
            assert numpy(body.get_dofs_kp())[0,0]==pytest.approx(100*(mass+.5),rel=1e-6)
            # Native force mode ignores the mapped position/velocity gains.
            w.apply_articulation_command(ArticulationCommand(w.resolve(e.path),CommandMode.EFFORT,ArrayValue((1,1),(1.,)),target_units=('N',)))
            before=articulated_mass_matrix(body,properties)
            w.step();assert numpy(body.get_mass_mat())==pytest.approx(before,rel=1e-5)
            assert numpy(body.get_dofs_velocity())[0,0]==pytest.approx((1/60)/(mass+.5),rel=1e-4)
            w.reset();assert w.read_articulation(w.resolve(e.path)).joint_positions.values[0]==pytest.approx(.1,abs=1e-7)
            w.apply_articulation_command(ArticulationCommand(w.resolve(e.path),CommandMode.POSITION,ArrayValue((1,1),(.2,)),target_units=('m',)))
            w.step(60);positions.append(w.read_articulation(w.resolve(e.path)).joint_positions.values[0])
        finally:session.close()
    assert positions[0]==pytest.approx(positions[1],abs=2e-6)
    assert positions[0]==pytest.approx(.2,abs=1e-3)


def test_zero_effort_cap_preserved(tmp_path):
    session,w,e=build_world(tmp_path,1.,cap=0.)
    try:
        body=w._root(e.path);lower,upper=body.get_dofs_force_range()
        assert numpy(lower)[0,0]==0 and numpy(upper)[0,0]==0
        w.apply_articulation_command(ArticulationCommand(w.resolve(e.path),CommandMode.POSITION,ArrayValue((1,1),(.8,)),target_units=('m',)))
        w.step(10)
        assert w.read_articulation(w.resolve(e.path)).joint_positions.values[0]==pytest.approx(.1,abs=1e-7)
    finally:session.close()


def test_coupled_mass_query_current_pose_and_reset(tmp_path):
    """Compare public Jacobian assembly to a native force-mode mass matrix."""
    p=tmp_path/'coupled.usda';make_slider(p,2.)
    stage=Usd.Stage.Open(str(p))
    b=UsdGeom.Xform.Define(stage,'/slider/arm');b.AddTranslateOp().Set((.3,0,0))
    UsdPhysics.RigidBodyAPI.Apply(b.GetPrim());ma=UsdPhysics.MassAPI.Apply(b.GetPrim())
    ma.CreateMassAttr(3.);ma.CreateCenterOfMassAttr((.2,.1,0));ma.CreateDiagonalInertiaAttr((.1,.2,.25))
    ma.CreatePrincipalAxesAttr(Gf.Quatf(float(np.cos(.2)),Gf.Vec3f(0,0,float(np.sin(.2)))))
    g=UsdGeom.Cube.Define(stage,'/slider/arm/collision');g.CreateSizeAttr(.1);UsdPhysics.CollisionAPI.Apply(g.GetPrim())
    j=UsdPhysics.RevoluteJoint.Define(stage,'/slider/hinge');j.CreateBody0Rel().SetTargets(['/slider/tip']);j.CreateBody1Rel().SetTargets(['/slider/arm']);j.CreateLocalPos0Attr((.3,0,0));j.CreateAxisAttr('Z');j.CreateLowerLimitAttr(-90.);j.CreateUpperLimitAttr(90.)
    stage.GetRootLayer().Save();build_input=build_input_for(p)
    e=EntitySpec(EntityPath('/robot'),EntityKind.ARTICULATION,asset_uri=p.as_uri(),joint_names=('slide','hinge'),initial_joint_positions=(.1,.3),joint_position_units=('m','rad'))
    spec=WorldSpec('coupled',(e,),physics=PhysicsSpec(time_step_seconds=1/600,gravity_m_s2=(0.,0.,0.)),schema_version='unirobosim.world/v0alpha6',build_resource_manifest_sha256=build_input.manifest.sha256)
    provider=create_provider(GenesisAdapterConfig(device=os.environ.get('GENESIS_TEST_DEVICE','cpu')))
    provider=usd_profile_provider(p,provider)
    with provider.open() as session:
        with session.build(spec,build_input=build_input) as w:
            body=w._root(e.path)
            properties={link.name:w._usd_robots[e.path]['bodies'][link.name.rsplit('/',1)[-1]] for link in body.links}
            first=articulated_mass_matrix(body,properties)
            assert abs(first[0,0,1])>.01 and np.linalg.eigvalsh(first).min()>0
            for q in ((.1,.3),(.2,-.5)):
                body.set_qpos(np.asarray([q]),qs_idx_local=w._q_indices[e.path])
                expected=articulated_mass_matrix(body,properties)
                w.apply_articulation_command(ArticulationCommand(w.resolve(e.path),CommandMode.EFFORT,ArrayValue((1,2),(0.,0.)),target_units=('N','N*m')))
                w.step()
                assert numpy(body.get_mass_mat())==pytest.approx(expected,rel=2e-5,abs=2e-6)
            assert not np.allclose(expected,first)
            w.reset()
            assert articulated_mass_matrix(body,properties)==pytest.approx(first,rel=2e-5,abs=2e-6)
