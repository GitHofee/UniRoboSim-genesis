"""Real Genesis engine contract tests (no fake native objects)."""
import hashlib
import os
from pathlib import Path
import numpy as np
import pytest
from unirobosim import (
    ArrayValue, ArticulationCommand, BoxGeometrySpec, CommandMode, EntityKind,
    EntityPath, EntitySpec, EnvironmentSpec, KinematicTarget, PhysicsSpec, Pose,
    SceneCommand, SceneCommandKind, SceneCommandStatus, WorldSpec,
    PlanningGeometryResourceRevokedError, StaleHandleError,
)
from unirobosim_genesis import create_provider, GenesisAdapterConfig

pytestmark=pytest.mark.engine

def build_input_for(path):
    from unirobosim import BuildInput, BuildResourceEntry, BuildResourceManifest, BuildSourceEntry, LocalSourceIdentity
    raw=path.read_bytes();digest=hashlib.sha256(raw).hexdigest();st=path.stat()
    entry=BuildResourceEntry("robot","robot.test","robot.model","simulation","model/vnd.urdf+xml",
        path.as_uri(),path.as_uri(),f"sha256:{digest}",len(raw),digest,True,
        ("collision","planning","simulation","visual"),path.name)
    source=BuildSourceEntry(entry.resource_id,"local-file",str(path.parent),path.name,
        LocalSourceIdentity(st.st_dev,st.st_ino,st.st_mode,st.st_size,st.st_mtime_ns,st.st_ctime_ns),digest)
    return BuildInput(manifest=BuildResourceManifest((entry,)),sources=(source,))

URDF='''<robot name="slider">
<link name="base"><inertial><mass value="1"/><inertia ixx=".01" ixy="0" ixz="0" iyy=".01" iyz="0" izz=".01"/></inertial><collision><geometry><box size=".1 .1 .1"/></geometry></collision></link>
<link name="tip"><inertial><mass value=".1"/><inertia ixx=".001" ixy="0" ixz="0" iyy=".001" iyz="0" izz=".001"/></inertial><collision><geometry><box size=".05 .05 .05"/></geometry></collision></link>
<joint name="slide" type="prismatic"><parent link="base"/><child link="tip"/><origin xyz="0 0 .2"/><axis xyz="1 0 0"/><limit lower="-.5" upper=".5" effort="100" velocity="2"/></joint>
</robot>'''

@pytest.fixture(scope="module")
def provider():
    device=os.environ.get("GENESIS_TEST_DEVICE","cuda")
    return create_provider(GenesisAdapterConfig(device=device,enable_cameras=False,seed=17))

@pytest.fixture
def robot_world(tmp_path,provider):
    path=tmp_path/'robot.urdf';path.write_text(URDF)
    build_input=build_input_for(path)
    entities=(EntitySpec(EntityPath('/robot'),EntityKind.ARTICULATION,
        pose=Pose((0.,0.,1.)),asset_uri=path.as_uri(),joint_names=('slide',),
        initial_joint_positions=(.1,),joint_position_units=('m',)),
        EntitySpec(EntityPath('/cube'),EntityKind.RIGID_BODY,pose=Pose((.3,0.,1.)),
            box=BoxGeometrySpec(dimensions_m=(.05,.05,.05),mass_kg=.2)))
    with provider.open() as session:
        with session.build(WorldSpec('native-test',entities,physics=PhysicsSpec(time_step_seconds=1/240),
                schema_version='unirobosim.world/v0alpha6',build_resource_manifest_sha256=build_input.manifest.sha256),build_input=build_input) as world:
            yield world


def test_native_articulation_kinematics_planning_and_weld(robot_world):
    w=robot_world;rp=EntityPath('/robot');cp=EntityPath('/cube');rh=w.resolve(rp);ch=w.resolve(cp)
    assert w.read_articulation(rh).joint_positions.rows()[0][0]==pytest.approx(.1,abs=1e-6)
    frames=w.read_selected_kinematics((KinematicTarget('tip',rp,'tip'),))
    assert frames[0].pose.position==pytest.approx((.1,0.,1.2),abs=1e-6)
    catalog=w.planning_scene_catalog();state=w.planning_scene_state();state.validate_against(catalog)
    geometry=catalog.geometries[0];lease=w.resolve_planning_geometry(geometry.geometry_id)
    assert hashlib.sha256(lease.read()).hexdigest()==geometry.sha256
    w.apply_articulation_command(ArticulationCommand(rh,CommandMode.POSITION,
        ArrayValue((1,1),(.3,)),target_units=('m',)))
    w.step(120)
    assert w.read_articulation(rh).joint_positions.rows()[0][0]==pytest.approx(.3,abs=.025)
    w.apply_scene_command(SceneCommand('reset-cube','test','test',w.generation,SceneCommandKind.SET_POSE,cp,
        target_pose=Pose((.3,0.,1.))))
    attach=SceneCommand('attach-cube','test','test',w.generation,SceneCommandKind.ATTACH,cp,
        attachment_id='grasp',parent_entity_path=rp,parent_link_name='base',
        parent_T_child=Pose((.3,0.,0.)))
    result=w.apply_scene_command(attach);assert result.status is SceneCommandStatus.APPLIED
    assert w.apply_scene_command(attach).status is SceneCommandStatus.DUPLICATE
    w.step(120)
    held=w.read_rigid_body(ch).positions_m.rows()[0]
    assert held==pytest.approx((.3,0.,1.),abs=.001)
    state=w.planning_scene_state();state.validate_against(catalog);assert len(state.attachments)==1
    result=w.apply_scene_command(SceneCommand('detach-cube','test','test',w.generation,SceneCommandKind.DETACH,cp,attachment_id='grasp'))
    assert result.status is SceneCommandStatus.APPLIED
    w.step(120)
    assert w.read_rigid_body(ch).positions_m.rows()[0][2]<held[2]-.5
    assert not w.planning_scene_state().attachments
    before=w.tick;w.reset();assert w.tick==before
    assert w.read_rigid_body(ch).positions_m.rows()[0][2]==pytest.approx(1.,abs=1e-6)
    lease.close()
    with pytest.raises(PlanningGeometryResourceRevokedError):lease.read()


def test_environment_selection_and_persistent_wrench(provider):
    from unirobosim import RigidBodyCommand
    spec=WorldSpec('batch',(
        EntitySpec(EntityPath('/box'),EntityKind.RIGID_BODY,pose=Pose((0.,0.,1.)),
            box=BoxGeometrySpec(dimensions_m=(.1,.1,.1),mass_kg=1.)),),
        physics=PhysicsSpec(time_step_seconds=1/240,gravity_m_s2=(0.,0.,0.)),
        environments=EnvironmentSpec(2))
    with provider.open() as session:
        with session.build(spec) as w:
            h=w.resolve(EntityPath('/box'))
            w.apply_rigid_body_command(RigidBodyCommand(h,ArrayValue((1,3),(1.,0.,0.)),
                ArrayValue((1,3),(0.,0.,0.)),environment_indices=(1,)))
            w.step(120)
            positions=w.read_rigid_body(h).positions_m.rows()
            assert positions[0][0]==pytest.approx(0.,abs=1e-6)
            assert positions[1][0]==pytest.approx(.126,abs=.003)
            w.reset((1,));assert w.read_rigid_body(h).positions_m.rows()[1][0]==pytest.approx(0.,abs=1e-6)


def test_high_seed_bits_determinism_and_generation_lifetimes(provider):
    import random
    import torch
    import quadrants as qd
    from unirobosim import FrozenMap, LifecycleError
    seed=2551680430
    spec=WorldSpec('high-seed',(
        EntitySpec(EntityPath('/box'),EntityKind.RIGID_BODY,pose=Pose((0.,0.,1.)),
            box=BoxGeometrySpec(dimensions_m=(.1,.1,.1),mass_kg=1.)),),
        metadata=FrozenMap({'fastsim_initial_generation_seed':seed}))
    samples=[];positions=[]
    with provider.open() as session:
        for _ in range(2):
            with session.build(spec) as w:
                assert w._gs.SEED==seed & 0x7fffffff
                assert w._physics_provenance['seed']=={'original_uint32':seed,'native_seed':seed & 0x7fffffff,'mapping':'uint32-low31-v1','host_seed':seed}
                assert w._spec.metadata.to_dict()['fastsim_initial_generation_seed']==seed
                if samples:
                    with pytest.raises(StaleHandleError):w.read_rigid_body(old_handle)
                    with pytest.raises(PlanningGeometryResourceRevokedError):lease.read()
                old_handle=w.resolve(EntityPath('/box'))
                lease=w.resolve_planning_geometry(w.planning_scene_catalog().geometries[0].geometry_id)
                samples.append((random.random(),float(np.random.random()),float(torch.rand(()))))
                w.step(20);positions.append(w.read_rigid_body(old_handle).positions_m.rows())
            with pytest.raises(LifecycleError):w.read_rigid_body(old_handle)
    assert samples[0]==samples[1]
    assert positions[0]==positions[1]


def test_usd_offset_joint_named_frame_and_collision_roles(tmp_path,provider):
    from pxr import Usd, UsdGeom, UsdPhysics, Sdf
    from unirobosim import FrozenMap
    from unirobosim_genesis.math import rotation
    path=tmp_path/'offset.usda';stage=Usd.Stage.CreateNew(str(path))
    robot=UsdGeom.Xform.Define(stage,'/robot');stage.SetDefaultPrim(robot.GetPrim())
    UsdGeom.SetStageMetersPerUnit(stage,1.);UsdGeom.SetStageUpAxis(stage,'Z')
    UsdPhysics.ArticulationRootAPI.Apply(robot.GetPrim())
    for name,z in [('base',0.),('child',1.)]:
        link=UsdGeom.Xform.Define(stage,'/robot/'+name)
        link.AddTranslateOp().Set((.5*np.sin(.2),0.,.5+.5*np.cos(.2)) if name=='child' else (0.,0.,z))
        if name=='child':link.AddRotateYOp().Set(float(np.degrees(.2)))
        UsdPhysics.RigidBodyAPI.Apply(link.GetPrim())
        mass=UsdPhysics.MassAPI.Apply(link.GetPrim());mass.CreateMassAttr(1.)
        mass.CreateDiagonalInertiaAttr((.01,.01,.01))
        cube=UsdGeom.Cube.Define(stage,str(link.GetPath())+'/shape');cube.CreateSizeAttr(.1)
        cube.AddScaleOp().Set((1.,2.,3.));cube.CreateVisibilityAttr('invisible')
        UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
    tool=UsdGeom.Xform.Define(stage,'/robot/child/tool');tool.AddTranslateOp().Set((.1,0.,.25))
    fixed=UsdPhysics.FixedJoint.Define(stage,'/robot/fixed');fixed.CreateBody1Rel().SetTargets(['/robot/base'])
    joint=UsdPhysics.RevoluteJoint.Define(stage,'/robot/hinge')
    joint.CreateBody0Rel().SetTargets(['/robot/base']);joint.CreateBody1Rel().SetTargets(['/robot/child'])
    joint.CreateAxisAttr('Y');joint.CreateLocalPos0Attr((0.,0.,.5));joint.CreateLocalPos1Attr((0.,0.,-.5))
    joint.CreateLowerLimitAttr(-90.);joint.CreateUpperLimitAttr(90.)
    joint.GetPrim().CreateAttribute('state:angular:physics:position',Sdf.ValueTypeNames.Float).Set(float(np.degrees(.2)))
    closure=UsdPhysics.SphericalJoint.Define(stage,'/robot/closure')
    closure.CreateBody0Rel().SetTargets(['/robot/base']);closure.CreateBody1Rel().SetTargets(['/robot/child'])
    closure.CreateLocalPos0Attr((0.,.1,.5));closure.CreateLocalPos1Attr((0.,.1,-.5))
    closure.CreateExcludeFromArticulationAttr(True)
    drive=UsdPhysics.DriveAPI.Apply(joint.GetPrim(),'angular')
    drive.CreateStiffnessAttr(200.);drive.CreateDampingAttr(20.)
    stage.GetRootLayer().Save();before=path.read_bytes()
    from usd_profile_fixture import usd_profile_provider
    provider=usd_profile_provider(path,provider)
    build_input=build_input_for(path)
    entity=EntitySpec(EntityPath('/robot'),EntityKind.ARTICULATION,asset_uri=path.as_uri(),
        joint_names=('hinge',),initial_joint_positions=(.4,),joint_position_units=('rad',),
        metadata=FrozenMap({'planning_frame_declarations':{'entries':[
            {'name':'tool','owner_link':'child','source':{'kind':'native_named','name':'tool'}}]}}))
    spec=WorldSpec('offset-usd',(entity,),physics=PhysicsSpec(time_step_seconds=1/240,substeps=8),schema_version='unirobosim.world/v0alpha6',
        build_resource_manifest_sha256=build_input.manifest.sha256)
    with provider.open() as session:
        with session.build(spec,build_input=build_input) as w:
            state=w.planning_scene_state();catalog=w.planning_scene_catalog();state.validate_against(catalog)
            assert len(catalog.geometries)==2  # hidden colliders survive; visual meshes are not colliders
            assert len(catalog.point_closures)==1
            assert catalog.point_closures[0].anchor_a_m==pytest.approx((0.,.1,.5))
            assert catalog.point_closures[0].anchor_b_m==pytest.approx((0.,.1,-.5))
            assert catalog.point_closures[0].coupled_joint_ids==tuple(j.joint_id for j in catalog.joints if j.authored_name=='hinge')
            j=next(j for j in catalog.joints if j.authored_name=='hinge')
            anchor=next(f.world_pose for f in state.frames if f.frame_id==j.axis_frame_id)
            child=w.read_selected_kinematics((KinematicTarget('child',entity.path,'child'),))[0].pose
            assert anchor.position_m==pytest.approx((0.,0.,.5),abs=1e-6)
            assert np.linalg.norm(np.asarray(anchor.position_m)-child.position)==pytest.approx(.5,abs=1e-6)
            q=.4;r=np.array([[np.cos(q),0,np.sin(q)],[0,1,0],[-np.sin(q),0,np.cos(q)]])
            assert child.position==pytest.approx(np.array([0,0,.5])+r@np.array([0,0,.5]),abs=1e-6)
            tool_pose=w.read_selected_kinematics((KinematicTarget('tool',entity.path,'tool'),))[0].pose
            assert tool_pose.position==pytest.approx(np.array(child.position)+r@np.array([.1,0,.25]),abs=1e-6)
            assert rotation(child.orientation_xyzw)==pytest.approx(r,abs=1e-6)
            assert w.read_articulation(w.resolve(entity.path)).joint_positions.values[0]==pytest.approx(.4,abs=1e-6)
            w.apply_articulation_command(ArticulationCommand(w.resolve(entity.path),CommandMode.POSITION,
                ArrayValue((1,1),(.6,)),target_units=('rad',)))
            w.step(180)
            assert w.read_articulation(w.resolve(entity.path)).joint_positions.values[0]==pytest.approx(.6,abs=.015)
            w.reset()
            assert w.read_articulation(w.resolve(entity.path)).joint_positions.values[0]==pytest.approx(.4,abs=1e-6)
            assert path.read_bytes()==before


def test_usd_asset_root_and_physical_link_pose_twist_are_distinct(tmp_path,provider):
    from pxr import Usd, UsdGeom, UsdPhysics
    from unirobosim import RigidBodyCommand
    from unirobosim_genesis.math import compose, inverse
    path=tmp_path/'root_offset.usda';stage=Usd.Stage.CreateNew(str(path))
    root=UsdGeom.Xform.Define(stage,'/asset');stage.SetDefaultPrim(root.GetPrim())
    UsdGeom.SetStageMetersPerUnit(stage,1.);UsdGeom.SetStageUpAxis(stage,'Z')
    body=UsdGeom.Xform.Define(stage,'/asset/body');body.AddTranslateOp().Set((.2,-.1,.3))
    body.AddRotateZOp().Set(30.)
    UsdPhysics.RigidBodyAPI.Apply(body.GetPrim())
    mass=UsdPhysics.MassAPI.Apply(body.GetPrim());mass.CreateMassAttr(1.)
    mass.CreateDiagonalInertiaAttr((.01,.01,.01))
    cube=UsdGeom.Cube.Define(stage,'/asset/body/cube');cube.CreateSizeAttr(.1)
    UsdPhysics.CollisionAPI.Apply(cube.GetPrim());stage.GetRootLayer().Save()
    from usd_profile_fixture import usd_profile_provider
    provider=usd_profile_provider(path,provider,articulation=False,prim_path='/asset/body')
    build_input=build_input_for(path)
    pose=Pose((1.,2.,3.),(0.,0.,float(np.sqrt(.5)),float(np.sqrt(.5))))
    entity=EntitySpec(EntityPath('/object'),EntityKind.RIGID_BODY,pose=pose,asset_uri=path.as_uri())
    spec=WorldSpec('root-offset',(entity,),physics=PhysicsSpec(time_step_seconds=1/240,gravity_m_s2=(0.,0.,0.)),
        schema_version='unirobosim.world/v0alpha6',build_resource_manifest_sha256=build_input.manifest.sha256)
    with provider.open() as session:
        with session.build(spec,build_input=build_input) as w:
            handle=w.resolve(entity.path);initial=w.read_rigid_body(handle)
            source_offset=Pose((.2,-.1,.3),(0.,0.,float(np.sin(np.pi/12)),float(np.cos(np.pi/12))))
            expected=compose(pose,source_offset)
            physical=w.read_selected_kinematics((KinematicTarget('body',entity.path,'body'),))[0]
            assert physical.pose.position==pytest.approx(expected.position,abs=1e-6)
            assert physical.pose.orientation_xyzw==pytest.approx(expected.orientation_xyzw,abs=1e-6)
            assert initial.positions_m.rows()[0]==pytest.approx(expected.position,abs=1e-6)
            assert initial.orientations_xyzw.rows()[0]==pytest.approx(expected.orientation_xyzw,abs=1e-6)
            assert w.scene_snapshot().entities[0].pose.position==pytest.approx(pose.position,abs=1e-6)
            w.apply_rigid_body_command(RigidBodyCommand(handle,ArrayValue((1,3),(0.,0.,0.)),ArrayValue((1,3),(0.,0.,.02))))
            w.step(20)
            rigid=w.read_rigid_body(handle);scene=w.scene_snapshot().entities[0]
            planning=w.planning_scene_state().entities[0]
            assert planning.pose.position_m==pytest.approx(scene.pose.position,abs=1e-6)
            assert planning.pose.orientation_xyzw==pytest.approx(scene.pose.orientation_xyzw,abs=1e-6)
            physical=w.read_selected_kinematics((KinematicTarget('body',entity.path,'body'),))[0]
            offset=np.asarray(scene.pose.position)-physical.pose.position
            expected_velocity=np.asarray(physical.linear_velocity_m_s)+np.cross(physical.angular_velocity_rad_s,offset)
            assert np.linalg.norm(expected_velocity-physical.linear_velocity_m_s)>1e-3
            assert rigid.linear_velocities_m_s.rows()[0]==pytest.approx(physical.linear_velocity_m_s,abs=1e-6)
            assert scene.linear_velocity_m_s==pytest.approx(expected_velocity,abs=1e-6)
            assert planning.twist.linear_m_s==pytest.approx(expected_velocity,abs=1e-6)


def test_authored_self_collision_policy_preserves_external_contacts(tmp_path,provider):
    from pxr import Usd, UsdGeom, UsdPhysics, Sdf
    path=tmp_path/'collision_policy.usda';stage=Usd.Stage.CreateNew(str(path))
    root=UsdGeom.Xform.Define(stage,'/robot');stage.SetDefaultPrim(root.GetPrim())
    UsdGeom.SetStageMetersPerUnit(stage,1.);UsdGeom.SetStageUpAxis(stage,'Z')
    UsdPhysics.ArticulationRootAPI.Apply(root.GetPrim())
    root.GetPrim().CreateAttribute('physxArticulation:enabledSelfCollisions',Sdf.ValueTypeNames.Bool).Set(False)
    for name,z in [('base',0.),('slider',.3),('tip',0.)]:
        body=UsdGeom.Xform.Define(stage,'/robot/'+name);body.AddTranslateOp().Set((0.,0.,z))
        UsdPhysics.RigidBodyAPI.Apply(body.GetPrim())
        mass=UsdPhysics.MassAPI.Apply(body.GetPrim());mass.CreateMassAttr(1.);mass.CreateDiagonalInertiaAttr((.01,.01,.01))
        shape=UsdGeom.Cube.Define(stage,'/robot/'+name+'/shape');shape.CreateSizeAttr(.2)
        UsdPhysics.CollisionAPI.Apply(shape.GetPrim())
    fixed=UsdPhysics.FixedJoint.Define(stage,'/robot/anchor');fixed.CreateBody1Rel().SetTargets(['/robot/base'])
    slide=UsdPhysics.PrismaticJoint.Define(stage,'/robot/slide')
    slide.CreateBody0Rel().SetTargets(['/robot/base']);slide.CreateBody1Rel().SetTargets(['/robot/slider']);slide.CreateAxisAttr('X')
    slide.CreateLowerLimitAttr(-1.);slide.CreateUpperLimitAttr(1.)
    tip=UsdPhysics.FixedJoint.Define(stage,'/robot/tip_mount')
    tip.CreateBody0Rel().SetTargets(['/robot/slider']);tip.CreateBody1Rel().SetTargets(['/robot/tip'])
    tip.CreateLocalPos0Attr((0.,0.,-.3));stage.GetRootLayer().Save()
    from usd_profile_fixture import usd_profile_provider
    provider=usd_profile_provider(path,provider)
    build_input=build_input_for(path)
    robot=EntitySpec(EntityPath('/robot'),EntityKind.ARTICULATION,asset_uri=path.as_uri(),joint_names=('slide',),initial_joint_positions=(0.,),joint_position_units=('m',))
    box=EntitySpec(EntityPath('/box'),EntityKind.RIGID_BODY,pose=Pose((.14,0.,0.)),box=BoxGeometrySpec(dimensions_m=(.1,.1,.1),mass_kg=1.))
    spec=WorldSpec('collision-policy',(robot,box),physics=PhysicsSpec(gravity_m_s2=(0.,0.,0.)),
        schema_version='unirobosim.world/v0alpha6',build_resource_manifest_sha256=build_input.manifest.sha256)
    with provider.open() as session:
        with session.build(spec,build_input=build_input) as w:
            assert w._physics_provenance['enable_self_collision'] is False
            assert w._scene.rigid_solver._enable_self_collision is False
            catalog=w.planning_scene_catalog()
            assert any(j.authored_name=='tip_mount' and j.joint_type.value=='fixed' for j in catalog.joints)
            w.step()
            contacts=w._root(robot.path).get_contacts()
            valid=contacts['valid_mask'].cpu().numpy()[0]
            a=contacts['link_a'].cpu().numpy()[0][valid];b=contacts['link_b'].cpu().numpy()[0][valid]
            links={link.idx for link in w._root(robot.path).links}
            assert len(a)>0
            assert not any(int(x) in links and int(y) in links for x,y in zip(a,b))
            assert w.read_contact(w.resolve(box.path)).in_contact.values[0]


def test_native_weld_capacity_rejection_rolls_back_and_partial_reset(tmp_path,provider):
    from unirobosim import RigidBodyCommand
    path=tmp_path/'slider.urdf';path.write_text(URDF);build_input=build_input_for(path)
    robot=EntitySpec(EntityPath('/robot'),EntityKind.ARTICULATION,pose=Pose((0.,0.,1.)),
        asset_uri=path.as_uri(),joint_names=('slide',),initial_joint_positions=(0.,),joint_position_units=('m',))
    boxes=tuple(EntitySpec(EntityPath(f'/box{i}'),EntityKind.RIGID_BODY,pose=Pose((.3+.1*i,0.,1.)),
        box=BoxGeometrySpec(dimensions_m=(.05,.05,.05),mass_kg=.2)) for i in range(9))
    spec=WorldSpec('weld-capacity',(robot,*boxes),environments=EnvironmentSpec(2),
        physics=PhysicsSpec(time_step_seconds=1/240,gravity_m_s2=(0.,0.,0.)),
        schema_version='unirobosim.world/v0alpha6',build_resource_manifest_sha256=build_input.manifest.sha256)
    with provider.open() as session:
        with session.build(spec,build_input=build_input) as w:
            for i,box in enumerate(boxes[:8]):
                result=w.apply_scene_command(SceneCommand(f'attach{i}','test','test',w.generation,
                    SceneCommandKind.ATTACH,box.path,environment_index=1,attachment_id=f'weld{i}',
                    parent_entity_path=robot.path,parent_link_name='base',parent_T_child=Pose((.3+.1*i,0.,0.))))
                assert result.status is SceneCommandStatus.APPLIED
            last=w.resolve(boxes[-1].path)
            w.apply_rigid_body_command(RigidBodyCommand(last,ArrayValue((1,3),(1.,0.,0.)),
                ArrayValue((1,3),(0.,0.,.01)),environment_indices=(1,)))
            w.step(3);before=w.read_rigid_body(last)
            result=w.apply_scene_command(SceneCommand('overflow','test','test',w.generation,
                SceneCommandKind.ATTACH,boxes[-1].path,environment_index=1,attachment_id='overflow',
                parent_entity_path=robot.path,parent_link_name='base',parent_T_child=Pose((4.,0.,0.))))
            assert result.status is SceneCommandStatus.REJECTED
            after=w.read_rigid_body(last)
            assert after.positions_m.values==pytest.approx(before.positions_m.values,abs=1e-6)
            assert after.orientations_xyzw.values==pytest.approx(before.orientations_xyzw.values,abs=1e-6)
            assert after.linear_velocities_m_s.values==pytest.approx(before.linear_velocities_m_s.values,abs=1e-6)
            assert after.angular_velocities_rad_s.values==pytest.approx(before.angular_velocities_rad_s.values,abs=1e-6)
            assert len(w.planning_scene_state(1).attachments)==8
            assert not w.planning_scene_state(0).attachments
            w.reset((1,))
            assert not w.planning_scene_state(1).attachments
            result=w.apply_scene_command(SceneCommand('after-reset','test','test',w.generation,
                SceneCommandKind.ATTACH,boxes[-1].path,environment_index=1,attachment_id='new',
                parent_entity_path=robot.path,parent_link_name='base'))
            assert result.status is SceneCommandStatus.APPLIED


def test_official_same_weld_pair_cross_environment_rejection(tmp_path,provider):
    path=tmp_path/'slider.urdf';path.write_text(URDF);build_input=build_input_for(path)
    robot=EntitySpec(EntityPath('/robot'),EntityKind.ARTICULATION,pose=Pose((0.,0.,1.)),
        asset_uri=path.as_uri(),joint_names=('slide',),initial_joint_positions=(0.,),joint_position_units=('m',))
    cube=EntitySpec(EntityPath('/cube'),EntityKind.RIGID_BODY,pose=Pose((.3,0.,1.)),
        box=BoxGeometrySpec(dimensions_m=(.05,.05,.05),mass_kg=.2))
    spec=WorldSpec('independent-welds',(robot,cube),environments=EnvironmentSpec(2),
        physics=PhysicsSpec(time_step_seconds=1/240),schema_version='unirobosim.world/v0alpha6',
        build_resource_manifest_sha256=build_input.manifest.sha256)
    with provider.open() as session:
        with session.build(spec,build_input=build_input) as w:
            handle=w.resolve(cube.path)
            def attach(env,command_id):
                return w.apply_scene_command(SceneCommand(command_id,'test','test',w.generation,
                    SceneCommandKind.ATTACH,cube.path,environment_index=env,attachment_id='grasp',
                    parent_entity_path=robot.path,parent_link_name='base',parent_T_child=Pose((.3,0.,0.))))
            assert attach(0,'attach-0').status is SceneCommandStatus.APPLIED
            before=w.read_rigid_body(handle)
            rejected=attach(1,'attach-1')
            assert rejected.status is SceneCommandStatus.REJECTED
            assert rejected.error_code=='native_cross_environment_pair_unsupported'
            assert w.read_rigid_body(handle)==before
            assert len(w.planning_scene_state(0).attachments)==1
            assert not w.planning_scene_state(1).attachments
            w.reset((0,))
            assert attach(1,'after-reset-1').status is SceneCommandStatus.APPLIED
            w.step(60)
            positions=w.read_rigid_body(handle).positions_m.rows()
            assert positions[1]==pytest.approx((.3,0.,1.),abs=.001)
            assert positions[0][2]<1.
