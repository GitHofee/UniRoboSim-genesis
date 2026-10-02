"""Native render-state transactions: no physics advance and exact root/joint contracts."""
import os
from dataclasses import replace
import numpy as np
import pytest
from unirobosim import (
    ArrayValue, BoxGeometrySpec, EntityKind, EntityPath, EntitySpec, EnvironmentSpec,
    PhysicsSpec, Pose, RenderArticulationState, RenderRigidBodyState, RenderStateFrame,
    StaleHandleError, ValidationError, UnsupportedCapabilityError, WorldSpec,
)
from unirobosim_genesis import create_provider, GenesisAdapterConfig
from test_native import URDF, build_input_for

pytestmark = pytest.mark.engine


def av(rows):
    a = np.asarray(rows, dtype=float)
    return ArrayValue(a.shape, tuple(a.flat))


@pytest.fixture
def world(tmp_path):
    path=tmp_path/'slider.urdf';path.write_text(URDF)
    build_input=build_input_for(path)
    config=GenesisAdapterConfig(device=os.environ.get('GENESIS_TEST_DEVICE','cuda'),enable_cameras=False)
    spec=WorldSpec('render-native',(
        EntitySpec(EntityPath('/robot'),EntityKind.ARTICULATION,asset_uri=path.as_uri(),
            joint_names=('slide',),initial_joint_positions=(.1,),joint_position_units=('m',)),
        EntitySpec(EntityPath('/cube'),EntityKind.RIGID_BODY,pose=Pose((.3,0.,1.)),
            box=BoxGeometrySpec(dimensions_m=(.1,.2,.3),mass_kg=.2))),
        environments=EnvironmentSpec(2),physics=PhysicsSpec(gravity_m_s2=(0.,0.,0.)),
        schema_version='unirobosim.world/v0alpha6',build_resource_manifest_sha256=build_input.manifest.sha256)
    with create_provider(config).open() as session:
        with session.build(spec,build_input=build_input) as result:
            yield result


def rigid(world):
    return RenderRigidBodyState(world.resolve(EntityPath('/cube')),av([[2,3,4]]),
        av([[.5,.5,.5,.5]]),av([[.4,.5,.6]]),av([[.7,.8,.9]]),environment_indices=(1,))


def test_render_joint_and_root_pose_twist_atomic_selection(world):
    w=world;rh=w.resolve(EntityPath('/robot'));ch=w.resolve(EntityPath('/cube'))
    w.step(2);before=w.tick
    a=RenderArticulationState(rh,av([[.3]]),av([[.2]]),environment_indices=(1,))
    result=w.apply_render_state(RenderStateFrame(articulations=(a,),rigid_bodies=(rigid(w),)))
    assert result.tick==before==w.tick and result.state_revision==1
    assert (result.articulation_count,result.rigid_body_count)==(1,1)
    joint=w.read_articulation(rh)
    assert np.asarray(joint.joint_positions.rows())==pytest.approx(np.array([[.1],[.3]]),abs=1e-6)
    assert np.asarray(joint.joint_velocities.rows())==pytest.approx(np.array([[0.],[.2]]),abs=1e-6)
    root=w.read_rigid_body(ch)
    assert np.asarray(root.positions_m.rows())==pytest.approx(np.array([[.3,0,1],[2,3,4]]),abs=1e-6)
    assert np.asarray(root.linear_velocities_m_s.rows())[1]==pytest.approx([.4,.5,.6],abs=1e-6)
    assert np.asarray(root.angular_velocities_rad_s.rows())[1]==pytest.approx([.7,.8,.9],abs=1e-6)
    assert w.apply_render_state(RenderStateFrame(rigid_bodies=(rigid(w),))).state_revision==2
    w.reset();assert w.read_rigid_body(ch).positions_m.rows()[1]==pytest.approx((.3,0,1),abs=1e-6)


def test_render_prevalidates_entire_frame_and_quality(world):
    w=world;rh=w.resolve(EntityPath('/robot'));before=w.scene_snapshot()
    good=RenderArticulationState(rh,av([[.3]]),av([[.2]]),environment_indices=(0,))
    bad=replace(rigid(w),environment_indices=(2,))
    with pytest.raises(ValidationError):w.apply_render_state(RenderStateFrame(articulations=(good,),rigid_bodies=(bad,)))
    assert w.scene_snapshot()==before
    stale=replace(rigid(w),handle=replace(rigid(w).handle,generation=999))
    with pytest.raises(StaleHandleError):w.apply_render_state(RenderStateFrame(articulations=(good,),rigid_bodies=(stale,)))
    assert w.scene_snapshot()==before
    assert w.configure_render_quality(enable_global_illumination=False,enable_ambient_occlusion=False)==(False,False)
    with pytest.raises(UnsupportedCapabilityError):w.configure_render_quality(enable_global_illumination=True,enable_ambient_occlusion=False)
    with pytest.raises(ValidationError):w.configure_render_quality(enable_global_illumination=0,enable_ambient_occlusion=False)


def test_native_setter_failure_rolls_back_earlier_entity(world,monkeypatch):
    w=world;rh=w.resolve(EntityPath('/robot'));before=w.scene_snapshot();tick=w.tick
    a=RenderArticulationState(rh,av([[.31]]),av([[.2]]),environment_indices=(1,))
    def fail_after_pose(plan):
        body,_,envs,*_=plan
        body.set_pos(np.array([[8.,9.,10.]]),envs_idx=envs)
        raise RuntimeError('injected adapter transaction failure')
    monkeypatch.setattr(w,'_write_render_root',fail_after_pose)
    with pytest.raises(RuntimeError,match='transaction failure'):
        w.apply_render_state(RenderStateFrame(articulations=(a,),rigid_bodies=(rigid(w),)))
    assert w.tick==tick and w.scene_snapshot()==before
    assert getattr(w,'_render_state_revision',0)==0


def test_articulation_root_pose_and_axis_selection(world):
    w=world;rh=w.resolve(EntityPath('/robot'));before=w.tick
    state=RenderArticulationState(rh,av([[.4]]),av([[.3]]),environment_indices=(0,),
        degree_of_freedom_indices=(0,),root_positions_m=av([[1,2,3]]),
        root_orientations_xyzw=av([[0,0,0,1]]),root_linear_velocities_m_s=av([[0,0,0]]),
        root_angular_velocities_rad_s=av([[0,0,0]]))
    w.apply_render_state(RenderStateFrame(articulations=(state,)))
    root=w.read_rigid_body(rh)
    assert root.positions_m.rows()[0]==pytest.approx((1,2,3),abs=1e-6)
    assert w.read_articulation(rh).joint_positions.rows()[0]==pytest.approx((.4,),abs=1e-6)
    assert w.tick==before
