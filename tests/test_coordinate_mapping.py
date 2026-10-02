"""Pure coordinate formatting regression, separate from native physics gates."""
from types import SimpleNamespace
import numpy as np
from unirobosim import EntityKind,EntityPath,Pose
from unirobosim_genesis.world import GenesisWorld


def test_snapshot_uses_shared_absolute_usd_coordinates_and_native_asset_offset():
    path=EntityPath('/robot');entity=SimpleNamespace(path=path,kind=EntityKind.ARTICULATION,pose=Pose(),joint_names=('a','b'))
    body=SimpleNamespace(get_pos=lambda:np.array([[0,0,-.125]]),get_quat=lambda:np.array([[1,0,0,0]]),
        get_vel=lambda:np.array([[0,0,0]]),get_ang=lambda:np.array([[0,0,0]]),get_qpos=lambda i:np.array([[.1,-.2]]))
    w=GenesisWorld.__new__(GenesisWorld);w._spec=SimpleNamespace(entities=(entity,),environments=SimpleNamespace(count=1))
    w._bodies={path:[body]};w._q_indices={path:(0,1)};w._q_read_offsets={path:np.array([2.9,.4])}
    w._position_references=w._q_read_offsets;w._init_root_offsets={path:Pose((0,0,.125))}
    state=w._scene_entities()[0]
    assert np.allclose(state.joint_positions,[3.,.2])
    assert state.pose.position==(0.,0.,0.)
    assert np.allclose(w._position_targets(path,[3.,.2]),[.1,-.2])
    assert np.allclose(w._absolute_positions(path,[.1],indices=(0,)),[3.])
