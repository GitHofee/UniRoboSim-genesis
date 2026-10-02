"""Batched public readback retains authored frames and selected environments."""
import json
import os
import time
from types import SimpleNamespace

import numpy as np
import pytest
from unirobosim import BoxGeometrySpec, EntityKind, EntityPath, EntitySpec, PhysicsSpec, WorldSpec
from unirobosim_genesis import GenesisAdapterConfig, create_provider
from unirobosim_genesis.math import numpy, xyzw
from unirobosim_genesis.planning import PlanningMixin


def _previous_read(bodies, env):
    return {b.idx: (numpy(b.get_links_pos())[env], xyzw(b.get_links_quat())[env],
        numpy(b.get_links_vel())[env], numpy(b.get_links_ang())[env],
        numpy(b.get_qpos())[env] if b.n_dofs else (),
        numpy(b.get_dofs_velocity())[env] if b.n_dofs else ()) for b in bodies}


@pytest.mark.engine
def test_batched_public_readback_matches_per_body_exactly_in_offset_world():
    # A normal adapter world owns initialization/teardown. The second small
    # public native scene tests offsets not exposed by portable BoxGeometrySpec.
    provider=create_provider(GenesisAdapterConfig(device=os.environ.get('GENESIS_TEST_DEVICE','cpu'),enable_cameras=False))
    spec=WorldSpec('readback-owner',(EntitySpec(EntityPath('/box'),EntityKind.RIGID_BODY,
        box=BoxGeometrySpec()),),physics=PhysicsSpec(gravity_m_s2=(0.,0.,0.)))
    with provider.open() as session, session.build(spec) as owner:
        gs=owner._gs
        scene=gs.Scene(show_viewer=False,sim_options=gs.options.SimOptions(dt=1/240,gravity=(0.,0.,0.)),
            rigid_options=gs.options.RigidOptions(enable_collision=False))
        try:
            bodies=[scene.add_entity(gs.morphs.Box(size=(.05,.06,.07),pos=(i*.1,0.,0.),fixed=True))
                for i in range(32)]
            moving=scene.add_entity(gs.morphs.Box(size=(.1,.2,.3),pos=(1.,2.,3.),
                quat=(float(np.cos(.2)),0.,0.,float(np.sin(.2))),
                offset_pos=(.13,-.09,.17),offset_quat=(float(np.cos(.1)),float(np.sin(.1)),0.,0.)))
            bodies.append(moving)
            scene.build(n_envs=2)
            moving.set_pos(np.asarray([[1.,2.,3.],[-2.,1.,4.]]))
            moving.set_dofs_velocity(np.asarray([[.2,.3,.4,.5,.6,.7],[-.4,.2,.1,-.2,.6,.3]]))
            reads=[]
            class CountReads:
                def __getattr__(self,name):
                    method=getattr(scene.rigid_solver,name)
                    def read(*args,**kwargs):
                        reads.append((name,kwargs));return method(*args,**kwargs)
                    return read
            view=SimpleNamespace(_scene=SimpleNamespace(rigid_solver=CountReads()),_bodies={'all':bodies})
            timings=[]
            for stage in ('initial','stepped'):
                if stage=='stepped':scene.step()
                for env in (1,0):
                    previous=_previous_read(bodies,env)
                    reads.clear()
                    current=PlanningMixin._read_planning_native_states(view,env)
                    assert len(reads)==4
                    assert all(kwargs['envs_idx']==[env] for _,kwargs in reads)
                    assert all(kwargs.get('relative') is True for name,kwargs in reads if name!='get_links_ang')
                    assert current.keys()==previous.keys()
                    for idx in previous:
                        for actual,expected in zip(current[idx],previous[idx],strict=True):
                            np.testing.assert_array_equal(actual,expected)
                # No simulation occurs while timing either read method.
                old=[];new=[]
                for _ in range(5):
                    start=time.perf_counter();_previous_read(bodies,1);old.append(time.perf_counter()-start)
                    start=time.perf_counter();PlanningMixin._read_planning_native_states(view,1);new.append(time.perf_counter()-start)
                timings.append(dict(stage=stage,old_median_s=float(np.median(old)),new_median_s=float(np.median(new))))
            assert not np.array_equal(numpy(moving.get_links_pos(relative=True)),numpy(moving.get_links_pos(relative=False)))
            assert not np.array_equal(numpy(moving.get_links_vel(relative=True)),numpy(moving.get_links_vel(relative=False)))
            print(json.dumps({'batch_readback':{'device':provider.config.device if hasattr(provider,'config') else os.environ.get('GENESIS_TEST_DEVICE','cpu'),
                'native_bodies':len(bodies),'environments':2,'bitwise_equal':True,
                'old_link_getter_calls':4*len(bodies),'new_link_getter_calls':4,'timings':timings}}))
        finally:
            scene.destroy()
