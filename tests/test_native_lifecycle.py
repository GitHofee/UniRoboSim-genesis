"""Process-exit regression: genuine CUDA teardown must happen on its owner thread."""
import os
import subprocess
import sys

import pytest


_WORKER = r'''
import os
from concurrent.futures import ThreadPoolExecutor
import genesis as gs
import torch
from unirobosim import (BoxGeometrySpec, EntityKind, EntityPath, EntitySpec, FrozenMap,
    LifecycleError, Pose, SessionState, WorldBuildError, WorldSpec)
from unirobosim_genesis import GenesisAdapterConfig, create_provider

device=os.environ.get('GENESIS_TEST_DEVICE','cuda')
config=GenesisAdapterConfig(device=device,logging_level='warning')
spec=WorldSpec('thread-owner',(EntitySpec(EntityPath('/box'),EntityKind.RIGID_BODY,
    pose=Pose((0.,0.,1.)),box=BoxGeometrySpec(dimensions_m=(.1,.1,.1),mass_kg=1.)),))

def native_scene():
    scene=gs.Scene(show_viewer=False)
    scene.add_entity(gs.morphs.Box(size=(.1,.1,.1),pos=(0.,0.,1.)))
    scene.build(n_envs=1)
    return scene

def wrong_thread(world,session):
    for call in (world.step,world.close,session.close):
        try:call()
        except LifecycleError:pass
        else:raise AssertionError('cross-thread native operation was accepted')

def wrong_thread_build(session):
    try:session.build(spec)
    except WorldBuildError:pass
    else:raise AssertionError('cross-thread runtime acquisition was accepted')

def run():
    mode=os.environ['GENESIS_LIFECYCLE_CASE']
    if mode=='owned':
        first=create_provider(config).open();w1=first.build(spec)
        w1.close()
        assert gs._initialized  # world close preserves runtime for session rebuild
        with ThreadPoolExecutor(max_workers=1) as foreign:
            foreign.submit(wrong_thread_build,first).result()
        assert gs._initialized and first.state is SessionState.OPEN
        w1=first.build(spec)
        second=create_provider(config).open();w2=second.build(spec)
        with ThreadPoolExecutor(max_workers=1) as foreign:
            foreign.submit(wrong_thread,w2,second).result()
        assert second.state is SessionState.READY
        first.close();first.close()
        assert gs._initialized
        w2.step();assert w2.read_rigid_body(w2.resolve(EntityPath('/box'))).positions_m.values[2]<1.
        second.close();second.close()
        assert not gs._initialized
        assert getattr(torch._GLOBAL_DEVICE_CONTEXT,'device_context',None) is None
    elif mode=='borrowed':
        gs.init(backend=gs.cpu if device=='cpu' else gs.cuda,logging_level='warning')
        external_storage=gs.use_ndarray
        external_device=torch.get_default_device();external_dtype=torch.get_default_dtype()
        external=native_scene()
        with create_provider(config).open() as session:
            with session.build(spec) as world:world.step()
        assert gs._initialized
        assert gs.use_ndarray is external_storage
        assert torch.get_default_device()==external_device
        assert torch.get_default_dtype()==external_dtype
        external.step();gs.destroy()
        # Direct Genesis callers own its Torch device context as well; SDK4
        # destroy does not restore it. The borrowing adapter must not remove it.
        torch.set_default_device(None)
    elif mode=='external_scene':
        session=create_provider(config).open();world=session.build(spec)
        external=native_scene()
        external_device=torch.get_default_device();external_dtype=torch.get_default_dtype()
        session.close()
        assert gs._initialized
        assert torch.get_default_device()==external_device
        assert torch.get_default_dtype()==external_dtype
        external.step();gs.destroy()
        torch.set_default_device(None)
    elif mode=='failed_build':
        with create_provider(config).open() as session:
            invalid=WorldSpec('bad-seed',spec.entities,
                metadata=FrozenMap({'fastsim_initial_generation_seed':-1}))
            try:session.build(invalid)
            except WorldBuildError:pass
            else:raise AssertionError('invalid build should fail')
            assert not gs._initialized
            with session.build(spec) as world:world.step()
        assert not gs._initialized
    elif mode=='failed_native_scene':
        from unirobosim_genesis.world import GenesisWorld
        original=GenesisWorld._build_native
        def build_then_fail(self):
            original(self)
            assert gs._scene_registry
            raise RuntimeError('injected failure after real native scene build')
        with create_provider(config).open() as session:
            GenesisWorld._build_native=build_then_fail
            try:
                try:session.build(spec)
                except WorldBuildError:pass
                else:raise AssertionError('injected failure should propagate')
            finally:GenesisWorld._build_native=original
            assert not gs._initialized
            assert not any(ref() is not None for ref in gs._scene_registry)
            with session.build(spec) as world:world.step()
        assert not gs._initialized
    elif mode=='prior_defaults':
        original_dtype=torch.get_default_dtype()
        torch.set_default_dtype(torch.float64)
        torch.set_default_device('cpu')
        with create_provider(config).open() as session:
            with session.build(spec) as world:world.step()
        assert torch.get_default_dtype() is torch.float64
        assert torch._GLOBAL_DEVICE_CONTEXT.device_context is not None
        assert torch.get_default_device()==torch.device('cpu')
        torch.set_default_device(None);torch.set_default_dtype(original_dtype)
    else:raise AssertionError(mode)
    print('Owner thread cleaned up successfully',flush=True)

with ThreadPoolExecutor(max_workers=1) as executor:
    executor.submit(run).result()
assert not gs._initialized
print('Main thread exiting normally',flush=True)
'''


@pytest.mark.engine
@pytest.mark.parametrize('case',['owned','borrowed','external_scene','failed_build','failed_native_scene','prior_defaults'])
def test_native_owner_thread_process_exit(case):
    # Match production. Quadrants' pytest plugin otherwise injects kernel
    # coverage even without --cov, adding a second native reset/atexit hook.
    # This test validates production lifecycle, not that optional SDK plugin.
    environment={**os.environ,'GENESIS_LIFECYCLE_CASE':case,'QD_KERNEL_COVERAGE':'0'}
    result=subprocess.run([sys.executable,'-c',_WORKER],env=environment,
        capture_output=True,text=True,timeout=180)
    assert result.returncode==0,result.stdout+'\n'+result.stderr
    assert 'Main thread exiting normally' in result.stdout
    assert 'CUDA_ERROR_INVALID_CONTEXT' not in result.stderr
