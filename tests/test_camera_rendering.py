"""Real public USD rasterization with calibrated rays and camera-local exclusion."""
import os
os.environ.setdefault('PYOPENGL_PLATFORM','egl')
import numpy as np
import pytest
from unirobosim import (
    CameraCalibrationSpec,CameraModality,CameraRenderExclusion,CameraSpec,
    EntityKind,EntityPath,EntitySpec,PhysicsSpec,Pose,WorldSpec,
    RenderRigidBodyState,RenderStateFrame,ArrayValue,
)
from unirobosim_genesis import create_provider,GenesisAdapterConfig
from test_native import build_input_for

pytestmark=pytest.mark.engine


def test_camera_exclusion_and_state_refresh(tmp_path, monkeypatch):
    from pxr import Usd,UsdGeom,UsdPhysics
    import trimesh
    file=tmp_path/'cube.usda';stage=Usd.Stage.CreateNew(str(file));root=UsdGeom.Xform.Define(stage,'/Object')
    stage.SetDefaultPrim(root.GetPrim());UsdGeom.SetStageMetersPerUnit(stage,1.);UsdGeom.SetStageUpAxis(stage,'Z')
    UsdPhysics.RigidBodyAPI.Apply(root.GetPrim());UsdPhysics.MassAPI.Apply(root.GetPrim()).CreateMassAttr(1.)
    mesh=trimesh.creation.box(extents=(.5,.5,.1));prim=UsdGeom.Mesh.Define(stage,'/Object/shape')
    prim.CreatePointsAttr(mesh.vertices);prim.CreateFaceVertexCountsAttr([3]*len(mesh.faces));prim.CreateFaceVertexIndicesAttr(mesh.faces.reshape(-1))
    prim.CreateDisplayColorAttr([(1.,0.,0.)]);UsdPhysics.CollisionAPI.Apply(prim.GetPrim())
    hidden=UsdGeom.Mesh.Define(stage,'/Object/hidden')
    hidden.CreatePointsAttr(mesh.vertices);hidden.CreateFaceVertexCountsAttr([3]*len(mesh.faces));hidden.CreateFaceVertexIndicesAttr(mesh.faces.reshape(-1))
    hidden.CreateVisibilityAttr(UsdGeom.Tokens.invisible)
    subset=UsdGeom.Subset.Define(stage,'/Object/hidden/material_faces')
    subset.CreateElementTypeAttr(UsdGeom.Tokens.face);subset.CreateIndicesAttr([0])
    stage.GetRootLayer().Save()
    build=build_input_for(file)
    target=EntityPath('/target');calibration=CameraCalibrationSpec('opencv_pinhole',(90.,0.,48.,0.,86.,37.,0.,0.,1.),(.05,-.005,.001,-.002,0.,0.,0.,0.))
    def camera(path,excluded):
        return EntitySpec(EntityPath(path),EntityKind.CAMERA_SENSOR,pose=Pose((0.,0.,2.)),
            camera=CameraSpec(width_px=100,height_px=80,modalities=(CameraModality.RGB,),calibration=calibration,
                render_exclusions=(CameraRenderExclusion(target,'shape'),) if excluded else ()))
    spec=WorldSpec('visual-exclusion',(EntitySpec(target,EntityKind.RIGID_BODY,asset_uri=file.as_uri()),
        camera('/camera/a',False),camera('/camera/b',True)),physics=PhysicsSpec(gravity_m_s2=(0.,0.,0.)),
        schema_version='unirobosim.world/v0alpha6',build_resource_manifest_sha256=build.manifest.sha256)
    cfg=GenesisAdapterConfig(device=os.environ.get('GENESIS_TEST_DEVICE','cuda'))
    with create_provider(cfg).open() as session:
        with session.build(spec,build_input=build) as world:
            def read(path):
                value=world.read_sensor(world.resolve(EntityPath(path))).channels[0].data
                return np.frombuffer(value.to_bytes(),np.uint8).reshape(value.shape)[0]
            a=read('/camera/a');b=read('/camera/b')
            assert a.shape==b.shape==(80,100,3)
            assert float(np.mean(np.abs(a.astype(float)-b)))>1
            # The red front face projects to the calibrated principal point.
            red=(a[...,0]>a[...,1]+20)&(a[...,0]>a[...,2]+20)
            y,x=np.nonzero(red)
            assert abs(float(x.mean())-48)<1.5 and abs(float(y.mean())-37)<1.5
            assert world._cameras[EntityPath('/camera/a')].scene.entities[0].n_vgeoms==1
            assert world._cameras[EntityPath('/camera/b')].scene.entities[0].n_vgeoms==1
            renderer_a=world._cameras[EntityPath('/camera/a')]
            renderer_b=world._cameras[EntityPath('/camera/b')]
            assert renderer_a.scene is renderer_b.scene
            from unirobosim_genesis.math import numpy as native_array
            visual=renderer_a.scene.entities[0]
            vertices=native_array(visual.get_vverts()).copy()
            assert np.array_equal(read('/camera/a'),a)
            assert np.array_equal(vertices,native_array(visual.get_vverts()))
            original=renderer_b._render_native
            def fail(*args,**kwargs): raise RuntimeError('injected native camera failure')
            monkeypatch.setattr(renderer_b,'_render_native',fail)
            with pytest.raises(RuntimeError,match='injected native camera failure'): read('/camera/b')
            assert np.array_equal(vertices,native_array(visual.get_vverts()))
            monkeypatch.setattr(renderer_b,'_render_native',original)
            assert np.array_equal(read('/camera/a'),a)
            before=world.tick
            def av(x):return ArrayValue((1,len(x)),tuple(float(v) for v in x))
            state=RenderRigidBodyState(world.resolve(target),av((.5,0,0)),av((0,0,0,1)),av((0,0,0)),av((0,0,0)))
            world.apply_render_state(RenderStateFrame(rigid_bodies=(state,)))
            moved=read('/camera/a');hidden=read('/camera/b')
            assert not np.array_equal(moved,a)
            assert np.array_equal(hidden,b)
            assert world.tick==before
