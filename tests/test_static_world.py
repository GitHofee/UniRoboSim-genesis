"""Native adapter routing for exact planar and volumetric static geometry."""
from dataclasses import replace
import os
import numpy as np
import pytest
from pxr import Usd, UsdGeom, UsdPhysics
from unirobosim import CapabilityNegotiationError, EntityKind, EntityPath, EntitySpec, FrozenMap, PhysicsSpec, Pose, WorldSpec
from unirobosim_genesis import GenesisAdapterConfig, create_provider
from unirobosim_genesis.math import numpy, rotation
from test_native import build_input_for


@pytest.mark.engine
def test_native_usd_stage_keeps_planar_and_volume_geometry_and_pose(tmp_path):
    source=tmp_path/'static.usda'
    stage=Usd.Stage.CreateNew(str(source))
    root=UsdGeom.Xform.Define(stage,'/scene');stage.SetDefaultPrim(root.GetPrim())
    UsdGeom.SetStageMetersPerUnit(stage,1.);UsdGeom.SetStageUpAxis(stage,'Z')
    cube=UsdGeom.Cube.Define(stage,'/scene/box');cube.CreateSizeAttr(.1)
    UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
    UsdPhysics.RigidBodyAPI.Apply(cube.GetPrim())
    points=np.asarray(((-1.,-1.,0.),(1.,-1.,0.),(1.,1.,0.),(-1.,1.,0.)))
    plane=UsdGeom.Mesh.Define(stage,'/scene/plane')
    plane.CreatePointsAttr(points);plane.CreateFaceVertexCountsAttr([3,3])
    plane.CreateFaceVertexIndicesAttr([0,1,2,0,2,3])
    plane.AddTranslateOp().Set((2.,3.,.4));plane.AddRotateXOp().Set(30.)
    UsdPhysics.CollisionAPI.Apply(plane.GetPrim())
    UsdPhysics.RigidBodyAPI.Apply(plane.GetPrim())
    UsdPhysics.MeshCollisionAPI.Apply(plane.GetPrim()).CreateApproximationAttr('none')
    transform=np.asarray(UsdGeom.XformCache().GetLocalToWorldTransform(plane.GetPrim())).T
    local=points@transform[:3,:3].T+transform[:3,3]
    stage.GetRootLayer().Save();build_input=build_input_for(source)
    pose=Pose((5.,-4.,2.),(0.,0.,float(np.sin(.3)),float(np.cos(.3))))
    # The public production capability is a composite scene with explicit static
    # unbound bodies; a standalone STATIC_SCENE capability is not advertised.
    entity=EntitySpec(EntityPath('/robot'),EntityKind.COMPOSITE_SCENE,pose=pose,asset_uri=source.as_uri(),
        metadata=FrozenMap({'composite_unbound_rigid_mode':'static'}))
    spec=WorldSpec('static-routing',(entity,),physics=PhysicsSpec(gravity_m_s2=(0.,0.,0.)),
        schema_version='unirobosim.world/v0alpha6',build_resource_manifest_sha256=build_input.manifest.sha256)
    provider=create_provider(GenesisAdapterConfig(device=os.environ.get('GENESIS_TEST_DEVICE','cpu'),enable_cameras=False))
    from usd_profile_fixture import usd_profile_provider
    provider=usd_profile_provider(source,provider,articulation=False,static=True)
    with provider.open() as session:
        unsupported=WorldSpec('standalone-static',(replace(entity,kind=EntityKind.STATIC_SCENE),),
            physics=spec.physics,schema_version=spec.schema_version,
            build_resource_manifest_sha256=build_input.manifest.sha256)
        with pytest.raises(CapabilityNegotiationError):
            session.build(unsupported,build_input=build_input)
        with session.build(spec,build_input=build_input) as world:
            bodies=world._bodies[entity.path]
            assert len(bodies)==2
            assert all(link.is_fixed for body in bodies for link in body.links)
            surface=next(g for body in bodies for g in body.geoms if not g.is_convex)
            actual=numpy(surface.get_verts()).reshape(-1,3)
            expected=local@rotation(pose.orientation_xyzw).T+pose.position
            assert actual.min(axis=0)==pytest.approx(expected.min(axis=0),abs=2e-6)
            assert actual.max(axis=0)==pytest.approx(expected.max(axis=0),abs=2e-6)
            world.step()
            assert np.isfinite(numpy(surface.get_verts())).all()
