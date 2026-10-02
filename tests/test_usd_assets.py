"""Native USD inspection preserves source physics and requires explicit asset repair."""
import hashlib
import pytest
from pathlib import Path
from pxr import Usd, UsdGeom, UsdPhysics
from unirobosim import EntityKind, EntityPath, EntitySpec, FrozenMap
from unirobosim_genesis.usd_assets import materialize_usd
from unirobosim_genesis.usd_assets import _collision_signature
from unirobosim_genesis.config import ENGINE_VERSION


def make_scene(path):
    stage=Usd.Stage.CreateNew(str(path));root=UsdGeom.Xform.Define(stage,'/scene')
    stage.SetDefaultPrim(root.GetPrim());UsdGeom.SetStageMetersPerUnit(stage,1.)
    body=UsdGeom.Xform.Define(stage,'/scene/body');body.AddTranslateOp().Set((1.,2.,3.))
    UsdPhysics.RigidBodyAPI.Apply(body.GetPrim())
    cube=UsdGeom.Cube.Define(stage,'/scene/body/collider');cube.CreateSizeAttr(2.)
    cube.AddScaleOp().Set((.2,.3,.4));cube.CreateVisibilityAttr('invisible')
    UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
    visual=UsdGeom.Cube.Define(stage,'/scene/body/visual_only');visual.CreateSizeAttr(.2)
    joint=UsdPhysics.FixedJoint.Define(stage,'/scene/private_joint')
    joint.CreateBody1Rel().SetTargets([body.GetPath()])
    stage.GetRootLayer().Save()


def test_static_usd_inspection_preserves_colliders_physics_and_source(tmp_path):
    source=tmp_path/'source.usda';make_scene(source);before=source.read_bytes()
    entity=EntitySpec(EntityPath('/scene'),EntityKind.COMPOSITE_SCENE,asset_uri=source.as_uri(),
        metadata=FrozenMap({'composite_unbound_rigid_mode':'static'}))
    output=tmp_path/'derived';output.mkdir()
    path,named,provenance=materialize_usd(entity,None,output)
    assert path==source
    assert not tuple(output.iterdir())
    assert provenance['runtime_format_conversion'] is False
    assert source.read_bytes()==before
    assert provenance['used_layers'][0]['sha256']==hashlib.sha256(before).hexdigest()
    stage=Usd.Stage.Open(str(path))
    assert _collision_signature(stage)['collider_count']==1
    assert stage.GetPrimAtPath('/scene/body').HasAPI(UsdPhysics.RigidBodyAPI)
    joint=UsdPhysics.Joint(stage.GetPrimAtPath('/scene/private_joint'))
    assert joint and joint.GetJointEnabledAttr().Get() is True
    collision=stage.GetPrimAtPath('/scene/body/collider')
    visual=stage.GetPrimAtPath('/scene/body/visual_only')
    assert collision.HasAPI(UsdPhysics.CollisionAPI)
    assert UsdGeom.Imageable(collision).ComputeVisibility()=='invisible'
    assert UsdGeom.Imageable(visual).ComputeVisibility()=='inherited'
    assert not visual.HasAPI(UsdPhysics.CollisionAPI)
    assert not stage.GetPrimAtPath('/scene/body/collider/__unirobosim_collision')
    assert str(UsdGeom.XformCache().GetLocalToWorldTransform(collision))==str(
        UsdGeom.XformCache().GetLocalToWorldTransform(Usd.Stage.Open(str(source)).GetPrimAtPath('/scene/body/collider')))


@pytest.mark.parametrize('authored', [False, True])
def test_usd_inspection_reports_authored_or_schema_units_without_rewriting(tmp_path, authored):
    source=tmp_path/'units.usda'
    stage=Usd.Stage.CreateNew(str(source));root=UsdGeom.Xform.Define(stage,'/root')
    stage.SetDefaultPrim(root.GetPrim());root.AddTranslateOp().Set((1.,2.,3.))
    if authored:
        UsdGeom.SetStageMetersPerUnit(stage,.01);UsdGeom.SetStageUpAxis(stage,'Y')
    stage.GetRootLayer().Save();before=source.read_bytes()
    entity=EntitySpec(EntityPath('/body'),EntityKind.RIGID_BODY,asset_uri=source.as_uri())
    output=tmp_path/'derived';output.mkdir()
    path,_,provenance=materialize_usd(entity,None,output)
    derived=Usd.Stage.Open(str(path))
    assert path==source and not tuple(output.iterdir())
    assert UsdGeom.GetStageMetersPerUnit(derived)==.01
    assert UsdGeom.GetStageUpAxis(derived)=='Y'
    assert derived.HasAuthoredMetadata('metersPerUnit') is authored
    assert derived.HasAuthoredMetadata('upAxis') is authored
    assert provenance['meters_per_unit']==.01 and provenance['up_axis']=='Y'
    assert provenance['runtime_format_conversion'] is False
    assert source.read_bytes()==before
    assert UsdGeom.XformCache().GetLocalToWorldTransform(derived.GetDefaultPrim())==UsdGeom.XformCache().GetLocalToWorldTransform(root.GetPrim())


def test_collision_provenance_covers_composed_instance_proxies(tmp_path):
    prototype=tmp_path/'prototype.usda'
    stage=Usd.Stage.CreateNew(str(prototype))
    root=UsdGeom.Xform.Define(stage,'/asset');stage.SetDefaultPrim(root.GetPrim())
    collider=UsdGeom.Cube.Define(stage,'/asset/collider');collider.CreateSizeAttr(.2)
    UsdPhysics.CollisionAPI.Apply(collider.GetPrim())
    visual=UsdGeom.Cube.Define(stage,'/asset/visual');visual.CreateSizeAttr(.5)
    disabled=UsdGeom.Cube.Define(stage,'/asset/disabled')
    UsdPhysics.CollisionAPI.Apply(disabled.GetPrim()).CreateCollisionEnabledAttr(False)
    stage.GetRootLayer().Save()
    source=tmp_path/'instances.usda';stage=Usd.Stage.CreateNew(str(source))
    root=UsdGeom.Xform.Define(stage,'/scene');stage.SetDefaultPrim(root.GetPrim())
    for name,x in [('left',1.),('right',3.)]:
        instance=UsdGeom.Xform.Define(stage,'/scene/'+name)
        instance.GetPrim().GetReferences().AddReference(str(prototype))
        instance.GetPrim().SetInstanceable(True)
        instance.AddTranslateOp().Set((x,0.,0.))
    stage.GetRootLayer().Save()
    assert stage.GetPrimAtPath('/scene/left/collider').IsInstanceProxy()
    before={p:p.read_bytes() for p in (source,prototype)}
    signature=_collision_signature(stage)
    assert signature['collider_count']==2
    entity=EntitySpec(EntityPath('/scene'),EntityKind.STATIC_SCENE,asset_uri=source.as_uri())
    output=tmp_path/'derived';output.mkdir()
    derived,_,provenance=materialize_usd(entity,None,output)
    assert derived==source and not tuple(output.iterdir())
    assert _collision_signature(Usd.Stage.Open(str(derived)))==signature
    assert provenance['physics_importer']==f'official genesis-world=={ENGINE_VERSION}'
    assert {Path(row['path']) for row in provenance['used_layers']}==set(before)
    assert all(path.read_bytes()==raw for path,raw in before.items())
    # Instance-local transforms and prototype geometry both affect the inventory.
    instance=UsdGeom.Xformable(stage.GetPrimAtPath('/scene/right'))
    instance.GetOrderedXformOps()[0].Set((4.,0.,0.))
    assert _collision_signature(stage)!=signature
