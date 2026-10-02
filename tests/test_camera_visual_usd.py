"""Visual USD filtering preserves visible surfaces when removing parent subtrees."""
from pathlib import Path
from types import SimpleNamespace
import hashlib

import pytest
from unirobosim import EntityPath
from unirobosim_genesis.camera_rendering import _visual_usd


def _filter(source, directory, *, static):
    path = EntityPath('/asset')
    world = SimpleNamespace(
        _asset_selections={path: {'visual_usd': {'file': str(source)}, 'loads': [{}]}},
        _static_intent=lambda entity: static,
        _usd_robots={},
    )
    return _visual_usd(world, SimpleNamespace(path=path), directory)


def _mesh(stage, path, *, invisible=False, guide=False):
    from pxr import UsdGeom, UsdPhysics
    mesh = UsdGeom.Mesh.Define(stage, path)
    mesh.CreatePointsAttr([(0., 0., 0.), (1., 0., 0.), (0., 1., 0.)])
    mesh.CreateFaceVertexCountsAttr([3])
    mesh.CreateFaceVertexIndicesAttr([0, 1, 2])
    mesh.CreateDisplayColorAttr([(1., .25, 0.)])
    UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
    if invisible:
        mesh.CreateVisibilityAttr(UsdGeom.Tokens.invisible)
    if guide:
        mesh.CreatePurposeAttr(UsdGeom.Tokens.guide)
    subset = UsdGeom.Subset.Define(stage, path+'/material_faces')
    subset.CreateElementTypeAttr(UsdGeom.Tokens.face)
    subset.CreateIndicesAttr([0])
    subset.GetPrim().CreateRelationship('material:binding').SetTargets(['/Object/Material'])
    return mesh


@pytest.mark.parametrize('static', [False, True])
def test_nonvisual_parent_removal_preserves_visible_mesh_and_subset(tmp_path, static):
    pytest.importorskip('pxr')
    from pxr import Usd, UsdGeom, UsdPhysics
    source = tmp_path/'source.usda'
    stage = Usd.Stage.CreateNew(str(source))
    root = UsdGeom.Xform.Define(stage, '/Object').GetPrim()
    stage.SetDefaultPrim(root)
    UsdPhysics.RigidBodyAPI.Apply(root)
    UsdPhysics.MassAPI.Apply(root).CreateMassAttr(1.)
    visible = _mesh(stage, '/Object/visible')
    _mesh(stage, '/Object/hidden', invisible=True)
    # Both a hidden parent and its nested Gprim are queued for removal.
    _mesh(stage, '/Object/hidden/nested')
    _mesh(stage, '/Object/guide', guide=True)
    stage.GetRootLayer().Save()
    original = source.read_bytes()
    opts, provenance = _filter(source, tmp_path, static=static)
    result = Usd.Stage.Open(opts['file'])
    assert not result.GetPrimAtPath('/Object/hidden')
    assert not result.GetPrimAtPath('/Object/guide')
    assert provenance['visible_prims'] == ['/Object/visible']
    assert set(provenance['nonvisual_removed']) == {'/Object/hidden', '/Object/hidden/nested', '/Object/guide'}
    # A visible collider remains a visual, including its material subset.
    for prim in Usd.PrimRange(visible.GetPrim()):
        other = result.GetPrimAtPath(prim.GetPath())
        assert other
        for attr in prim.GetAttributes():
            assert attr.Get() == other.GetAttribute(attr.GetName()).Get()
        for rel in prim.GetRelationships():
            assert rel.GetTargets() == other.GetRelationship(rel.GetName()).GetTargets()
    assert result.GetDefaultPrim().HasAPI(UsdPhysics.RigidBodyAPI) is (not static)
    assert source.read_bytes() == original
    assert provenance['source_sha256'] == hashlib.sha256(original).hexdigest()


def test_static_joint_removal_does_not_visit_expired_descendants(tmp_path):
    pytest.importorskip('pxr')
    from pxr import Usd, UsdGeom, UsdPhysics
    source = tmp_path/'joints.usda'
    stage = Usd.Stage.CreateNew(str(source))
    root = UsdGeom.Xform.Define(stage, '/Object').GetPrim()
    stage.SetDefaultPrim(root)
    UsdPhysics.RigidBodyAPI.Apply(root)
    UsdPhysics.FixedJoint.Define(stage, '/Object/Joint')
    UsdPhysics.FixedJoint.Define(stage, '/Object/Joint/NestedJoint')
    _mesh(stage, '/Object/Joint/child')
    _mesh(stage, '/Object/visible')
    stage.GetRootLayer().Save()
    original = source.read_bytes()
    opts, provenance = _filter(source, tmp_path, static=True)
    result = Usd.Stage.Open(opts['file'])
    assert not result.GetPrimAtPath('/Object/Joint')
    assert not result.GetDefaultPrim().HasAPI(UsdPhysics.RigidBodyAPI)
    assert provenance['visible_prims'] == ['/Object/visible']
    assert result.GetPrimAtPath('/Object/visible/material_faces')
    assert source.read_bytes() == original
