import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import pytest
from pxr import Gf, Usd, UsdGeom, UsdPhysics, UsdShade
import trimesh
from unirobosim import EntityKind, EntityPath, EntitySpec
from unirobosim_genesis.static_scene import materialize_static_scene, _load_cooked_cache, _write_cooked_cache, _normalize_declared_convex


def _mesh(stage,path,geometry,approximation='none'):
    mesh=UsdGeom.Mesh.Define(stage,path)
    mesh.CreatePointsAttr([Gf.Vec3f(*p) for p in geometry.vertices])
    mesh.CreateFaceVertexCountsAttr([3]*len(geometry.faces))
    mesh.CreateFaceVertexIndicesAttr(geometry.faces.reshape(-1).tolist())
    UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
    UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim()).CreateApproximationAttr(approximation)
    return mesh


def _scene(tmp_path):
    path=tmp_path/'source.usda';stage=Usd.Stage.CreateNew(str(path))
    root=UsdGeom.Xform.Define(stage,'/root');stage.SetDefaultPrim(root.GetPrim())
    UsdGeom.SetStageMetersPerUnit(stage,1.);UsdGeom.SetStageUpAxis(stage,'Z')
    return path,stage


def _convert(path,stage,tmp_path,**kwargs):
    stage.GetRootLayer().Save()
    entity=EntitySpec(EntityPath('/environment'),EntityKind.STATIC_SCENE,asset_uri=path.as_uri())
    return materialize_static_scene(entity,path,tmp_path,{'meters_per_unit':1.,'up_axis':'Z'},**kwargs)


def test_static_groups_preserve_transformed_bounds_and_source(tmp_path):
    path,stage=_scene(tmp_path)
    box=_mesh(stage,'/root/box',trimesh.creation.box(extents=(1,2,3)),'boundingCube')
    box.AddTranslateOp().Set((3,4,5));box.AddRotateZOp().Set(30);box.AddScaleOp().Set((2,1,1))
    parts=[]
    for x in (-1,1):
        part=trimesh.creation.box(extents=(.3,.3,.3));part.apply_translation((x,0,0));parts.append(part)
    raw=_mesh(stage,'/root/nonconvex',trimesh.util.concatenate(parts),'meshSimplification')
    disabled=_mesh(stage,'/root/off',trimesh.creation.box(),'none')
    UsdPhysics.CollisionAPI(disabled.GetPrim()).CreateCollisionEnabledAttr(False)
    marker=UsdGeom.Xform.Define(stage,'/root/marker');UsdPhysics.CollisionAPI.Apply(marker.GetPrim())
    stage.GetRootLayer().Save();digest=hashlib.sha256(path.read_bytes()).hexdigest()
    converted=_convert(path,stage,tmp_path,static_convex_sdf_max_res=16)
    assert len(converted.groups)==2
    assert converted.groups[0].material_kwargs=={'sdf_min_res':16,'sdf_max_res':16}
    assert converted.groups[1].material_kwargs=={}
    p=converted.provenance
    assert p['source_colliders']==2 and p['derived_geometry_count']==2
    assert p['disabled_collision_prims']==['/root/off']
    assert p['nongeometric_collision_markers']==['/root/marker']
    primitive=p['physics_geometry'][0]['geometry'][0]['primitive']
    assert np.fromstring(primitive['pos'],sep=' ')==pytest.approx((3,4,5))
    assert np.fromstring(primitive['size'],sep=' ')==pytest.approx((1,1,1.5))
    assert p['physics_geometry'][1]['derived_policy']=='unsimplified-source-triangles'
    assert p['physics_geometry'][1]['geometry'][0]['faces']==len(raw.GetFaceVertexCountsAttr().Get())
    assert hashlib.sha256(path.read_bytes()).hexdigest()==digest
    assert converted.visual_source_usd==path
    for group in converted.groups:
        root=ET.parse(group.morph_kwargs['file']).getroot()
        assert not root.findall('.//joint') and not root.findall('.//freejoint')


def test_convex_cooking_never_merges_distinct_primitives(tmp_path):
    path,stage=_scene(tmp_path)
    for i in range(2):
        g=_mesh(stage,f'/root/box{i}',trimesh.creation.box(),'convexDecomposition')
        g.AddTranslateOp().Set((i*10,0,0))
    converted=_convert(path,stage,tmp_path)
    assert converted.provenance['derived_geometry_count']==2
    assert len(converted.groups[0].geometry_names)==2
    assert {r['source'] for r in converted.provenance['physics_geometry']}=={'/root/box0','/root/box1'}
    a,b=[r['geometry'][0]['world_aabb'] for r in converted.provenance['physics_geometry']]
    assert a[1][0]<b[0][0]


def test_mesh_local_coordinates_and_pose_reconstruct_exact_world_geometry(tmp_path):
    from scipy.spatial.transform import Rotation
    path,stage=_scene(tmp_path)
    shape=trimesh.creation.box(extents=(.002,.2,.7))
    shape.apply_translation((3,2,-1))
    mesh=_mesh(stage,'/root/box',shape,'convexHull')
    mesh.AddTranslateOp().Set((130,40,15))
    mesh.AddRotateZOp().Set(37)
    mesh.AddScaleOp().Set((2,1,1))
    converted=_convert(path,stage,tmp_path)
    root=ET.parse(converted.groups[0].morph_kwargs['file']).getroot()
    geom=root.find('.//geom');asset=root.find('.//asset/mesh')
    local=trimesh.load(asset.attrib['file'],process=False)
    assert np.max(np.abs(local.vertices.mean(axis=0)))<1e-12
    pos=np.fromstring(geom.attrib['pos'],sep=' ')
    quat=np.fromstring(geom.attrib['quat'],sep=' ')
    world=local.vertices@Rotation.from_quat(quat[[1,2,3,0]]).as_matrix().T+pos
    transform=np.asarray(UsdGeom.XformCache().GetLocalToWorldTransform(mesh.GetPrim())).T
    expected=np.asarray(mesh.GetPointsAttr().Get())@transform[:3,:3].T+transform[:3,3]
    assert np.sort(world,axis=0)==pytest.approx(np.sort(expected,axis=0),abs=1e-12)
    assert [world.min(axis=0),world.max(axis=0)]==pytest.approx(np.asarray(converted.provenance['physics_geometry'][0]['geometry'][0]['world_aabb']))


def test_authored_physics_material_mapping_is_explicit_and_warned(tmp_path):
    path,stage=_scene(tmp_path)
    mesh=_mesh(stage,'/root/box',trimesh.creation.box(),'boundingCube')
    material=UsdShade.Material.Define(stage,'/root/material')
    api=UsdPhysics.MaterialAPI.Apply(material.GetPrim())
    api.CreateStaticFrictionAttr(.8);api.CreateDynamicFrictionAttr(.7);api.CreateRestitutionAttr(.05)
    UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(material,materialPurpose='physics')
    with pytest.warns(RuntimeWarning,match='rigid-rigid restitution'):
        converted=_convert(path,stage,tmp_path)
    mapping=converted.provenance['physics_geometry'][0]['material_mapping']
    assert mapping['effective_friction']==pytest.approx(.7)
    assert mapping['source_static_friction']==pytest.approx(.8)
    assert mapping['nonzero_restitution_unsupported'] is True
    assert converted.provenance['material_limitations']=={'colliders_with_distinct_friction':1,'colliders_with_unsupported_restitution':1}


def test_corrupt_or_wrong_input_cache_cannot_replace_collision_geometry(tmp_path):
    mesh=trimesh.creation.box();path=tmp_path/'entry.npz'
    _write_cooked_cache(path,'input-key',[(mesh.vertices,mesh.faces)])
    assert _load_cooked_cache(path,'different-input') is None
    assert np.array_equal(_load_cooked_cache(path,'input-key')[0][0],mesh.vertices)
    with np.load(path,allow_pickle=False) as stored:
        data={k:stored[k] for k in stored.files}
    data['v0']=data['v0']+10
    np.savez_compressed(path,**data)
    assert _load_cooked_cache(path,'input-key') is None
    assert list(tmp_path.glob('*.tmp'))==[]


def test_missing_dynamic_friction_uses_authored_static_not_schema_zero(tmp_path):
    path,stage=_scene(tmp_path)
    mesh=_mesh(stage,'/root/box',trimesh.creation.box(),'boundingCube')
    material=UsdShade.Material.Define(stage,'/root/material')
    UsdPhysics.MaterialAPI.Apply(material.GetPrim()).CreateStaticFrictionAttr(.8)
    UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(material,materialPurpose='physics')
    converted=_convert(path,stage,tmp_path)
    mapping=converted.provenance['physics_geometry'][0]['material_mapping']
    assert mapping['source_dynamic_friction'] is None
    assert mapping['source_restitution'] is None
    assert mapping['effective_friction']==pytest.approx(.8)


def test_real_coacd_near_convex_output_is_normalized_with_bounded_evidence():
    path=Path(__file__).parent/'fixtures'/'static-coacd-near-convex.json'
    data=json.loads(path.read_text());v=np.asarray(data['vertices']);f=np.asarray(data['faces'])
    assert not trimesh.Trimesh(v,f,process=False).is_convex
    out_v,out_f,evidence=_normalize_declared_convex(v,f,'fixture')
    assert trimesh.Trimesh(out_v,out_f,process=False).is_convex
    assert evidence['relative_volume_difference']<1e-8
    assert evidence['bounds_difference_m']==0
    assert evidence['source_geometry_sha256']!=evidence['derived_geometry_sha256']


@pytest.mark.engine
def test_public_convex_import_repairs_float32_roundoff_without_merging_primitives(tmp_path):
    gs=pytest.importorskip('genesis')
    from scipy.spatial.transform import Rotation
    from scipy.spatial import cKDTree
    fixture=json.loads((Path(__file__).parent/'fixtures/static-mjcf-convex-roundoff.json').read_text())
    shape=trimesh.Trimesh(fixture['vertices'],fixture['faces'],process=False)
    assert shape.is_convex
    gs.init(backend=gs.cpu,logging_level='warning',seed=0)
    scene=None
    try:
        path,stage=_scene(tmp_path)
        for i in range(2):
            mesh=_mesh(stage,f'/root/convex{i}',shape,'convexHull')
            mesh.AddTranslateOp().Set((i*10,4,2))
        converted=_convert(path,stage,tmp_path,static_convex_sdf_max_res=16)
        group=converted.groups[0]
        assert group.morph_kwargs['convexify'] is True
        assert group.morph_kwargs['decompose_object_error_threshold']==float('inf')
        scene=gs.Scene(show_viewer=False)
        body=scene.add_entity(gs.morphs.MJCF(**group.morph_kwargs),material=gs.materials.Rigid(**group.material_kwargs))
        assert len(body.geoms)==2
        for i,g in enumerate(body.geoms):
            assert g.is_convex and g.link.is_fixed
            assert g.metadata['name']==group.geometry_names[i]
            q=np.asarray(g.init_quat)
            actual=g.init_verts@Rotation.from_quat(q[[1,2,3,0]]).as_matrix().T+g.init_pos
            expected=np.asarray(shape.vertices,dtype=np.float32).astype(float)+(i*10,4,2)
            assert cKDTree(expected).query(actual)[0].max()<1e-6
    finally:
        if scene is not None:scene.destroy()
        gs.destroy()


@pytest.mark.engine
def test_planar_authored_surface_uses_public_mesh_without_inventing_thickness(tmp_path):
    gs=pytest.importorskip('genesis')
    path,stage=_scene(tmp_path)
    vertices=np.array([[-1,-1,0],[1,-1,0],[1,1,0],[-1,1,0]],dtype=float)
    faces=np.array([[0,1,2],[0,2,3]])
    mesh=_mesh(stage,'/root/floor',trimesh.Trimesh(vertices,faces,process=False),'none')
    mesh.AddTranslateOp().Set((13,4,2))
    converted=_convert(path,stage,tmp_path,static_convex_sdf_max_res=16)
    assert len(converted.groups)==1
    group=converted.groups[0]
    assert group.kind=='Mesh' and not group.convex
    assert group.material_kwargs=={}
    assert group.relative_pose.position==pytest.approx((13,4,2))
    shape=trimesh.load(group.morph_kwargs['file'],process=False)
    assert np.ptp(shape.vertices[:,2])==0
    assert len(shape.faces)==2
    gs.init(backend=gs.cpu,logging_level='warning',seed=0)
    scene=None
    try:
        scene=gs.Scene(show_viewer=False)
        body=scene.add_entity(gs.morphs.Mesh(pos=group.relative_pose.position,**group.morph_kwargs))
        assert len(body.geoms)==1
        geom=body.geoms[0]
        assert geom.link.is_fixed and not geom.is_convex
        assert geom.n_faces==2 and np.ptp(geom.init_verts[:,2])==0
    finally:
        if scene is not None:scene.destroy()
        gs.destroy()


@pytest.mark.engine
def test_public_mjcf_box_accepts_16_grid_with_exact_convex_contacts(tmp_path):
    gs=pytest.importorskip('genesis')
    gs.init(backend=gs.cpu,logging_level='warning',seed=0)
    scene=None
    try:
        path,stage=_scene(tmp_path)
        _mesh(stage,'/root/box',trimesh.creation.box(extents=(.6,3,.0039)),'boundingCube')
        converted=_convert(path,stage,tmp_path,static_convex_sdf_max_res=16)
        group=converted.groups[0]
        scene=gs.Scene(show_viewer=False,sim_options=gs.options.SimOptions(dt=.001,gravity=(0,0,0)))
        wall=scene.add_entity(gs.morphs.MJCF(**group.morph_kwargs),material=gs.materials.Rigid(**group.material_kwargs))
        mover=scene.add_entity(gs.morphs.Box(size=(.01,.01,.01),pos=(0,0,.0039/2+.004)))
        assert list(wall.geoms[0].sdf_res)==[16,16,16]
        assert wall.geoms[0].is_convex and wall.geoms[0].link.is_fixed
        scene.build();scene.step()
        assert list(wall.geoms[0].sdf_res)==[16,16,16]
        contacts=mover.get_contacts(with_entity=wall)
        assert len(contacts['penetration'])>0
        assert contacts['penetration'].cpu().numpy()==pytest.approx(.001,abs=1e-6)
    finally:
        if scene is not None:scene.destroy()
        gs.destroy()
