import numpy as np
import pytest
import trimesh
from PIL import Image
from unirobosim_genesis.appearance import effective_mesh_material,srgb_to_linear


def test_constant_mesh_captures_effective_rasterizer_brdf():
    mesh=trimesh.creation.box();mesh.visual.vertex_colors=np.tile([128,64,32,255],(len(mesh.vertices),1))
    color,roughness,metallic,image,uv=effective_mesh_material(mesh)
    assert color==srgb_to_linear(np.array([128,64,32,255])/255)
    assert (roughness,metallic)==(.8,.2)
    assert image is uv is None


def test_base_color_texture_preserves_png_and_uv():
    mesh=trimesh.creation.box();pixels=np.zeros((2,2,4),dtype=np.uint8);pixels[:,:,3]=255
    image=Image.fromarray(pixels)
    uv=np.zeros((len(mesh.vertices),2))
    mesh.visual=trimesh.visual.texture.TextureVisuals(uv=uv,material=trimesh.visual.material.SimpleMaterial(image=image,diffuse=[255]*4))
    color,r,m,out,outuv=effective_mesh_material(mesh)
    assert color==(1,1,1,1)
    assert m==1
    assert np.array_equal(out,pixels)
    assert np.array_equal(outuv,uv)


def test_nonuniform_mesh_vertex_colors_are_explicitly_unsupported():
    mesh=trimesh.creation.box();colors=np.tile([128,64,32,255],(len(mesh.vertices),1));colors[0]=[0,0,0,255];mesh.visual.vertex_colors=colors
    with pytest.raises(ValueError,match='nonuniform'):
        effective_mesh_material(mesh)


def test_textured_capture_keeps_stable_string_binding_and_roundtrips(tmp_path):
    from types import SimpleNamespace as NS
    from unirobosim import EntityKind,EntityPath,EntitySpec,AppearanceSnapshot
    from unirobosim_genesis.appearance import AppearanceMixin
    mesh=trimesh.creation.box()
    image=Image.fromarray(np.full((2,2,4),255,dtype=np.uint8))
    mesh.visual=trimesh.visual.texture.TextureVisuals(uv=np.zeros((len(mesh.vertices),2)),material=trimesh.visual.material.SimpleMaterial(image=image,diffuse=[255]*4))
    asset=tmp_path/'one.urdf';asset.write_text('<robot name="r"><link name="panel"><visual/></link></robot>')
    entity=EntitySpec(EntityPath('/panel'),EntityKind.STATIC_SCENE,asset_uri=asset.as_uri())
    world=AppearanceMixin();world._ensure=lambda op:None;world._local_path=lambda entity:asset
    world._spec=NS(entities=(entity,));world._derived_root=tmp_path
    world._bodies={entity.path:[NS(links=[NS(name='panel',vgeoms=[NS(get_trimesh=lambda:mesh)])])]}
    world._scene=NS(visualizer=NS(context=NS(lights=[],shadow=True,background_color=(0,0,0),ambient_light=(.1,.1,.1))))
    first=world.capture_appearance();second=world.capture_appearance()
    assert first.to_dict()==second.to_dict()
    assert first.bindings[0].mesh_key=='link:panel/visual:0'
    assert first.bindings[0].uv_topology_sha256
    assert len(world._appearance_texture_cache)==1
    assert AppearanceSnapshot.from_dict(first.to_dict()).to_dict()==first.to_dict()
