"""Read effective rasterizer appearance through official public Genesis objects.

No renderer graph mutation, private native attributes, or engine patches.
"""
from pathlib import Path
import hashlib
import io
import xml.etree.ElementTree as ET
import numpy as np
from unirobosim import EntityKind, FrozenMap
from .particle_colors import ParticlePartitions


def srgb_to_linear(values):
    values=np.asarray(values,dtype=float).copy()
    rgb=values[..., :3]
    values[..., :3]=np.where(rgb<=.04045,rgb/12.92,((rgb+.055)/1.055)**2.4)
    return tuple(float(x) for x in values)


def effective_mesh_material(mesh):
    """Match official pyrender's public trimesh conversion, without calling internals."""
    visual=mesh.visual
    if visual.kind in ('vertex','face'):
        colors=np.asarray(visual.vertex_colors if visual.kind=='vertex' else visual.face_colors)
        unique=np.unique(colors,axis=0)
        if len(unique)!=1:
            raise ValueError('nonuniform mesh vertex colors require a vertex-color stream')
        return srgb_to_linear(unique[0]/255.), .8, .2, None, None
    if visual.kind=='texture':
        from trimesh.visual.material import SimpleMaterial, PBRMaterial
        mat=visual.material
        if isinstance(mat,SimpleMaterial):
            ns=mat.kwargs.get('Ns',1.)
            if isinstance(ns,list):ns=ns[0]
            return srgb_to_linear(np.asarray(mat.diffuse)/255.), (2/(float(ns)+2))**.25, 1., mat.image, visual.uv
        if isinstance(mat,PBRMaterial):
            if any(getattr(mat,k,None) is not None for k in ('normalTexture','occlusionTexture','emissiveTexture','metallicRoughnessTexture')):
                raise ValueError('PBR auxiliary texture graphs are not supported')
            if mat.emissiveFactor is not None and np.any(mat.emissiveFactor):
                raise ValueError('emissive PBR is not supported')
            color=np.asarray(mat.baseColorFactor if mat.baseColorFactor is not None else [255]*4)/255.
            return srgb_to_linear(color), float(mat.roughnessFactor if mat.roughnessFactor is not None else 1.),float(mat.metallicFactor if mat.metallicFactor is not None else 1.),mat.baseColorTexture,visual.uv
    raise ValueError('unsupported native trimesh material representation')


class AppearanceMixin:
    def capture_appearance(self):
        from unirobosim import (AppearanceSnapshot,AppearanceMaterial,AppearanceBinding,
            AppearanceLight,AppearanceEnvironment,AppearanceRenderer,AppearanceTexture)
        self._ensure('genesis.appearance.capture')
        materials=[];bindings=[]
        def add(entity,key,mesh=None,particle=False):
            mid=f'material:{len(materials)}'
            textures=()
            captured_uv=None
            uv_topology_sha256=None
            if particle:
                # Public SPH surface is converted to vertex colors by the official rasterizer.
                color=(1.,1.,1.,1.);roughness=.8;metallic=.2
            else:
                try:color,roughness,metallic,image,uv=effective_mesh_material(mesh)
                except ValueError as exc:self._unsupported(str(exc),'genesis.appearance.capture')
                if image is not None:
                    if uv is None or len(uv)!=len(mesh.vertices):
                        self._unsupported('base-color texture has no lossless vertex UV mapping','genesis.appearance.capture')
                    texture_key=(image.mode,image.size,hashlib.sha256(image.tobytes()).hexdigest())
                    cache=getattr(self,'_appearance_texture_cache',None)
                    if cache is None:cache=self._appearance_texture_cache={}
                    if texture_key not in cache:
                        buf=io.BytesIO();image.save(buf,format='PNG');raw=buf.getvalue();digest=hashlib.sha256(raw).hexdigest()
                        path=self._derived_root/f'appearance-{digest}.png';path.write_bytes(raw)
                        cache[texture_key]=(path,digest)
                    path,digest=cache[texture_key]
                    from unirobosim import appearance_topology_sha256
                    uv_topology_sha256=appearance_topology_sha256(mesh.vertices,mesh.faces.reshape(-1),np.full(len(mesh.faces),3))
                    captured_uv=tuple(tuple(float(v) for v in row) for row in uv)
                    textures=(AppearanceTexture('base_color',path.as_uri(),'srgb',sha256=digest),)
            materials.append(AppearanceMaterial(mid,color,roughness,metallic,
                color_source='particle_colors' if particle else 'constant',textures=textures,
                source_parameters=FrozenMap({'native_renderer':'Genesis Rasterizer/pyrender','native_material_representation':'particle vertex colors' if particle else str(mesh.visual.kind),'effective_brdf':True})))
            bindings.append(AppearanceBinding(entity.path.value,key,mid,uv_coordinates=captured_uv,uv_topology_sha256=uv_topology_sha256))
        for entity in self._spec.entities:
            if entity.kind is EntityKind.CAMERA_SENSOR:continue
            if entity.kind is EntityKind.PARTICLE_FLUID:
                native=self._soft_entities[entity.path]
                groups=native.groups if isinstance(native,ParticlePartitions) else ((None,native),)
                for _,part in groups:
                    if part.surface.vis_mode!='particle':self._unsupported('only SPH particle rasterization is supported','genesis.appearance.capture')
                    texture=part.surface.get_rgba()
                    if not hasattr(texture,'color'):self._unsupported('textured SPH particles unsupported','genesis.appearance.capture')
                add(entity,'particles',particle=True)
            elif entity.kind is EntityKind.VOLUME_DEFORMABLE:
                geoms=self._soft_entities[entity.path].vgeoms
                if len(geoms)!=1:self._unsupported('multi-surface deformable appearance unsupported','genesis.appearance.capture')
                add(entity,'body/visual:0',geoms[0].vmesh.trimesh)
            elif entity.box is not None:
                geoms=[g for b in self._bodies[entity.path] for g in b.vgeoms]
                if len(geoms)!=1:self._unsupported('procedural visual binding is ambiguous','genesis.appearance.capture')
                add(entity,'body/visual:0',geoms[0].get_trimesh())
            elif entity.asset_uri:
                path=self._local_path(entity)
                if path.suffix.lower()!='.urdf':self._unsupported('appearance stable binding requires URDF or procedural geometry','genesis.appearance.capture')
                author={n.attrib['name']:n.findall('visual') for n in ET.parse(path).getroot().findall('link')}
                for body in self._bodies[entity.path]:
                    for link in body.links:
                        name=link.name
                        if name not in author or len(author[name])!=len(link.vgeoms):
                            self._unsupported('URDF visual to native geometry mapping is not one-to-one','genesis.appearance.capture')
                        for i,g in enumerate(link.vgeoms):add(entity,f'link:{name}/visual:{i}',g.get_trimesh())
            else:self._unsupported('appearance entity kind unsupported','genesis.appearance.capture')
        context=self._scene.visualizer.context
        lights=[]
        for i,light in enumerate(context.lights):
            if type(light).__name__!='DirectionalLight':self._unsupported('only directional rasterizer lights supported','genesis.appearance.capture')
            direction=np.array(light.dir,dtype=float,copy=True);direction/=np.linalg.norm(direction)
            lights.append(AppearanceLight(f'light:{i}','directional',tuple(light.color),float(light.intensity),'linear_rgb_irradiance',direction_world=tuple(direction),shadows=bool(context.shadow),source_parameters=FrozenMap({'native_intensity':float(light.intensity),'native_unit':'pyrender linear RGB BRDF input'})))
        bg=tuple(float(v)**2.2 for v in context.background_color[:3])
        return AppearanceSnapshot(tuple(materials),tuple(bindings),tuple(lights),
            AppearanceEnvironment(bg,tuple(context.ambient_light)),
            AppearanceRenderer('genesis','Rasterizer',source_parameters=FrozenMap({'surface_input_color_space':'sRGB','official_gamma':2.2})))

    def apply_appearance(self,snapshot):
        self._ensure('genesis.appearance.apply')
        self._unsupported('official Genesis rasterizer has no supported atomic live material/light replacement API','genesis.appearance.apply')
