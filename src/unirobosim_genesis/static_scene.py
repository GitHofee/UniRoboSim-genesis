"""Adapter-owned static USD collision conversion using public asset formats.

Each authored collider is cooked independently. The original USD remains the
visual resource; this module makes no claim to convert its materials or textures.
No Genesis implementation module is imported or patched.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any
import warnings
import xml.etree.ElementTree as ET

import numpy as np
from unirobosim import EntityKind, Pose, UnsupportedCapabilityError, ValidationError


@dataclass(frozen=True, slots=True)
class StaticSceneGroup:
    name: str
    morph_kwargs: dict[str, Any]
    material_kwargs: dict[str, Any]
    convex: bool
    geometry_names: tuple[str, ...]
    kind: str = "MJCF"
    relative_pose: Pose = Pose()


@dataclass(frozen=True, slots=True)
class StaticSceneMaterialization:
    groups: tuple[StaticSceneGroup, ...]
    provenance: dict[str, Any]
    visual_source_usd: Path


def _fail(message, **details):
    raise UnsupportedCapabilityError(message, operation="genesis.static_scene", details=details)


def _fmt(values):
    return " ".join(format(float(v), ".17g") for v in np.asarray(values).reshape(-1))


def _array_hash(vertices, faces):
    digest = hashlib.sha256()
    digest.update(np.ascontiguousarray(vertices, dtype="<f8").tobytes())
    digest.update(np.ascontiguousarray(faces, dtype="<i8").tobytes())
    return digest.hexdigest()


def _load_cooked_cache(path, key):
    if not path.exists():
        return None
    try:
        with np.load(path, allow_pickle=False) as data:
            if str(data['input_key']) != key:
                return None
            count = int(data['count'])
            if count < 1 or len(data.files) != 2+3*count:
                return None
            pieces = []
            for i in range(count):
                v,f = data[f'v{i}'],data[f'f{i}']
                if (v.ndim != 2 or v.shape[1:] != (3,) or f.ndim != 2 or f.shape[1:] != (3,)
                        or len(f) == 0 or not np.isfinite(v).all() or f.dtype.kind not in 'iu'
                        or f.min() < 0 or f.max() >= len(v)
                        or str(data[f'h{i}']) != _array_hash(v,f)):
                    return None
                pieces.append((v,f))
            return pieces
    except (OSError, ValueError, KeyError, EOFError):
        return None


def _write_cooked_cache(path, key, pieces):
    data = {'count':np.array(len(pieces)), 'input_key':np.array(key)}
    for i,(v,f) in enumerate(pieces):
        data.update({f'v{i}':np.asarray(v),f'f{i}':np.asarray(f),f'h{i}':np.array(_array_hash(v,f))})
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=path.stem+'.',suffix='.tmp',delete=False) as output:
            temporary = Path(output.name)
            np.savez_compressed(output, **data)
            output.flush();os.fsync(output.fileno())
        os.replace(temporary,path)
    finally:
        if temporary is not None:temporary.unlink(missing_ok=True)


def _normalize_declared_convex(vertices, faces, source):
    """Repair only a numerical triangulation defect in a convex cook output."""
    import trimesh
    mesh = trimesh.Trimesh(vertices,faces,process=False)
    if mesh.is_convex:
        return vertices,faces,None
    hull = mesh.convex_hull
    volume_difference = abs(float(hull.volume)-float(mesh.volume))
    relative_difference = volume_difference/max(abs(float(hull.volume)),1e-15)
    bounds_difference = float(np.max(np.abs(hull.bounds-mesh.bounds)))
    if (not mesh.is_watertight or not mesh.is_winding_consistent or not hull.is_convex
            or relative_difference > 1e-6 or bounds_difference != 0):
        _fail("convex cooking did not produce a convex surface",source=source,
              relative_volume_difference=relative_difference,bounds_difference_m=bounds_difference)
    record=dict(source_geometry_sha256=_array_hash(vertices,faces),
                derived_geometry_sha256=_array_hash(hull.vertices,hull.faces),
                source_volume_m3=float(mesh.volume),derived_volume_m3=float(hull.volume),
                relative_volume_difference=relative_difference,bounds_difference_m=bounds_difference,
                policy='public-convex-hull-numerical-normalization; declared-convex-only')
    return np.asarray(hull.vertices),np.asarray(hull.faces),record


def _triangles(mesh, path):
    """Retain authored points; triangulate triangles/quads without processing."""
    from pxr import UsdGeom
    vertices = np.asarray(mesh.GetPointsAttr().Get(), dtype=float)
    counts = np.asarray(mesh.GetFaceVertexCountsAttr().Get(), dtype=np.int64)
    indices = np.asarray(mesh.GetFaceVertexIndicesAttr().Get(), dtype=np.int64)
    holes = set(mesh.GetHoleIndicesAttr().Get() or [])
    if (vertices.ndim != 2 or vertices.shape[1:] != (3,) or not len(vertices)
            or not np.isfinite(vertices).all() or counts.sum() != len(indices)
            or len(indices) == 0 or indices.min() < 0 or indices.max() >= len(vertices)):
        _fail("invalid authored collision mesh", source=path)
    faces = []
    offset = 0
    for fi, count in enumerate(counts):
        polygon = indices[offset:offset+count]
        offset += count
        if fi in holes:
            continue
        if count not in (3, 4):
            _fail("static collision polygon requires explicit triangulation", source=path, vertex_count=int(count))
        # USD's triangle and quad topology is retained without vertex welding.
        if count == 3:
            faces.append(polygon)
        else:
            # Choose a diagonal whose two triangles agree with polygon winding;
            # a concave quad must not introduce an exterior triangle.
            p = vertices[polygon]
            normal = np.cross(p[1]-p[0], p[2]-p[0]) + np.cross(p[2]-p[0], p[3]-p[0])
            candidates = (((0,1,2),(0,2,3)), ((0,1,3),(1,2,3)))
            selected = next((c for c in candidates if all(
                np.dot(np.cross(p[b]-p[a], p[c_]-p[a]), normal) >= 0 for a,b,c_ in c)), None)
            if selected is None:
                _fail("self-intersecting collision quad", source=path)
            faces.extend(polygon[list(t)] for t in selected)
    if not faces:
        _fail("collision mesh has no surface triangles", source=path)
    faces = np.asarray(faces, dtype=np.int64)
    if mesh.GetOrientationAttr().Get() == UsdGeom.Tokens.leftHanded:
        faces = faces[:, ::-1]
    return vertices, faces


def _cook_mesh(vertices, faces, approximation, options, cache_dir):
    """Public trimesh/CoACD operations, never combine distinct source prims."""
    import trimesh
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    if approximation not in ("convexHull", "convexDecomposition"):
        return [(vertices, faces)], "source-triangles-retained"
    if approximation == "convexHull":
        hull = mesh.convex_hull
        limit = options.get("max_ch_vertex")
        if limit is None or len(hull.vertices) <= limit:
            return [(np.asarray(hull.vertices), np.asarray(hull.faces))], "per-prim-convex-hull"
        options = dict(options, max_convex_hull=1)
    elif mesh.is_convex and (not options.get("max_ch_vertex") or len(vertices) <= options["max_ch_vertex"]):
        return [(vertices, faces)], "already-convex-source"
    import coacd
    parameters = dict(threshold=.1, max_convex_hull=-1, preprocess_mode="auto", preprocess_resolution=30,
                      resolution=1000, mcts_nodes=20, mcts_iterations=100, mcts_max_depth=3,
                      pca=False, merge=True, decimate=False, max_ch_vertex=256, extrude=False,
                      extrude_margin=.1, apx_mode="ch", seed=0)
    parameters.update(options)
    if "max_ch_vertex" in options:
        parameters["decimate"] = True
    key = hashlib.sha256((_array_hash(vertices, faces)+json.dumps(parameters, sort_keys=True)
                          +importlib.metadata.version("coacd")).encode()).hexdigest()
    cached = cache_dir/(key+".npz")
    pieces = _load_cooked_cache(cached,key)
    if pieces is None:
        pieces = coacd.run_coacd(coacd.Mesh(vertices, faces.astype(np.int32)), **parameters)
        if not pieces:
            _fail("per-primitive convex decomposition produced no geometry")
        _write_cooked_cache(cached,key,pieces)
    if parameters["max_convex_hull"] > 0 and len(pieces) > parameters["max_convex_hull"]:
        _fail("CoACD exceeded authored convex hull count")
    if "max_ch_vertex" in options and any(len(v) > options["max_ch_vertex"] for v,f in pieces):
        _fail("CoACD exceeded authored convex hull vertex limit")
    return pieces, "per-prim-coacd:"+key


def materialize_static_scene(entity, source_usd: Path, derived_root: Path, provenance,
                             *, static_convex_sdf_max_res: int | None = None) -> StaticSceneMaterialization:
    """Convert a declared static entity; caller validates cap eligibility first.

    Reduced grids are admitted only for fixed convex groups. The world owner
    must verify every potentially movable collider is convex before requesting
    a cap, and leave all nonconvex/movable grids at their native defaults.
    """
    import trimesh
    from pxr import Usd, UsdGeom, UsdPhysics, UsdShade
    from scipy.spatial.transform import Rotation
    metadata = entity.metadata.to_dict()
    if not (entity.kind is EntityKind.STATIC_SCENE or
            entity.kind is EntityKind.COMPOSITE_SCENE and metadata.get("composite_unbound_rigid_mode") == "static"):
        _fail("static conversion requires an explicit static entity", entity=entity.path.value)
    if static_convex_sdf_max_res is not None and (type(static_convex_sdf_max_res) is not int or static_convex_sdf_max_res < 16):
        raise ValidationError("static convex SDF cap must be an integer >=16", operation="genesis.static_scene")
    source_usd = Path(source_usd)
    stage = Usd.Stage.Open(str(source_usd))
    if stage is None:
        _fail("cannot open static USD", source=str(source_usd))
    unit = float(provenance["meters_per_unit"])
    if not math.isfinite(unit) or unit <= 0:
        _fail("invalid static stage length unit")
    axis = provenance["up_axis"]
    if axis not in ("Z", "Y"):
        _fail("unsupported static USD up axis", axis=axis)
    axis_rotation = np.eye(3) if axis == "Z" else Rotation.from_euler("x", math.pi/2).as_matrix()
    prefix = np.diag(entity.scale_xyz) @ axis_rotation * unit
    prims = list(Usd.PrimRange(stage.GetPseudoRoot(), Usd.TraverseInstanceProxies()))
    # Unsupported filters must not silently become colliding pairs.
    for prim in prims:
        if prim.HasAPI(UsdPhysics.FilteredPairsAPI) and UsdPhysics.FilteredPairsAPI(prim).GetFilteredPairsRel().GetTargets():
            _fail("static USD filtered pairs need an explicit portable mapping", source=str(prim.GetPath()))
    root_hash = hashlib.sha256(source_usd.read_bytes()).hexdigest()
    key = hashlib.sha256((root_hash+repr(tuple(entity.scale_xyz))+str(unit)+axis).encode()).hexdigest()[:20]
    outdir = Path(derived_root)/("static-mjcf-"+key)
    outdir.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(os.environ.get('XDG_CACHE_HOME',str(Path.home()/'.cache')))/'unirobosim-genesis'/'coacd-v1'
    cache_dir.mkdir(parents=True,exist_ok=True)
    xforms = UsdGeom.XformCache()
    physics_materials = any(p.HasAPI(UsdPhysics.MaterialAPI) for p in prims)
    roots, assets, bodies, names = {}, {}, {}, {True:[], False:[]}
    for convex in (True, False):
        root = ET.Element("mujoco", model="static_convex" if convex else "static_nonconvex")
        ET.SubElement(root,"compiler",angle="radian",fusestatic="false",inertiafromgeom="false")
        roots[convex] = root
        assets[convex] = ET.SubElement(root,"asset")
        bodies[convex] = ET.SubElement(ET.SubElement(root,"worldbody"),"body",name="static_scene")
    inventory, markers, disabled, direct_groups = [], [], [], []
    for prim in prims:
        if not prim.HasAPI(UsdPhysics.CollisionAPI):
            continue
        path = str(prim.GetPath())
        if UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get() is False:
            disabled.append(path)
            continue
        if not prim.IsA(UsdGeom.Gprim):
            markers.append(path)
            continue
        if not prim.IsA(UsdGeom.Mesh):
            _fail("static primitive type requires explicit conversion", source=path, type=prim.GetTypeName())
        mesh = UsdGeom.Mesh(prim)
        source_vertices, faces = _triangles(mesh, path)
        transform = np.asarray(xforms.GetLocalToWorldTransform(prim), dtype=float).T
        linear = prefix @ transform[:3,:3]
        translation = prefix @ transform[:3,3]
        if not np.isfinite(linear).all() or abs(np.linalg.det(linear)) < 1e-15:
            _fail("singular or nonfinite collision transform", source=path)
        # Separate a proper rotation to reuse cooking for translated/rotated instances.
        u,_,vt = np.linalg.svd(linear)
        rotation = u@vt
        if np.linalg.det(rotation) < 0:
            u[:,-1] *= -1
            rotation = u@vt
        deformation = rotation.T@linear
        vertices = source_vertices@deformation.T
        if np.linalg.det(deformation) < 0:
            faces = faces[:,::-1]
        approximation = str(UsdPhysics.MeshCollisionAPI(prim).GetApproximationAttr().Get() or "none")
        if approximation not in ("none","sdf","meshSimplification","convexHull","convexDecomposition","boundingCube","boundingSphere"):
            _fail("unsupported authored collision approximation", source=path, approximation=approximation)
        source_options = {a.GetName():a.Get() for a in prim.GetAttributes()
                          if a.HasAuthoredValueOpinion() and any(t in a.GetName() for t in
                              ("physxConvexDecompositionCollision:","physxConvexHullCollision:","physxSDFMeshCollision:"))}
        options = {}
        for token,target in (("physxConvexDecompositionCollision:maxConvexHulls","max_convex_hull"),
                             ("physxConvexDecompositionCollision:hullVertexLimit","max_ch_vertex"),
                             ("physxConvexHullCollision:hullVertexLimit","max_ch_vertex")):
            if token in source_options:
                options[target] = int(source_options[token])
        friction = None
        material_mapping = None
        if physics_materials:
            # Physics-purpose bindings may fall back to a visual material; only
            # a true PhysicsMaterial schema supplies physical coefficients.
            mat,_ = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial("physics")
            if mat and mat.GetPrim().HasAPI(UsdPhysics.MaterialAPI):
                physics_mat = UsdPhysics.MaterialAPI(mat.GetPrim())
                def authored_value(attribute):
                    return attribute.Get() if attribute and attribute.HasAuthoredValueOpinion() else None
                value = authored_value(physics_mat.GetDynamicFrictionAttr())
                static_value = authored_value(physics_mat.GetStaticFrictionAttr())
                friction = float(value if value is not None else static_value) if value is not None or static_value is not None else None
                restitution = authored_value(physics_mat.GetRestitutionAttr())
                material_mapping = dict(source_material=str(mat.GetPath()),
                    source_static_friction=None if static_value is None else float(static_value),
                    source_dynamic_friction=None if value is None else float(value),
                    source_restitution=None if restitution is None else float(restitution),
                    effective_friction=friction,effective_restitution=None,
                    policy='official-Genesis-USD: dynamic friction, fallback static; rigid-rigid restitution unsupported',
                    distinct_friction_approximated=static_value is not None and value is not None and not math.isclose(float(static_value),float(value)),
                    nonzero_restitution_unsupported=restitution not in (None,0,0.0))
        record = dict(source=path, approximation=approximation,
                      source_geometry_sha256=_array_hash(source_vertices, faces),
                      stage_world_transform=transform.tolist(), portable_linear=linear.tolist(),
                      portable_translation=translation.tolist(), authored_engine_options=source_options,
                      material_mapping=material_mapping,
                      effective_cooking_options=options,
                      unmapped_engine_options={k:{"value":v,"reason":"PhysX-specific cooking control has no equivalent public CoACD/Genesis option"}
                                               for k,v in source_options.items() if not any(k.endswith(t) for t in (":maxConvexHulls",":hullVertexLimit"))},
                      geometry=[])
        label = "g_"+hashlib.sha256(path.encode()).hexdigest()[:20]
        primitive = None
        if approximation == "boundingCube":
            # Fit in the authored local frame before the composed transform.
            lo,hi = source_vertices.min(axis=0),source_vertices.max(axis=0)
            center=(lo+hi)/2
            lengths=np.linalg.norm(linear,axis=0)
            axes=linear/lengths
            if np.allclose(axes.T@axes,np.eye(3),atol=1e-7):
                if np.linalg.det(axes)<0:axes[:,0]*=-1
                primitive=dict(type="box",size=_fmt((hi-lo)*lengths/2),pos=_fmt(linear@center+translation),
                               quat=_fmt(Rotation.from_matrix(axes).as_quat()[[3,0,1,2]]))
            else:
                box=trimesh.creation.box(extents=hi-lo)
                vertices=(box.vertices+center)@deformation.T
                faces=box.faces[:,::-1] if np.linalg.det(deformation)<0 else box.faces
        elif approximation == "boundingSphere":
            lengths=np.linalg.norm(linear,axis=0)
            if not np.allclose(lengths,lengths[0]) or not np.allclose(deformation.T@deformation,np.eye(3)*lengths[0]**2):
                _fail("nonuniformly transformed bounding sphere needs ellipsoid mapping",source=path)
            center=(source_vertices.min(axis=0)+source_vertices.max(axis=0))/2
            radius=np.linalg.norm(source_vertices-center,axis=1).max()*lengths[0]
            primitive=dict(type="sphere",size=str(radius),pos=_fmt(linear@center+translation))
        if primitive is not None:
            pieces=[(None,None)]
            record['derived_policy']='authored-bounding-primitive'
        else:
            pieces,policy = _cook_mesh(vertices,faces,approximation,options,cache_dir)
            record['derived_policy'] = ('unsimplified-source-triangles' if approximation=='meshSimplification' else policy)
        for piece_index,(v,f) in enumerate(pieces):
            name=label+f"_{piece_index}"
            attrs=dict(name=name,group="3",contype="1",conaffinity="1")
            if friction is not None:attrs['friction']=_fmt((friction,.005,.0001))
            if primitive is not None:
                convex=True;attrs.update(primitive)
                derived=dict(native_name=name,type=primitive['type'],primitive=dict(primitive),convex=True)
            else:
                v=np.asarray(v,dtype=float);f=np.asarray(f,dtype=np.int64)
                if not np.isfinite(v).all() or len(f)==0:
                    _fail("invalid cooked collider",source=path)
                normalization = None
                if approximation in ('convexHull','convexDecomposition','boundingCube'):
                    v,f,normalization = _normalize_declared_convex(v,f,path)
                convex=bool(trimesh.Trimesh(v,f,process=False).is_convex)
                if approximation in ('convexHull','convexDecomposition','boundingCube') and not convex:
                    _fail("convex cooking did not produce a convex surface",source=path)
                world_vertices=v@rotation.T+translation
                # Keep the mesh close to its own origin. MuJoCo's public MJCF
                # loader stores mesh vertices in float32; baking a whole-house
                # translation into them can collapse small convex hull facets.
                center=(v.min(axis=0)+v.max(axis=0))/2
                local_vertices=v-center
                geom_position=rotation@center+translation
                geom_quaternion=Rotation.from_matrix(rotation).as_quat()[[3,0,1,2]]
                mesh_path=outdir/(name+'.obj')
                mesh_path.write_text(''.join('v '+_fmt(p)+'\n' for p in local_vertices)
                                     +''.join('f '+' '.join(str(int(i)+1) for i in face)+'\n' for face in f))
                attrs.update(type='mesh',mesh=name,pos=_fmt(geom_position),quat=_fmt(geom_quaternion))
                derived=dict(native_name=name,type='mesh',convex=convex,vertices=len(v),faces=len(f),
                             geometry_sha256=_array_hash(world_vertices,f),file_sha256=hashlib.sha256(mesh_path.read_bytes()).hexdigest(),
                             local_geometry_sha256=_array_hash(local_vertices,f),
                             geom_position=geom_position.tolist(),geom_quaternion_wxyz=geom_quaternion.tolist(),
                             world_aabb=[world_vertices.min(axis=0).tolist(),world_vertices.max(axis=0).tolist()])
                if normalization is not None:derived['convex_output_normalization']=normalization
                if not convex and np.linalg.matrix_rank(local_vertices-local_vertices.mean(axis=0))<3:
                    # MuJoCo always computes a 3-D mesh hull and cannot import
                    # planar collision surfaces. The public Genesis Mesh path
                    # accepts the exact same open triangles without that step.
                    material={} if friction is None else {'friction':friction}
                    direct_groups.append(StaticSceneGroup('fixed_surface_'+name,
                        dict(file=str(mesh_path),fixed=True,scale=1.,align=False,convexify=False,
                             decimate=False,watertighten=None,visualization=False,collision=True),
                        material,False,(name,),kind='Mesh',
                        relative_pose=Pose(tuple(geom_position),tuple(geom_quaternion[[1,2,3,0]]))))
                    derived['native_import_morph']='Mesh'
                    record['geometry'].append(derived)
                    continue
                ET.SubElement(assets[convex],'mesh',name=name,file=str(mesh_path))
            ET.SubElement(bodies[convex],'geom',**attrs)
            names[convex].append(name);record['geometry'].append(derived)
        inventory.append(record)
    groups=[]
    for convex in (True,False):
        if not names[convex]:continue
        group_name='fixed_convex' if convex else 'fixed_nonconvex'
        destination=outdir/(group_name+'.xml')
        ET.indent(roots[convex]);ET.ElementTree(roots[convex]).write(destination,encoding='unicode')
        material={}
        if convex and static_convex_sdf_max_res is not None:
            material=dict(sdf_min_res=min(32,static_convex_sdf_max_res),sdf_max_res=static_convex_sdf_max_res)
        morph=dict(file=str(destination),scale=1.,align=False,convexify=convex,decimate=False,watertighten=None,
                   visualization=False,collision=True,default_armature=None)
        if convex:
            # Re-establish each already-convex surface after MJCF float32 vertex
            # loading. Infinite thresholds disable decomposition and the native
            # fusion path; distinct authored pieces remain distinct geometries.
            morph.update(decompose_object_error_threshold=float('inf'),
                         decompose_robot_error_threshold=float('inf'))
        groups.append(StaticSceneGroup(group_name,morph,material,convex,tuple(names[convex])))
    groups.extend(direct_groups)
    material_limitations = dict(
        colliders_with_distinct_friction=sum(bool(r['material_mapping'] and r['material_mapping']['distinct_friction_approximated']) for r in inventory),
        colliders_with_unsupported_restitution=sum(bool(r['material_mapping'] and r['material_mapping']['nonzero_restitution_unsupported']) for r in inventory))
    if any(material_limitations.values()):
        warnings.warn('Static USD uses official Genesis material mapping: '+json.dumps(material_limitations)
                      +'; dynamic friction is used and rigid-rigid restitution is unsupported.',RuntimeWarning,stacklevel=2)
    conversion=dict(format='MJCF',source_usd=str(source_usd),source_usd_sha256=root_hash,
                    source_provenance=dict(provenance),source_colliders=len(inventory),
                    nongeometric_collision_markers=markers,disabled_collision_prims=disabled,
                    source_approximation_counts=dict(Counter(r['approximation'] for r in inventory)),
                    derived_geometry_count=sum(len(r['geometry']) for r in inventory),
                    static_convex_sdf_max_res=static_convex_sdf_max_res,
                    cooking_cache=str(cache_dir),
                    convex_import_policy='public per-geometry convexify; decomposition disabled; centered local vertices and MJCF pose',
                    cap_precondition='caller verified all potentially movable native colliders convex; rigid contact only',
                    physics_geometry=inventory,visual_source_usd=str(source_usd),
                    material_limitations=material_limitations,
                    visual_status='authored USD retained; native visual binding not implemented',
                    groups=[dict(name=g.name,kind=g.kind,convex=g.convex,geometry_count=len(g.geometry_names),
                                 relative_pose=dict(position=g.relative_pose.position,orientation_xyzw=g.relative_pose.orientation_xyzw),
                                 file=g.morph_kwargs['file'],file_sha256=hashlib.sha256(Path(g.morph_kwargs['file']).read_bytes()).hexdigest(),
                                 material_kwargs=g.material_kwargs) for g in groups])
    if hashlib.sha256(source_usd.read_bytes()).hexdigest()!=root_hash:
        raise ValidationError('static source changed during conversion',operation='genesis.static_scene')
    (outdir/'conversion.json').write_text(json.dumps(conversion,indent=2)+'\n')
    return StaticSceneMaterialization(tuple(groups),conversion,source_usd)
