"""Portable planning resources must preserve mesh values and reject invalid indices."""
import numpy as np
import pytest
from unirobosim import PlanningGeometryDType, PlanningSceneRepresentationError
from unirobosim_genesis.planning import _encode_mesh_resource


def test_native_signed_indices_export_as_portable_unsigned_bytes():
    vertices=np.asarray([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.]],dtype=np.float64)
    faces=np.asarray([[0,2,1]],dtype=np.int64)
    payload,layout=_encode_mesh_resource(vertices,faces)
    assert layout.index_dtype is PlanningGeometryDType.UINT32
    assert layout.vertex_dtype is PlanningGeometryDType.FLOAT32
    assert layout.decoded_byte_size==len(payload)
    assert payload==vertices.astype('<f4').tobytes()+faces.astype('<i4').tobytes()
    assert np.array_equal(np.frombuffer(payload,offset=36,dtype='<u4').reshape(1,3),faces)


@pytest.mark.parametrize('faces',[
    [[0,-1,2]], [[0,1,3]], [[0,1,2**32]],
    [[0.,1.,2.]], [[0.,1.5,2.]], [[True,False,True]],
    [[0,1]], [],
])
def test_invalid_native_indices_are_rejected_before_unsigned_cast(faces):
    with pytest.raises(PlanningSceneRepresentationError):
        _encode_mesh_resource(np.eye(3),faces)


def test_noncontiguous_big_endian_native_mesh_is_canonicalized():
    vertices=np.arange(18,dtype='>f8').reshape(3,6)[:,::2]
    vertices[2,2]+=1.
    faces=np.asarray([[0,1,2],[2,1,0]],dtype='>u4')[::2]
    payload,layout=_encode_mesh_resource(vertices,faces)
    assert layout.index_shape==(1,3)
    assert payload==np.asarray(vertices,dtype='<f4').tobytes()+np.asarray(faces,dtype='<u4').tobytes()


def test_only_exact_zero_area_faces_are_removed_without_mutating_native_mesh():
    vertices=np.asarray([[0.,0.,0.],[1.,0.,0.],[2.,0.,0.],[0.,1.,0.],[0.,1e-30,0.]])
    faces=np.asarray([[0,1,3],[0,1,1],[0,1,2],[0,1,4],[3,1,0]],dtype=np.int64)
    before_vertices=vertices.copy();before_faces=faces.copy()
    payload,layout,evidence=_encode_mesh_resource(vertices,faces,provenance=True)
    exported=np.frombuffer(payload,offset=vertices.size*4,dtype='<u4').reshape(-1,3)
    assert np.array_equal(exported,faces[[0,3,4]])
    assert evidence['removed_triangle_indices']==[1,2]
    assert evidence['removed_cross_product_exactly_zero'] is True
    assert evidence['native_triangle_count']==5 and evidence['exported_triangle_count']==3
    assert evidence['physics_geometry_modified'] is False
    assert np.array_equal(vertices,before_vertices) and np.array_equal(faces,before_faces)


def test_float32_collapsed_real_surface_is_preserved_in_float64():
    vertices=np.asarray([[1.,0.,0.],[1.+1e-10,0.,0.],[1.,1.,0.]])
    faces=np.asarray([[0,1,2]])
    payload,layout,evidence=_encode_mesh_resource(vertices,faces,provenance=True)
    assert layout.vertex_dtype is PlanningGeometryDType.FLOAT64
    assert np.array_equal(np.frombuffer(payload[:72],dtype='<f8').reshape(3,3),vertices)
    assert evidence['removed_triangle_indices']==[]
    assert evidence['float32_collapse_triangle_indices']==[0]
    assert layout.index_shape==(1,3)


def test_mesh_with_no_surface_is_rejected_instead_of_publishing_empty_resource():
    with pytest.raises(PlanningSceneRepresentationError,match='nondegenerate'):
        _encode_mesh_resource(np.asarray([[0.,0.,0.],[1.,0.,0.],[2.,0.,0.]]),[[0,1,2]])
