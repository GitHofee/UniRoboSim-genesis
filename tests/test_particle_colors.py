import numpy as np
from unirobosim_genesis.particle_colors import ParticlePartitions,linear_to_srgb

class Native:
    def __init__(self,points):self.pos=np.asarray(points,dtype=np.float32)[None].copy()
    def get_particles_pos(self,envs_idx=None):return self.pos if envs_idx is None else self.pos[list(envs_idx)]
    def set_particles_pos(self,value,particles_idx_local=None,envs_idx=None):
        for i,e in enumerate(envs_idx or [0]):self.pos[e,list(particles_idx_local)]=value[i] if np.ndim(value)==3 else value

def test_palette_partition_preserves_noncontiguous_authored_order():
    a=Native([[10,0,0],[30,0,0]])
    b=Native([[20,0,0],[40,0,0]])
    p=ParticlePartitions(((np.asarray([0,2]),a),(np.asarray([1,3]),b)),4)
    np.testing.assert_array_equal(p.get_particles_pos()[0,:,0],[10,20,30,40])
    p.set_particles_pos(np.asarray([[[44,0,0],[11,0,0]]]),particles_idx_local=(3,0),envs_idx=(0,))
    np.testing.assert_array_equal(p.get_particles_pos()[0,:,0],[11,20,30,44])

def test_linear_to_srgb_preserves_alpha_and_decodes_back():
    linear=np.asarray((.03,.2,.85,.4))
    encoded=np.asarray(linear_to_srgb(linear))
    decoded=np.where(encoded[:3]<=.04045,encoded[:3]/12.92,((encoded[:3]+.055)/1.055)**2.4)
    np.testing.assert_allclose(decoded,linear[:3],rtol=1e-12)
    assert encoded[3]==linear[3]

def test_palette_limit_rejects_before_creating_native_entities():
    import pytest
    from types import SimpleNamespace
    from unirobosim import ParticleFluidSpec,ArrayValue,UnsupportedCapabilityError
    from unirobosim_genesis import GenesisAdapterConfig
    from unirobosim_genesis.soft import SoftMixin
    class Subject(SoftMixin):
        def _unsupported(self,message,operation):raise UnsupportedCapabilityError(message,operation=operation)
    s=Subject();s._session=SimpleNamespace(config=GenesisAdapterConfig(max_fluid_color_groups=2))
    spec=ParticleFluidSpec(ArrayValue.from_nested([[0,0,1],[0,0,2],[0,0,3]]),initial_particle_colors_rgba=ArrayValue.from_nested([[1,0,0,1],[0,1,0,1],[0,0,1,1]],dtype='float32'))
    with pytest.raises(UnsupportedCapabilityError,match='palette'):
        s._add_fluid(SimpleNamespace(particle_fluid=spec))
