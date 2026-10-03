from types import SimpleNamespace
import pytest
from unirobosim import ValidationError, UnsupportedCapabilityError, EntityKind, ParticleFluidSpec, ArrayValue
from unirobosim_genesis import GenesisAdapterConfig
from unirobosim_genesis.soft import SoftMixin

class Subject(SoftMixin):
    def _unsupported(self,message,operation):
        raise UnsupportedCapabilityError(message,operation=operation)

@pytest.mark.parametrize('kw',[
 {'fem_mass_policy':'guess'}, {'fem_young_modulus_pa':False},
 {'fem_young_modulus_pa':0}, {'fem_poisson_ratio':.5}, {'fem_poisson_ratio':float('nan')}])
def test_fem_configuration_rejects_invalid_physics(kw):
    with pytest.raises(ValidationError):GenesisAdapterConfig(**kw)

def test_default_fem_policy_rejects_remeshing_before_native_calls():
    s=Subject();s._session=SimpleNamespace(config=GenesisAdapterConfig())
    with pytest.raises(UnsupportedCapabilityError,match='preserve_total_mass'):
        s._add_deformable(SimpleNamespace(kind=EntityKind.VOLUME_DEFORMABLE,deformable=object()))

def test_fem_requires_explicit_material():
    s=Subject();s._session=SimpleNamespace(config=GenesisAdapterConfig(fem_mass_policy='preserve_total_mass'))
    with pytest.raises(UnsupportedCapabilityError,match='explicit'):
        s._add_deformable(SimpleNamespace(kind=EntityKind.VOLUME_DEFORMABLE,deformable=object()))

def test_sph_rejects_mass_that_native_cannot_preserve():
    s=Subject()
    f=ParticleFluidSpec(ArrayValue.from_nested([[0.,0.,1.]]),particle_mass_kg=1.)
    with pytest.raises(UnsupportedCapabilityError,match='particle mass'):
        s._add_fluid(SimpleNamespace(particle_fluid=f))

def test_sph_rejects_mixed_radii():
    s=Subject();s._spec=SimpleNamespace(entities=[SimpleNamespace(particle_fluid=SimpleNamespace(particle_radius_m=r)) for r in (.01,.02)])
    with pytest.raises(UnsupportedCapabilityError,match='shared particle radius'):
        s._soft_options()
