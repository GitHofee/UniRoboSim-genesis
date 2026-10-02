import os
import subprocess
import sys
import pytest
from unirobosim import CapabilityId,ValidationError
from unirobosim_genesis import create_provider,GenesisAdapterConfig


def test_import_does_not_import_genesis_or_torch():
    result=subprocess.run([sys.executable,'-c',
        'import sys; import unirobosim_genesis; assert "genesis" not in sys.modules; assert "torch" not in sys.modules'],capture_output=True,text=True)
    assert result.returncode==0,result.stderr


def test_launch_profiles_and_capabilities():
    provider=create_provider(launch_profile='headless-physics')
    assert provider.descriptor.provider_id=='genesis-world.genesis'
    assert provider.descriptor.version=='0.1.0'
    assert provider.descriptor.contract_version=='v0alpha6'
    assert provider.descriptor.capabilities.get(CapabilityId('sensor.camera@1')) is None
    assert provider.descriptor.capabilities.get(CapabilityId('state.kinematics.selected@1')) is not None
    assert create_provider(launch_profile='visible').config.headless is False
    with pytest.raises(ValueError):create_provider(launch_profile='invalid')


@pytest.mark.parametrize('kwargs',[{'device':'fake'},{'seed':True},{'seed':-1},{'precision':'16'},
    {'headless':1},{'position_stiffness':float('nan')},{'position_damping':-1},{'max_cached_commands':0}])
def test_invalid_configuration(kwargs):
    with pytest.raises(ValidationError):GenesisAdapterConfig(**kwargs)


def test_reused_asset_selects_exact_entity_resource(tmp_path):
    from dataclasses import replace
    import hashlib
    from unirobosim import BuildInput, BuildResourceEntry, BuildResourceManifest, BuildSourceEntry, LocalSourceIdentity
    from unirobosim_genesis.build_assets import BuildAssetLease
    source=tmp_path/'shared.usda';source.write_text('#usda 1.0\n')
    raw=source.read_bytes();digest=hashlib.sha256(raw).hexdigest();st=source.stat()
    first=BuildResourceEntry('cup-a','asset.cup','model.a','simulation','model/vnd.usd',
        source.as_uri(),source.as_uri(),f'sha256:{digest}',len(raw),digest,True,
        ('collision','simulation'),'a/model.usda')
    second=replace(first,entity_id='cup-b',resource_id='model.b',relative_bundle_path='b/model.usda')
    identity=LocalSourceIdentity(st.st_dev,st.st_ino,st.st_mode,st.st_size,st.st_mtime_ns,st.st_ctime_ns)
    sources=tuple(BuildSourceEntry(e.resource_id,'local-file',str(tmp_path),source.name,identity,digest)
        for e in (first,second))
    build=BuildInput(manifest=BuildResourceManifest((first,second)),sources=sources)
    lease=BuildAssetLease(tmp_path,build,{'model.a':tmp_path/'a/model.usda','model.b':tmp_path/'b/model.usda'})
    assert lease.selected_path(entity_id='cup-a',asset_uri=source.as_uri())==tmp_path/'a/model.usda'
    assert lease.selected_path(entity_id='cup-b',asset_uri=source.as_uri())==tmp_path/'b/model.usda'
    with pytest.raises(ValidationError):lease.selected_path(entity_id='unknown',asset_uri=source.as_uri())
