"""Installed asset selections are explicit and pinned across composition layers."""
import hashlib,json
from pathlib import Path
import pytest
from unirobosim import ValidationError
from unirobosim_genesis.asset_profile import AssetProfile
from unirobosim_genesis import GenesisAdapterConfig,GenesisProvider


def pin(p):return {'file':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}

def fixture(tmp_path):
    pytest.importorskip('pxr.Usd')
    layer=tmp_path/'layer.usda';layer.write_text('#usda 1.0\ndef Xform "root" {}\n')
    source=tmp_path/'source.usda';source.write_text('#usda 1.0\n( subLayers = [@layer.usda@] )\n')
    prepared=tmp_path/'prepared.usda';prepared.write_text('#usda 1.0\ndef Xform "prepared" {}\n')
    entry={'source_sha256':pin(source)['sha256'],'source_files':[pin(source),pin(layer)],'pinned_files':[pin(prepared)],'loads':[dict(pin(prepared),mode='add_entity',morph_kwargs={})]}
    profile=tmp_path/'profile.json';profile.write_text(json.dumps({'schema':'unirobosim-genesis-asset-profile/v1','entries':[entry]}))
    return source,layer,prepared,profile


def test_explicit_profile_and_unknown_hash(tmp_path):
    source,layer,prepared,profile=fixture(tmp_path);p=AssetProfile(str(profile))
    selected=p.select(source)
    assert selected['loads'][0]['file']==str(prepared)
    assert selected['profile_sha256']==pin(profile)['sha256']
    assert p.select(prepared) is None


@pytest.mark.parametrize('changed',('source_layer','prepared'))
def test_mutation_rejected(tmp_path,changed):
    source,layer,prepared,profile=fixture(tmp_path)
    target=layer if changed=='source_layer' else prepared
    target.write_text(target.read_text()+'\n# changed\n')
    with pytest.raises(ValidationError,match='digest mismatch|dependency changed'):AssetProfile(str(profile)).select(source)


def test_unpinned_layer_rejected(tmp_path):
    source,layer,prepared,profile=fixture(tmp_path);data=json.loads(profile.read_text())
    data['entries'][0]['source_files']=[pin(source)];profile.write_text(json.dumps(data))
    with pytest.raises(ValidationError,match='unpinned dependency'):AssetProfile(str(profile)).select(source)


def test_profile_environment_and_explicit_precedence(monkeypatch):
    monkeypatch.setenv('UNIROBOSIM_GENESIS_ASSET_PROFILE','/tmp/env-profile.json')
    assert GenesisProvider().config.asset_profile=='/tmp/env-profile.json'
    assert GenesisProvider(GenesisAdapterConfig(asset_profile='/tmp/explicit.json')).config.asset_profile=='/tmp/explicit.json'


def test_visual_usd_requires_pin_and_complete_composition(tmp_path):
    source,layer,prepared,profile=fixture(tmp_path)
    visual=tmp_path/'visual.usda';visual.write_text('#usda 1.0\ndef Xform "visual" {}\n')
    data=json.loads(profile.read_text());entry=data['entries'][0]
    entry['visual_usd']=pin(visual);profile.write_text(json.dumps(data))
    with pytest.raises(ValidationError,match='visual USD is not pinned'):AssetProfile(str(profile)).select(source)
    entry['pinned_files'].append(pin(visual));profile.write_text(json.dumps(data))
    assert AssetProfile(str(profile)).select(source)['visual_usd']['file']==str(visual)
    visual.write_text('#usda 1.0\n( subLayers = [@layer.usda@] )\n')
    entry['visual_usd']=pin(visual);entry['pinned_files'][-1]=pin(visual);profile.write_text(json.dumps(data))
    with pytest.raises(ValidationError,match='visual USD has an unpinned layer'):AssetProfile(str(profile)).select(source)
