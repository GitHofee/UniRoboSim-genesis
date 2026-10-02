"""Planning limits describe representable native state, including USD references."""
import math
import os
import numpy as np
import pytest
from pxr import Sdf, Usd, UsdGeom, UsdPhysics
from unirobosim import (EntityKind, EntityPath, EntitySpec, EnvironmentSpec,
    PhysicsSpec, PlanningJointType, WorldSpec)
from unirobosim_genesis import create_provider, GenesisAdapterConfig
from unirobosim_genesis.math import numpy
from test_native import build_input_for
from usd_profile_fixture import usd_profile_provider


@pytest.mark.engine
def test_effective_native_limits_contain_exact_boundary_states_and_references(tmp_path):
    source=tmp_path/'limits.usda';stage=Usd.Stage.CreateNew(str(source))
    root=UsdGeom.Xform.Define(stage,'/robot');stage.SetDefaultPrim(root.GetPrim())
    UsdGeom.SetStageMetersPerUnit(stage,1.);UsdGeom.SetStageUpAxis(stage,'Z')
    UsdPhysics.ArticulationRootAPI.Apply(root.GetPrim())
    root.GetPrim().CreateAttribute('newton:selfCollisionEnabled',Sdf.ValueTypeNames.Bool).Set(False)
    settings=(('yaw',True,-360.,360.,0.),('offset',True,-90.,180.,17.),
              ('slide',False,-.3,.5,.123),('unbounded',True,-math.inf,math.inf,0.))
    for index,name in enumerate(('base',)+tuple(s[0] for s in settings)):
        link=UsdGeom.Xform.Define(stage,'/robot/'+name)
        if index:
            joint_name,angular,lo,hi,reference=settings[index-1]
            link.AddTranslateOp().Set((reference if not angular else 0.,index*.2,0.))
            if angular and reference:link.AddRotateZOp().Set(reference)
        UsdPhysics.RigidBodyAPI.Apply(link.GetPrim())
        mass=UsdPhysics.MassAPI.Apply(link.GetPrim());mass.CreateMassAttr(1.);mass.CreateDiagonalInertiaAttr((.01,.01,.01))
        geom=UsdGeom.Cube.Define(stage,str(link.GetPath())+'/collision');geom.CreateSizeAttr(.05)
        UsdPhysics.CollisionAPI.Apply(geom.GetPrim())
    fixed=UsdPhysics.FixedJoint.Define(stage,'/robot/fixed');fixed.CreateBody1Rel().SetTargets(['/robot/base'])
    for index,(name,angular,lo,hi,reference) in enumerate(settings,1):
        joint=(UsdPhysics.RevoluteJoint if angular else UsdPhysics.PrismaticJoint).Define(stage,'/robot/joint_'+name)
        joint.CreateBody0Rel().SetTargets(['/robot/base']);joint.CreateBody1Rel().SetTargets(['/robot/'+name])
        joint.CreateLocalPos0Attr((0.,index*.2,0.));joint.CreateAxisAttr('Z' if angular else 'X')
        joint.CreateLowerLimitAttr(lo);joint.CreateUpperLimitAttr(hi)
        kind='angular' if angular else 'linear'
        joint.GetPrim().CreateAttribute(f'state:{kind}:physics:position',Sdf.ValueTypeNames.Float).Set(reference)
        drive=UsdPhysics.DriveAPI.Apply(joint.GetPrim(),kind);drive.CreateTypeAttr('force')
        drive.CreateStiffnessAttr(0.);drive.CreateDampingAttr(0.);drive.CreateMaxForceAttr(0.)
    stage.GetRootLayer().Save();before=source.read_bytes();build_input=build_input_for(source)
    entity=EntitySpec(EntityPath('/robot'),EntityKind.ARTICULATION,asset_uri=source.as_uri(),
        joint_names=tuple('joint_'+s[0] for s in settings),
        initial_joint_positions=tuple(math.radians(s[4]) if s[1] else s[4] for s in settings),
        joint_position_units=tuple('rad' if s[1] else 'm' for s in settings))
    spec=WorldSpec('limits',(entity,),environments=EnvironmentSpec(2),
        physics=PhysicsSpec(time_step_seconds=1/60,substeps=8,gravity_m_s2=(0.,0.,0.)),
        schema_version='unirobosim.world/v0alpha6',build_resource_manifest_sha256=build_input.manifest.sha256)
    provider=create_provider(GenesisAdapterConfig(device=os.environ.get('GENESIS_TEST_DEVICE','cpu')))
    provider=usd_profile_provider(source,provider)
    with provider.open() as session, session.build(spec,build_input=build_input) as world:
        body=world._root(entity.path);dofs=world._dofs[entity.path]
        # Prove selected-environment readback, not descriptor construction data.
        body.set_dofs_limit(np.asarray([[-.7]]),np.asarray([[.9]]),dofs_idx_local=[dofs[2]],envs_idx=[1])
        raw_lower,raw_upper=(numpy(x) for x in body.get_dofs_limit(dofs))
        references=world._q_read_offsets[entity.path]
        assert references[1]!=0. and references[2]!=0.
        catalogs=[world.planning_scene_catalog(env) for env in (0,1)]
        for env in (0,1):
            by_name={j.authored_name:j for j in catalogs[env].joints}
            for axis,name in enumerate(entity.joint_names):
                joint=by_name[name]
                if axis==3:
                    assert joint.joint_type is PlanningJointType.CONTINUOUS
                    assert joint.lower is None and joint.upper is None
                    continue
                # Exact equality is required at a hard boundary, not isclose.
                assert joint.lower==float(np.float64(raw_lower[env,axis])+references[axis])
                assert joint.upper==float(np.float64(raw_upper[env,axis])+references[axis])
            for boundary in (raw_lower,raw_upper):
                values=boundary[env].copy();values[3]=0.
                body.set_qpos(values[None,:],qs_idx_local=world._q_indices[entity.path],envs_idx=[env])
                state=world.read_articulation(world.resolve(entity.path))
                positions=state.joint_positions.rows()[env]
                for axis,name in enumerate(entity.joint_names[:3]):
                    joint=by_name[name];q=positions[axis]
                    assert joint.lower<=q<=joint.upper
                    assert joint.lower-q<=0.<=joint.upper-q
                    expected=joint.lower if boundary is raw_lower else joint.upper
                    assert q==expected
        evidence=world.asset_provenance['physics']['planning_joint_limit_readback']
        for env in (0,1):
            rows=evidence[str(env)]['entities'][entity.path.value]
            assert rows['joint_yaw']['authoring_or_import_description_limits']==[-math.tau,math.tau]
            assert rows['joint_yaw']['effective_absolute_limits']==[float(raw_lower[env,0]),float(raw_upper[env,0])]
            assert rows['joint_unbounded']['effective_absolute_limits']==[None,None]
            assert rows['joint_offset']['position_read_offset']==references[1]
        assert source.read_bytes()==before
