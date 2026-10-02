"""Minimal unmodified-SDK import regressions; emits evidence, no production claim."""
import argparse,json,os,tempfile
from pathlib import Path
from pxr import Usd,UsdGeom,UsdPhysics
import genesis as gs


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--result',required=True);a=parser.parse_args()
    gs.init(backend=gs.cpu,logging_level='error')
    from genesis.utils.usd import parse_usd_rigid_entity
    from genesis.utils.usd.usd_context import find_rigid_bodies_in_range,resolve_rigid_body_link_path,find_joints_in_range
    with tempfile.TemporaryDirectory(prefix='genesis-usd-regressions-') as folder:
        path=Path(folder)/'roles.usda';stage=Usd.Stage.CreateNew(str(path))
        root=UsdGeom.Xform.Define(stage,'/scene');stage.SetDefaultPrim(root.GetPrim())
        UsdGeom.SetStageMetersPerUnit(stage,1.);UsdGeom.SetStageUpAxis(stage,'Z')
        for name,visible,enabled in [('hidden_collider',False,True),('disabled_collider',True,False),('visual_only',True,None)]:
            shape=UsdGeom.Cube.Define(stage,'/scene/'+name);shape.CreateSizeAttr(.1)
            if not visible:shape.CreateVisibilityAttr('invisible')
            if enabled is not None:UsdPhysics.CollisionAPI.Apply(shape.GetPrim()).CreateCollisionEnabledAttr(enabled)
        stage.GetRootLayer().Save()
        _,_,geoms,_=parse_usd_rigid_entity(gs.morphs.USD(file=str(path),fixed=True,align=False),gs.surfaces.Default())
        collisions=[g for group in geoms for g in group if g.get('contype') or g.get('conaffinity')]
        names=[g['mesh'].metadata.get('name') for g in collisions]
        results={'hidden_collision_and_disabled_visual':{'passed':names==['/scene/hidden_collider'],'actual_collision_names':names}}
        path=Path(folder)/'nested.usda';stage=Usd.Stage.CreateNew(str(path))
        root=UsdGeom.Xform.Define(stage,'/robot');stage.SetDefaultPrim(root.GetPrim())
        first=UsdGeom.Xform.Define(stage,'/robot/base');second=UsdGeom.Xform.Define(stage,'/robot/base/child')
        for link in [first,second]:UsdPhysics.RigidBodyAPI.Apply(link.GetPrim())
        bodies=sorted(find_rigid_bodies_in_range(Usd.PrimRange(root.GetPrim())))
        results['nested_bodies']={'passed':len(bodies)==2,'actual_bodies':bodies}
        resolved=resolve_rigid_body_link_path(stage,'/robot')
        results['world_wrapper']={'passed':resolved is None,'actual_body':resolved}
        joint=UsdPhysics.FixedJoint.Define(stage,'/robot/disabled');joint.CreateJointEnabledAttr(False)
        discovered=find_joints_in_range(Usd.PrimRange(root.GetPrim()))
        results['disabled_joint']={'passed':len(discovered)==0,'actual_joints':[str(p.GetPath()) for p in discovered]}
        Path(a.result).write_text(json.dumps(results,indent=2)+'\n');print(json.dumps(results,indent=2))

if __name__=='__main__':main()
