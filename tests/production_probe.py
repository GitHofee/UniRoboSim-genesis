"""Diagnostic build of an unchanged FastSim project; not generation acceptance."""
import argparse
from dataclasses import replace
import json
import importlib.metadata
from collections import Counter
import resource
import numpy as np
import time
from pathlib import Path
import yaml
from fastsim.config.project import load_project
from fastsim.integrations.unirobosim.projection import project_execution_plan
from fastsim_assets_client.client import resolve_asset_references
from unirobosim_genesis import create_provider
from unirobosim import KinematicTarget
from unirobosim_genesis.math import numpy, xyzw, rotation


def process_resources():
    import torch
    stats={'process_max_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024}
    for line in Path('/proc/self/status').read_text().splitlines():
        if line.startswith('VmRSS:'):
            stats['process_rss_bytes']=int(line.split()[1])*1024
    if torch.cuda.is_available():
        free,total=torch.cuda.mem_get_info()
        stats.update(cuda_free_bytes=free,cuda_total_bytes=total,
            torch_cuda_allocated_bytes=torch.cuda.memory_allocated(),
            torch_cuda_reserved_bytes=torch.cuda.memory_reserved(),
            torch_cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated())
    return stats


def geometry_allocation(world):
    solver=world._scene.rigid_solver
    sdf=solver.collider._sdf
    return {'native_geoms':len(solver.geoms),
        'native_geoms_by_type':dict(Counter(g.type.name for g in solver.geoms)),
        'logical_cells':sum(int(g.n_cells) for g in solver.geoms),
        'packed_SDF_cells':sum(int(g.n_cells) for g in sdf._unique_geoms),
        'unique_grid_count':len(sdf._unique_geoms),
        'analytic_box_count':sum(g.type.name=='BOX' for g in solver.geoms),
        'sdf_active':sdf.is_active}


def closure_residuals(body):
    positions=numpy(body.get_links_pos())[0];quaternions=xyzw(body.get_links_quat())[0]
    errors={}
    for equality in body.equalities:
        if equality.type.name!='CONNECT':continue
        a=equality.eq_obj1id-body.link_start;b=equality.eq_obj2id-body.link_start
        anchors=np.asarray(equality.eq_data)[:6].reshape((2,3))
        delta=positions[a]+rotation(quaternions[a])@anchors[0]-positions[b]-rotation(quaternions[b])@anchors[1]
        errors[equality.name]=float(np.linalg.norm(delta))
    return errors


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',required=True);parser.add_argument('--project',required=True)
    parser.add_argument('--assets',required=True);parser.add_argument('--result',required=True)
    parser.add_argument('--robot-only',action='store_true',help='isolated articulation diagnostic, not production acceptance')
    a=parser.parse_args();assets=Path(a.assets)
    config=resolve_asset_references(yaml.safe_load(Path(a.config).read_text()),assets)
    plan=load_project(assets,project_file=Path(a.project)).compiler().compile_mapping(config,source='probe.yaml',source_base=assets).execution_plan
    projection=project_execution_plan(plan,base_seed=config['runtime']['seed'])
    spec=projection.world_spec
    if a.robot_only:spec=replace(spec,entities=tuple(e for e in spec.entities if e.kind.value=='articulation'))
    started=time.monotonic();p=create_provider(launch_profile='headless-physics')
    with p.open() as s:
        resources_before_build=process_resources()
        print('Building Genesis production world',flush=True)
        with s.build(spec,build_input=projection.build_input) as w:
            print('Built',time.monotonic()-started,flush=True)
            build_seconds=time.monotonic()-started
            allocation=geometry_allocation(w)
            resources_after_build=process_resources()
            print('Native geometry allocation',json.dumps(allocation),flush=True)
            catalog=w.planning_scene_catalog();state=w.planning_scene_state();state.validate_against(catalog)
            print('Catalog',len(catalog.links),len(catalog.geometries),flush=True)
            reports=[]
            for entity in spec.entities:
                if entity.kind.value=='articulation':
                    targets=tuple(KinematicTarget(x['name'],entity.path,x['name']) for x in entity.metadata.to_dict()['planning_frame_declarations']['entries'])
                    frames=w.read_selected_kinematics(targets)
                    body=w._root(entity.path)
                    articulation=w.read_articulation(w.resolve(entity.path))
                    reports.append({'entity':entity.path.value,'kinematic_frames':len(frames),
                        'joints':len(articulation.joint_names),'joint_names':articulation.joint_names,
                        'joint_positions':articulation.joint_positions.values,
                        'max_config_joint_error':float(np.max(np.abs(np.asarray(articulation.joint_positions.values)-entity.initial_joint_positions))),
                        'native_links':body.n_links,'native_dofs':body.n_dofs,
                        'native_equalities':body.n_equalities,
                        'equality_types':[e.type.name for e in body.equalities],
                        'closure_residuals_before_step_m':closure_residuals(body),
                        'frame_poses':{f.target_id:{'position':f.pose.position,'orientation_xyzw':f.pose.orientation_xyzw} for f in frames}})
            w.step(3)
            state=w.planning_scene_state();state.validate_against(catalog)
            for report in reports:
                body=next(b for path,bodies in w._bodies.items() if path.value==report['entity'] for b in bodies)
                report['closure_residuals_after_step_m']=closure_residuals(body)
            result={'passed':True,'seconds':time.monotonic()-started,'catalog_links':len(catalog.links),
                'catalog_geometries':len(catalog.geometries),'catalog_point_closures':len(catalog.point_closures),
                'articulations':reports,'usd_provenance':w._provenance,
                'physics_provenance':w._physics_provenance,
                'build_seconds':build_seconds,'geometry_allocation':allocation,
                'resources_before_build':resources_before_build,
                'resources_after_build':resources_after_build,
                'resources_after_steps':process_resources(),
                'engine_version':importlib.metadata.version('genesis-world'),'engine_module':w._gs.__file__}
            Path(a.result).write_text(json.dumps(result,indent=2)+'\n')
            print(json.dumps({k:v for k,v in result.items() if k!='usd_provenance'}),flush=True)

if __name__=='__main__':main()
