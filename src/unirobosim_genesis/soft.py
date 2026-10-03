"""Soft-body and SPH access through official public Genesis APIs only."""
import numpy as np
from unirobosim import EntityKind, ParticleFluidState, DeformableState, PointCommandMode, ValidationError
from .math import array, numpy, rotation


class SoftMixin:
    def _soft_options(self):
        fluids = [e.particle_fluid for e in self._spec.entities if e.particle_fluid is not None]
        radii = {f.particle_radius_m for f in fluids}
        if len(radii) > 1:
            self._unsupported("Genesis SPH requires one shared particle radius per world", "genesis.fluid.build")
        return {"sph_options": self._gs.options.SPHOptions(particle_size=2 * next(iter(radii)))} if radii else {}

    def _add_fluid(self, entity):
        f = entity.particle_fluid
        if f.material_id is not None:
            self._unsupported("named fluid materials are not implemented", "genesis.fluid.build")
        mass = f.rest_density_kg_m3 * .8 * (2 * f.particle_radius_m) ** 3
        if f.particle_mass_kg is not None and not np.isclose(f.particle_mass_kg, mass, rtol=1e-6, atol=0):
            self._unsupported("Genesis SPH particle mass is density * 0.8 * diameter^3", "genesis.fluid.build")
        from .particle_colors import ParticlePartitions, linear_to_srgb
        authored=getattr(f,"initial_particle_colors_rgba",None)
        colors=np.asarray(authored.nested(),dtype=np.float32) if authored is not None else np.tile(
            np.asarray(f.color_rgba or (.25,.55,.95,1),dtype=np.float32),(f.particle_count,1))
        palette={}
        for i,rgba in enumerate(colors):palette.setdefault(tuple(float(v) for v in rgba),[]).append(i)
        if len(palette)>self._session.config.max_fluid_color_groups:
            self._unsupported("static particle palette exceeds max_fluid_color_groups; arbitrary unique colors are unsupported", "genesis.fluid.build")
        groups=[]
        for rgba,indices in palette.items():
            native=self._scene.add_emitter(
                material=self._gs.materials.SPH.Liquid(rho=f.rest_density_kg_m3,mu=f.dynamic_viscosity_pa_s,
                    gamma=f.surface_tension_n_m),max_particles=len(indices),
                surface=self._gs.surfaces.Default(color=linear_to_srgb(rgba),roughness=self._session.config.material_roughness,metallic=0.0,vis_mode="particle")).entity
            groups.append((np.asarray(indices,dtype=int),native))
        self._soft_entities[entity.path]=groups[0][1] if len(groups)==1 else ParticlePartitions(groups,f.particle_count)
        self._fluid_colors[entity.path]=colors
        self._provenance.append({"entity":entity.path.value,"solver":"SPH", "particle_mass_kg":mass,
            "creation":"public Scene.add_emitter", "particle_radius_m":f.particle_radius_m,
            "static_color_groups":len(groups),"color_semantics":"linear RGBA; surface input encoded to sRGB",
            "dynamic_color_updates":False})

    def _add_deformable(self, entity):
        from .particle_colors import linear_to_srgb
        d=entity.deformable; cfg=self._session.config
        if entity.kind is not EntityKind.VOLUME_DEFORMABLE:
            self._unsupported("Genesis surface cloth requires a separately supported IPC runtime", "genesis.soft.build")
        if cfg.fem_mass_policy != "preserve_total_mass":
            self._unsupported("Genesis remeshes FEM geometry; explicitly select fem_mass_policy=preserve_total_mass", "genesis.soft.build")
        if cfg.fem_young_modulus_pa is None or cfg.fem_poisson_ratio is None:
            self._unsupported("FEM requires explicit fem_young_modulus_pa and fem_poisson_ratio", "genesis.soft.build")
        if d.material_id is not None or d.kinematic_node_indices or d.self_collision or d.linear_damping_per_s:
            self._unsupported("named materials, pinned nodes, self collision and linear damping are not mapped for FEM", "genesis.soft.build")
        verts=np.asarray(d.rest_positions_m.nested()) * np.asarray(entity.scale_xyz)
        tets=np.asarray(d.tetrahedra.nested(),dtype=int)
        volume=np.abs(np.linalg.det(verts[tets[:,1:]]-verts[tets[:,0,None]] )).sum()/6
        if not np.isfinite(volume) or volume <= 0:
            raise ValidationError("FEM input volume must be positive",operation="genesis.soft.build")
        if d.surface_triangles is not None:
            faces=np.asarray(d.surface_triangles.nested(),dtype=int)
        else:
            all_faces={}
            for a,b,c,e in tets:
                for face in ((a,c,b),(a,b,e),(a,e,c),(b,c,e)):
                    key=tuple(sorted(face));all_faces.setdefault(key,[]).append(face)
            faces=np.asarray([f[0] for f in all_faces.values() if len(f)==1],dtype=int)
        import trimesh
        mesh=trimesh.Trimesh(vertices=verts@rotation(entity.pose.orientation_xyzw).T,faces=faces,process=False)
        if not mesh.is_watertight:
            raise ValidationError("FEM boundary must be watertight",operation="genesis.soft.build")
        mesh.fix_normals()
        path=self._derived_root / (str(len(self._soft_entities))+"-fem.obj")
        mesh.export(path)
        mass=d.node_count*d.node_mass_kg
        native=self._scene.add_entity(self._gs.morphs.Mesh(file=str(path),pos=entity.pose.position,
            quat=(1.,0.,0.,0.),decimate=False,convexify=False,align=False,watertighten=None),
            material=self._gs.materials.FEM.Elastic(E=cfg.fem_young_modulus_pa,nu=cfg.fem_poisson_ratio,
                rho=mass/volume,model="stable_neohookean"),
            surface=self._gs.surfaces.Default(color=linear_to_srgb(cfg.fem_color_linear_rgba),roughness=cfg.material_roughness,metallic=0.0))
        native_verts=numpy(native.init_positions);native_tets=np.asarray(native.elems,dtype=int)
        native_volume=np.abs(np.linalg.det(native_verts[native_tets[:,1:]]-native_verts[native_tets[:,0,None]])).sum()/6
        if not np.isclose(native_volume,volume,rtol=1e-4,atol=1e-12):
            self._unsupported("native FEM meshing changed volume; total mass cannot be preserved", "genesis.soft.build")
        self._soft_entities[entity.path]=native
        self._provenance.append({"entity":entity.path.value,"solver":"FEM.Elastic","remeshed":True,
            "author_nodes":d.node_count,"native_nodes":native.n_vertices,"mass_policy":"preserve_total_mass",
            "total_mass_kg":mass,"native_total_mass_kg":float(native_volume*mass/volume),"density_kg_m3":mass/volume,"young_modulus_pa":cfg.fem_young_modulus_pa,
            "poisson_ratio":cfg.fem_poisson_ratio})

    def _initialize_soft(self):
        for path,native in self._soft_entities.items():
            e=self._entities[path]
            if e.kind is EntityKind.PARTICLE_FLUID:
                f=e.particle_fluid;r=rotation(e.pose.orientation_xyzw)
                p=np.asarray(f.initial_particle_positions_m.nested()) * np.asarray(e.scale_xyz)
                v=np.asarray(f.initial_velocities().nested())
                native.set_particles_pos(p@r.T+np.asarray(e.pose.position))
                native.set_particles_vel(v@r.T)
                native.set_particles_active(np.ones(f.particle_count,dtype=bool))
            else:
                v=np.asarray(e.deformable.initial_velocities().nested())
                if np.any(v):
                    if not np.allclose(v,v[0]):
                        self._unsupported("remeshed FEM only supports uniform initial velocity", "genesis.soft.build")
                    native.set_velocity(v[0]@rotation(e.pose.orientation_xyzw).T)

    def read_particle_fluid(self, handle):
        e=self._entity(handle,"genesis.fluid.read")
        if e.kind is not EntityKind.PARTICLE_FLUID:
            raise ValidationError("target is not fluid",operation="genesis.fluid.read")
        n=self._soft_entities[e.path]
        pos=n.get_particles_pos();vel=n.get_particles_vel()
        return ParticleFluidState(array(pos,str(numpy(pos).dtype)),array(vel,str(numpy(vel).dtype)),self.tick,
            particle_colors_rgba=array(np.broadcast_to(self._fluid_colors[e.path],(self._spec.environments.count,*self._fluid_colors[e.path].shape)),"float32"))

    def read_deformable(self, handle):
        e=self._entity(handle,"genesis.soft.read")
        if e.kind is not EntityKind.VOLUME_DEFORMABLE:
            raise ValidationError("target is not supported deformable",operation="genesis.soft.read")
        state=self._soft_entities[e.path].get_state()
        return DeformableState(array(state.pos,str(numpy(state.pos).dtype)),array(state.vel,str(numpy(state.vel).dtype)),self.tick)

    def apply_particle_fluid_command(self, command):
        e=self._entity(command.handle,"genesis.fluid.command")
        if e.kind is not EntityKind.PARTICLE_FLUID:
            raise ValidationError("target is not fluid",operation="genesis.fluid.command")
        if command.mode not in (PointCommandMode.POSITION,PointCommandMode.VELOCITY):
            self._unsupported("SPH point force command is unsupported", "genesis.fluid.command")
        envs=self._indices(command.environment_indices,self._spec.environments.count,"genesis.fluid.command")
        points=self._indices(command.particle_indices,e.particle_fluid.particle_count,"genesis.fluid.command")
        values=np.asarray(command.targets.nested())
        if values.shape != (len(envs),len(points),3) or not np.isfinite(values).all():
            raise ValidationError("point target shape must match selected environments and particles",operation="genesis.fluid.command")
        n=self._soft_entities[e.path]
        fn=n.set_particles_pos if command.mode is PointCommandMode.POSITION else n.set_particles_vel
        fn(values,particles_idx_local=points,envs_idx=envs)

    def read_deformable_topology(self, handle):
        from unirobosim import DeformableTopologySnapshot, DeformableTopology, ArrayValue
        e=self._entity(handle,"genesis.soft.topology")
        if e.kind is not EntityKind.VOLUME_DEFORMABLE:
            raise ValidationError("target is not supported deformable",operation="genesis.soft.topology")
        n=self._soft_entities[e.path]
        local=(numpy(n.init_positions)-np.asarray(e.pose.position))@rotation(e.pose.orientation_xyzw)
        local=local/np.asarray(e.scale_xyz)
        return DeformableTopologySnapshot(DeformableTopology.VOLUME,array(local,str(numpy(n.init_positions).dtype)),
            ArrayValue.from_nested(np.asarray(n.surface_triangles).tolist(),dtype="int64"),
            ArrayValue.from_nested(np.asarray(n.elems).tolist(),dtype="int64"))
