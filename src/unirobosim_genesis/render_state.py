"""Render-only state transactions composed from public Genesis entity APIs."""
from __future__ import annotations

import numpy as np
from unirobosim import (
    EntityKind, LifecycleError, Pose, RenderStateFrame, RenderStateResult,
    UnsupportedCapabilityError, ValidationError,
)
from .math import compose, inverse, multiply, numpy, rotation, wxyz, xyzw


class RenderStateMixin:
    def _render_array(self, value, shape):
        if value.shape != shape:
            raise ValidationError("render-state shape does not match selected entity axes/environments", operation="genesis.render_state")
        result = np.asarray(value.rows(), dtype=np.float64)
        if not np.isfinite(result).all():
            raise ValidationError("render state must be finite", operation="genesis.render_state")
        return result

    def _render_root_plan(self, path, envs, values):
        body = self._root(path)
        if len(self._bodies[path]) != 1:
            self._unsupported("render root state requires a single native entity", "genesis.render_state")
        p, q, v, a = (self._render_array(value, (len(envs), width))
                      for value, width in zip(values, (3, 4, 3, 3), strict=True))
        link = self._logical_root_link(path)
        free = [j for j in body.joints if j.type == self._gs.JOINT_TYPE.FREE and j.link is body.base_link]
        if not free and (np.any(v != 0) or np.any(a != 0)):
            self._unsupported("fixed-root render state cannot carry a nonzero root twist", "genesis.render_state")
        if free and link is not body.base_link:
            self._unsupported("floating render roots must identify the native base link", "genesis.render_state")
        return body, path, envs, p, q, v, a, free[0] if free else None

    def _write_render_root(self, plan):
        body, path, envs, positions, quats, linear, angular, free = plan
        native_positions, native_quats = [], []
        for row, env in enumerate(envs):
            current = self._native_pose(body, env)
            logical = self._link_state(path, None, env)[0]
            target = Pose(tuple(positions[row]), tuple(quats[row]))
            native = compose(target, inverse(compose(inverse(current), logical)))
            native_positions.append(native.position)
            native_quats.append(wxyz(native.orientation_xyzw))
        body.set_pos(np.asarray(native_positions), envs_idx=envs, zero_velocity=False)
        body.set_quat(np.asarray(native_quats), envs_idx=envs, zero_velocity=False)
        if free is not None:
            # Genesis free-joint translation is world-frame; its angular DOFs
            # rotate with the internal base frame. Transport the authored-origin
            # linear velocity to that same internal origin before assignment.
            internal_p = numpy(body.get_links_pos([body.base_link.idx_local], envs_idx=envs, relative=False))[:, 0]
            internal_q = xyzw(body.get_links_quat([body.base_link.idx_local], envs_idx=envs, relative=False))[:, 0]
            vel = linear + np.cross(angular, internal_p - positions)
            ang = np.asarray([rotation(q).T @ a for q, a in zip(internal_q, angular, strict=True)])
            body.set_dofs_velocity(np.concatenate((vel, ang), axis=1), dofs_idx_local=free.dofs_idx_local, envs_idx=envs)

    def _align_render_auxiliaries(self, path, envs):
        metadata = self._usd_robots.get(path, {})
        if not metadata.get("closures"):
            return
        body = self._root(path)
        joints = {j.name.rsplit("/", 1)[-1]: j for j in body.joints}
        for closure in metadata["closures"]:
            auxiliary = self._links[path][closure["auxiliary_name"]]
            original = self._links[path][closure["original_body_name"]]
            joint = joints[closure["name"]]
            qs = range(joint.q_start - body.q_start, joint.q_start - body.q_start + 4)
            current = xyzw(body.get_qpos(qs, envs_idx=envs))
            poses = xyzw(body.get_links_quat([auxiliary.idx_local, original.idx_local], envs_idx=envs))
            updated = []
            for q, (a, b) in zip(current, poses, strict=True):
                delta = multiply(tuple(-a[:3]) + (a[3],), tuple(b))
                updated.append(wxyz(multiply(tuple(q), delta)))
            body.set_qpos(np.asarray(updated), qs_idx_local=qs, envs_idx=envs, zero_velocity=False)

    def apply_render_state(self, frame):
        self._ensure("genesis.render_state")
        if not isinstance(frame, RenderStateFrame):
            raise ValidationError("expected RenderStateFrame", operation="genesis.render_state")
        if getattr(frame, "deformables", ()):
            self._unsupported("deformable render state is not implemented", "genesis.render_state")
        if frame.particle_fluids:
            self._unsupported("particle render state is not implemented", "genesis.render_state")
        articulation_plans, root_plans, affected = [], [], {}
        # Resolve and validate the complete transaction before the first setter.
        for state in frame.articulations:
            entity = self._entity(state.handle, "genesis.render_state")
            if entity.kind is not EntityKind.ARTICULATION:
                raise ValidationError("render articulation target has the wrong kind", operation="genesis.render_state")
            envs = self._indices(state.environment_indices, self._spec.environments.count, "genesis.render_state")
            indices = self._indices(state.degree_of_freedom_indices, len(entity.joint_names), "genesis.render_state")
            q = self._render_array(state.joint_positions, (len(envs), len(indices)))
            v = self._render_array(state.joint_velocities, q.shape)
            articulation_plans.append((entity.path, envs, indices, q, v))
            affected[entity.path] = envs
            if state.root_positions_m is not None:
                root_plans.append(self._render_root_plan(entity.path, envs, (
                    state.root_positions_m, state.root_orientations_xyzw,
                    state.root_linear_velocities_m_s, state.root_angular_velocities_rad_s)))
        for state in frame.rigid_bodies:
            entity = self._entity(state.handle, "genesis.render_state")
            if entity.kind is not EntityKind.RIGID_BODY:
                raise ValidationError("render rigid target has the wrong kind", operation="genesis.render_state")
            envs = self._indices(state.environment_indices, self._spec.environments.count, "genesis.render_state")
            root_plans.append(self._render_root_plan(entity.path, envs, (
                state.positions_m, state.orientations_xyzw,
                state.linear_velocities_m_s, state.angular_velocities_rad_s)))
            affected[entity.path] = envs
        backups = []
        root_paths = {plan[1] for plan in root_plans}
        for path, envs in affected.items():
            body = self._root(path)
            backups.append((body, envs, path in root_paths, numpy(body.get_pos(envs_idx=envs)).copy(),
                numpy(body.get_quat(envs_idx=envs)).copy(), numpy(body.get_qpos(envs_idx=envs)).copy(),
                numpy(body.get_dofs_velocity(envs_idx=envs)).copy()))
        try:
            for path, envs, indices, q, v in articulation_plans:
                body = self._root(path)
                body.set_qpos(q-self._q_read_offsets[path][list(indices)],
                    qs_idx_local=[self._q_indices[path][i] for i in indices], envs_idx=envs, zero_velocity=False)
                body.set_dofs_velocity(v, dofs_idx_local=[self._dofs[path][i] for i in indices], envs_idx=envs)
            for plan in root_plans:
                self._write_render_root(plan)
            for path, envs, *_ in articulation_plans:
                self._align_render_auxiliaries(path, envs)
        except Exception:
            try:
                for body, envs, root_changed, p, q, qs, vs in reversed(backups):
                    if root_changed:
                        body.set_pos(p, envs_idx=envs, zero_velocity=False)
                        body.set_quat(q, envs_idx=envs, zero_velocity=False)
                    if qs.shape[-1]:
                        body.set_qpos(qs, envs_idx=envs, zero_velocity=False)
                    if vs.shape[-1]:
                        body.set_dofs_velocity(vs, envs_idx=envs)
            except Exception as restore_error:
                self.close()
                raise LifecycleError("native render-state rollback failed; world closed", operation="genesis.render_state") from restore_error
            raise
        self._render_state_revision = getattr(self, "_render_state_revision", 0) + 1
        self._scene_sequence += 1
        self._planning_cache.clear()
        return RenderStateResult(self.generation, self.tick, self._render_state_revision,
            len(frame.articulations), len(frame.rigid_bodies), 0)

    def configure_render_quality(self, *, enable_global_illumination, enable_ambient_occlusion):
        self._ensure("genesis.render_quality")
        if type(enable_global_illumination) is not bool or type(enable_ambient_occlusion) is not bool:
            raise ValidationError("render-quality flags must be boolean", operation="genesis.render_quality")
        if enable_global_illumination or enable_ambient_occlusion:
            raise UnsupportedCapabilityError("official Genesis Rasterizer has no global illumination or ambient occlusion controls", operation="genesis.render_quality")
        return False, False
