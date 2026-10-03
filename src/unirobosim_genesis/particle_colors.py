"""Static per-particle color partitions using public SPH emitters."""
import numpy as np
from .math import numpy


def linear_to_srgb(rgba):
    value=np.asarray(rgba,dtype=float).copy()
    rgb=value[..., :3]
    value[..., :3]=np.where(rgb<=.0031308,12.92*rgb,1.055*np.power(rgb,1/2.4)-.055)
    return tuple(value.tolist())


class ParticlePartitions:
    """Preserve authored point order across same-solver native color groups."""
    def __init__(self, groups, count):
        self.groups=tuple(groups)
        self.count=count

    def _get(self, method, envs_idx=None):
        result=None
        for indices,native in self.groups:
            part=numpy(getattr(native,method)(envs_idx=envs_idx))
            if result is None:result=np.empty((*part.shape[:-2],self.count,part.shape[-1]),dtype=part.dtype)
            result[...,indices,:]=part
        return result

    def get_particles_pos(self,envs_idx=None):return self._get('get_particles_pos',envs_idx)
    def get_particles_vel(self,envs_idx=None):return self._get('get_particles_vel',envs_idx)

    def _set(self,method,values,particles_idx_local=None,envs_idx=None):
        values=numpy(values)
        selected=np.arange(self.count) if particles_idx_local is None else np.asarray(particles_idx_local,dtype=int)
        vector=method!='set_particles_active'
        for indices,native in self.groups:
            mapping={int(p):i for i,p in enumerate(indices)}
            selected_offsets=[i for i,p in enumerate(selected) if int(p) in mapping]
            if not selected_offsets:continue
            native_indices=[mapping[int(selected[i])] for i in selected_offsets]
            part=values[...,selected_offsets,:] if vector else values[...,selected_offsets]
            getattr(native,method)(part,particles_idx_local=native_indices,envs_idx=envs_idx)

    def set_particles_pos(self,values,particles_idx_local=None,envs_idx=None):
        self._set('set_particles_pos',values,particles_idx_local,envs_idx)
    def set_particles_vel(self,values,particles_idx_local=None,envs_idx=None):
        self._set('set_particles_vel',values,particles_idx_local,envs_idx)
    def set_particles_active(self,values,particles_idx_local=None,envs_idx=None):
        self._set('set_particles_active',values,particles_idx_local,envs_idx)
