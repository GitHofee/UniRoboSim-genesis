"""Mass-normalized native PD using public state and Jacobian queries only."""
import numpy as np
from .math import numpy, xyzw, rotation


def articulated_mass_matrix(body, authored_bodies):
    """Current-pose generalized inertia, including the public motor armature.

    Genesis 1.4.2's get_mass_mat reads a previously assembled buffer, including
    neutral-pose values after parameter setters. Spatial Jacobians instead query
    the current kinematics and allow an explicit, current-pose inertia query.
    This is a mass-property calculation, not a dynamics integrator.
    """
    quats=xyzw(body.get_links_quat())
    if quats.ndim==2:quats=quats[None]
    matrix=np.zeros((len(quats),body.n_dofs,body.n_dofs),dtype=np.float64)
    for link in body.links:
        data=authored_bodies.get(link.name)
        if data is None:continue
        jac=numpy(body.get_jacobian(link,local_point=data["com"]))
        if jac.ndim==2:jac=jac[None]
        principal=rotation(data["principal_axes_xyzw"])
        for env in range(len(quats)):
            r=rotation(quats[env,link.idx_local])@principal
            inertia=r@np.diag(data["principal_inertia"])@r.T
            jv,jw=jac[env,:3],jac[env,3:]
            matrix[env]+=data["mass"]*(jv.T@jv)+jw.T@inertia@jw
    armature=numpy(body.get_dofs_armature())
    matrix[:,np.arange(body.n_dofs),np.arange(body.n_dofs)]+=armature
    return matrix


def normalized_gains(matrix, dofs, properties):
    """Return force gains preserving the uncoupled acceleration response."""
    inverse=np.linalg.inv(matrix)
    response=np.diagonal(inverse,axis1=-2,axis2=-1)[:,list(dofs)]
    if not np.isfinite(response).all() or np.min(response)<=0:
        raise ValueError("mass-normalized drive requires a finite positive articulated response")
    factor=np.where(np.asarray([p["drive_type"]=="acceleration" for p in properties]),1/response,1.)
    return factor*np.asarray([p["stiffness"] for p in properties]),factor*np.asarray([p["damping"] for p in properties])
