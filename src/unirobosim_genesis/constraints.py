"""Solver parameters for build-time equalities through public Genesis APIs."""

def rigid_equality_params(substep_dt):
    """Critical damping at the native public time-constant stability floor."""
    return [2*substep_dt,1.,.9999,.9999,.001,.5,2.]


class ConstraintRestoreError(RuntimeError):
    """A public setter failed to restore unrelated physics parameters."""


def harden_dynamic_welds(solver, substep_dt):
    """Use public bulk setters and restore every non-weld parameter exactly.

    The official equality-index setter addresses authored slots only. The public
    global setter includes dynamic slots; composition with public typed setters
    confines the result to the dynamic welds owned by this adapter's scene.
    There is no simulation step or externally visible read during the operation.
    """
    import numpy as np
    from .math import numpy
    groups=[]
    for count,selector in ((solver.n_geoms,"geoms_idx"),(solver.n_joints,"joints_idx"),(solver.n_equalities,"eqs_idx")):
        if not count:continue
        selection={selector:np.arange(count,dtype=np.int32)}
        values=numpy(solver.get_sol_params(**selection)).copy()
        groups.append((selection,values))
    try:
        solver.set_global_sol_params(np.asarray(rigid_equality_params(substep_dt)))
    finally:
        failures=[]
        for selection,values in groups:
            try:
                solver.set_sol_params(values,**selection)
                if not np.array_equal(numpy(solver.get_sol_params(**selection)),values):
                    raise RuntimeError("public parameter restore did not preserve exact values")
            except Exception as exc:failures.append(exc)
        if failures:
            raise ConstraintRestoreError("unrelated solver parameters could not be restored") from failures[0]
