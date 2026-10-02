"""Small SI/xyzw boundary conversions, independent of Genesis and Torch."""
import numpy as np
from unirobosim import ArrayValue, Pose

def numpy(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)

def array(value, dtype="float64"):
    value = numpy(value)
    cast = bool if dtype == "bool" else float
    return ArrayValue(tuple(value.shape), tuple(cast(v) for v in value.reshape(-1)), dtype=dtype)

def wxyz(quat):
    x,y,z,w = quat
    return (w,x,y,z)

def xyzw(quat):
    value = numpy(quat)
    q = value[..., [1,2,3,0]]
    return q / np.linalg.norm(q, axis=-1, keepdims=True)

def rotation(q):
    x,y,z,w = q
    return np.array([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],
        [2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],
        [2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]])

def multiply(a,b):
    x,y,z,w = a; X,Y,Z,W = b
    q = np.array([w*X+x*W+y*Z-z*Y,w*Y-x*Z+y*W+z*X,w*Z+x*Y-y*X+z*W,w*W-x*X-y*Y-z*Z])
    return tuple(float(v) for v in q/np.linalg.norm(q))

def compose(a,b):
    p = np.asarray(a.position)+rotation(a.orientation_xyzw)@b.position
    return Pose(tuple(float(v) for v in p), multiply(a.orientation_xyzw,b.orientation_xyzw))

def inverse(p):
    q=tuple(-v for v in p.orientation_xyzw[:3])+(p.orientation_xyzw[3],)
    xyz=-rotation(q)@p.position
    return Pose(tuple(float(v) for v in xyz), q)
