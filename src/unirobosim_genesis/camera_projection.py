"""Calibrated OpenCV rays sampled from an overscanned public pinhole camera.

This preserves the requested projection, including rational distortion. It adds
one explicit image-resampling stage; it does not claim Isaac pixel equivalence.
"""
from dataclasses import dataclass
import math
import numpy as np
from unirobosim import UnsupportedCapabilityError


@dataclass(frozen=True)
class CameraProjection:
    native_resolution: tuple[int, int]
    vertical_fov_degrees: float
    map_x: np.ndarray | None
    map_y: np.ndarray | None
    max_roundtrip_error_px: float = 0.0

    def sample(self, image, *, depth=False):
        if self.map_x is None:
            return image
        import cv2
        return cv2.remap(image, self.map_x, self.map_y,
            cv2.INTER_NEAREST if depth else cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def prepare_projection(camera):
    width, height = camera.width_px, camera.height_px
    calibration = camera.calibration
    if calibration is None:
        vfov = math.degrees(2*math.atan(math.tan(math.radians(camera.horizontal_fov_degrees)/2)*height/width))
        return CameraProjection((width, height), vfov, None, None)
    import cv2
    k = np.asarray(calibration.intrinsics, dtype=np.float64).reshape(3, 3)
    d = np.asarray(calibration.distortion_coefficients, dtype=np.float64)
    yy, xx = np.mgrid[:height, :width]
    pixels = np.stack((xx, yy), axis=-1).reshape(-1, 1, 2).astype(np.float64)
    criteria = (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 100, 1e-12)
    if hasattr(cv2, "undistortPointsIter"):
        rays = cv2.undistortPointsIter(pixels, k, d, None, None, criteria).reshape(-1, 2)
    else:
        rays = cv2.undistortPoints(pixels, k, d, criteria=criteria).reshape(-1, 2)
    projected, _ = cv2.projectPoints(np.column_stack((rays, np.ones(len(rays)))),
        np.zeros(3), np.zeros(3), k, d)
    error = np.linalg.norm(projected.reshape(-1, 2)-pixels.reshape(-1, 2), axis=1)
    if not np.isfinite(rays).all() or not np.isfinite(error).all() or error.max() > 1e-5:
        raise UnsupportedCapabilityError("camera rational distortion is not invertible to the required pixel accuracy", operation="genesis.camera", details={"max_roundtrip_error_px":float(error.max())})
    focal = max(k[0, 0], k[1, 1])
    # Symmetric native frustum must cover every requested ray, including
    # principal-point offset and the complete interpolation footprint.
    extent = np.max(np.abs(rays), axis=0)
    native_width, native_height = (int(2*math.ceil(focal*x+2)) for x in extent)
    if native_width > 8192 or native_height > 8192:
        raise UnsupportedCapabilityError("calibrated overscan exceeds the supported 8192 pixel rasterizer dimension", operation="genesis.camera")
    maps = rays*np.array([focal, focal])+np.array([native_width/2, native_height/2])
    vfov = math.degrees(2*math.atan(native_height/(2*focal)))
    return CameraProjection((native_width, native_height), vfov,
        maps[:, 0].reshape(height, width).astype(np.float32),
        maps[:, 1].reshape(height, width).astype(np.float32),float(error.max()))
