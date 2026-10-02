"""Independent camera-ray contracts and boundary failures, without native runtime."""
import numpy as np
import pytest
from unirobosim import CameraCalibrationSpec, CameraSpec, CameraModality, UnsupportedCapabilityError
from unirobosim_genesis.camera_projection import prepare_projection


def camera(distortion=(0.,)*8):
    return CameraSpec(width_px=100,height_px=80,modalities=(CameraModality.RGB,),
        calibration=CameraCalibrationSpec('opencv_pinhole',(71.,0.,44.,0.,79.,37.,0.,0.,1.),distortion))


def test_asymmetric_intrinsics_preserve_requested_rays():
    c=camera();p=prepare_projection(c)
    yy,xx=np.mgrid[:80,:100]
    focal=max(c.calibration.intrinsics[0],c.calibration.intrinsics[4])
    x=(p.map_x-p.native_resolution[0]/2)/focal
    y=(p.map_y-p.native_resolution[1]/2)/focal
    assert np.max(np.abs(x-(xx-44)/71))<1e-7
    assert np.max(np.abs(y-(yy-37)/79))<1e-7
    assert p.map_x.min()>0 and p.map_y.min()>0
    assert p.map_x.max()<p.native_resolution[0]-1
    assert p.map_y.max()<p.native_resolution[1]-1


def test_rational_tangential_distortion_roundtrips_all_output_pixels():
    c=camera((-.12,.025,.002,-.004,.001,.005,-.003,.002));p=prepare_projection(c)
    yy,xx=np.mgrid[:80,:100];f=79
    x=(p.map_x.astype(float)-p.native_resolution[0]/2)/f
    y=(p.map_y.astype(float)-p.native_resolution[1]/2)/f
    k1,k2,p1,p2,k3,k4,k5,k6=c.calibration.distortion_coefficients
    r=x*x+y*y;radial=(1+k1*r+k2*r*r+k3*r*r*r)/(1+k4*r+k5*r*r+k6*r*r*r)
    distorted_x=x*radial+2*p1*x*y+p2*(r+2*x*x)
    distorted_y=y*radial+p1*(r+2*y*y)+2*p2*x*y
    assert np.max(np.abs(71*distorted_x+44-xx))<1e-5
    assert np.max(np.abs(79*distorted_y+37-yy))<1e-5
    assert p.max_roundtrip_error_px<1e-8


def test_native_projection_and_sampling_contract():
    c=CameraSpec(width_px=100,height_px=80,modalities=(CameraModality.RGB,))
    p=prepare_projection(c);image=np.zeros((80,100,3),np.uint8)
    assert p.sample(image) is image
    q=prepare_projection(camera())
    image=np.full((*q.native_resolution[::-1],3),123,np.uint8)
    assert q.sample(image).shape==(80,100,3)
    assert np.all(q.sample(image)==123)


def test_noninvertible_projection_is_explicitly_rejected():
    with pytest.raises(UnsupportedCapabilityError):prepare_projection(camera((-2.,0.,0.,0.,0.,0.,0.,0.)))
