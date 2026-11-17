"""Spherical camera motion and pixel-centred rectilinear inverse projection.

World axes are right, up, forward. Positive yaw turns right, positive pitch
looks up, positive roll tilts the picture clockwise. Quaternions are xyzw.
The saved quaternion actively maps a camera ray into world space. Adjacent
keys take the shortest constant-angular-speed (spherical linear) path, so an
orientation recorded with either quaternion sign produces the same camera.
"""
import numpy as np
from scipy.spatial.transform import Rotation


def canonical(values):
    """Normalize a quaternion and pick one sign for identical orientations."""
    values = np.asarray(values, dtype=float)
    norm = float(np.linalg.norm(values))
    if norm < 1e-8 or not np.isfinite(norm):
        raise ValueError('orientation has no finite direction')
    values = values/norm
    sign = next((float(v) for v in values if v != 0), 1.0)
    values = values if sign > 0 else -values
    values[values == 0] = 0.0
    return values


def orientation(keys, milliseconds):
    times = np.asarray([k['at_ms'] for k in keys], dtype=float)
    index = min(len(keys)-2, max(0, int(np.searchsorted(times, milliseconds)-1)))
    span = times[index+1]-times[index]
    amount = min(1.0, max(0.0, (float(milliseconds)-times[index])/span))
    first = canonical(keys[index]['quaternion'])
    second = canonical(keys[index+1]['quaternion'])
    dot = float(np.dot(first, second))
    if dot < 0:
        second, dot = -second, -dot
    dot = min(1.0, max(-1.0, dot))
    if dot > 1-1e-12:
        blended = first*(1-amount)+second*amount
    else:
        angle = np.arccos(dot)
        blended = (np.sin((1-amount)*angle)*first
                   + np.sin(amount*angle)*second)/np.sin(angle)
    return Rotation.from_quat(blended/np.linalg.norm(blended)).as_matrix()


def rays(width, height, horizontal_fov):
    """Pixel-centred camera rays in the camera's own right/up/forward frame."""
    yy, xx = np.mgrid[:height, :width]
    focal = (width/2)/np.tan(np.deg2rad(horizontal_fov)/2)
    return np.stack((xx+.5-width/2, height/2-yy-.5,
                     np.full(xx.shape, focal, dtype=float)), axis=-1)


def coordinates(camera_rays, rotation, width, height):
    """Equirectangular sample coordinates of every ray in world direction.

    Longitude wraps modulo one turn. Beyond either pole the true direction
    naturally continues on the opposite meridian instead of clamping to the
    top or bottom source row.
    """
    world = camera_rays @ rotation.T
    right, up, forward = world[...,0], world[...,1], world[...,2]
    longitude = np.arctan2(right, forward)
    latitude = np.arctan2(up, np.hypot(right, forward))
    x = (longitude/(2*np.pi)+.5)*width-.5
    y = (.5-latitude/np.pi)*height-.5
    return x, y
