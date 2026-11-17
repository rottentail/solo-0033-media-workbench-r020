"""Spherical camera motion and pixel-centred rectilinear inverse projection.

World axes are right, up, forward. Positive yaw turns right, positive pitch
looks up, positive roll tilts the picture clockwise. Quaternions are xyzw.
"""
import numpy as np
from scipy.spatial.transform import Rotation, Slerp


def orientation(keys, milliseconds):
    times = np.asarray([k['at_ms'] for k in keys], dtype=float)
    index = min(len(keys)-2, max(0, int(np.searchsorted(times, milliseconds)-1)))
    amount = (milliseconds-times[index])/(times[index+1]-times[index])
    first = np.asarray(keys[index]['quaternion'])
    second = np.asarray(keys[index+1]['quaternion'])
    blended = first*(1-amount)+second*amount
    return Rotation.from_quat(blended).as_matrix()


def rays(width, height, horizontal_fov):
    yy, xx = np.mgrid[:height, :width]
    horizontal = (xx+.5-width/2)/width*np.deg2rad(horizontal_fov)
    vertical = (height/2-yy-.5)/width*np.deg2rad(horizontal_fov)
    return np.stack((horizontal, vertical, np.ones_like(horizontal)), axis=-1)


def coordinates(camera_rays, rotation, width, height):
    centre = rotation @ np.array([0., 0., 1.])
    longitude = np.arctan2(centre[0], centre[2]) + camera_rays[...,0]
    latitude = np.arcsin(np.clip(centre[1], -1, 1)) + camera_rays[...,1]
    x = (longitude/(2*np.pi)+.5)*width-.5
    y = (.5-latitude/np.pi)*height-.5
    return x,y
