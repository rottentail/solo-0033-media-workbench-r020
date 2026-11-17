"""Bilinear viewport sampling in scene-linear sRGB light on the sphere.

The equirectangular source is periodic in longitude: crossing either side seam
wraps to the opposite meridian. Latitude is naturally limited to the poles by
the inverse projection, so no texel is replicated along the top or bottom row.
Interpolation happens in linear light, preserving the photometric weight of
bright/dark boundaries and introducing no seam or exposure change.
"""
import numpy as np


def linear(values):
    values = np.asarray(values, dtype=float)/255
    return np.where(values <= 0.04045, values/12.92, ((values+0.055)/1.055)**2.4)


def encoded(values):
    values = np.clip(values, 0, 1)
    code = np.where(values <= 0.0031308, 12.92*values, 1.055*values**(1/2.4)-0.055)
    return np.rint(code*255).astype('uint8')


# Decoded 8-bit codes expressed as scene-linear light, as one shared lookup.
TABLE = linear(np.arange(256)).astype(np.float32)


def sample(frame, coordinates):
    """Sample a frame at (x, y) equirectangular cell-centre coordinates.

    Works on float32 arrays so a 640x360 viewport stays well inside the
    delivery memory budget.
    """
    height, width = frame.shape[:2]
    x = np.mod(np.asarray(coordinates[0], dtype=np.float32), width)
    y = np.clip(np.asarray(coordinates[1], dtype=np.float32), 0, height-1)
    x0 = np.floor(x).astype(np.int64) % width
    y0 = np.floor(y).astype(np.int64)
    x1 = (x0+1) % width
    y1 = np.minimum(y0+1, height-1)
    dx = (x-x0).astype(np.float32)[..., None]
    dy = (y-y0).astype(np.float32)[..., None]

    top = np.subtract(TABLE[frame[y0, x1]], TABLE[frame[y0, x0]], dtype=np.float32)
    top *= dx
    top += TABLE[frame[y0, x0]]
    bottom = np.subtract(TABLE[frame[y1, x1]], TABLE[frame[y1, x0]], dtype=np.float32)
    bottom *= dx
    bottom += TABLE[frame[y1, x0]]
    bottom -= top
    bottom *= dy
    bottom += top
    np.clip(bottom, 0, 1, out=bottom)

    low = bottom <= 0.0031308
    np.multiply(bottom, 12.92, out=bottom, where=low)
    np.power(bottom, 1/2.4, out=bottom, where=~low)
    np.multiply(bottom, 1.055, out=bottom, where=~low)
    np.subtract(bottom, 0.055, out=bottom, where=~low)
    np.multiply(bottom, 255, out=bottom)
    np.rint(bottom, out=bottom)
    return bottom.astype(np.uint8)
