"""Bilinear viewport texture interpolation."""
import numpy as np


def texels(frame,x,y):
    height,width=frame.shape[:2]
    return frame[np.clip(y,0,height-1),np.clip(x,0,width-1)].astype(float)


def sample(frame,coordinates):
    x,y=coordinates
    x0,y0=np.floor(x).astype(int),np.floor(y).astype(int)
    dx,dy=(x-x0)[...,None],(y-y0)[...,None]
    top=texels(frame,x0,y0)*(1-dx)+texels(frame,x0+1,y0)*dx
    bottom=texels(frame,x0,y0+1)*(1-dx)+texels(frame,x0+1,y0+1)*dx
    return np.rint(top*(1-dy)+bottom*dy).astype('uint8')
