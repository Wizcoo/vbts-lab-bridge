"""Frozen single-axis lab estimator. Positive output is compression in newtons."""
import ctypes
import hashlib
import json
import os
from pathlib import Path

import numpy as np

PACKAGE = Path(__file__).resolve().parent

def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024), b''): h.update(block)
    return h.hexdigest()

def descriptor(uv):
    if uv.shape != (69,82,2) or uv.dtype != np.float32 or not np.isfinite(uv).all():
        raise ValueError('invalid native displacement grid')
    u,v=uv[:,:,0],uv[:,:,1]
    m=np.hypot(u,v).astype('float64'); w=m*m
    total=w.sum()
    mx=(w*GRID_X).sum()/total if total else 0.
    my=(w*GRID_Y).sum()/total if total else 0.
    q=np.quantile(m,[.5,.75,.9,.95,.98,.99,.995,.999,1])
    mean=m.mean(); rms=np.sqrt(w.mean())
    features=[*q,mean,rms,float(np.abs(u).mean()),float(np.abs(v).mean())]
    for amplitude in [q[6],mean,rms]:
        features.extend(amplitude*p for p in (mx,my,mx*mx,my*my,mx*my))
    features.extend(np.square([q[6],mean,rms]))
    return np.array(features)

GRID_X,GRID_Y=np.meshgrid(np.linspace(-1,1,82),np.linspace(-1,1,69))

def plane(bgr):
    if not isinstance(bgr,np.ndarray) or bgr.dtype!=np.uint8 or bgr.shape!=(480,640,3):
        raise ValueError('expected lossless camera BGR uint8 640x480; check camera format')
    return np.ascontiguousarray(bgr[:,:,0])

class Engine:
    def __init__(self, opencv_dll):
        self.binding=json.loads((PACKAGE/'manifest.json').read_text(encoding='utf8'))
        dll=Path(opencv_dll).expanduser().resolve()
        native=PACKAGE/'native/seed17_tracker.dll';weights=PACKAGE/'seed17.json'
        for file,key in [(dll,'opencv_sha256'),(native,'tracker_sha256'),(weights,'weights_sha256')]:
            if digest(file)!=self.binding[key]:raise ValueError(f'{file.name} SHA256 mismatch')
        self.binding=dict(self.binding,opencv_loaded_path=str(dll),tracker_loaded_path=str(native),weights_loaded_path=str(weights))
        if os.name!='nt' or ctypes.sizeof(ctypes.c_void_p)!=8:
            raise RuntimeError('This verified tracker requires 64-bit Windows and x64 Python.')
        self._runtime_dir=os.add_dll_directory(str(dll.parent))
        self._opencv=ctypes.CDLL(str(dll))
        self.lib=ctypes.CDLL(str(native))
        self.lib.seed17_open.argtypes=[ctypes.c_void_p];self.lib.seed17_open.restype=ctypes.c_void_p
        self.lib.seed17_track.argtypes=[ctypes.c_void_p,ctypes.c_void_p,ctypes.c_void_p,ctypes.c_size_t];self.lib.seed17_track.restype=ctypes.c_int
        self.lib.seed17_close.argtypes=[ctypes.c_void_p];self.lib.seed17_close.restype=None
        self.handle=None;self.uv=np.empty((69,82,2),np.float32)
        m=json.loads(weights.read_text(encoding='utf8'))
        self.mu,self.std,self.w,self.b,self.v=[np.asarray(m[k],dtype='float64') for k in ['mean','std','w','b','v']]
        self.bias,self.ym,self.ys=m['bias'],m['ym'],m['ys']
        if self.w.shape!=(31,32) or not (self.std>0).all():raise ValueError('invalid seed17 parameters')

    def reference(self,bgr):
        p=plane(bgr)
        self.clear()
        self.handle=self.lib.seed17_open(p.ctypes.data)
        if not self.handle:raise RuntimeError('failed to construct 000011 tracker reference')

    def predict(self,bgr):
        if not self.handle:raise RuntimeError('an unloaded reference is required')
        p=plane(bgr)
        code=self.lib.seed17_track(self.handle,p.ctypes.data,self.uv.ctypes.data,self.uv.size)
        if code:raise RuntimeError(f'native tracking failed ({code})')
        z=(descriptor(self.uv)-self.mu)/self.std
        y=float((np.tanh(z@self.w+self.b)@self.v+self.bias)*self.ys+self.ym)
        if not np.isfinite(y):raise RuntimeError('invalid prediction')
        return y

    def clear(self):
        if self.handle:self.lib.seed17_close(self.handle);self.handle=None
    def close(self):
        self.clear()
        self._runtime_dir.close()
