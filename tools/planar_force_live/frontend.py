"""Portable wrapper around byte-identical, fixed-reference DIS source files."""
from pathlib import Path
import hashlib
import importlib.util
import json
import cv2
import numpy as np

ROOT=Path(__file__).resolve().parent
SENSOR_ID='HSMVBTS_G01_H20_000011'
VENDOR_FILES={'vendor/float_tracker.py','vendor/protocol.json','vendor/sealed_baseline/uv_frontend.py',
              'vendor/sealed_baseline/protocol.json','vendor/sealed_baseline/vendor/preprocessing.py','vendor/sealed_baseline/vendor/profile.txt'}

def sha_bytes(blob): return hashlib.sha256(blob).hexdigest()
def sha_file(path): return sha_bytes(Path(path).read_bytes())
def checked_image(image):
    if not isinstance(image,np.ndarray) or image.dtype!=np.uint8 or image.shape!=(480,640,3):
        raise ValueError('Expected camera BGR uint8 image exactly 640x480; no resize/conversion fallback')
    return image

def read_image(path, expected_hash=None):
    blob=Path(path).read_bytes(); digest=sha_bytes(blob)
    if expected_hash is not None and digest!=expected_hash: raise ValueError('Image SHA256 differs')
    image=cv2.imdecode(np.frombuffer(blob,np.uint8),cv2.IMREAD_UNCHANGED)
    return checked_image(image),digest

class PairFrontend:
    def __init__(self):
        self.contract_bytes=(ROOT/'frontend_contract.json').read_bytes()
        self.contract=json.loads(self.contract_bytes.decode('utf-8-sig'))
        fixed=dict(schema='planar_dis_raw_camera_uv_v1',sensor_id=SENSOR_ID,image_shape=[480,640,3],image_dtype='uint8',image_channel='B',
                   uv_shape=[2,69,82],uv_dtype='float32',uv_units='pixel',valid_shape=[1,69,82],valid_dtype='bool',
                   geometry_hash='be7f9795d5fdebee10e6f65202e1f7e6ef20458b08f8ba5f1489f77962e0aaa6',method='old_dis_fixed_reference_pairwise',
                   valid_fraction_minimum=.2,temporal_filter=False,amplitude_normalization=False,rigid_correction=False,reference_updates=False,qualification='NOT_READY')
        if any(self.contract.get(k)!=v for k,v in fixed.items()): raise ValueError('Frontend contract constant differs')
        if set(self.contract.get('vendor_sha256',{}))!=VENDOR_FILES: raise ValueError('Exactly six vendored dependencies must be hash bound')
        self.sources={ROOT/'frontend_contract.json':sha_bytes(self.contract_bytes),Path(__file__).resolve():sha_file(__file__)}
        for relative,digest in self.contract['vendor_sha256'].items():
            path=(ROOT/relative).resolve()
            if not path.is_relative_to(ROOT/'vendor') or sha_file(path)!=digest: raise ValueError('Vendored frontend hash mismatch: '+relative)
            self.sources[path]=digest
        cv2.setNumThreads(1)
        spec=importlib.util.spec_from_file_location('planar_fixed_float_tracker',ROOT/'vendor/float_tracker.py')
        self.old=importlib.util.module_from_spec(spec);spec.loader.exec_module(self.old)
        self.geometry=self.old.base.Geometry.legacy_canonical()
        if self.geometry.hash!=self.contract['geometry_hash']: raise ValueError('Frontend geometry differs')
        self.forward=self.old.GridMatcher('old_dis');self.reverse=self.old.GridMatcher('old_dis')
        self.assert_unchanged()

    def assert_unchanged(self):
        if any(sha_file(p)!=h for p,h in self.sources.items()): raise ValueError('Frontend source/contract changed')

    def process(self,reference,current):
        result=self.old.match_pair(checked_image(reference),checked_image(current),self.geometry,'old_dis',self.forward,self.reverse)
        uv=np.transpose(result['raw_uv_px'],(2,0,1)).astype(np.float32)[None]
        valid=result['valid'].astype(bool)[None,None]
        if uv.shape!=(1,2,69,82) or valid.shape!=(1,1,69,82) or not np.isfinite(uv).all(): raise ValueError('Invalid raw UV result')
        return dict(uv=uv,valid=valid,confidence=result['confidence'].astype(np.float32),reason_bits=result['reason_bits'],
                    quality_valid_fraction=float(valid.mean()))
