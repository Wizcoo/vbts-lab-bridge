"""Versioned offline UV engineering prototype. Never calls a camera or force API.

E2 bridges the copied E1 native float UV to a continuous float encoder. E3 is a
separate compound frontend: fixed-reference pairwise DIS, audited corrections,
endpoint geometry and explicit quality channels. It is NOT production qualified.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from types import MappingProxyType

import cv2
import numpy as np

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("uv_frontend_vendor", HERE / "vendor/preprocessing.py")
legacy = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(legacy)
PROTOCOL = json.loads((HERE / "protocol.json").read_text(encoding="utf-8"))


def sha_array(a):
    a = np.ascontiguousarray(a)
    h = hashlib.sha256(str(a.dtype).encode() + repr(a.shape).encode())
    h.update(a.tobytes())
    return h.hexdigest()


def object_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def letterbox_dimensions(w, h, target=112):
    scale = min(np.float32(target)/np.float32(w), np.float32(target)/np.float32(h))
    rw, rh = int(np.float32(w)*scale), int(np.float32(h)*scale)
    left, top = (target-rw)//2, (target-rh)//2
    return rw, rh, left, top


def encode_e2_float(displacement, contract=None, return_stages=False):
    """Only remove initial byte quantization and byte resize output quantization."""
    c = contract or legacy.read_contract()
    d = np.asarray(displacement, np.float32)
    if d.shape != (69,82,2) or not np.isfinite(d).all():
        raise ValueError("E2 requires finite native UV with shape (69,82,2)")
    rw, rh, left, top = letterbox_dimensions(82,69)
    continuous = np.empty((2,69,82), np.float32)
    encoded = np.full((2,112,112), np.float32(c["pad"]), np.float32)
    clipped = []
    for axis, bounds in enumerate((c["dx_range"],c["dy_range"])):
        lo, hi = np.float32(bounds[0]), np.float32(bounds[1])
        if not np.isfinite([lo,hi]).all() or lo >= hi:
            raise ValueError("Invalid encoding limits")
        clipped.append(int(np.count_nonzero((d[...,axis]<lo)|(d[...,axis]>hi))))
        continuous[axis] = (np.clip(d[...,axis],lo,hi)-lo)/(hi-lo)*np.float32(255)
        resized = cv2.resize(continuous[axis],(rw,rh),interpolation=cv2.INTER_LINEAR)
        encoded[axis,top:top+rh,left:left+rw] = resized
    tensor = encoded * np.float32(1.0/255.0)
    tensor = (tensor-np.asarray(c["mean"],np.float32)[:,None,None])/np.asarray(c["std"],np.float32)[:,None,None]
    if not np.isfinite(tensor).all():
        raise ValueError("Non-finite E2 tensor")
    info = {"shape":list(tensor.shape),"clipped_grid_values_per_axis":clipped,
            "zero_continuous":127.5,"padding_continuous":128.0,
            "resize_shape":[rh,rw],"per_frame_amplitude_normalization":False}
    return (tensor,info,{"continuous":continuous,"letterboxed":encoded}) if return_stages else (tensor,info)


class LegacyFloatBridge:
    """One old state evolution and native postprocess, then both encoder branches."""
    def __init__(self, reference):
        self.old = legacy.ProductionPreprocessor(reference)
    def process(self, image):
        e1, old_info = self.old.process(image)
        d = self.old.last_displacement.copy()
        e2, e2_info, stages = encode_e2_float(d,self.old.contract,True)
        return {"e1":e1,"e2":e2,"native_uv":d,"e1_bytes":self.old.last_encoded_uint8.copy(),
                "e2_continuous":stages["letterboxed"],"e1_info":old_info,"e2_info":e2_info}


def project_points(points, matrix):
    p = np.asarray(points,np.float64)
    H = np.asarray(matrix,np.float64)
    if p.shape[-1] != 2 or H.shape != (3,3) or not np.isfinite(H).all():
        raise ValueError("Expected xy points and finite 3x3 transform")
    ph = np.concatenate([p,np.ones(p.shape[:-1]+(1,))],axis=-1) @ H.T
    good = np.isfinite(ph).all(axis=-1) & (np.abs(ph[...,2])>1e-10)
    out = np.full(p.shape,np.nan,np.float64)
    np.divide(ph[...,:2],ph[...,2,None],out=out,where=good[...,None])
    return out,good


class Geometry:
    """Input images are already in this camera coordinate system.

    No distortion model is silently assumed calibrated. A real calibrated
    frontend must provide rectified images and transform metadata together.
    """
    def __init__(self, camera_to_standard, grid_xy, *, units="canonical_px", calibration_id="engineering_fixture",
                 physical_calibration_verified=False, rectification="not_calibrated",
                 independent_rigid_calibration_id=None):
        H = np.asarray(camera_to_standard,np.float64)
        grid = np.asarray(grid_xy,np.float64)
        if H.shape!=(3,3) or not np.isfinite(H).all() or abs(np.linalg.det(H))<1e-12:
            raise ValueError("Singular/non-finite geometry")
        if grid.shape!=(69,82,2) or not np.isfinite(grid).all():
            raise ValueError("Geometry grid must be finite 69x82 xy")
        if units not in ("canonical_px","mm"):
            raise ValueError("Unknown coordinate unit")
        if units=="mm" and not physical_calibration_verified:
            raise ValueError("mm requires independently verified physical calibration")
        if type(physical_calibration_verified) is not bool:
            raise ValueError("Physical calibration assertion must be a boolean")
        if not isinstance(calibration_id,str) or not calibration_id.strip():
            raise ValueError("Nonempty geometry calibration ID required")
        if not isinstance(rectification,str) or not rectification.strip():
            raise ValueError("Rectification provenance required")
        if independent_rigid_calibration_id is not None and (not isinstance(independent_rigid_calibration_id,str) or not independent_rigid_calibration_id.strip()):
            raise ValueError("Invalid independent rigid calibration ID")
        self.H=H.copy(); self.grid=grid.copy(); self.units=units
        self.calibration_id=calibration_id
        self.physical_calibration_verified=bool(physical_calibration_verified)
        self.rectification=rectification
        self.independent_rigid_calibration_id=independent_rigid_calibration_id
        self.points,self.points_valid=project_points(grid,np.linalg.inv(H))
        self.hash=object_hash(self.metadata())
        self._points_hash=sha_array(self.points)
        for a in (self.H,self.grid,self.points,self.points_valid):a.flags.writeable=False
    def metadata(self):
        return {"H":self.H.tolist(),"grid_sha256":sha_array(self.grid),"units":self.units,
                "calibration_id":self.calibration_id,"physical_calibration_verified":self.physical_calibration_verified,
                "rectification":self.rectification,
                "independent_rigid_calibration_id":self.independent_rigid_calibration_id}
    def assert_integrity(self):
        if object_hash(self.metadata())!=self.hash or sha_array(self.points)!=self._points_hash:
            raise ValueError("Geometry changed after provenance was frozen")
    @classmethod
    def legacy_canonical(cls):
        c=legacy.read_contract()
        rect=np.asarray([[156,69],[561,69],[561,411],[156,411]],np.float64)
        H,_=cv2.findHomography(rect,np.asarray(c["roi"],np.float32),cv2.RANSAC,3.0)
        x,y=np.meshgrid(np.arange(156,562,5),np.arange(69,412,5))
        return cls(np.linalg.inv(H),np.stack([x,y],axis=-1),calibration_id="legacy_roi_canonical_not_mm")


def bilinear_sample(image, points):
    """Exact bilinear interpolation, not OpenCV remap's 1/32-pixel table."""
    a=np.asarray(image); p=np.asarray(points,np.float64)
    h,w=a.shape[:2]
    ok=np.isfinite(p).all(axis=-1)&(p[...,0]>=0)&(p[...,0]<=w-1)&(p[...,1]>=0)&(p[...,1]<=h-1)
    x=np.clip(np.nan_to_num(p[...,0],nan=0,posinf=w-1,neginf=0),0,w-1)
    y=np.clip(np.nan_to_num(p[...,1],nan=0,posinf=h-1,neginf=0),0,h-1)
    x0=np.floor(x).astype(int); y0=np.floor(y).astype(int)
    x1=np.minimum(x0+1,w-1); y1=np.minimum(y0+1,h-1)
    dx=x-x0; dy=y-y0
    if a.ndim==3: dx=dx[...,None];dy=dy[...,None]
    out=(1-dy)*((1-dx)*a[y0,x0]+dx*a[y0,x1])+dy*((1-dx)*a[y1,x0]+dx*a[y1,x1])
    finite=np.isfinite(out).all(axis=-1) if out.ndim>ok.ndim else np.isfinite(out)
    return out.astype(np.float32),ok&finite


def apply_endpoint_geometry(points, raw_uv_px, geometry, *, rigid_transform=None):
    """No correction is inferred from the elastic displacement field."""
    geometry.assert_integrity()
    p=np.asarray(points,np.float64); raw=np.asarray(raw_uv_px,np.float64)
    endpoint=p+raw
    if rigid_transform is None:
        corrected_px=raw.copy(); corr_px=np.zeros_like(raw)
    else:
        endpoint,_=project_points(endpoint,rigid_transform)
        corrected_px=endpoint-p
        corr_px=raw-corrected_px
    q0,g0=project_points(p,geometry.H)
    q1,g1=project_points(p+raw,geometry.H)
    q2,g2=project_points(p+corrected_px,geometry.H)
    raw_std=q1-q0; corrected_std=q2-q0
    return {"raw_uv_px":raw.astype(np.float32),"correction_uv_px":corr_px.astype(np.float32),
            "corrected_uv_px":corrected_px.astype(np.float32),"raw_uv_standard":raw_std.astype(np.float32),
            "correction_uv_standard":(raw_std-corrected_std).astype(np.float32),
            "corrected_uv_standard":corrected_std.astype(np.float32),"geometry_valid":g0&g1&g2}


def encode_e3_tensor(corrected_uv, valid, confidence, *, fixed_scale=1.0):
    d=np.asarray(corrected_uv,np.float32); v=np.asarray(valid,bool); q=np.asarray(confidence,np.float32)
    if d.shape!=(69,82,2) or v.shape!=(69,82) or q.shape!=v.shape:
        raise ValueError("E3 field shape mismatch")
    if not np.isfinite(fixed_scale) or fixed_scale<=0:
        raise ValueError("Fixed scale must be positive")
    if not np.isfinite(q).all() or np.any((q<0)|(q>1)):
        raise ValueError("Confidence must be finite in [0,1]")
    if np.any(v & ~np.isfinite(d).all(axis=-1)):
        raise ValueError("Valid UV cannot contain non-finite values")
    rw,rh,left,top=letterbox_dimensions(82,69)
    support=cv2.resize(v.astype(np.float32),(rw,rh),interpolation=cv2.INTER_LINEAR)
    # Require all contributing neighbors valid; an interpolated hole is not measured data.
    out_valid=support>=np.float32(1-1e-6)
    tensor=np.zeros((4,112,112),np.float32)
    region=(slice(top,top+rh),slice(left,left+rw))
    for a in range(2):
        src=np.where(v,d[...,a],0).astype(np.float32)
        num=cv2.resize(src,(rw,rh),interpolation=cv2.INTER_LINEAR)
        value=np.divide(num,support,out=np.zeros_like(num),where=support>0)
        tensor[a][region]=np.where(out_valid,value/np.float32(fixed_scale),0)
    conf=cv2.resize(np.where(v,q,0).astype(np.float32),(rw,rh),interpolation=cv2.INTER_LINEAR)
    conf=np.divide(conf,support,out=np.zeros_like(conf),where=support>0)
    tensor[2][region]=out_valid.astype(np.float32)
    tensor[3][region]=np.where(out_valid,conf,0)
    return tensor


def _make_dis():
    d=cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_ULTRAFAST)
    d.setFinestScale(2);d.setPatchSize(8);d.setPatchStride(4)
    d.setGradientDescentIterations(10);d.setVariationalRefinementIterations(0)
    d.setUseSpatialPropagation(True);d.setUseMeanNormalization(True)
    return d


def _texture_and_saturation(gray, quality):
    f=gray.astype(np.float32); k=(int(quality["window"]),)*2
    mean=cv2.boxFilter(f,-1,k,normalize=True)
    var=np.maximum(cv2.boxFilter(f*f,-1,k,normalize=True)-mean*mean,0)
    sat=((gray<=quality["saturation_low"])|(gray>=quality["saturation_high"])).astype(np.float32)
    return np.sqrt(var),cv2.boxFilter(sat,-1,k,normalize=True)


def validate_quality(value):
    q=dict(value)
    if set(q)!=set(PROTOCOL["quality"]):raise ValueError("Quality configuration whitelist mismatch")
    numeric=set(q)-{"confidence_interpretation"}
    for k in numeric:
        if isinstance(q[k],(bool,np.bool_)) or not isinstance(q[k],(int,float,np.integer,np.floating)) or not np.isfinite(q[k]):
            raise ValueError("Quality values must be finite numbers: "+k)
    if type(q["window"]) is not int or not 3<=q["window"]<=255 or q["window"]%2!=1:
        raise ValueError("Quality window must be odd integer in [3,255]")
    if not 0<q["texture_std_min"]<=255 or not 0<q["photometric_abs_max"]<=255 or q["forward_backward_error_max_px"]<=0:
        raise ValueError("Quality thresholds must be positive and in intensity range")
    if not 0<q["saturation_fraction_max"]<=1:
        raise ValueError("Saturation fraction threshold must be in (0,1]")
    if any(type(q[k]) is not int for k in ("saturation_low","saturation_high")) or not 0<=q["saturation_low"]<q["saturation_high"]<=255:
        raise ValueError("Invalid saturation intensity bounds")
    if not 0<=q["invalid_frame_fraction"]<q["valid_frame_fraction"]<=1:
        raise ValueError("Frame quality thresholds must satisfy 0 <= invalid < valid <= 1")
    if not isinstance(q["confidence_interpretation"],str) or not q["confidence_interpretation"].strip():
        raise ValueError("Confidence interpretation required")
    return q


def inspect_unverified_pair(reference, current, geometry=None):
    """Read-only matching diagnostic; deliberately returns no model tensor.

    This function does not authorize a reference, cannot make force inputs, and
    does not bypass AuditedUVFrontend's reference gate. It exposes observations
    for real data whose physical unloaded reference has not been established.
    """
    ref=legacy.blue_channel(reference);cur=legacy.blue_channel(current)
    g=geometry or Geometry.legacy_canonical();q=PROTOCOL["quality"]
    h,w=ref.shape
    flow=_make_dis().calc(ref,cur,np.zeros((h,w,2),np.float32))
    back=_make_dis().calc(cur,ref,np.zeros((h,w,2),np.float32))
    raw,ok=bilinear_sample(flow,g.points);end=g.points+raw
    rev,eok=bilinear_sample(back,end)
    fb=np.linalg.norm(raw+rev,axis=-1)
    rp,rok=bilinear_sample(ref,g.points);cp,cok=bilinear_sample(cur,end)
    residual=np.abs(rp-cp)
    rt,rs=_texture_and_saturation(ref,q);ct,cs=_texture_and_saturation(cur,q)
    rt,_=bilinear_sample(rt,g.points);rs,_=bilinear_sample(rs,g.points)
    ct,_=bilinear_sample(ct,end);cs,_=bilinear_sample(cs,end)
    audit=apply_endpoint_geometry(g.points,raw,g)
    valid=ok&eok&rok&cok&audit["geometry_valid"]&(np.minimum(rt,ct)>=q["texture_std_min"])&(np.maximum(rs,cs)<q["saturation_fraction_max"])&(residual<=q["photometric_abs_max"])&(fb<=q["forward_backward_error_max_px"])
    confidence=np.minimum(np.clip(np.minimum(rt,ct)/(2*q["texture_std_min"]),0,1),np.minimum(np.exp(-residual/q["photometric_abs_max"]),np.exp(-fb/q["forward_backward_error_max_px"])))
    confidence=np.where(valid,np.nan_to_num(confidence,nan=0),0).astype(np.float32)
    return {**audit,"valid":valid,"confidence":confidence,"forward_backward_error_px":fb,
            "photometric_residual":residual,"reference_status":"UNVERIFIED_REFERENCE",
            "force_input_eligible":False,"model_tensor":None,"drop_evaluation_row":False,
            "contact_status":"UNKNOWN","geometry_units":g.units}


class AuditedUVFrontend:
    REFERENCE_FIELDS={"sensor_id","session_id","reference_id","timestamp_ns","sha256","unloaded_confirmed",
                      "unloaded_evidence","role","allowed_reference_hashes"}
    FRAME_FIELDS={"sensor_id","session_id","timestamp_ns","frame_id"}
    def __init__(self, reference, reference_binding, geometry=None, quality=None):
        b=dict(reference_binding)
        if set(b)!=self.REFERENCE_FIELDS:
            raise ValueError("Reference fields must exactly match inference whitelist")
        self.reference=legacy.blue_channel(reference).copy()
        digest=sha_array(self.reference)
        for k in ("sensor_id","session_id","reference_id","unloaded_evidence"):
            if not isinstance(b[k],str) or not b[k].strip():raise ValueError("Reference requires nonempty string: "+k)
        if type(b["unloaded_confirmed"]) is not bool or b["unloaded_confirmed"] is not True:
            raise ValueError("Unloaded confirmation must be literal true")
        if type(b["allowed_reference_hashes"]) is not list or not b["allowed_reference_hashes"] or any(not isinstance(x,str) or len(x)!=64 or any(c not in "0123456789abcdef" for c in x) for x in b["allowed_reference_hashes"]):
            raise ValueError("Allowed reference hashes must be a nonempty list of sha256 strings")
        if not isinstance(b["sha256"],str) or len(b["sha256"])!=64:
            raise ValueError("Invalid reference digest type")
        if b["sha256"]!=digest or digest not in b["allowed_reference_hashes"]:
            raise ValueError("Unapproved/reference hash mismatch")
        if not b["unloaded_confirmed"] or not b["unloaded_evidence"].strip():
            raise ValueError("Missing independent unloaded reference evidence")
        if b["role"] not in ("pre_sequence_unloaded","calibration_unloaded"):
            raise ValueError("after_lift/future reference is not an online input")
        if type(b["timestamp_ns"]) is not int or b["timestamp_ns"]<0:
            raise ValueError("Invalid reference timestamp")
        b["allowed_reference_hashes"]=tuple(b["allowed_reference_hashes"])
        self.binding=MappingProxyType(b); self.geometry=geometry or Geometry.legacy_canonical()
        self.quality=MappingProxyType(validate_quality(PROTOCOL["quality"] if quality is None else quality))
        self.config_hash=object_hash(dict(self.quality))
        self.ref_texture,self.ref_sat=_texture_and_saturation(self.reference,self.quality)
        ref_texture,ok=bilinear_sample(self.ref_texture,self.geometry.points)
        ref_sat,_=bilinear_sample(self.ref_sat,self.geometry.points)
        if np.mean(ok&(ref_texture>=self.quality["texture_std_min"])&(ref_sat<self.quality["saturation_fraction_max"]))<.20:
            raise ValueError("Reference unusable: no trackable texture or saturation")
        self.last_timestamp=b["timestamp_ns"];self.frame_count=0
        self.reference.flags.writeable=False
    def _rigid(self, correction, frame):
        if correction is None: return None,"NONE"
        fields={"matrix","source","calibration_id","sensor_id","session_id","timestamp_ns","reference_sha256","evidence_id"}
        if set(correction)!=fields: raise ValueError("Rigid correction evidence whitelist mismatch")
        if type(correction["timestamp_ns"]) is not int or any(not isinstance(correction[k],str) or not correction[k].strip() for k in fields-{"matrix","timestamp_ns"}):
            raise ValueError("Invalid rigid evidence field types")
        g=self.geometry
        if correction["source"]!="independent_rigid_reference" or not g.independent_rigid_calibration_id:
            raise ValueError("Elastic-field/common-mode bias correction is forbidden")
        if correction["calibration_id"]!=g.independent_rigid_calibration_id or not correction["evidence_id"]:
            raise ValueError("Unapproved independent rigid reference calibration")
        if any(correction[k]!=frame[k] for k in ("sensor_id","session_id","timestamp_ns")) or correction["reference_sha256"]!=self.binding["sha256"]:
            raise ValueError("Rigid evidence is stale/future/wrong reference")
        m=np.asarray(correction["matrix"],np.float64)
        if m.shape!=(3,3) or not np.isfinite(m).all() or not np.allclose(m[2],[0,0,1],atol=1e-10):
            raise ValueError("Rigid correction requires finite affine homogeneous matrix")
        if not np.allclose(m[:2,:2].T@m[:2,:2],np.eye(2),atol=1e-6) or abs(np.linalg.det(m[:2,:2])-1)>1e-6:
            raise ValueError("Independent correction must be rigid, not arbitrary deformation")
        return m,"INDEPENDENT_RIGID_REFERENCE"
    def process(self,image,frame_binding,*,correction=None):
        f=dict(frame_binding)
        self.geometry.assert_integrity()
        if object_hash(dict(self.quality))!=self.config_hash:raise ValueError("Quality changed after provenance was frozen")
        if sha_array(self.reference)!=self.binding["sha256"]:raise ValueError("Reference changed after binding")
        if set(f)!=self.FRAME_FIELDS: raise ValueError("Frame metadata contains unapproved inference fields")
        if any(f[k]!=self.binding[k] for k in ("sensor_id","session_id")):
            raise ValueError("Wrong sensor/session reference")
        if any(not isinstance(f[k],str) or not f[k].strip() for k in ("sensor_id","session_id")) or type(f["frame_id"]) is not int or f["frame_id"]<0:
            raise ValueError("Invalid frame identity")
        if type(f["timestamp_ns"]) is not int or f["timestamp_ns"]<=self.last_timestamp:
            raise ValueError("Non-causal/non-increasing frame timestamp")
        current=legacy.blue_channel(image)
        rigid,reason=self._rigid(correction,f)
        h,w=current.shape
        # No previous frame, adaptive reference or temporal flow seed enters these solves.
        raw_dense=_make_dis().calc(self.reference,current,np.zeros((h,w,2),np.float32))
        reverse=_make_dis().calc(current,self.reference,np.zeros((h,w,2),np.float32))
        p=self.geometry.points
        raw,inside=bilinear_sample(raw_dense,p)
        endpoint=p+raw
        backward,inside_end=bilinear_sample(reverse,endpoint)
        fb=np.linalg.norm(raw+backward,axis=-1)
        current_pixels,current_ok=bilinear_sample(current,endpoint)
        reference_pixels,ref_ok=bilinear_sample(self.reference,p)
        photo=np.abs(current_pixels-reference_pixels)
        cur_texture,cur_sat=_texture_and_saturation(current,self.quality)
        rt,_=bilinear_sample(self.ref_texture,p);ct,_=bilinear_sample(cur_texture,endpoint)
        rs,_=bilinear_sample(self.ref_sat,p);cs,_=bilinear_sample(cur_sat,endpoint)
        texture=np.minimum(rt,ct);sat=np.maximum(rs,cs)
        audit=apply_endpoint_geometry(p,raw,self.geometry,rigid_transform=rigid)
        finite=inside&inside_end&current_ok&ref_ok&audit["geometry_valid"]
        q=self.quality
        tests=[finite,texture>=q["texture_std_min"],photo<=q["photometric_abs_max"],
               fb<=q["forward_backward_error_max_px"],sat<q["saturation_fraction_max"]]
        names=["geometry_or_bounds","texture","photometric","forward_backward","saturation"]
        valid=np.logical_and.reduce(tests)
        reason_bits=np.zeros(valid.shape,np.uint16)
        for i,t in enumerate(tests): reason_bits[~t]|=np.uint16(1<<i)
        confidence=np.minimum(np.clip(texture/(2*q["texture_std_min"]),0,1),
                    np.minimum(np.exp(-photo/q["photometric_abs_max"]),np.exp(-fb/q["forward_backward_error_max_px"])))
        confidence=np.where(valid,np.nan_to_num(confidence,nan=0),0).astype(np.float32)
        tensor=encode_e3_tensor(audit["corrected_uv_standard"],valid,confidence)
        fraction=float(valid.mean())
        status="VALID" if fraction>=q["valid_frame_fraction"] else "INVALID" if fraction<q["invalid_frame_fraction"] else "DEGRADED"
        self.frame_count+=1;self.last_timestamp=f["timestamp_ns"]
        info={**f,"reference_id":self.binding["reference_id"],"reference_sha256":self.binding["sha256"],
              "geometry_hash":self.geometry.hash,"geometry_units":self.geometry.units,"frame_number":self.frame_count,
              "measurement_status":status,"contact_status":"UNKNOWN","center":None,"center_valid":False,
              "valid_grid_fraction":fraction,"valid_tensor_fraction":float(tensor[2].mean()),
              "quality_fail_counts":{n:int((~t).sum()) for n,t in zip(names,tests)},"correction_reason":reason,
              "reference_update":False,"per_frame_amplitude_normalization":False,
              "temporal_policy":"independent_zero_seed_reference_current_pair",
              "confidence_calibrated_probability":False,"force_prediction_available":False,
              "drop_evaluation_row":False}
        return {**audit,"raw_dense_uv_px":raw_dense,"valid":valid,"confidence":confidence,
                "reason_bits":reason_bits,"forward_backward_error_px":fb.astype(np.float32),
                "photometric_residual":photo.astype(np.float32),"texture_std":texture.astype(np.float32),
                "saturation_fraction":sat.astype(np.float32),"tensor":tensor,"info":info}
    def state_dict(self):
        return {"reference_sha256":self.binding["sha256"],"sensor_id":self.binding["sensor_id"],
                "session_id":self.binding["session_id"],"geometry_hash":self.geometry.hash,
                "quality_hash":self.config_hash,"frame_count":self.frame_count,"last_timestamp_ns":self.last_timestamp,
                "temporal_policy":"independent_zero_seed_reference_current_pair"}
    def load_state_dict(self,state):
        base=self.state_dict()
        if set(state)!=set(base): raise ValueError("State field mismatch")
        for k in set(base)-{"frame_count","last_timestamp_ns"}:
            if state[k]!=base[k]: raise ValueError("State/reference/config mismatch: "+k)
        if type(state["frame_count"]) is not int or state["frame_count"]<0:
            raise ValueError("Invalid checkpoint frame count")
        t=state["last_timestamp_ns"]
        if type(t) is not int or t<self.binding["timestamp_ns"]:
            raise ValueError("Invalid checkpoint timestamp")
        self.frame_count=state["frame_count"];self.last_timestamp=t


def synthetic_reference_binding(reference):
    return {"sensor_id":"synthetic_sensor","session_id":"engineering_sequence","reference_id":"fixed_unloaded_fixture",
            "timestamp_ns":0,"sha256":sha_array(legacy.blue_channel(reference)),"unloaded_confirmed":True,
            "unloaded_evidence":"synthetic construction; no physical unloaded claim","role":"pre_sequence_unloaded",
            "allowed_reference_hashes":[sha_array(legacy.blue_channel(reference))]}


def synthetic_frame_binding(i):
    return {"sensor_id":"synthetic_sensor","session_id":"engineering_sequence","timestamp_ns":int(i)*1000000,"frame_id":int(i)}
