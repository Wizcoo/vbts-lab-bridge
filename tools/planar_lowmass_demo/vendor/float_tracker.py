"""Fixed-reference grid matching; no force labels, time filtering or UV rescaling."""
from __future__ import annotations
import importlib.util
import json
from pathlib import Path
import sys
from types import MappingProxyType

sys.dont_write_bytecode = True
import cv2
import numpy as np
from scipy.ndimage import map_coordinates, spline_filter

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("sealed_uv_baseline", HERE / "sealed_baseline/uv_frontend.py")
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)
PROTOCOL = json.loads((HERE / "protocol.json").read_text(encoding="utf-8"))


def gray(image):
    a = np.asarray(image)
    if a.dtype != np.uint8 or a.shape not in ((480, 640), (480, 640, 3)):
        raise ValueError("Expected uint8 camera observations 480x640, monochrome or BGR")
    return base.legacy.blue_channel(a)


def sample_cubic(coeff, points):
    shape = points.shape[:-1]
    coords = np.stack((points[..., 1].ravel(), points[..., 0].ravel()))
    return map_coordinates(coeff, coords, order=3, prefilter=False, mode="mirror").reshape(shape)


class GridMatcher:
    def __init__(self, method="float_iclk"):
        if method not in ("float_iclk", "pyr_lk", "old_dis"):
            raise ValueError("Unknown fixed matcher")
        self._method = method
        value = PROTOCOL["options"].get(method)
        self.options = MappingProxyType(dict(value)) if isinstance(value, dict) else value
        self.config_hash = base.object_hash(value)
        self._cache = None

    @property
    def method(self):
        return self._method

    def assert_integrity(self):
        value = dict(self.options) if isinstance(self.options, MappingProxyType) else self.options
        if self.method not in PROTOCOL["options"] or base.object_hash(value) != self.config_hash or base.object_hash(PROTOCOL["options"][self.method]) != self.config_hash:
            raise ValueError("Matcher configuration changed after freeze")

    def track(self, reference, current, points):
        self.assert_integrity()
        ref, cur = gray(reference), gray(current)
        p = np.asarray(points, np.float64)
        if p.ndim != 3 or p.shape[-1] != 2 or not np.isfinite(p).all():
            raise ValueError("Finite grid of camera xy coordinates required")
        if self.method == "pyr_lk":
            opt = self.options
            pp = p.astype(np.float32).reshape(-1, 1, 2)
            end, status, residual = cv2.calcOpticalFlowPyrLK(ref, cur, pp, None,
                winSize=(opt["patch_size"],)*2, maxLevel=opt["max_level"],
                criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, opt["iterations"], opt["epsilon_px"]),
                minEigThreshold=opt["min_eigenvalue"])
            d = (end.reshape(p.shape).astype(np.float64)-p).astype(np.float32)
            return d, status.reshape(p.shape[:-1]).astype(bool), {"photometric_gain": None, "photometric_offset": None}
        dense = base._make_dis().calc(ref, cur, np.zeros(ref.shape+(2,), np.float32))
        seed, inside = base.bilinear_sample(dense, p)
        if self.method == "old_dis":
            return seed, inside, {"photometric_gain": None, "photometric_offset": None}
        opt = self.options
        radius = opt["patch_size"]//2
        key = (base.sha_array(ref), base.sha_array(p))
        if self._cache is not None and self._cache[0] == key:
            offsets, weight, coords, t, tm, gx, gy, xx, yy, xy, det, tvar = self._cache[1]
        else:
            y, x = np.mgrid[-radius:radius+1, -radius:radius+1]
            offsets = np.stack([x.ravel(), y.ravel()], -1).astype(np.float64)
            # Fixed spatial apodization; does not normalize displacement amplitudes.
            h = np.hanning(opt["patch_size"])
            weight = (h[:, None]*h[None, :]).ravel(); weight /= weight.sum()
            coords = p.reshape(-1, 1, 2)+offsets
            coeff = spline_filter(ref.astype(np.float64), order=3, mode="mirror")
            template = sample_cubic(coeff, coords)
            tm = np.sum(template*weight, axis=1)
            t = template-tm[:, None]
            gx = (sample_cubic(coeff, coords+[.01, 0])-sample_cubic(coeff, coords-[.01, 0]))/.02
            gy = (sample_cubic(coeff, coords+[0, .01])-sample_cubic(coeff, coords-[0, .01]))/.02
            tvar = np.sum(t*t*weight, axis=1)
            for grad in (gx, gy):
                grad -= np.sum(grad*weight, axis=1)[:, None]
                grad -= t*(np.sum(grad*t*weight, axis=1)/np.maximum(tvar, 1e-9))[:, None]
            xx = np.sum(gx*gx*weight, axis=1)
            yy = np.sum(gy*gy*weight, axis=1)
            xy = np.sum(gx*gy*weight, axis=1)
            det = xx*yy-xy*xy
            self._cache = key, (offsets, weight, coords, t, tm, gx, gy, xx, yy, xy, det, tvar)
        coeff_cur = spline_filter(cur.astype(np.float64), order=3, mode="mirror")
        displacement = seed.reshape(-1, 2).astype(np.float64)
        stable = (det > 1e-6) & (tvar > 1)
        gain = np.ones(len(displacement)); offset = np.zeros(len(displacement))
        for iteration in range(opt["iterations"]):
            observed = sample_cubic(coeff_cur, coords+displacement[:, None, :])
            cm = np.sum(observed*weight, axis=1)
            centered = observed-cm[:, None]
            gain = np.sum(t*centered*weight, axis=1)/np.maximum(tvar, 1e-9)
            offset = cm-gain*tm
            error = centered-gain[:, None]*t
            bx = np.sum(gx*error*weight, axis=1)
            by = np.sum(gy*error*weight, axis=1)
            denom = np.where(stable, det, 1)*np.where(np.abs(gain)>.25, gain, 1)
            step = np.stack([(yy*bx-xy*by)/denom, (xx*by-xy*bx)/denom], -1)
            step[~stable] = 0
            norm = np.linalg.norm(step, axis=1)
            step *= np.minimum(1, opt["step_max_px"]/np.maximum(norm, 1e-12))[:, None]
            displacement -= step
            if np.max(np.abs(step)) < opt["epsilon_px"]:
                break
        shape = p.shape[:-1]
        endpoint = p+displacement.reshape(p.shape)
        valid = stable.reshape(shape) & inside & np.isfinite(endpoint).all(axis=-1)
        valid &= (endpoint[..., 0]>=radius) & (endpoint[..., 0]<640-radius)
        valid &= (endpoint[..., 1]>=radius) & (endpoint[..., 1]<480-radius)
        valid &= (gain.reshape(shape)>.25) & (gain.reshape(shape)<4)
        return displacement.reshape(p.shape).astype(np.float32), valid, {
            "photometric_gain": gain.reshape(shape).astype(np.float32),
            "photometric_offset": offset.reshape(shape).astype(np.float32),
            "iterations": iteration+1}


def match_pair(reference, current, geometry=None, method="float_iclk", matcher=None, reverse_matcher=None, quality=None):
    """Numerical matcher, no reference eligibility asserted and no force tensor returned."""
    ref, cur = gray(reference), gray(current)
    g = geometry or base.Geometry.legacy_canonical(); g.assert_integrity()
    p = g.points; q = base.PROTOCOL["quality"] if quality is None else quality
    forward = matcher or GridMatcher(method)
    reverse = reverse_matcher or GridMatcher(method)
    raw, good, photo_fit = forward.track(ref, cur, p)
    end = p+raw
    back, back_good, _ = reverse.track(cur, ref, end)
    fb = np.linalg.norm(raw+back, axis=-1)
    a, aok = base.bilinear_sample(ref, p); b, bok = base.bilinear_sample(cur, end)
    # Compensate only the independently fitted local photometric nuisance model.
    if photo_fit["photometric_gain"] is None:
        residual = np.abs(b-a)
    else:
        residual = np.abs(b-(photo_fit["photometric_gain"]*a+photo_fit["photometric_offset"]))
    rt, rs = base._texture_and_saturation(ref, q); ct, cs = base._texture_and_saturation(cur, q)
    rt, _ = base.bilinear_sample(rt, p); rs, _ = base.bilinear_sample(rs, p)
    ct, _ = base.bilinear_sample(ct, end); cs, _ = base.bilinear_sample(cs, end)
    audit = base.apply_endpoint_geometry(p, raw, g)
    tests = [good & back_good & aok & bok & audit["geometry_valid"],
             np.minimum(rt, ct)>=q["texture_std_min"], residual<=q["photometric_abs_max"],
             fb<=q["forward_backward_error_max_px"], np.maximum(rs, cs)<q["saturation_fraction_max"]]
    valid = np.logical_and.reduce(tests)
    bits = np.zeros(valid.shape, np.uint16)
    for i, check in enumerate(tests): bits[~check] |= 1<<i
    confidence = np.minimum(np.clip(np.minimum(rt, ct)/(2*q["texture_std_min"]), 0, 1),
                           np.minimum(np.exp(-residual/q["photometric_abs_max"]), np.exp(-fb/q["forward_backward_error_max_px"])))
    confidence = np.where(valid, np.nan_to_num(confidence, nan=0), 0).astype(np.float32)
    return {**audit, "valid": valid, "confidence": confidence, "reason_bits": bits,
            "forward_backward_error_px": fb, "photometric_residual": residual, "photometric_fit": photo_fit,
            "method": method, "reference_status": "UNVERIFIED_REFERENCE", "force_input_eligible": False,
            "model_tensor": None, "drop_evaluation_row": False, "contact_status": "UNKNOWN"}


class AuditedFloatFrontend(base.AuditedUVFrontend):
    """Strict causal/reference constructor inherited unchanged from sealed v1."""
    def __init__(self, reference, reference_binding, geometry=None, method="float_iclk"):
        super().__init__(reference, reference_binding, geometry)
        self.method = method
        self.forward_matcher = GridMatcher(method)
        self.reverse_matcher = GridMatcher(method)
        self.matcher_hash = self.forward_matcher.config_hash

    def process(self, image, frame_binding, correction=None):
        self.geometry.assert_integrity()
        if base.sha_array(self.reference) != self.binding["sha256"]:
            raise ValueError("Reference changed after provenance was frozen")
        for matcher in (self.forward_matcher, self.reverse_matcher):
            matcher.assert_integrity()
            if matcher.method != self.method or matcher.config_hash != self.matcher_hash:
                raise ValueError("Actual matcher and frontend provenance differ")
        if base.object_hash(dict(self.quality)) != self.config_hash:
            raise ValueError("Quality changed after freeze")
        f = dict(frame_binding)
        if set(f) != self.FRAME_FIELDS: raise ValueError("Frame inference whitelist mismatch")
        if any(f[k] != self.binding[k] for k in ("sensor_id", "session_id")): raise ValueError("Wrong sensor/session")
        if type(f["timestamp_ns"]) is not int or f["timestamp_ns"]<=self.last_timestamp: raise ValueError("Noncausal timestamp")
        if type(f["frame_id"]) is not int or f["frame_id"]<0: raise ValueError("Invalid frame identity")
        rigid, reason = self._rigid(correction, f)
        r = match_pair(self.reference, image, self.geometry, self.method, self.forward_matcher, self.reverse_matcher, self.quality)
        if rigid is not None:
            r.update(base.apply_endpoint_geometry(self.geometry.points, r["raw_uv_px"], self.geometry, rigid_transform=rigid))
            r["valid"] &= r["geometry_valid"]
            r["reason_bits"][~r["geometry_valid"]] |= np.uint16(1)
            r["confidence"] = np.where(r["valid"], r["confidence"], 0).astype(np.float32)
        tensor = base.encode_e3_tensor(r["corrected_uv_standard"], r["valid"], r["confidence"])
        fraction = float(r["valid"].mean()); q = self.quality
        status = "VALID" if fraction>=q["valid_frame_fraction"] else "INVALID" if fraction<q["invalid_frame_fraction"] else "DEGRADED"
        self.frame_count += 1; self.last_timestamp = f["timestamp_ns"]
        r.update({"tensor": tensor, "model_tensor": tensor, "reference_status": "BOUND_APPROVED_REFERENCE", "force_input_eligible": True,
            "info": {**f, "reference_sha256": self.binding["sha256"], "reference_id": self.binding["reference_id"],
                     "geometry_hash": self.geometry.hash, "geometry_units": self.geometry.units, "matcher_hash": self.matcher_hash,
                     "measurement_status": status, "contact_status": "UNKNOWN", "center": None, "center_valid": False,
                     "valid_grid_fraction": fraction, "reference_update": False, "per_frame_amplitude_normalization": False,
                     "correction_reason": reason, "drop_evaluation_row": False, "confidence_calibrated_probability": False,
                     "temporal_policy": "fixed_reference_current_pair_no_temporal_seed"}})
        return r

    def state_dict(self):
        return {**super().state_dict(), "matcher_hash": self.matcher_hash, "method": self.method}

    def load_state_dict(self, state):
        # Parent implementation calls self.state_dict, therefore includes method/config binding.
        super().load_state_dict(state)
