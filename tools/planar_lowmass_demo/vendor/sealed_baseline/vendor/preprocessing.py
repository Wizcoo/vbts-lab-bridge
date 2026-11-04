"""Offline, source-aligned flat-gripper input extraction. No hardware APIs.

This is NOT a claim of binary parity with the deployed OpenCV 4.4 SDK. The
installed Python OpenCV runs DIS; the actual production C++ helper functions
are compiled unchanged against a minimal array/Point2f adapter. Every cache
must record provenance(), including OpenCV/build/source hashes. Full binary
parity remains a release blocker until a production tensor dump is compared.
"""
from __future__ import annotations

import ctypes
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE
DEFAULT_PROFILE = HERE / "profile.txt"
TRACKER_SOURCE = HERE / "cpp_sources/VT_Tracker.cpp"
ENCODER_SOURCE = HERE / "cpp_sources/ForceInputEncoder.cpp"
PIPELINE_SOURCE = HERE / "cpp_sources/SensorPipeline.cpp"
VERSION = "source-aligned-b-channel-native-helpers-v1"


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_contract(path: str | Path = DEFAULT_PROFILE) -> dict:
    values = {}
    for line in Path(path).read_text(encoding="utf-8-sig").splitlines():
        line = line.split("#", 1)[0].strip()
        if "=" in line:
            key, value = line.split("=", 1)
            values[key.strip()] = value.strip()
    required = {"IMAGE_SIZE": "640x480", "GRID_SIZE": "82x69", "GRID_STEP": "5",
                "CHANNELS": "Dx,Dy", "INPUT_SHAPE": "1x2x112x112",
                "INPUT_TYPE": "float32", "LETTERBOX_PAD_VALUE": "128"}
    for key, expected in required.items():
        if values.get(key) != expected:
            raise ValueError(f"Unsupported production contract {key}={values.get(key)!r}; expected {expected!r}")
    parse = lambda key: [float(v) for v in values[key].split(",")]
    for key in ("MEAN", "STD", "FEATURE_DX_RANGE", "FEATURE_DY_RANGE"):
        nums=parse(key)
        if len(nums)!=2 or not np.isfinite(nums).all():
            raise ValueError(f"Invalid finite two-channel profile field: {key}")
    if any(v<=0 for v in parse("STD")):
        raise ValueError("STD must be strictly positive")
    if any(parse(key)[0]>=parse(key)[1] for key in ("FEATURE_DX_RANGE", "FEATURE_DY_RANGE")):
        raise ValueError("Displacement ranges must be strictly increasing")
    if len(parse("ROI_POINTS"))!=8 or not np.isfinite(parse("ROI_POINTS")).all():
        raise ValueError("ROI_POINTS must contain four finite xy pairs")
    return {"image_size": [640, 480], "grid_size": [82, 69], "grid_step": 5,
            "roi": np.asarray(parse("ROI_POINTS"), np.float32).reshape(4, 2).tolist(),
            "dx_range": parse("FEATURE_DX_RANGE"), "dy_range": parse("FEATURE_DY_RANGE"),
            "mean": parse("MEAN"), "std": parse("STD"), "pad": 128,
            "shape": [2, 112, 112], "image_channel": "B",
            "profile_sha256": sha256_file(path)}


_CPP_PREFIX = r'''
#include <algorithm>
#include <cmath>
#include <cstring>
#include <vector>
namespace cv {
struct Point2f {
    float x, y;
    Point2f(float x_=0, float y_=0): x(x_), y(y_) {}
    Point2f operator+(const Point2f& b) const { return {x+b.x,y+b.y}; }
    Point2f operator-(const Point2f& b) const { return {x-b.x,y-b.y}; }
    Point2f operator*(float v) const { return {x*v,y*v}; }
    Point2f& operator+=(const Point2f& b) {x+=b.x;y+=b.y;return *this;}
    Point2f& operator-=(const Point2f& b) {x-=b.x;y-=b.y;return *this;}
    Point2f& operator*=(float v) {x*=v;y*=v;return *this;}
};
struct Mat {
    int rows, cols; const float* data;
    template<class T> const T* ptr(int row) const {
        return reinterpret_cast<const T*>(data + row*cols*2);
    }
};
}
'''

_CPP_SUFFIX = r'''
#define API extern "C" __declspec(dllexport)
API void sample_flow(const float* flow, int h, int w, const float* points,
                     int n, float* result) {
    cv::Mat mat{h,w,flow};
    for(int i=0;i<n;i++) {
        bool ok=false;
        const auto p=vt_sampleFlow(mat,points[2*i],points[2*i+1],ok);
        result[2*i]=p.x;result[2*i+1]=p.y;
    }
}
API void encode_bytes(const float* values, int n, float lo, float hi,
                      unsigned char* result) {
    for(int i=0;i<n;i++) {
        const float v=(values[i]-lo)/(hi-lo)*255.0f;
        const long q=std::lrintf(v);
        result[i]=static_cast<unsigned char>(std::max(0L,std::min(255L,q)));
    }
}
API void postprocess(const float* flow, int h, int w, const float* points,
                     int rows, int cols, double meanAbsDiff,
                     float* result, float* diagnostics) {
    const int n=rows*cols;
    cv::Mat mat{h,w,flow};
    const auto* ref=reinterpret_cast<const cv::Point2f*>(points);
    std::vector<cv::Point2f> disp(n), tmp(n);
    std::vector<unsigned char> valid(n,0);
    for(int i=0;i<n;i++) {
        const auto base=vt_clampPt(ref[i],w,h);
        bool ok=false;
        const auto d=vt_sampleFlow(mat,base.x,base.y,ok);
        if(!ok || !vt_isFiniteVec(d) || vt_norm2Sq(d)>6400.0f) continue;
        const auto next=base+d;
        if(next.x<0 || next.y<0 || next.x>w-1 || next.y>h-1) continue;
        disp[i]=d;valid[i]=1;
    }
    vt_repairBorderLayer(disp.data(),valid.data(),rows,cols,0);
    if(rows>=8 && cols>=8) {
        vt_overwriteCornerByRef(disp.data(),valid.data(),rows,cols,0,1,0,1,2,4,2,4);
        vt_overwriteCornerByRef(disp.data(),valid.data(),rows,cols,0,1,cols-2,cols-1,2,4,cols-5,cols-3);
        vt_overwriteCornerByRef(disp.data(),valid.data(),rows,cols,rows-2,rows-1,0,1,rows-5,rows-3,2,4);
        vt_overwriteCornerByRef(disp.data(),valid.data(),rows,cols,rows-2,rows-1,cols-2,cols-1,rows-5,rows-3,cols-5,cols-3);
    }
    cv::Point2f bias; float coherence=0,meanMag=0;bool removed=false;
    const bool have=vt_estimateCenterBias(disp.data(),valid.data(),rows,cols,bias,coherence,meanMag);
    if(have) {
        const float mag=std::sqrt(vt_norm2Sq(bias));
        const float limit=meanAbsDiff<1.8?30.0f:1.20f;
        removed=coherence>.72f && mag>.03f && mag<limit && meanAbsDiff<8.0;
        if(removed) for(int i=0;i<n;i++) if(valid[i] && vt_isFiniteVec(disp[i])) disp[i]-=bias;
        if(meanAbsDiff<1.8) for(int i=0;i<n;i++)
            if(valid[i] && vt_isFiniteVec(disp[i]) && vt_norm2Sq(disp[i])<.10f*.10f) disp[i]={0,0};
    }
    vt_refineDirectionField(disp.data(),valid.data(),tmp.data(),rows,cols);
    auto* out=reinterpret_cast<cv::Point2f*>(result);
    int validCount=0;
    for(int i=0;i<n;i++) {
        cv::Point2f d;
        if(valid[i] && vt_isFiniteVec(tmp[i])) {d=tmp[i];++validCount;}
        out[i]=vt_clampPt(vt_clampPt(ref[i],w,h)+d,w,h)-ref[i];
    }
    diagnostics[0]=float(validCount)/n;diagnostics[1]=bias.x;
    diagnostics[2]=bias.y;diagnostics[3]=float(removed);
    diagnostics[4]=coherence;diagnostics[5]=meanMag;
}
'''


def _native_code() -> str:
    source = TRACKER_SOURCE.read_text(encoding="utf-8-sig")
    begin = source.index("static inline float vt_norm2Sq")
    end = source.index("struct VT_Tracker::FastTrackWS")
    return _CPP_PREFIX + source[begin:end] + _CPP_SUFFIX


_LIB = None
_NATIVE_HASH = None
_DLL_DIRECTORY_HANDLES = []


def ensure_native():
    """Compile offline helpers in this experiment directory, never the repo."""
    global _LIB, _NATIVE_HASH
    if _LIB is not None:
        return _LIB
    code = _native_code()
    _NATIVE_HASH = hashlib.sha256(code.encode()).hexdigest()
    directory = HERE / "native"
    directory.mkdir(exist_ok=True)
    source = directory / f"tracker_helpers_{_NATIVE_HASH[:16]}.cpp"
    library = source.with_suffix(".dll")
    if not library.exists():
        source.write_text(code, encoding="utf-8")
        compiler = shutil.which("g++") or r"C:\msys64\ucrt64\bin\g++.exe"
        temporary = library.with_suffix(f".{os.getpid()}.tmp.dll")
        command = [compiler, "-std=c++17", "-O2", "-shared", "-ffp-contract=off",
                   "-static-libgcc", "-static-libstdc++", str(source), "-o", str(temporary)]
        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode:
            raise RuntimeError(f"Offline native helper build failed:\n{result.stderr}")
        try:
            os.replace(temporary, library)
        except PermissionError:
            if not library.exists():
                raise
            temporary.unlink(missing_ok=True)
    compiler_directory = Path(shutil.which("g++") or r"C:\msys64\ucrt64\bin\g++.exe").parent
    if hasattr(os, "add_dll_directory"):
        _DLL_DIRECTORY_HANDLES.append(os.add_dll_directory(str(compiler_directory)))
    lib = ctypes.CDLL(str(library))
    fp = np.ctypeslib.ndpointer(dtype=np.float32, flags="C_CONTIGUOUS")
    bp = np.ctypeslib.ndpointer(dtype=np.uint8, flags="C_CONTIGUOUS")
    lib.postprocess.argtypes = [fp, ctypes.c_int, ctypes.c_int, fp, ctypes.c_int,
                               ctypes.c_int, ctypes.c_double, fp, fp]
    lib.postprocess.restype = None
    lib.sample_flow.argtypes = [fp, ctypes.c_int, ctypes.c_int, fp, ctypes.c_int, fp]
    lib.sample_flow.restype = None
    lib.encode_bytes.argtypes = [fp, ctypes.c_int, ctypes.c_float, ctypes.c_float, bp]
    lib.encode_bytes.restype = None
    _LIB = lib
    return lib


def provenance(profile_path: str | Path = DEFAULT_PROFILE) -> dict:
    ensure_native()
    contract = read_contract(profile_path)
    info = {"implementation": VERSION, "contract": contract,
            "opencv_version": cv2.__version__, "numpy_version": np.__version__,
            "tracker_cpp_sha256": sha256_file(TRACKER_SOURCE),
            "encoder_cpp_sha256": sha256_file(ENCODER_SOURCE),
            "pipeline_cpp_sha256": sha256_file(PIPELINE_SOURCE),
            "native_helper_sha256": _NATIVE_HASH,
            "preprocessing_sha256": sha256_file(__file__),
            "binary_production_parity": "NOT_VERIFIED",
            "binary_parity_limitations": [
                "Production OpenCV 4.4 SDK C API and tensor dumper are unavailable.",
                "Installed Python OpenCV DIS and resize may differ from OpenCV 4.4.",
                "First DIS flow is deterministically zero-initialized; source C++ allocates an uninitialized buffer.",
                "Point helper source is reused, but compiler and float evaluation may differ from the deployed MSVC binary."],
            "ordering": "stateful DIS; process every original frame sequentially within reference, including excluded frames"}
    info["contract_hash"] = hashlib.sha256(json.dumps(info, sort_keys=True).encode()).hexdigest()
    return info


def blue_channel(image: np.ndarray) -> np.ndarray:
    if image is None or image.dtype != np.uint8:
        raise ValueError("Expected decoded uint8 image")
    if image.shape == (480, 640, 3):
        return np.ascontiguousarray(image[:, :, 0])
    if image.shape == (480, 640):
        return np.ascontiguousarray(image)
    raise ValueError(f"Expected 640x480 BGR or gray image, got {image.shape}")


def read_image(path: str | Path) -> np.ndarray:
    image = cv2.imdecode(np.fromfile(path, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Cannot decode image: {path}")
    return image


def encode_displacements(displacement: np.ndarray, contract: dict | None = None,
                         *, return_bytes: bool = False):
    c = contract or read_contract()
    displacement = np.asarray(displacement, np.float32).reshape(69, 82, 2)
    if not np.isfinite(displacement).all():
        raise ValueError("Non-finite displacement cannot be encoded")
    encoded = np.empty((2, 112, 112), np.uint8)
    native = ensure_native()
    scale = min(np.float32(112) / np.float32(82), np.float32(112) / np.float32(69))
    rw, rh = int(np.float32(82) * scale), int(np.float32(69) * scale)
    for axis, limits in enumerate([c["dx_range"], c["dy_range"]]):
        channel = np.ascontiguousarray(displacement[:, :, axis])
        raw = np.empty(channel.shape, np.uint8)
        native.encode_bytes(channel, channel.size, limits[0], limits[1], raw)
        resized = cv2.resize(raw, (rw, rh), interpolation=cv2.INTER_LINEAR)
        pw, ph = 112-rw, 112-rh
        encoded[axis] = cv2.copyMakeBorder(resized, ph//2, ph-ph//2, pw//2, pw-pw//2,
                                         cv2.BORDER_CONSTANT, value=c["pad"])
    tensor = encoded.astype(np.float32) * np.float32(1.0 / 255.0)
    tensor -= np.asarray(c["mean"], np.float32)[:, None, None]
    tensor /= np.asarray(c["std"], np.float32)[:, None, None]
    return (tensor, encoded) if return_bytes else tensor


class ProductionPreprocessor:
    """Source-aligned research extractor; binary SDK parity is NOT_VERIFIED.

    One instance per reference sequence. Never shard a sequence across workers.
    process() returns tensor (2,112,112 float32), diagnostic dictionary.
    """
    def __init__(self, reference_bgr: np.ndarray, profile_path: str | Path = DEFAULT_PROFILE):
        self.contract = read_contract(profile_path)
        self.native = ensure_native()
        self.reference = blue_channel(reference_bgr).copy()
        rectangle = np.asarray([[156,69],[561,69],[561,411],[156,411]], np.float64)
        roi = np.asarray(self.contract["roi"], np.float32)
        matrix, _ = cv2.findHomography(rectangle, roi, cv2.RANSAC, 3.0)
        if matrix is None:
            raise RuntimeError("Production grid homography failed")
        xs, ys = np.meshgrid(np.arange(156,562,5), np.arange(69,412,5))
        x, y = xs.ravel().astype(np.float64), ys.ravel().astype(np.float64)
        tx = matrix[0,0]*x+matrix[0,1]*y+matrix[0,2]
        ty = matrix[1,0]*x+matrix[1,1]*y+matrix[1,2]
        tw = matrix[2,0]*x+matrix[2,1]*y+matrix[2,2]
        inv = np.divide(1.0, tw, out=np.zeros_like(tw), where=np.abs(tw)>=1e-12)
        self.points = np.ascontiguousarray(np.stack([tx*inv, ty*inv], axis=1), np.float32)
        self.dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_ULTRAFAST)
        self.dis.setFinestScale(2)
        self.dis.setPatchSize(8)
        self.dis.setPatchStride(4)
        self.dis.setGradientDescentIterations(10)
        self.dis.setVariationalRefinementIterations(0)
        self.dis.setUseSpatialPropagation(True)
        self.dis.setUseMeanNormalization(True)
        self.flow = None
        self.frame_count = 0
        self.last_displacement = None
        self.last_encoded_uint8 = None

    def process(self, frame_bgr: np.ndarray):
        current = blue_channel(frame_bgr)
        diff = cv2.absdiff(self.reference, current)
        mean_diff = float(cv2.sumElems(diff)[0] / diff.size)
        max_diff = float(cv2.minMaxLoc(diff)[1])
        static = mean_diff < .80 or (mean_diff < 1.80 and max_diff < 18.0)
        diagnostic = np.zeros(6, np.float32)
        displacement = np.zeros((69,82,2), np.float32)
        if not static:
            if self.flow is None:
                self.flow = np.zeros((480,640,2), np.float32)
            self.flow = self.dis.calc(self.reference, current, self.flow)
            cv2.blur(self.flow, (3,3), dst=self.flow, borderType=cv2.BORDER_REPLICATE)
            self.native.postprocess(self.flow, 480, 640, self.points, 69, 82,
                                    mean_diff, displacement, diagnostic)
        else:
            diagnostic[0] = 1.0
        self.frame_count += 1
        self.last_displacement = displacement
        tensor, encoded = encode_displacements(displacement, self.contract, return_bytes=True)
        self.last_encoded_uint8 = encoded
        info = {"frame_number": self.frame_count, "mean_abs_diff": mean_diff,
                "max_abs_diff": max_diff, "static_skipped": static,
                "valid_ratio": float(diagnostic[0]), "bias_x": float(diagnostic[1]),
                "bias_y": float(diagnostic[2]), "bias_removed": bool(diagnostic[3]),
                "bias_coherence": float(diagnostic[4]), "mean_motion": float(diagnostic[5]),
                "byte_saturation_fraction": float(np.mean((encoded==0)|(encoded==255)))}
        return tensor, info

    def close(self):
        self.dis = None
        self.flow = None

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
