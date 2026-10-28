"""Read explicit asset paths and environment metadata. Never opens hardware."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import struct
import sys
import time


def sha256(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def inventory(profile, ft_dll, sdk_dll=None, model=None, assets=()):
    named = [('profile', profile), ('ft_dll', ft_dll)]
    named += [('sdk_dll', sdk_dll), ('model', model)]
    named += [('declared_asset', p) for p in assets]
    named += [('bridge_source', p) for p in sorted(Path(__file__).parent.glob('*.py'))]
    rows = []
    for kind, raw in named:
        if raw is None:
            continue
        path = Path(raw).resolve()
        exists = path.is_file()
        rows.append(dict(kind=kind, path=str(path), exists=exists,
                         bytes=path.stat().st_size if exists else None,
                         sha256=sha256(path) if exists else None))
    packages = {}
    for name in ('numpy', 'opencv-python', 'onnxruntime'):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    mono_before = time.perf_counter_ns()
    utc_ns = time.time_ns()
    mono_after = time.perf_counter_ns()
    return dict(schema_version=1, purpose='preflight_inventory', hardware_opened=False,
                asset_identity_kind='declared paths, NOT proof of actually loaded modules',
                status='FILES_PRESENT' if all(r['exists'] for r in rows) else 'MISSING_FILES',
                platform=platform.platform(), python=sys.version, pointer_bits=struct.calcsize('P') * 8,
                packages=packages, clock=dict(api='time.perf_counter_ns', unit='ns',
                    monotonic_before_ns=mono_before, utc_ns=utc_ns, monotonic_after_ns=mono_after,
                    hardware_sync_proven=False), assets=rows,
                physical_calibration='UNKNOWN', device_identity='CONFIRM_ON_LAB_PC',
                sdk_abi_verified=False, qualification_status='NOT_READY')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile', required=True, type=Path)
    parser.add_argument('--ft-dll', required=True, type=Path)
    parser.add_argument('--sdk-dll', type=Path)
    parser.add_argument('--model', type=Path)
    parser.add_argument('--asset', type=Path, action='append', default=[])
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args(argv)
    report = inventory(args.profile, args.ft_dll, args.sdk_dll, args.model, args.asset)
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps({'report': str(args.output), 'status': report['status'], 'hardware_opened': False}))
    return 0 if report['status'] == 'FILES_PRESENT' else 1


if __name__ == '__main__':
    raise SystemExit(main())
