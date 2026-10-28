"""Copy a schema-v2 session without filtering, then export portable diagnostic CSVs."""
import argparse
import csv
import json
from pathlib import Path
import shutil
import sys
import zipfile

from audit_sync_session import audit_session
from inventory_device import sha256

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from session_integrity import inventory, seal, verify

AXES = ('Fx', 'Fy', 'Fz', 'Mx', 'My', 'Mz')


def write_json(path, content):
    with path.open('x', encoding='utf-8') as stream:
        json.dump(content, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def read_csv(path):
    with path.open(encoding='utf-8-sig', newline='') as stream:
        yield from csv.DictReader(stream)


def write_csv(path, fields, rows):
    with path.open('x', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def export_session(source, output, *, identity=None, actions=None, allow_audit_fail=False, archive=False, origin='unverified'):
    source, source_files = inventory(source)
    output = Path(output).resolve()
    if output.is_relative_to(source) or source.is_relative_to(output):
        raise ValueError('Output and original session must be disjoint')
    if output.exists() or (archive and (output.with_suffix('.zip').exists() or output.with_suffix('.receipt.json').exists())):
        raise FileExistsError('Output/package already exists; use a new directory')
    report = audit_session(source)
    if not report['passed'] and not allow_audit_fail:
        raise ValueError('Source audit failed; use --allow-audit-fail only for an explicitly failed diagnostic package')
    session = json.loads((source / 'session.json').read_text(encoding='utf-8-sig'))
    if session.get('schema_version') != 2:
        raise ValueError('Only schema-v2 sessions are supported; no legacy timestamp guessing')
    output.mkdir(parents=True, exist_ok=False)
    try:
        raw = output / 'raw_source'
        raw.mkdir()
        before_hashes = {name: sha256(path) for name, path in source_files.items()}
        for name, path in source_files.items():
            target = raw / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)
        _, after_files = inventory(source)
        after_hashes = {name: sha256(path) for name, path in after_files.items()}
        copied_hashes = {name: sha256(raw / name) for name in source_files}
        if before_hashes != after_hashes or before_hashes != copied_hashes:
            raise ValueError('Source changed during export or copied bytes differ')
        # If the original was already sealed, preserve its existing manifest too.
        if (source / 'checksums.sha256').exists():
            shutil.copyfile(source / 'checksums.sha256', raw / 'checksums.sha256')
        write_json(output / 'source_inventory.json', before_hashes)
        write_json(output / 'quality_report.json', report)
        pairs = list(read_csv(raw / 'capture_sync.csv'))
        camera_fields = ['frame_id', 'image_path', 'image_sha256', 'grab_start_ns', 'grab_end_ns',
                         'host_receive_monotonic_ns', 'wall_ns', 'exposure_ns', 'reference_id', 'quality_status']
        write_csv(output / 'camera_frames.csv', camera_fields, (
            dict(frame_id=r['frame_seq'], image_path='raw_source/' + r['image_filename'], image_sha256=r['image_sha256'],
                 grab_start_ns=r['grab_start_ns'], grab_end_ns=r['grab_end_ns'], host_receive_monotonic_ns=r['receive_ns'],
                 wall_ns=r['frame_wall_ns'], exposure_ns=r['exposure_ns'], reference_id=r['reference_id'], quality_status=r['sync_status'])
            for r in pairs))
        raw_fields = ['force_sample_id', 'read_start_ns', 'read_end_ns', 'host_call_midpoint_ns', 'wall_ns',
                      'ipoc', 'ret', 'code', 'code_info', 'tare_epoch', 'duplicate_ipoc', 'valid', 'status'] + ['raw_' + a for a in AXES]
        def force_rows():
            for r in read_csv(raw / 'force_raw.csv'):
                row = {k: r[k] for k in raw_fields if k in r}
                row.update(force_sample_id=r['read_seq'], host_call_midpoint_ns=r['midpoint_ns'])
                yield row
        write_csv(output / 'force_samples_raw.csv', raw_fields, force_rows())
        transformed_fields = ['force_sample_id', 'tare_epoch', 'valid', 'status'] + ['transformed_' + a for a in AXES]
        write_csv(output / 'force_samples_transformed.csv', transformed_fields, (
            dict(force_sample_id=r['read_seq'], **{k: r[k] for k in transformed_fields if k != 'force_sample_id'})
            for r in read_csv(raw / 'force_raw.csv')))
        align_fields = ['frame_id', 'time_basis', 'method', 'target_ns', 'force_before_id', 'force_after_id',
                        'weight_after', 'bracket_gap_ns', 'valid', 'reason', 'tare_epoch']
        write_csv(output / 'alignment.csv', align_fields, (
            dict(frame_id=r['frame_seq'], time_basis='camera_grab_end_and_FT_call_midpoint_host_perf_counter_ns',
                 method='two_sided_linear_interpolation', target_ns=r['target_ns'], force_before_id=r['force_left_read_seq'],
                 force_after_id=r['force_right_read_seq'], weight_after=r['force_interp_alpha'],
                 bracket_gap_ns=r['force_bracket_gap_ns'], valid=r['label_valid'], reason=r['sync_status'], tare_epoch=r['tare_epoch'])
            for r in pairs))
        prediction_fields = ['frame_id', 'prediction_json', 'extra_json'] + ['recorded_' + a for a in AXES] + ['post_' + a for a in AXES]
        def predictions():
            for r in pairs:
                row = dict(frame_id=r['frame_seq'], prediction_json=r.get('prediction_json', ''), extra_json=r.get('extra_json', ''))
                row.update({'recorded_' + a: r[a] for a in AXES})
                row.update({'post_' + a: r.get('pred_' + a, '') for a in AXES})
                yield row
        write_csv(output / 'predictions.csv', prediction_fields, predictions())
        if identity:
            write_json(output / 'device_inventory.json', json.loads(Path(identity).read_text(encoding='utf-8-sig')))
        if actions:
            shutil.copyfile(actions, output / 'actions.csv')
        limitations = [
            'Host call times, not hardware exposure/sample timestamps; physical synchronization unverified.',
            'Latest-frame capture may skip intermediate frames. camera_frames.csv covers saved pairs, not every camera exposure.',
            'force_raw.csv logs SDK-returned values in each hardware tare epoch; pre-hardware-tare values cannot be recovered.',
            'Raw units, physical calibration, axes and torque origin require external evidence despite software N/N*m declarations.',
            'No actual SDK input tensor is available through the copied interface. No tensors are fabricated.',
            'SDK prediction_json.raw_force is an SDK field, not independently proven unprocessed ONNX graph output.',
            'No force failure, invalid alignment, reference or saved frame was filtered from raw_source.',
            'assets in device_inventory are declared files; actual loaded-module identity still needs device logs.'
        ]
        if not actions:
            limitations.append('No operator action annotations supplied; actions.csv is absent.')
        if not identity:
            limitations.append('No preflight device identity report supplied.')
        manifest = dict(schema_version=1, session_id=session['session_id'],
                        provenance_kind=origin, source_schema_version=2,
                        source_status=session.get('status'), source_audit_passed=report['passed'],
                        qualification_status='NOT_READY', source_data_policy='unfiltered byte-identical copy',
                        axis_order=session['axis_order'], declared_software_units=session['units'],
                        transformed_force_formula='signed=raw*profile_signs; Mx+=signed_Fy*arm; My-=signed_Fx*arm',
                        counts=dict(saved_pairs=len(pairs), raw_force_reads=sum(1 for _ in read_csv(raw / 'force_raw.csv'))),
                        limitations=limitations, reference_records='raw_source/reference_*.json',
                        original_events='raw_source/events.jsonl')
        write_json(output / 'session_manifest.json', manifest)
        seal(output)
        verify(output)
        if archive:
            zip_path = output.with_suffix('.zip')
            with zipfile.ZipFile(zip_path, 'x', compression=zipfile.ZIP_DEFLATED) as bundle:
                for path in sorted(output.rglob('*')):
                    if path.is_file():
                        bundle.write(path, output.name + '/' + path.relative_to(output).as_posix())
            with zipfile.ZipFile(zip_path) as bundle:
                if bundle.testzip() is not None:
                    raise ValueError('ZIP CRC check failed')
            write_json(output.with_suffix('.receipt.json'), dict(archive=zip_path.name,
                sha256=sha256(zip_path), bytes=zip_path.stat().st_size, session_id=session['session_id'],
                source_audit_passed=report['passed'], qualification_status='NOT_READY'))
        return manifest
    except Exception as exc:
        write_json(output / 'EXPORT_FAILED.json', {'error': repr(exc), 'complete': False})
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('session', type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--identity', type=Path)
    parser.add_argument('--actions', type=Path)
    parser.add_argument('--allow-audit-fail', action='store_true')
    parser.add_argument('--zip', action='store_true')
    parser.add_argument('--origin', required=True, choices=['new_capture', 'historical', 'synthetic', 'unverified'],
                        help='Operator-provided provenance, not inferred from an audit PASS')
    args = parser.parse_args(argv)
    result = export_session(args.session, args.output, identity=args.identity, actions=args.actions,
                            allow_audit_fail=args.allow_audit_fail, archive=args.zip, origin=args.origin)
    print(json.dumps({'session_id': result['session_id'], 'source_audit_passed': result['source_audit_passed'],
                      'output': str(args.output), 'qualification_status': 'NOT_READY'}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
