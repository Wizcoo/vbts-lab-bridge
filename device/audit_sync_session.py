"""Read-only schema-v2 capture audit; never infer delay or repair source labels.

Usage: python audit_sync_session.py SESSION [--report NEW_REPORT.json]
       [--manifest NEW_MANIFEST.jsonl]
Exit 0 means logged host-time pairing/integrity checks passed, not hardware sync.
Exit 1 means integrity checks failed; argparse rejects invalid CLI arguments.
No camera, DLL, model, or image decoder is loaded. Legacy schemas are rejected.
"""
from __future__ import annotations

import argparse
from bisect import bisect_right
from collections import Counter
import csv
import hashlib
import json
import math
from pathlib import Path

from sync_acquisition import AXES, ForceBuffer, ForceSample, load_force_profile


def _bool(value):
    if value in (True, "True", "true", "1", 1):
        return True
    if value in (False, "False", "false", "0", 0):
        return False
    raise ValueError(f"Invalid boolean {value!r}")


def _csv(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        return list(reader), set(reader.fieldnames or ())


def _sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stats(values):
    values = sorted(values)
    if not values:
        return {"n": 0}
    def percentile(q):
        p = q * (len(values)-1)
        i = int(p)
        return values[i] + (values[min(i+1, len(values)-1)]-values[i]) * (p-i)
    return {"n": len(values), "min": values[0], "median": percentile(.5),
            "p95": percentile(.95), "p99": percentile(.99), "max": values[-1]}


def audit_session(directory):
    root = Path(directory).resolve()
    report = {"audit_schema_version": 1, "session_path": str(root), "passed": False,
              "errors": [], "warnings": [], "hardware_synchronization_proven": False,
              "hardware_force_timestamp_available": False,
              "scope": "Integrity and exact reconstruction of logged HOST timestamps/labels only; no delay estimation or physical hardware calibration."}
    def issue(code, where, detail):
        report["errors"].append({"code": code, "where": str(where), "detail": str(detail)})
    def check(condition, code, where, detail):
        if not condition:
            issue(code, where, detail)
    def eq(a, b):
        return math.isfinite(a) and math.isfinite(b) and math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-12)
    def asset(filename, expected, where):
        path = (root / str(filename)).resolve()
        if not path.is_relative_to(root) or path.suffix.lower() != ".png":
            issue("UNSAFE_OR_NON_PNG_ASSET", where, filename)
            return
        if not path.is_file():
            issue("MISSING_IMAGE", where, filename)
            return
        with path.open("rb") as stream:
            check(stream.read(8) == b"\x89PNG\r\n\x1a\n", "INVALID_PNG_SIGNATURE", where, filename)
        check(_sha(path) == expected, "IMAGE_HASH_MISMATCH", where, filename)
    try:
        session = json.loads((root / "session.json").read_text(encoding="utf-8-sig"))
        report["session_id"] = sid = session.get("session_id")
        report["source_schema_version"] = session.get("schema_version")
        if session.get("schema_version") != 2:
            issue("UNSUPPORTED_SCHEMA", "session.json", "Requires schema_version=2; legacy timestamp fields are never guessed.")
            return report
        check(bool(sid), "MISSING_SESSION_ID", "session.json", "Empty session_id")
        report["session_status"] = session.get("status")
        check(session.get("status") == "closed", "SESSION_NOT_CLOSED", "session.json", session.get("status"))
        check(session.get("axis_order") == list(AXES), "AXIS_ORDER", "session.json", session.get("axis_order"))
        check(session.get("units") == ["N"]*3+["N*m"]*3, "UNITS", "session.json", session.get("units"))
        profile_path = root / "sensor_profile_snapshot.txt"
        check(_sha(profile_path) == session["profile_sha256"], "PROFILE_HASH", profile_path.name, "Snapshot differs from session hash")
        profile = load_force_profile(profile_path)
        raw_rows, raw_fields = _csv(root / "force_raw.csv")
        pair_rows, pair_fields = _csv(root / "capture_sync.csv")
        required_raw = {"session_id", "read_seq", "read_start_ns", "read_end_ns", "midpoint_ns", "wall_ns", "ipoc", "ret", "code", "tare_epoch", "duplicate_ipoc", "valid", "status", "code_info"}
        required_raw |= {prefix+a for prefix in ("raw_", "transformed_") for a in AXES}
        required_pair = {"session_id", "pair_idx", "image_filename", "image_sha256", "image_written", "reference_id", "frame_seq", "grab_start_ns", "grab_end_ns", "receive_ns", "exposure_ns", "target_ns", "tare_epoch", "epoch_start_ns", "sync_status", "label_valid", "force_interp_alpha", "force_bracket_gap_ns", "extra_json", *AXES}
        endpoint_keys = ("read_seq", "read_start_ns", "read_end_ns", "midpoint_ns", "ipoc", "ret", "code", "tare_epoch", "valid", "status")
        required_pair |= {f"force_{side}_{key}" for side in ("left", "right") for key in endpoint_keys}
        check(required_raw <= raw_fields, "RAW_SCHEMA_COLUMNS", "force_raw.csv", sorted(required_raw-raw_fields))
        check(required_pair <= pair_fields, "PAIR_SCHEMA_COLUMNS", "capture_sync.csv", sorted(required_pair-pair_fields))
        if not required_raw <= raw_fields or not required_pair <= pair_fields:
            return report
        events = [json.loads(line) for line in (root / "events.jsonl").read_text(encoding="utf-8-sig").splitlines() if line.strip()]
        event_counts = Counter(e.get("event") for e in events)
        report["event_counts"] = dict(event_counts)
        epoch_starts = {0: None}
        last_event = -1
        current_epoch = 0
        reference_events = {}
        accepted_codes = (10000,)
        metadata = session.get("metadata", {})
        offset = metadata.get("align_offset_ns")
        if offset is None and "alignment_offset_ms" in metadata:
            offset = round(float(metadata["alignment_offset_ms"])*1e6)
        gap = metadata.get("max_pair_gap_ns")
        for index, event in enumerate(events, 1):
            where = f"events.jsonl:{index}"
            check(event.get("session_id") == sid, "EVENT_SESSION_ID", where, event.get("session_id"))
            now = int(event["mono_ns"])
            check(now >= last_event, "EVENT_TIME_ORDER", where, now)
            last_event = now
            details = event.get("details", {})
            if event["event"] == "force_open_result":
                accepted_codes = tuple(details.get("accepted_codes", [10000]))
                if details.get("ret") == 1:
                    check(details.get("axis_signs") == list(profile["axis_signs"]), "OPEN_PROFILE_SIGNS", where, details.get("axis_signs"))
                    check(eq(float(details.get("moment_arm", float("nan"))), profile["moment_arm"]), "OPEN_PROFILE_ARM", where, details.get("moment_arm"))
            if event["event"] == "alignment_policy":
                offset, gap = details["offset_ns"], details["max_gap_ns"]
            if event["event"] == "tare_result":
                if details.get("success"):
                    current_epoch += 1
                    check(details.get("ret") == 1 and details.get("tare_epoch") == current_epoch, "TARE_EPOCH_SEQUENCE", where, details)
                    epoch_starts[current_epoch] = int(details["end_ns"])
                    check(details["request_ns"] <= details["start_ns"] <= details["end_ns"] <= now, "TARE_TIME_ORDER", where, details)
                else:
                    check(details.get("tare_epoch") == current_epoch, "FAILED_TARE_CHANGED_EPOCH", where, details)
            if event["event"] == "reference_applied":
                rid = details["reference_id"]
                check(rid not in reference_events, "DUPLICATE_REFERENCE_EVENT", where, rid)
                reference_events[rid] = details
        check(event_counts["session_start"] == 1 and event_counts["session_end"] == 1, "SESSION_BOUNDARY_EVENTS", "events.jsonl", dict(event_counts))
        check(any(e["event"] == "force_open_result" and e.get("details",{}).get("ret") == 1 for e in events), "NO_SUCCESSFUL_FORCE_OPEN", "events.jsonl", "No successful FT open with coordinate contract")
        check(event_counts["tare_request"] == event_counts["tare_result"], "TARE_EVENT_PAIR_COUNT", "events.jsonl", "tare request/result count differs")
        report["tare_epochs"] = epoch_starts
        raw = {}
        buffers = {}
        previous_seq, previous_mid, previous_start, last_ipoc = 0, -1, -1, None
        raw_status = Counter()
        for index, row in enumerate(raw_rows, 2):
            where = f"force_raw.csv:{index}"
            try:
                seq, start, end, mid, epoch = (int(row[k]) for k in ("read_seq", "read_start_ns", "read_end_ns", "midpoint_ns", "tare_epoch"))
                check(row["session_id"] == sid, "RAW_SESSION_ID", where, row["session_id"])
                check(seq == previous_seq+1 and seq not in raw, "RAW_SEQUENCE", where, seq)
                check(previous_start <= start <= mid <= end and mid > previous_mid and mid == (start+end)//2, "RAW_TIME_ORDER", where, (start, mid, end))
                previous_seq, previous_mid, previous_start = seq, mid, start
                check(epoch in epoch_starts, "RAW_UNKNOWN_TARE_EPOCH", where, epoch)
                valid, duplicate = _bool(row["valid"]), _bool(row["duplicate_ipoc"])
                ret, code, ipoc = (int(row[k]) for k in ("ret", "code", "ipoc"))
                expected_duplicate = ret == 1 and last_ipoc is not None and ipoc == last_ipoc
                check(duplicate == expected_duplicate, "RAW_DUPLICATE_IPOC_FLAG", where, ipoc)
                if ret == 1:
                    last_ipoc = ipoc
                values = tuple(float(row["raw_"+a]) for a in AXES)
                finite = all(math.isfinite(v) for v in values)
                transformed = tuple(float(row["transformed_"+a]) for a in AXES) if all(row["transformed_"+a] != "" for a in AXES) else None
                check(valid == (ret == 1 and code in accepted_codes and finite and not duplicate), "RAW_VALIDITY", where, row["status"])
                check(valid == (row["status"] == "OK"), "RAW_STATUS_VALIDITY", where, row["status"])
                if ret == 1 and finite:
                    expected = [a*b for a,b in zip(values, profile["axis_signs"])]
                    expected[3] += expected[1]*profile["moment_arm"]
                    expected[4] -= expected[0]*profile["moment_arm"]
                    check(transformed is not None and all(eq(a,b) for a,b in zip(transformed, expected)), "RAW_TRANSFORM", where, "Raw-to-physical coordinate transform mismatch")
                else:
                    check(all(row["transformed_"+a] == "" for a in AXES), "INVALID_RAW_TRANSFORM_FILLED", where, "Failed/nonfinite raw transform must be empty")
                known = [e for e,t in epoch_starts.items() if t is None or t <= start]
                check(epoch == max(known), "RAW_TARE_TIME_EPOCH", where, epoch)
                sample = ForceSample(seq,start,end,mid,int(row["wall_ns"]),ipoc,ret,code,row["code_info"],values,transformed,epoch,duplicate,valid,row["status"])
                raw[seq] = sample
                raw_status[row["status"]] += 1
                if epoch not in buffers:
                    buffers[epoch] = ForceBuffer(maxlen=len(raw_rows)+1)
                    if epoch_starts.get(epoch) is not None:
                        buffers[epoch].reset(epoch, epoch_starts[epoch])
                buffers[epoch].add(sample)
            except (ValueError, TypeError, KeyError) as exc:
                issue("RAW_ROW_PARSE_OR_BUFFER", where, exc)

        indexed = {epoch: (tuple(buffer.samples), tuple(s.midpoint_ns for s in buffer.samples))
                   for epoch, buffer in buffers.items()}

        def validate_match(record, where, epoch, target, valid, status, left, right, alpha, label, max_gap):
            check(epoch in epoch_starts, "UNKNOWN_TARE_EPOCH", where, epoch)
            check(valid == (status == "OK"), "PAIR_STATUS_VALIDITY", where, status)
            if not valid:
                check(all(v in (None, "") for v in label) and alpha in (None, ""), "INVALID_LABEL_FILLED", where, "Invalid labels and interpolation alpha must be empty, never fake zero")
                return
            check(left is not None and right is not None, "MISSING_BRACKET", where, "Valid label requires both raw read_seq records")
            if left is None or right is None:
                return
            check(left.tare_epoch == right.tare_epoch == epoch and left.valid and right.valid and left.ipoc != right.ipoc, "BRACKET_EPOCH_OR_VALIDITY", where, (left.read_seq, right.read_seq))
            if max_gap is None:
                issue("MISSING_GAP_POLICY", where, "No recorded max gap; refusing to guess")
                return
            samples, times = indexed[epoch]
            i = bisect_right(times, target)
            # Reuse the shared core with adjacent records: O(log n) per frame.
            local = ForceBuffer(maxlen=2)
            if epoch_starts.get(epoch) is not None:
                local.reset(epoch, epoch_starts[epoch])
            for sample in samples[max(0,i-1):min(len(samples),i+1)]:
                local.add(sample)
            match = local.match(target, int(max_gap), expected_tare_epoch=epoch)
            check(match.valid, "BRACKET_RECONSTRUCTION", where, match.status)
            if match.valid:
                check((match.left.read_seq, match.right.read_seq) == (left.read_seq, right.read_seq), "NONADJACENT_BRACKET", where, "Recorded endpoints do not match full raw log bracketing")
                check(eq(float(alpha), match.alpha), "ALPHA_MISMATCH", where, (alpha, match.alpha))
                check(len(label) == 6 and all(eq(float(a),b) for a,b in zip(label,match.label)), "LABEL_MISMATCH", where, "Labels differ from exact raw interpolation")

        references = {}
        for path in sorted(root.glob("reference_*.json")):
            try:
                ref = json.loads(path.read_text(encoding="utf-8-sig"))
                rid = ref["reference_id"]
                check(rid not in references and path.stem == rid, "REFERENCE_ID", path.name, rid)
                references[rid] = ref
                check(ref == reference_events.get(rid), "REFERENCE_EVENT_MISMATCH", path.name, rid)
                check(ref["profile_sha256"] == session["profile_sha256"], "REFERENCE_PROFILE_HASH", path.name, rid)
                asset(ref["image_filename"], ref["image_sha256"], path.name)
                epoch = int(ref["tare_epoch"])
                match, packet = ref["force_match"], ref["packet"]
                check(epoch == int(match["tare_epoch"]), "REFERENCE_EPOCH", path.name, epoch)
                check(match.get("epoch_start_ns") == epoch_starts.get(epoch), "REFERENCE_EPOCH_START", path.name, epoch)
                check(packet["grab_start_ns"] <= packet["grab_end_ns"] <= packet["receive_ns"], "REFERENCE_TIME_ORDER", path.name, packet)
                if epoch_starts.get(epoch) is not None:
                    check(packet["grab_start_ns"] > epoch_starts[epoch], "REFERENCE_BEFORE_TARE", path.name, epoch)
                ends = []
                for side in ("left", "right"):
                    stored = match.get(side)
                    endpoint = raw.get(int(stored["read_seq"])) if stored else None
                    check(endpoint is not None and stored == endpoint.as_dict(), "REFERENCE_RAW_ENDPOINT", path.name, side)
                    ends.append(endpoint)
                validate_match(match,path.name,epoch,int(match["target_ns"]),_bool(match["valid"]),match["sync_status"],*ends,match["force_interp_alpha"],match.get("label") or [None]*6,gap)
            except (ValueError, TypeError, KeyError) as exc:
                issue("REFERENCE_PARSE", path.name, exc)
        check(set(references) == set(reference_events), "REFERENCE_FILES_EVENTS", "references", "Reference files and applied events differ")
        seen_frames, seen_images = set(), set()
        previous_frame, previous_grab, previous_receive, previous_target = -1, -1, -1, -1
        statuses, gaps, valid_count, exposure_count, missing_reference = Counter(), [], 0, 0, 0
        for index, row in enumerate(pair_rows, 2):
            where = f"capture_sync.csv:{index}"
            try:
                check(row["session_id"] == sid, "PAIR_SESSION_ID", where, row["session_id"])
                check(int(row["pair_idx"]) == index-1, "PAIR_SEQUENCE", where, row["pair_idx"])
                frame, start, end, receive, target, epoch = (int(row[k]) for k in ("frame_seq","grab_start_ns","grab_end_ns","receive_ns","target_ns","tare_epoch"))
                check(frame not in seen_frames and frame > previous_frame, "DUPLICATE_OR_UNORDERED_FRAME", where, frame)
                check(row["image_filename"] not in seen_images, "DUPLICATE_IMAGE_NAME", where, row["image_filename"])
                seen_frames.add(frame); seen_images.add(row["image_filename"])
                check(previous_grab < end and previous_receive < receive and previous_target < target and start <= end <= receive, "FRAME_TIME_ORDER", where, (start,end,receive,target))
                previous_frame, previous_grab, previous_receive, previous_target = frame,end,receive,target
                check(_bool(row["image_written"]), "IMAGE_NOT_WRITTEN", where, row["image_written"])
                asset(row["image_filename"],row["image_sha256"],where)
                valid = _bool(row["label_valid"])
                statuses[row["sync_status"]] += 1
                valid_count += int(valid)
                exposure_count += int(bool(row["exposure_ns"]))
                rid = row["reference_id"]
                if rid:
                    check(rid in references, "MISSING_REFERENCE", where, rid)
                    if rid in references:
                        check(int(references[rid]["tare_epoch"]) == epoch, "PAIR_REFERENCE_EPOCH", where, rid)
                        check(int(references[rid]["packet"]["frame_seq"]) <= frame, "FUTURE_REFERENCE", where, rid)
                else:
                    missing_reference += 1
                expected_start = epoch_starts.get(epoch)
                check(row["epoch_start_ns"] == (str(expected_start) if expected_start is not None else ""), "PAIR_EPOCH_START", where, row["epoch_start_ns"])
                extra = json.loads(row["extra_json"] or "{}")
                local_offset = extra.get("align_offset_ns", extra.get("alignment_offset_ns", offset))
                local_gap = extra.get("max_pair_gap_ns", gap)
                if local_offset is None:
                    issue("MISSING_OFFSET_POLICY", where, "No recorded offset policy; refusing to guess")
                else:
                    check(target == end-int(local_offset), "TARGET_OFFSET", where, (target,end,local_offset))
                ends = []
                for side in ("left","right"):
                    key = f"force_{side}_read_seq"
                    endpoint = raw.get(int(row[key])) if row[key] else None
                    if row[key]:
                        check(endpoint is not None, "RAW_ENDPOINT_NOT_FOUND", where, row[key])
                    if endpoint is not None:
                        for field in endpoint_keys:
                            check(row[f"force_{side}_{field}"] == str(getattr(endpoint,field)), "RAW_ENDPOINT_METADATA", where, f"{side}.{field}")
                    ends.append(endpoint)
                if all(ends):
                    measured_gap = ends[1].midpoint_ns-ends[0].midpoint_ns
                    check(int(row["force_bracket_gap_ns"]) == measured_gap, "GAP_MISMATCH", where, row["force_bracket_gap_ns"])
                    gaps.append(measured_gap/1e6)
                validate_match(row,where,epoch,target,valid,row["sync_status"],*ends,row["force_interp_alpha"],[row[a] for a in AXES],local_gap)
            except (ValueError, TypeError, KeyError) as exc:
                issue("PAIR_ROW_PARSE", where, exc)
        check(session.get("recorded_pairs") == len(pair_rows), "SESSION_PAIR_COUNT", "session.json", len(pair_rows))
        check(session.get("references") == len(references), "SESSION_REFERENCE_COUNT", "session.json", len(references))
        report["counts"] = {"raw_reads":len(raw_rows),"pairs":len(pair_rows),"references":len(references),"valid_labels":valid_count,"invalid_labels":len(pair_rows)-valid_count,"pairs_without_reference":missing_reference,"duplicate_ipoc_raw_reads":sum(_bool(r["duplicate_ipoc"]) for r in raw_rows),"unique_frame_sequences":len(seen_frames),"duplicate_frame_rows":len(pair_rows)-len(seen_frames),"repeated_image_hash_rows":len(pair_rows)-len({r["image_sha256"] for r in pair_rows})}
        report["sync_success_rate"] = valid_count/len(pair_rows) if pair_rows else None
        report["sync_status_counts"] = dict(statuses)
        report["raw_status_counts"] = dict(raw_status)
        report["bracket_gap_ms"] = _stats(gaps)
        report["camera_exposure_timestamp_rows"] = exposure_count
        report["training_compatibility"] = "Schema-v2 sessions cannot be passed directly to the historical dataset_manifest.py. Optional audited manifest is a new adapter interface; split remains unassigned and must be assigned by whole session."
        report["warnings"].append("Force hardware sample timestamps are unavailable in schema v2. Host interpolation success does not prove exposure/ADC synchronization.")
        if exposure_count == 0:
            report["warnings"].append("Camera exposure timestamps are unavailable; grab completion/receive timestamps are host timestamps.")
        if missing_reference:
            report["warnings"].append(f"{missing_reference} rows have no active reference and are not eligible for reference-based model training/comparison.")
        report["error_counts"] = dict(Counter(e["code"] for e in report["errors"]))
        report["source_hashes"] = {name:_sha(root/name) for name in ("session.json","force_raw.csv","capture_sync.csv","events.jsonl","sensor_profile_snapshot.txt")}
        report["warnings"].append("Repeated PNG content hashes can be legitimate static frames; frame_seq reuse is an error. No physical unloaded-reference condition is established by file integrity alone.")
        report["passed"] = not report["errors"]
        return report
    except (OSError, ValueError, KeyError, TypeError) as exc:
        issue("SESSION_READ_OR_SCHEMA", root, exc)
        return report


def _training_records(directory):
    """Call only after a successful audit; exclude invalid and reference-less rows."""
    root = Path(directory).resolve()
    session = json.loads((root / "session.json").read_text(encoding="utf-8-sig"))
    rows,_ = _csv(root / "capture_sync.csv")
    refs = {p.stem:json.loads(p.read_text(encoding="utf-8-sig")) for p in root.glob("reference_*.json")}
    for row in rows:
        if not _bool(row["label_valid"]) or row["reference_id"] not in refs:
            continue
        ref = refs[row["reference_id"]]
        yield {"manifest_schema":"audited_sync_v2_training_adapter_v1", "source_schema_version":2,
               "session_id":session["session_id"], "group_id":session["session_id"], "split":"unassigned",
               "image_path":str((root/row["image_filename"]).resolve()),
               "reference_path":str((root/ref["image_filename"]).resolve()),
               "image_sha256":row["image_sha256"], "reference_sha256":ref["image_sha256"],
               "profile_sha256":session["profile_sha256"], "reference_id":row["reference_id"],
               "frame_seq":int(row["frame_seq"]), "tare_epoch":int(row["tare_epoch"]),
               "target_ns":int(row["target_ns"]), "label":[float(row[a]) for a in AXES],
               "axis_order":list(AXES), "units":session["units"], "label_mask":[1]*6,
               "label_convention":"physical signed/arm-transformed force in recorded tare epoch; no reference-label subtraction",
               "hardware_synchronization_proven":False}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("session", type=Path)
    parser.add_argument("--report", type=Path, help="Optional new JSON path outside the source session; never overwritten")
    parser.add_argument("--manifest", type=Path, help="Optional new audited training-adapter JSONL; excludes invalid/reference-less frames; not the old trainer's schema")
    args = parser.parse_args(argv)
    outputs = [p.resolve() for p in (args.report,args.manifest) if p is not None]
    if len(set(outputs)) != len(outputs):
        parser.error("report and manifest must be distinct new files")
    for path in outputs:
        if path.is_relative_to(args.session.resolve()):
            parser.error("Outputs must be outside the source session; audit never writes into original data")
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite {path}")
    report = audit_session(args.session)
    if args.manifest and report["passed"]:
        records = list(_training_records(args.session))
        with args.manifest.resolve().open("x", encoding="utf-8", newline="") as stream:
            for record in records:
                stream.write(json.dumps(record,ensure_ascii=False,allow_nan=False)+"\n")
        report["training_manifest"] = {"path":str(args.manifest.resolve()),"rows":len(records),"split":"unassigned"}
    elif args.manifest:
        report["warnings"].append("Training manifest was NOT written because the session audit failed.")
    content = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)+"\n"
    if args.report:
        path = args.report.resolve()
        with path.open("x", encoding="utf-8", newline="") as stream:
            stream.write(content)
    print(content, end="")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
