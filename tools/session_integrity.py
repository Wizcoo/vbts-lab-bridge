"""Byte integrity only; never captures hardware or certifies physical truth."""
import argparse
import hashlib
from pathlib import Path, PurePosixPath
import re

MANIFEST = 'checksums.sha256'


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def inventory(root):
    root = Path(root).resolve(strict=True)
    if not root.is_dir():
        raise ValueError('Session root must be a directory')
    files = {}
    for path in sorted(root.rglob('*')):
        if path.is_symlink() or (hasattr(path, 'is_junction') and path.is_junction()):
            raise ValueError('Links/junctions are not allowed in a session package')
        if not path.is_file():
            continue
        name = path.relative_to(root).as_posix()
        if name == MANIFEST:
            continue
        safe_name(name)
        files[name] = path
    return root, files


def safe_name(name):
    parts = PurePosixPath(name).parts
    if (not name or '\\' in name or ':' in name or '\n' in name or '\r' in name
            or name.startswith('/') or any(p in ('.', '..') for p in parts)
            or PurePosixPath(name).as_posix() != name or name == MANIFEST):
        raise ValueError('Unsafe/noncanonical manifest path: ' + repr(name))


def seal(root):
    root, files = inventory(root)
    if not files:
        raise ValueError('Cannot seal an empty package')
    body = ''.join(f'{digest(path)}  {name}\n' for name, path in files.items())
    with (root / MANIFEST).open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(body)
    return len(files)


def verify(root):
    root, files = inventory(root)
    manifest_path = root / MANIFEST
    if manifest_path.is_symlink():
        raise ValueError('Manifest cannot be a symlink')
    expected = {}
    for line in manifest_path.read_text(encoding='utf-8').splitlines():
        match = re.fullmatch(r'([0-9a-f]{64})  (.+)', line)
        if not match:
            raise ValueError('Malformed checksum line')
        value, name = match.groups()
        safe_name(name)
        if name in expected:
            raise ValueError('Duplicate manifest entry: ' + name)
        expected[name] = value
    if not expected or set(expected) != set(files):
        raise ValueError('Manifest coverage mismatch: missing or extra files')
    for name, value in expected.items():
        if digest(files[name]) != value:
            raise ValueError('Hash mismatch: ' + name)
    return len(expected)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['seal', 'verify'])
    parser.add_argument('session', type=Path)
    args = parser.parse_args()
    try:
        count = (seal if args.action == 'seal' else verify)(args.session)
    except (ValueError, OSError) as exc:
        parser.exit(1, f'INTEGRITY_FAIL: {exc}\n')
    print(f'BYTE_INTEGRITY_PASS: {args.action}, {count} files; physical qualification NOT_EVALUATED')


if __name__ == '__main__':
    main()
