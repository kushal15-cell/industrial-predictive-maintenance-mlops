"""Restore the exact DVC snapshot mirrored in a public release, without credentials."""
import hashlib
import json
from pathlib import Path
import tempfile
import urllib.request
import zipfile


def restore(root, archive=None):
    root = Path(root)
    manifest = json.loads((root / 'release/bundle.json').read_text(encoding='utf-8'))
    with tempfile.TemporaryDirectory() as directory:
        bundle = Path(archive) if archive else Path(directory) / 'artifacts.zip'
        if archive is None:
            with urllib.request.urlopen(manifest['url'], timeout=120) as response, bundle.open('wb') as target:
                import shutil
                shutil.copyfileobj(response, target)
        if hashlib.sha256(bundle.read_bytes()).hexdigest() != manifest['sha256']:
            raise ValueError('Release bundle hash mismatch')
        with zipfile.ZipFile(bundle) as source:
            for name, expected in manifest['files'].items():
                target = (root / name).resolve()
                if not target.is_relative_to(root.resolve()):
                    raise ValueError('Invalid artifact path')
                contents = source.read(name)
                if hashlib.sha256(contents).hexdigest() != expected['sha256']:
                    raise ValueError(f'Artifact hash mismatch: {name}')
            for name in manifest['files']:
                target = root / name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(source.read(name))
    print('Verified release artifacts restored')


if __name__ == '__main__':
    restore(Path(__file__).resolve().parents[1])
