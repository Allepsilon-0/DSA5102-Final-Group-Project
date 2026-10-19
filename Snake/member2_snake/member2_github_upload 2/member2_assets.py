"""Extract missing experiment assets from the single uploaded archive."""
from pathlib import Path
import hashlib
import zipfile

ASSET_SHA256 = '2bb0b353c4851142304685e77bb9000891b222ff181d40e78556d76b1921b6a9'

def ensure_assets(root=None):
    root = Path(root or Path(__file__).resolve().parent).resolve()
    archive = root / 'member2_assets.zip'
    if not archive.is_file():
        if (root / 'runs/member2/final/results.json').is_file():
            return
        raise FileNotFoundError('Keep member2_assets.zip beside the notebook and Python files.')
    with zipfile.ZipFile(archive) as z:
        entries = [entry for entry in z.infolist() if not entry.is_dir()]
        missing = [entry for entry in entries if not (root / entry.filename).is_file()]
        if not missing:
            return
        if hashlib.sha256(archive.read_bytes()).hexdigest() != ASSET_SHA256:
            raise ValueError('member2_assets.zip checksum does not match the supplied package.')
        for entry in missing:
            target = (root / entry.filename).resolve()
            if not target.is_relative_to(root):
                raise ValueError('Invalid archive path')
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(z.read(entry))
    print('Experiment assets extracted; no training required.')
