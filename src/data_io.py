import json
from pathlib import Path
import pandas as pd

def load_descriptors(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if 'SMILES' not in df.columns:
        raise ValueError("Input CSV must have a 'SMILES' column")
    return df

def load_or_init_cache(path: Path) -> dict[str, float]:
    if path.exists():
        raw = json.loads(path.read_text())
        # if the file is a list of { "smiles":..., "energy":... } records
        if isinstance(raw, list):
            return { rec['SMILES']: rec['energy'] for rec in raw }
        # otherwise assume it's already a dict
        return raw
    return {}

def append_to_cache(smiles: str, energy: float, cache_path: Path):
    cache = load_or_init_cache(cache_path)
    cache[smiles] = energy
    cache_path.write_text(json.dumps(cache, indent=2))