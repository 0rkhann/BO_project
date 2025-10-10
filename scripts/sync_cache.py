"""
Script to synchronize energy cache with available descriptors.
"""

import json
import pandas as pd
from pathlib import Path


def sync_cache(descriptor_file: str, cache_file: str):
    """Synchronize energy cache with available descriptors in place."""
    # Load descriptor file
    df = pd.read_csv(descriptor_file)
    available_smiles = set(df["SMILES"].tolist())

    # Load cache
    with open(cache_file, "r") as f:
        cache_list = json.load(f)

    # Filter cache in place
    original_size = len(cache_list)
    filtered_cache_list = [
        item for item in cache_list if item["SMILES"] in available_smiles
    ]

    # Save filtered cache back to original file
    with open(cache_file, "w") as f:
        json.dump(filtered_cache_list, f, indent=2)

    print(f"Original cache size: {original_size}")
    print(f"Filtered cache size: {len(filtered_cache_list)}")
    print(f"Removed {original_size - len(filtered_cache_list)} entries")
    print(f"Updated {cache_file} in place")


if __name__ == "__main__":
    data_dir = Path(__file__).parents[1] / "data"
    sync_cache(
        descriptor_file=str(data_dir / "mordred.csv"),
        cache_file=str(data_dir / "energy_cache.json"),
    )
