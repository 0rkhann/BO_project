"""
Script to verify synchronization between energy cache and mordred descriptors.
"""

import json
import pandas as pd
from pathlib import Path


def verify_data_sync():
    """Verify that energy cache and mordred descriptors are synchronized."""
    data_dir = Path(__file__).parents[1] / "data"

    # Load mordred descriptors
    print("Loading mordred descriptors...")
    mordred_df = pd.read_csv(data_dir / "mordred.csv")
    mordred_smiles = set(mordred_df["SMILES"].tolist())
    print(f"Mordred descriptors: {len(mordred_smiles)} unique SMILES")

    # Load original energy cache
    print("\nLoading original energy cache...")
    with open(data_dir / "energy_cache.json", "r") as f:
        original_cache = json.load(f)
    original_smiles = set(item["SMILES"] for item in original_cache)
    print(f"Original energy cache: {len(original_smiles)} unique SMILES")

    # Load current energy cache (after filtering)
    print("\nLoading current energy cache...")
    with open(data_dir / "energy_cache.json", "r") as f:
        current_cache = json.load(f)

    # Handle both formats: list of dicts or dict
    if isinstance(current_cache, list):
        current_smiles = set(item["SMILES"] for item in current_cache)
    else:
        current_smiles = set(current_cache.keys())
    print(f"Current energy cache: {len(current_smiles)} unique SMILES")

    # Analysis
    print("\n" + "=" * 60)
    print("SYNCHRONIZATION ANALYSIS")
    print("=" * 60)

    # Check overlap
    mordred_in_current = mordred_smiles & current_smiles

    print(f"SMILES in both Mordred and current cache: {len(mordred_in_current)}")

    # Check missing
    mordred_missing_from_cache = mordred_smiles - current_smiles
    cache_missing_from_mordred = current_smiles - mordred_smiles

    print(
        f"\nSMILES in Mordred but NOT in current cache: {len(mordred_missing_from_cache)}"
    )
    print(
        f"SMILES in current cache but NOT in Mordred: {len(cache_missing_from_mordred)}"
    )

    # Show some examples if there are mismatches
    if mordred_missing_from_cache:
        print(f"\nFirst 5 SMILES missing from cache:")
        for i, smi in enumerate(list(mordred_missing_from_cache)[:5]):
            print(f"  {i+1}. {smi}")

    if cache_missing_from_mordred:
        print(f"\nFirst 5 SMILES missing from Mordred:")
        for i, smi in enumerate(list(cache_missing_from_mordred)[:5]):
            print(f"  {i+1}. {smi}")

    # Perfect sync check
    if len(mordred_missing_from_cache) == 0 and len(cache_missing_from_mordred) == 0:
        print(
            f"\n✅ PERFECT SYNC: All {len(current_smiles)} SMILES match between Mordred and current cache!"
        )
    else:
        print(
            f"\n⚠️  SYNC ISSUES: {len(mordred_missing_from_cache) + len(cache_missing_from_mordred)} mismatches found"
        )

    # Statistics
    print(f"\n" + "=" * 60)
    print("STATISTICS")
    print("=" * 60)
    print(f"Current cache size: {len(current_cache):,}")
    print(f"Mordred descriptor molecules: {len(mordred_smiles):,}")
    print(
        f"Coverage: {len(mordred_in_current) / len(mordred_smiles) * 100:.1f}% of Mordred molecules have energies"
    )

    return {
        "mordred_count": len(mordred_smiles),
        "current_cache_count": len(current_smiles),
        "perfect_sync": len(mordred_missing_from_cache) == 0
        and len(cache_missing_from_mordred) == 0,
        "coverage_percent": len(mordred_in_current) / len(mordred_smiles) * 100,
    }


if __name__ == "__main__":
    stats = verify_data_sync()

    if not stats["perfect_sync"]:
        print(f"\n⚠️  Consider running sync_cache.py again if needed")
    else:
        print(f"\n✅ Data is ready for BO experiments!")
