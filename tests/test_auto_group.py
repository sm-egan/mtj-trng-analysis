#!/usr/bin/env python3
"""Test script to verify auto_group functionality"""

from pathlib import Path
import json
import polars as pl
from src.parquet_data_manager import ParquetExperimentalDataManager
from src.analysis.entropy_evaluation_pipeline import EntropyEvaluationPipeline


def load_ea_binaries_path():
    """Load the SP800-90B EntropyAssessment binaries path from config.json"""
    config_path = Path(__file__).parents[1] / 'config' / 'config.json'
    if not config_path.exists():
        raise FileNotFoundError(
            "Config file not found. Please copy config.template.json to "
            "config.json and fill in the required values."
        )

    with open(config_path) as f:
        config = json.load(f)

    ea_binaries_path = config.get('ea_binaries', {}).get('path')
    if not ea_binaries_path:
        raise ValueError("ea_binaries.path not found in config.json")

    return ea_binaries_path



def test_auto_group():
    """Test that auto_group parameter works correctly"""

    dm = ParquetExperimentalDataManager()
    pipeline = EntropyEvaluationPipeline(
        dm,
        ea_binaries_path=load_ea_binaries_path()
    )

    print("=" * 80)
    print("Testing auto_group functionality")
    print("=" * 80)

    # Test 1: Verify that auto_group=True is accepted
    print("\nTest 1: Check that auto_group parameter is accepted")
    try:
        # This should work - auto_group=True will process each group separately
        print("  Calling evaluate_entropy with auto_group=True...")
        print("  (This will process 21 groups separately)")

        # Note: This will actually run! Set skip_existing=True to avoid reprocessing
        pipeline.evaluate_entropy(
            device_id="011_08",
            polars_expr=(pl.col("row_number").is_between(51,425)) & (pl.col("repeats") == 1_000_000),
            xor_levels=[1],  # Just use raw data for quick test
            tests=['non_iid'],  # Just one test for speed
            auto_group=True,
            skip_existing=True
        )
        print("✓ Test 1 PASSED: auto_group=True works correctly")
    except Exception as e:
        print(f"✗ Test 1 FAILED: {e}")
        return False

    # Test 2: Verify that auto_group=False raises an error with multiple groups
    print("\nTest 2: Check that auto_group=False raises error with multiple groups")
    try:
        pipeline.evaluate_entropy(
            device_id="011_08",
            polars_expr=(pl.col("row_number").is_between(51,425)) & (pl.col("repeats") == 1_000_000),
            xor_levels=[1],
            tests=['non_iid'],
            auto_group=False,  # Should raise error
            skip_existing=True
        )
        print("✗ Test 2 FAILED: Should have raised ValueError")
        return False
    except ValueError as e:
        if "different combinations" in str(e):
            print(f"✓ Test 2 PASSED: Correctly raised ValueError")
            print(f"  Error message: {str(e)[:100]}...")
        else:
            print(f"✗ Test 2 FAILED: Wrong error message: {e}")
            return False
    except Exception as e:
        print(f"✗ Test 2 FAILED: Wrong exception type: {type(e).__name__}: {e}")
        return False

    print("\n" + "=" * 80)
    print("All tests passed!")
    print("=" * 80)
    return True

if __name__ == "__main__":
    test_auto_group()
