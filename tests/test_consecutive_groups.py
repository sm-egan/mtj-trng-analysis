"""Test script for consecutive row number grouping in EntropyEvaluationPipeline"""
import polars as pl
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

def test_split_consecutive_groups():
    """Test the _split_into_consecutive_groups method"""

    # Create a minimal mock class with just the method we need
    class MockPipeline:
        def _split_into_consecutive_groups(self, df):
            """
            Split a dataframe into sub-groups with consecutive row_numbers.
            Uses vectorized Polars operations for efficiency.
            """
            if len(df) == 0:
                return []

            # Ensure sorted by row_number
            df = df.sort("row_number")

            # Create a group identifier: increment when there's a gap in row_numbers
            df = df.with_columns([
                (pl.col("row_number").diff().fill_null(1) > 1)
                .cast(pl.Int32)
                .cum_sum()
                .alias("_consec_group")
            ])

            # Split into groups based on the consecutive group identifier
            groups = []
            for _, group_df in df.group_by("_consec_group", maintain_order=True):
                groups.append(group_df.drop("_consec_group"))

            return groups

    pipeline = MockPipeline()

    print("="*80)
    print("Test 1: Gap in middle (your example: 34,35,36,37,201)")
    print("="*80)
    df1 = pl.DataFrame({
        "row_number": [34, 35, 36, 37, 201],
        "run_id": ["run_34", "run_35", "run_36", "run_37", "run_201"],
        "field": [10.0] * 5
    })

    groups1 = pipeline._split_into_consecutive_groups(df1)
    print(f"Input: {df1['row_number'].to_list()}")
    print(f"Number of groups: {len(groups1)}")
    for i, group in enumerate(groups1):
        row_nums = group['row_number'].to_list()
        print(f"  Group {i}: rows {row_nums[0]}-{row_nums[-1]} ({len(row_nums)} runs)")
        print(f"    Row numbers: {row_nums}")

    assert len(groups1) == 2, f"Expected 2 groups, got {len(groups1)}"
    assert groups1[0]['row_number'].to_list() == [34, 35, 36, 37]
    assert groups1[1]['row_number'].to_list() == [201]
    print("✓ Test 1 passed\n")

    print("="*80)
    print("Test 2: Multiple gaps")
    print("="*80)
    df2 = pl.DataFrame({
        "row_number": [1, 2, 3, 10, 11, 20, 21, 22],
        "run_id": [f"run_{i}" for i in [1, 2, 3, 10, 11, 20, 21, 22]],
        "field": [10.0] * 8
    })

    groups2 = pipeline._split_into_consecutive_groups(df2)
    print(f"Input: {df2['row_number'].to_list()}")
    print(f"Number of groups: {len(groups2)}")
    for i, group in enumerate(groups2):
        row_nums = group['row_number'].to_list()
        print(f"  Group {i}: rows {row_nums[0]}-{row_nums[-1]} ({len(row_nums)} runs)")
        print(f"    Row numbers: {row_nums}")

    assert len(groups2) == 3
    assert groups2[0]['row_number'].to_list() == [1, 2, 3]
    assert groups2[1]['row_number'].to_list() == [10, 11]
    assert groups2[2]['row_number'].to_list() == [20, 21, 22]
    print("✓ Test 2 passed\n")

    print("="*80)
    print("Test 3: All consecutive (no gaps)")
    print("="*80)
    df3 = pl.DataFrame({
        "row_number": [100, 101, 102, 103, 104],
        "run_id": [f"run_{i}" for i in range(100, 105)],
        "field": [10.0] * 5
    })

    groups3 = pipeline._split_into_consecutive_groups(df3)
    print(f"Input: {df3['row_number'].to_list()}")
    print(f"Number of groups: {len(groups3)}")
    for i, group in enumerate(groups3):
        row_nums = group['row_number'].to_list()
        print(f"  Group {i}: rows {row_nums[0]}-{row_nums[-1]} ({len(row_nums)} runs)")

    assert len(groups3) == 1
    assert groups3[0]['row_number'].to_list() == [100, 101, 102, 103, 104]
    print("✓ Test 3 passed\n")

    print("="*80)
    print("Test 4: All separate (maximum gaps)")
    print("="*80)
    df4 = pl.DataFrame({
        "row_number": [1, 10, 100, 1000],
        "run_id": ["run_1", "run_10", "run_100", "run_1000"],
        "field": [10.0] * 4
    })

    groups4 = pipeline._split_into_consecutive_groups(df4)
    print(f"Input: {df4['row_number'].to_list()}")
    print(f"Number of groups: {len(groups4)}")
    for i, group in enumerate(groups4):
        row_nums = group['row_number'].to_list()
        print(f"  Group {i}: row {row_nums[0]}")

    assert len(groups4) == 4
    for i, expected_row in enumerate([1, 10, 100, 1000]):
        assert groups4[i]['row_number'].to_list() == [expected_row]
    print("✓ Test 4 passed\n")

    print("="*80)
    print("Test 5: Empty DataFrame")
    print("="*80)
    df5 = pl.DataFrame({
        "row_number": [],
        "run_id": [],
        "field": []
    })

    groups5 = pipeline._split_into_consecutive_groups(df5)
    print(f"Input: empty DataFrame")
    print(f"Number of groups: {len(groups5)}")
    assert len(groups5) == 0
    print("✓ Test 5 passed\n")

    print("="*80)
    print("Test 6: Unsorted input (should still work)")
    print("="*80)
    df6 = pl.DataFrame({
        "row_number": [37, 201, 34, 36, 35],  # Unsorted
        "run_id": ["run_37", "run_201", "run_34", "run_36", "run_35"],
        "field": [10.0] * 5
    })

    groups6 = pipeline._split_into_consecutive_groups(df6)
    print(f"Input (unsorted): {df6['row_number'].to_list()}")
    print(f"Number of groups: {len(groups6)}")
    for i, group in enumerate(groups6):
        row_nums = group['row_number'].to_list()
        print(f"  Group {i}: rows {row_nums[0]}-{row_nums[-1]} ({len(row_nums)} runs)")
        print(f"    Row numbers: {row_nums}")

    assert len(groups6) == 2
    assert groups6[0]['row_number'].to_list() == [34, 35, 36, 37]
    assert groups6[1]['row_number'].to_list() == [201]
    print("✓ Test 6 passed\n")

    print("="*80)
    print("✅ All tests passed!")
    print("="*80)

if __name__ == "__main__":
    test_split_consecutive_groups()
