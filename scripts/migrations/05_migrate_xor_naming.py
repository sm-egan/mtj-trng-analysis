#!/usr/bin/env python
"""
Migration script to rename xor processing folders and update references.

PROBLEM:
    When you have folders named xor2, xor4, xor8 alongside xor12 and xor16,
    alphabetical sorting results in: xor12, xor16, xor2, xor4, xor8
    Similarly, overlap naming like o1, o2, o3 sorts incorrectly with o10, o11, o12
    This makes data import and plotting difficult.

SOLUTION:
    - Rename xor2->xor02, xor4->xor04, xor8->xor08 for proper sorting
    - Rename overlap suffixes: _o1->_o01, _o2->_o02, _o3->_o03, etc.
    Result: xor02_o01, xor04_o03, xor08_o07, xor12_o12, xor16_o15

WHAT THIS SCRIPT DOES:
    1. Renames folders in bin/ and esv/ directories
       - Handles base names (e.g., xor2) and overlapping versions (e.g., xor2_o0, xor2_o1)
       - Applies both xor number padding AND overlap number padding

    2. Updates code references in Python files, Jupyter notebooks, and CSV files
       - Uses regex with word boundaries to avoid false matches (won't replace xor2 in xor24)
       - Updates strings, variable names, path references, and CSV values

    3. Safety features:
       - Dry-run mode to preview all changes before applying
       - Git status check to warn about uncommitted changes
       - Conflict detection (won't rename if target folder exists)
       - Detailed logging of all changes

USAGE:
    # Preview changes (RECOMMENDED FIRST STEP)
    python scripts/migrations/05_migrate_xor_naming.py --dry-run

    # Execute migration
    python scripts/migrations/05_migrate_xor_naming.py

    # Migrate specific devices only
    python scripts/migrations/05_migrate_xor_naming.py --devices 011_08 023_07

    # Only rename folders, don't update file contents
    python scripts/migrations/05_migrate_xor_naming.py --skip-files

    # Only update file contents, don't rename folders
    python scripts/migrations/05_migrate_xor_naming.py --skip-folders

AFTER MIGRATION:
    1. Review changes with: git diff
    2. Run tests to ensure everything works
    3. Commit changes:
       git add -A
       git commit -m "Migrate xor naming to zero-padded format"
"""

import re
import sys
import argparse
import shutil
from pathlib import Path
from typing import List, Tuple, Dict
import json

# Project root
PROJECT_ROOT = Path(__file__).parent.parent.parent

# Mapping of old names to new names
XOR_MIGRATIONS = {
    'xor2': 'xor02',
    'xor4': 'xor04',
    'xor8': 'xor08',
}


class MigrationTracker:
    """Track all changes made during migration"""

    def __init__(self):
        self.folders_renamed = []
        self.files_updated = []
        self.errors = []

    def add_folder_rename(self, old_path: Path, new_path: Path):
        self.folders_renamed.append((str(old_path), str(new_path)))

    def add_file_update(self, file_path: Path, num_replacements: int):
        self.files_updated.append((str(file_path), num_replacements))

    def add_error(self, error_msg: str):
        self.errors.append(error_msg)

    def print_summary(self):
        """Print summary of all changes"""
        print("\n" + "="*80)
        print("MIGRATION SUMMARY")
        print("="*80)

        if self.folders_renamed:
            print(f"\nFolders renamed: {len(self.folders_renamed)}")
            for old, new in self.folders_renamed:
                print(f"  {old} -> {new}")

        if self.files_updated:
            print(f"\nFiles updated: {len(self.files_updated)}")
            total_replacements = sum(n for _, n in self.files_updated)
            print(f"Total replacements: {total_replacements}")
            for file, num in self.files_updated:
                print(f"  {file}: {num} replacement(s)")

        if self.errors:
            print(f"\nErrors encountered: {len(self.errors)}")
            for error in self.errors:
                print(f"  ERROR: {error}")

        if not self.folders_renamed and not self.files_updated:
            print("\nNo changes needed - all folders already use correct naming.")


def find_xor_folders(base_dir: Path, devices: List[str] = None) -> List[Path]:
    """
    Find all xor folders that need renaming in bin/ and esv/ directories.

    Args:
        base_dir: Base directory to search (bin or esv)
        devices: Optional list of specific devices to process

    Returns:
        List of folder paths that need renaming
    """
    folders_to_rename = []

    if not base_dir.exists():
        return folders_to_rename

    # Get device directories
    device_dirs = [d for d in base_dir.iterdir() if d.is_dir() and '_' in d.name]

    if devices:
        device_dirs = [d for d in device_dirs if d.name in devices]

    for device_dir in device_dirs:
        # Look for xor folders
        for folder in device_dir.iterdir():
            if not folder.is_dir():
                continue

            folder_name = folder.name

            # Check if folder matches old naming pattern
            # Match: xor2, xor4, xor8 (with optional _o suffix)
            # Don't match: xor02, xor04, xor08, xor12, xor16
            for old_name in XOR_MIGRATIONS.keys():
                if folder_name == old_name or folder_name.startswith(old_name + '_'):
                    folders_to_rename.append(folder)
                    break

    return sorted(folders_to_rename)


def rename_folder(old_path: Path, dry_run: bool = True) -> Tuple[Path, bool]:
    """
    Rename a single folder according to migration mapping.
    Handles both xor number padding (xor2->xor02) and overlap padding (_o1->_o01).

    Args:
        old_path: Path to folder to rename
        dry_run: If True, only simulate the rename

    Returns:
        Tuple of (new_path, success)
    """
    old_name = old_path.name

    # Determine new name - apply xor number padding first
    new_name = old_name
    for old_prefix, new_prefix in XOR_MIGRATIONS.items():
        if old_name == old_prefix:
            new_name = new_prefix
            break
        elif old_name.startswith(old_prefix + '_'):
            new_name = old_name.replace(old_prefix + '_', new_prefix + '_', 1)
            break

    # Apply overlap number padding (_o1->_o01, _o2->_o02, etc.)
    # Match _o followed by single digit (0-9)
    overlap_pattern = re.compile(r'_o(\d)(?=$)')
    match = overlap_pattern.search(new_name)
    if match:
        overlap_num = match.group(1)
        new_name = overlap_pattern.sub(f'_o0{overlap_num}', new_name)

    new_path = old_path.parent / new_name

    if new_path.exists():
        print(f"  WARNING: Target already exists: {new_path}")
        return new_path, False

    if not dry_run:
        try:
            old_path.rename(new_path)
            print(f"  Renamed: {old_path.relative_to(PROJECT_ROOT)} -> {new_path.relative_to(PROJECT_ROOT)}")
            return new_path, True
        except Exception as e:
            print(f"  ERROR renaming {old_path}: {e}")
            return new_path, False
    else:
        print(f"  Would rename: {old_path.relative_to(PROJECT_ROOT)} -> {new_path.relative_to(PROJECT_ROOT)}")
        return new_path, True


def update_file_content(file_path: Path, dry_run: bool = True) -> int:
    """
    Update references in a file (Python, Jupyter notebook, or CSV).
    Handles both xor number padding and overlap number padding.

    Args:
        file_path: Path to file to update
        dry_run: If True, only simulate the update

    Returns:
        Number of replacements made
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        original_content = content
        total_replacements = 0

        # Step 1: Apply xor number padding (xor2->xor02, xor4->xor04, xor8->xor08)
        # We need to be careful to match whole words or patterns like "xor2_o0", "xor2", etc.
        # but not match things like "xor24"
        for old_name, new_name in XOR_MIGRATIONS.items():
            # Pattern: match old_name followed by underscore, quote, space, or end of word
            # This prevents matching xor2 in xor24, xor12, etc.
            pattern = re.compile(
                r'\b' + re.escape(old_name) + r'(?=_|"|\'|\s|,|\)|]|\}|$)',
                re.MULTILINE
            )

            # Count and replace
            matches = pattern.findall(content)
            if matches:
                content = pattern.sub(new_name, content)
                total_replacements += len(matches)

        # Step 2: Apply overlap number padding (_o1->_o01, _o2->_o02, etc.)
        # Match _o followed by single digit, but not double digit
        # Use negative lookahead to avoid matching _o1 in _o10, _o11, etc.
        overlap_pattern = re.compile(
            r'_o(\d)(?!\d)',  # _o followed by single digit, not followed by another digit
            re.MULTILINE
        )

        # Find all matches and replace
        overlap_matches = overlap_pattern.findall(content)
        if overlap_matches:
            content = overlap_pattern.sub(r'_o0\1', content)
            total_replacements += len(overlap_matches)

        if total_replacements > 0:
            if not dry_run:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                print(f"  Updated {file_path.relative_to(PROJECT_ROOT)}: {total_replacements} replacement(s)")
            else:
                print(f"  Would update {file_path.relative_to(PROJECT_ROOT)}: {total_replacements} replacement(s)")

        return total_replacements

    except Exception as e:
        print(f"  ERROR updating {file_path}: {e}")
        return 0


def find_files_to_update() -> List[Path]:
    """Find all Python, notebook, and CSV files that might contain xor references."""
    files = []

    # Python, notebook, and CSV files
    for pattern in ['**/*.py', '**/*.ipynb', '**/*.csv']:
        files.extend(PROJECT_ROOT.glob(pattern))

    # Exclude certain directories
    excluded = {'.git', '.venv', 'venv', '__pycache__', '.ipynb_checkpoints', 'node_modules'}
    files = [f for f in files if not any(excl in f.parts for excl in excluded)]

    # Exclude this migration script itself
    script_path = Path(__file__).resolve()
    files = [f for f in files if f.resolve() != script_path]

    return sorted(files)


def check_git_status() -> bool:
    """Check if we're in a git repo with uncommitted changes."""
    import subprocess

    try:
        result = subprocess.run(
            ['git', 'status', '--porcelain'],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True
        )

        if result.returncode != 0:
            return True  # Not a git repo, allow migration

        if result.stdout.strip():
            print("\nWARNING: You have uncommitted changes in your git repository.")
            print("It's recommended to commit or stash your changes before running this migration.")
            response = input("Continue anyway? (y/N): ")
            return response.lower() == 'y'

        return True

    except FileNotFoundError:
        # Git not installed
        return True


def main():
    parser = argparse.ArgumentParser(
        description="Migrate xor folder naming from single-digit to zero-padded format"
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Preview changes without making them'
    )
    parser.add_argument(
        '--devices',
        nargs='+',
        help='Specific devices to migrate (e.g., 011_08 023_07)'
    )
    parser.add_argument(
        '--skip-files',
        action='store_true',
        help='Only rename folders, skip updating file contents'
    )
    parser.add_argument(
        '--skip-folders',
        action='store_true',
        help='Only update files, skip renaming folders'
    )

    args = parser.parse_args()

    print("="*80)
    print("XOR NAMING MIGRATION SCRIPT")
    print("="*80)
    print(f"Mode: {'DRY RUN (no changes will be made)' if args.dry_run else 'LIVE (changes will be applied)'}")
    print(f"Project root: {PROJECT_ROOT}")

    if args.devices:
        print(f"Target devices: {', '.join(args.devices)}")

    # Check git status if not dry run
    if not args.dry_run and not check_git_status():
        print("\nMigration cancelled.")
        return 1

    tracker = MigrationTracker()

    # Step 1: Rename folders in bin/ and esv/
    if not args.skip_folders:
        print("\n" + "-"*80)
        print("STEP 1: Renaming folders")
        print("-"*80)

        for base_dir_name in ['bin', 'esv']:
            base_dir = PROJECT_ROOT / base_dir_name
            print(f"\nScanning {base_dir_name}/ directory...")

            folders = find_xor_folders(base_dir, args.devices)

            if not folders:
                print(f"  No folders to rename in {base_dir_name}/")
                continue

            print(f"  Found {len(folders)} folder(s) to rename")

            for folder in folders:
                new_path, success = rename_folder(folder, dry_run=args.dry_run)
                if success:
                    tracker.add_folder_rename(folder, new_path)
                else:
                    tracker.add_error(f"Failed to rename {folder}")

    # Step 2: Update file contents
    if not args.skip_files:
        print("\n" + "-"*80)
        print("STEP 2: Updating file references")
        print("-"*80)

        files = find_files_to_update()
        print(f"\nFound {len(files)} Python and notebook files to check")

        for file_path in files:
            num_replacements = update_file_content(file_path, dry_run=args.dry_run)
            if num_replacements > 0:
                tracker.add_file_update(file_path, num_replacements)

    # Print summary
    tracker.print_summary()

    if args.dry_run:
        print("\n" + "="*80)
        print("This was a DRY RUN. No changes were made.")
        print("Run without --dry-run to apply these changes.")
        print("="*80)
    else:
        print("\n" + "="*80)
        print("Migration complete!")
        print("="*80)

    return 0


if __name__ == '__main__':
    sys.exit(main())
