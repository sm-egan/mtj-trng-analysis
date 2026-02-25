# src/analysis_registry.py
"""
Registry for analysis filter configurations.

Stores named, described filter expressions in SQLite so that analysis
subsets are reproducible and traceable across notebooks.
"""

import sqlite3
import polars as pl
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional, Tuple, List


class AnalysisRegistry:
    """
    Registry for saving and loading analysis filter configurations.

    Each registered filter gets a unique integer ID that can be used to:
    - Quickly reload the same data subset in any notebook
    - Tag saved figures for traceability (e.g., fid001_analog_hist.tiff)

    Usage:
        reg = AnalysisRegistry(data_manager)

        # Register a new filter
        fid = reg.register(
            name="011_08 baseline 1M repeats",
            device_id="011_08",
            expr_str='(pl.col("row_number").is_between(51, 425)) & (pl.col("repeats") == 1_000_000)',
            description="Main dataset for entropy analysis.",
        )

        # Load it back in any notebook
        run_ids, info = reg.load(fid)
        measurements = data_manager.get_measurements(run_ids)

        # Generate a figure filename
        fig.savefig(figpath / reg.figname(fid, "analog_hist", ext=".tiff"))
    """

    def __init__(self, data_manager):
        """
        Initialize the registry.

        Args:
            data_manager: A ParquetExperimentalDataManager instance.
                          Uses its db_path for storage and its methods for run lookups.
        """
        self.dm = data_manager
        self.db_path = data_manager.db_path
        self.logger = logging.getLogger(__name__)
        self._initialize_table()

    def _initialize_table(self):
        """Create the analysis_filters table if it doesn't exist."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS analysis_filters (
                    filter_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    description TEXT,
                    device_id TEXT,
                    expr_str TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.commit()

    def register(
        self,
        name: str,
        device_id: str,
        expr_str: str,
        description: str = "",
    ) -> int:
        """
        Register a new analysis filter.

        Args:
            name: Short human-readable label (e.g., "011_08 baseline 1M repeats")
            device_id: Device this filter applies to
            expr_str: Polars filter expression as a string.
                      Must be a valid expression using pl.col(), e.g.:
                      '(pl.col("row_number").is_between(51, 425)) & (pl.col("repeats") == 1_000_000)'
            description: Longer notes about what this data subset represents

        Returns:
            int: The assigned filter_id
        """
        # Validate that the expression string can be parsed
        self._reconstruct_expr(expr_str)

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                """
                INSERT INTO analysis_filters (name, description, device_id, expr_str)
                VALUES (?, ?, ?, ?)
                """,
                (name, description, device_id, expr_str),
            )
            conn.commit()
            filter_id = cursor.lastrowid

        self.logger.info(f"Registered filter {filter_id}: '{name}'")
        return filter_id

    def load(self, fid: int) -> Tuple[list, dict]:
        """
        Load a registered filter and return matching run IDs.

        Args:
            fid: The filter_id to load

        Returns:
            Tuple of (run_ids, info) where:
                - run_ids: list of matching run_id strings, sorted by row_number
                - info: dict with filter metadata (name, description, device_id, expr_str, created_at)
        """
        # Fetch filter metadata
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT name, description, device_id, expr_str, created_at "
                "FROM analysis_filters WHERE filter_id = ?",
                (fid,),
            ).fetchone()

        if row is None:
            raise KeyError(f"No filter found with filter_id={fid}")

        name, description, device_id, expr_str, created_at = row
        info = {
            "filter_id": fid,
            "name": name,
            "description": description,
            "device_id": device_id,
            "expr_str": expr_str,
            "created_at": created_at,
        }

        # Reconstruct the Polars expression and query runs
        expr = self._reconstruct_expr(expr_str)
        device_filter = (pl.col("device_id") == device_id) & expr

        with sqlite3.connect(self.db_path) as conn:
            runs_df = pl.read_database("SELECT * FROM runs", conn)

        matched = runs_df.filter(device_filter).sort("row_number")
        run_ids = matched["run_id"].to_list()

        self.logger.info(f"Filter {fid} ('{name}'): {len(run_ids)} runs matched")
        return run_ids, info

    def list_filters(self, device_id: Optional[str] = None) -> pl.DataFrame:
        """
        List all registered filters.

        Args:
            device_id: If provided, only show filters for this device

        Returns:
            Polars DataFrame with all filter metadata
        """
        with sqlite3.connect(self.db_path) as conn:
            if device_id:
                df = pl.read_database(
                    "SELECT * FROM analysis_filters WHERE device_id = ? ORDER BY filter_id",
                    conn,
                    execute_options={"parameters": [device_id]},
                )
            else:
                df = pl.read_database(
                    "SELECT * FROM analysis_filters ORDER BY filter_id",
                    conn,
                )
        return df

    def figname(self, fid: int, label: str, ext: str = ".tiff") -> str:
        """
        Generate a standardized figure filename tagged with filter ID.

        Args:
            fid: The filter_id
            label: Descriptive label for the plot (e.g., "analog_hist", "xor4_scatter")
            ext: File extension (default: ".tiff")

        Returns:
            str: Filename like "fid001_analog_hist.tiff"

        Example:
            fig.savefig(figpath / reg.figname(1, "analog_hist"))
        """
        return f"fid{fid:03d}_{label}{ext}"

    def update(self, fid: int, **kwargs) -> None:
        """
        Update fields of an existing filter.

        Args:
            fid: The filter_id to update
            **kwargs: Fields to update. Valid keys: name, description, device_id, expr_str
        """
        valid_fields = {"name", "description", "device_id", "expr_str"}
        invalid = set(kwargs.keys()) - valid_fields
        if invalid:
            raise ValueError(f"Cannot update fields: {invalid}. Valid: {valid_fields}")

        if "expr_str" in kwargs:
            self._reconstruct_expr(kwargs["expr_str"])

        set_clause = ", ".join(f"{key} = ?" for key in kwargs)
        values = list(kwargs.values()) + [fid]

        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                f"UPDATE analysis_filters SET {set_clause} WHERE filter_id = ?",
                values,
            )
            conn.commit()

        self.logger.info(f"Updated filter {fid}: {list(kwargs.keys())}")

    def delete(self, fid: int) -> None:
        """
        Delete a registered filter.

        Args:
            fid: The filter_id to delete
        """
        with sqlite3.connect(self.db_path) as conn:
            result = conn.execute(
                "DELETE FROM analysis_filters WHERE filter_id = ?", (fid,)
            )
            conn.commit()

        if result.rowcount == 0:
            raise KeyError(f"No filter found with filter_id={fid}")

        self.logger.info(f"Deleted filter {fid}")

    @staticmethod
    def _reconstruct_expr(expr_str: str) -> pl.Expr:
        """
        Reconstruct a Polars expression from its string representation.

        The eval namespace is restricted to only the polars module.

        Args:
            expr_str: String representation of a Polars expression

        Returns:
            pl.Expr: The reconstructed expression

        Raises:
            ValueError: If the string cannot be evaluated as a valid Polars expression
        """
        try:
            expr = eval(expr_str, {"__builtins__": {}, "pl": pl})
        except Exception as e:
            raise ValueError(
                f"Could not reconstruct Polars expression from: {expr_str!r}\n"
                f"Error: {e}"
            )

        if not isinstance(expr, pl.Expr):
            raise ValueError(
                f"Expression did not produce a pl.Expr, got {type(expr).__name__}"
            )

        return expr
