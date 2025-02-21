import re
from pathlib import Path
import logging
import numpy as np
from typing import Optional, Tuple, Dict
import hashlib
from dataclasses import dataclass

@dataclass
class RunIdentity:
    """Class to hold run identification information"""
    full_hash: str
    short_hash: str
    is_valid: bool
    validation_message: str

class DataValidator:
    def __init__(self, filename_pattern: str = r'Hz_.*Row.*RAmp.*_RWid.*_SAmp.*_SWid.*_Field.*_Source.*_Repeats.*\.txt$'):
        """
        Initialize the data validator with configurable filename pattern.
        
        Parameters:
        -----------
        filename_pattern : str
            Regular expression pattern that valid filenames must match.
            Default pattern updated to match new run ID format.
        """
        self.filename_pattern = filename_pattern
        self.logger = logging.getLogger(__name__)
    

    def compute_run_identity(self, 
                            metadata: Dict, 
                            measurements: np.ndarray,
                            hash_length: int = 12,
                            n_points: int = 1000) -> RunIdentity:
        """
        Compute run identity and validate data integrity.
        
        Parameters:
        -----------
        metadata : dict
            Dictionary containing run metadata
        measurements : numpy.ndarray
            Array of measurement values
        hash_length : int
            Length of the short hash to return
        n_points : int
            Number of points to use for hash computation
        """
        if measurements.ndim > 1:
            measurements = measurements.flatten()

        # Check for NaN or infinite values in first n_points
        check_measurements = measurements[:n_points]
        #Return invalid hash if any of the values are NaN or inf
        if not np.all(np.isfinite(check_measurements)):
            return RunIdentity(
                full_hash="",
                short_hash="",
                is_valid=False,
                validation_message="Measurements contain NaN or infinite values"
            )
        
        # Create deterministic string representation of data
        metadata_str = '_'.join(f"{k}:{v}" for k, v in sorted(metadata.items()))
        measurements_str = '_'.join(f"{x:.6f}" for x in check_measurements)
        content_to_hash = f"{metadata_str}|{measurements_str}"
        
        # Compute full hash
        full_hash = hashlib.sha256(content_to_hash.encode()).hexdigest()
        short_hash = full_hash[:hash_length]
        
        return RunIdentity(
            full_hash=full_hash,
            short_hash=short_hash,
            is_valid=True,
            validation_message="Data validated successfully"
        )
    
    def is_valid_filename(self, filename: str) -> bool:
        """
        Check if filename matches the expected pattern.
        Now updated to match new run ID format.
        
        Parameters:
        -----------
        filename : str
            Filename to validate
        
        Returns:
        --------
        bool
            True if filename matches pattern, False otherwise
        """
        if not bool(re.match(self.filename_pattern, Path(filename).name)):
            self.logger.warning(f"Invalid filename format: {filename}")
            return False
        return True
    
    def compare_runs(self, 
                    run1_identity: RunIdentity,
                    run2_identity: RunIdentity) -> bool:
        """
        Compare two runs using their hash identities.
        
        Parameters:
        -----------
        run1_identity : RunIdentity
            Identity information for first run
        run2_identity : RunIdentity
            Identity information for second run
            
        Returns:
        --------
        bool
            True if runs are identical
        """
        if not (run1_identity.is_valid and run2_identity.is_valid):
            return False
        
        return run1_identity.full_hash == run2_identity.full_hash