import json
import numpy as np
import polars as pl
import subprocess
import hashlib
import numpy as np
from pathlib import Path
import polars as pl
#from datetime import datetime
#import logging
#from typing import List, Optional, Tuple
from src.analysis.processing import xorn_whitening
from src.parquet_data_manager import ParquetExperimentalDataManager
from src.analysis.entropy_evaluation_pipeline import EntropyEvaluationPipeline


def load_ea_binaries_path():
    """Load the SP800-90B EntropyAssessment binaries path from config.json"""
    config_path = Path(__file__).parents[2] / 'config' / 'config.json'
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



if __name__ == "__main__":
    
    dm = ParquetExperimentalDataManager()
    pipeline = EntropyEvaluationPipeline(
        dm, 
        ea_binaries_path=load_ea_binaries_path()
    )

    import sys
    print("Python path:")
    for p in sys.path:
        print(f"  {p}")

    from src.analysis import EntropyEvaluationPipeline
    print(f"Imported from: {EntropyEvaluationPipeline.__module__}")
    print(f"File location: {EntropyEvaluationPipeline.__module__.__file__ if hasattr(EntropyEvaluationPipeline.__module__, '__file__') else 'N/A'}")
    

    ###################################################
    ## Device 011_08
    # Command to run full XOR-16 analysis
    pipeline.evaluate_entropy(
        device_id="011_08",
        # 51 to 475 includes 3 different reset pulse amplitudes: -0.6,-0.7,-0.8
        polars_expr=(pl.col("row_number").is_between(51,425)) & (pl.col("repeats") == 1_000_000),
        xor_levels=[1, 2, 4, 8, 12, 16],
        xor_overlaps={2 : [0,1], 4:[0,1,2,3], 8:[0,2,4,5,6,7], 12:[0,2,4,6,8,10,11], 16:[0,2,4,6,8,10,12,13,14,15]},
        tests=['non_iid', 'iid', 'restart'],
        skip_existing=True
    )

    ###############################################################
    ### Run entropy evaluation for D023_07

    # pipeline.evaluate_entropy(
    #     device_id="023_07",
    #     #First run of 10 million repeats only 
    #     polars_expr=(pl.col("row_number") == 290),
    #     xor_levels=[2,4,8],
    #     tests=['non_iid', 'iid', 'restart'],
    #     skip_existing=False
    # )
