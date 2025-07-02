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

if __name__ == "__main__":
    
    dm = ParquetExperimentalDataManager()
    pipeline = EntropyEvaluationPipeline(
        dm, 
        ea_binaries_path="/Users/shannonegan/Documents/DSV/StochastIQ/umn/SP800-90B_EntropyAssessment/cpp"
    )

    import sys
    print("Python path:")
    for p in sys.path:
        print(f"  {p}")

    from src.analysis import EntropyEvaluationPipeline
    print(f"Imported from: {EntropyEvaluationPipeline.__module__}")
    print(f"File location: {EntropyEvaluationPipeline.__module__.__file__ if hasattr(EntropyEvaluationPipeline.__module__, '__file__') else 'N/A'}")
    
    # Loop over perturb pulse widths for Device 07_08
    # for s_width_ns in np.arange(2,8,1):

    #     pipeline.evaluate_entropy(
    #         device_id="07_08",
    #         # str -> float approach necessary to avoid least significant digits from being introduced which prevent the runs from being found.
    #         ## I encountered this problem when using np.linspace(2e-9,8e-9,1e-9) or even multiplying s_width_ns*1e-9 to define the s_width condition
    #         polars_expr=(pl.col("s_amplitude") == 4.73) & (pl.col("s_width") == float(f"{s_width_ns}e-9")) & (pl.col("row_number") < 319),
    #         tests=['non_iid', 'iid'],
    #         skip_existing=False,
    #     )

    ###################################################
    # Loop over perturb pulse widths for Device 011_08
    # for s_width_ns in np.arange(2,7,1):
    #     pipeline.evaluate_entropy(
    #         device_id="011_08",
    #         # str -> float approach necessary to avoid least significant digits from being introduced which prevent the runs from being found.
    #         ## I encountered this problem when using np.linspace(2e-9,8e-9,1e-9) or even multiplying s_width_ns*1e-9 to define the s_width condition
    #         polars_expr=(pl.col("s_amplitude") == 5.96) & (pl.col("s_width") == float(f"{s_width_ns}e-9")) & (pl.col("row_number") < 168),
    #         tests=['non_iid', 'iid'],
    #     )

    # pipeline.evaluate_entropy(
    #     device_id="011_08",
    #     polars_expr=(pl.col("s_amplitude") == 5.96) & (pl.col("s_width") == 1e-8) & (pl.col("row_number") < 168),
    #     tests=['non_iid', 'iid'],
    # )


    # 011_08 runs with r_amplitude = -0.7
    # pipeline.evaluate_entropy(
    #     device_id="011_08",
    #     polars_expr=(pl.col("row_number").is_between(243, 260)),
    #     tests=['non_iid', 'iid']
    # )

    ###################################################
    # Loop over perturb pulse widths for Device 011_08 with r_amplitude = -0.6, -0.7, -0.8
    # for r_amplitude_V in [-0.6, -0.7, -0.8]:
    #     for s_width_ns in np.arange(2,7,1):
    #         pipeline.evaluate_entropy(
    #             device_id="011_08",
    #             # str -> float approach necessary to avoid least significant digits from being introduced which prevent the runs from being found.
    #             ## I encountered this problem when using np.linspace(2e-9,8e-9,1e-9) or even multiplying s_width_ns*1e-9 to define the s_width condition
    #             polars_expr=(pl.col("r_amplitude") == r_amplitude_V) 
    #                     & (pl.col("s_amplitude") == 5.96) 
    #                     & (pl.col("s_width") == float(f"{s_width_ns}e-9")) 
    #                     & (pl.col("row_number") < 426)
    #                     & (pl.col("repeats") == 1_000_000),
    #             tests=['non_iid', 'iid', 'restart'],
    #             skip_existing=True
    #         )

    #     pipeline.evaluate_entropy(
    #         device_id="011_08",
    #         # str -> float approach necessary to avoid least significant digits from being introduced which prevent the runs from being found.
    #         ## I encountered this problem when using np.linspace(2e-9,8e-9,1e-9) or even multiplying s_width_ns*1e-9 to define the s_width condition
    #         polars_expr=(
    #                 pl.col("r_amplitude") == r_amplitude_V) 
    #                 & (pl.col("s_amplitude") == 5.96) 
    #                 & (pl.col("s_width") == 1e-8) 
    #                 & (pl.col("row_number") < 426),
    #         tests=['non_iid', 'iid', 'restart'],
    #         skip_existing=True
    #     )
    ###################################################
    # Loop over device 07_08

    for r_amplitude_V in [-0.9]:
        for s_width_ns in np.arange(2,7,1):
            pipeline.evaluate_entropy(
                device_id="07_08",
                # str -> float approach necessary to avoid least significant digits from being introduced which prevent the runs from being found.
                ## I encountered this problem when using np.linspace(2e-9,8e-9,1e-9) or even multiplying s_width_ns*1e-9 to define the s_width condition
                polars_expr=(pl.col("r_amplitude") == r_amplitude_V) 
                        & (pl.col("s_amplitude") == 4.73) 
                        & (pl.col("s_width") == float(f"{s_width_ns}e-9")) 
                        & (pl.col("row_number") < 555)
                        & (pl.col("repeats") == 1_000_000),
                tests=['non_iid', 'iid', 'restart'],
                skip_existing=True
            )

    # pipeline.evaluate_entropy(
    #     device_id="023_07",
    #     polars_expr=(pl.col("row_number") == 290),
    #     xor_levels=[8,4,2,1],
    #     skip_existing=False,
    #     tests=['non_iid', 'iid']
    # )
    
    # Process with specific run IDs (fast path)
    # pipeline.evaluate_entropy(
    #     device_id="07_08",
    #     run_ids=["D07_08_SW100.0_SA0.15_F10.0_abc123", "D07_08_SW100.0_SA0.15_F10.0_def456"],
    #     xor_levels=[1, 8]
    # )
    ###################################################
    # Create restart files for device 011_08

    #tasks = pipeline.create_restart_test_plan(device_id='011_08',file_hash='90c7317d8d67d14414f2ae742e8c60e63cfa8c7b403e216493f3ba36db77b5d0')
    #print(tasks)

    # Print summary
    # print("\nSummary of created restart files:")
    # for device, processes in summary.items():
    #     print(f"\nDevice: {device}")
    #     for process, files in processes.items():
    #         print(f"  Process: {process}")
    #         for file_hash, num_created in files.items():
    #             print(f"    {file_hash[:12]}...: {num_created} restart files")