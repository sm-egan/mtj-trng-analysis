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
    ## Device 011_08

    # pipeline.evaluate_entropy(
    #     device_id="011_08",
    #     # 51 to 475 includes 3 different reset pulse amplitudes: -0.6,-0.7,-0.8
    #     polars_expr=(pl.col("row_number").is_between(51,425)) & (pl.col("repeats") == 1_000_000),
    #     xor_levels=[1,2,4,8, 12, 16],
    #     xor_overlaps={2 : [0,1], 4:[0,1,2,3], 8:[0,2,4,5,6,7], 12:[0,2,4,6,8,10,11], 16:[0,2,4,6,8,10,12,13,14,15]},
    #     tests=['non_iid', 'iid', 'restart'],
    #     skip_existing=True
    # )

    # pipeline.evaluate_entropy(
    #     device_id="011_08",
    #     # 51 to 475 includes 3 different reset pulse amplitudes: -0.6,-0.7,-0.8
    #     polars_expr=(pl.col("row_number").is_between(51,425)) & (pl.col("repeats") == 1_000_000),
    #     xor_levels=[8],
    #     xor_overlaps={8:[0,1,2,3,4]},
    #     tests=['non_iid', 'iid', 'restart'],
    #     skip_existing=True
    # )

    # pipeline.evaluate_entropy(
    #     device_id="011_08",
    #     polars_expr=(pl.col("row_number").is_between(51,425)) & (pl.col("repeats") == 1_000_000) & (pl.col("r_amplitude") == -0.7),
    #     xor_levels=[12, 16],
    #     xor_overlaps={12 : [2,4,6], 16 : [2,6,10]},
    #     tests=['non_iid', 'iid', 'restart'],
    #     skip_existing=True
    # )
    # pipeline.evaluate_entropy(
    #     device_id="011_08",
    #     polars_expr=(pl.col("row_number").is_between(51,425)) & (pl.col("repeats") == 1_000_000) & (pl.col("r_amplitude") == -0.7) & (pl.col("s_width") == 4e-9),
    #     xor_levels=[16],
    #     xor_overlaps={16 : [4]},
    #     tests=['non_iid', 'iid', 'restart'],
    #     skip_existing=False
    # )

    # pipeline.evaluate_entropy(
    #     device_id="011_08",
    #     polars_expr=(pl.col("row_number").is_between(51,425)) & (pl.col("repeats") == 1_000_000) & (pl.col("r_amplitude") == -0.7),
    #     xor_levels=[6],
    #     xor_overlaps={6 : [0,2,4,5]},
    #     tests=['non_iid', 'iid', 'restart'],
    #     skip_existing=True
    # )



    # pipeline.evaluate_entropy(
    #     device_id="011_08",
    #     polars_expr=(pl.col("row_number").is_between(51,425)) & (pl.col("repeats") == 1_000_000),
    #     xor_levels=[2],
    #     xor_overlaps={2 : [0,1]},
    #     tests=['non_iid', 'iid', 'restart']
    # )

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
    #             #xor_method='rolling',
    #             tests=['non_iid', 'iid', 'restart'],
    #             skip_existing=True
    #         )

    #         pipeline.evaluate_entropy(
    #             device_id="011_08",
    #             # str -> float approach necessary to avoid least significant digits from being introduced which prevent the runs from being found.
    #             ## I encountered this problem when using np.linspace(2e-9,8e-9,1e-9) or even multiplying s_width_ns*1e-9 to define the s_width condition
    #             polars_expr=(
    #                     pl.col("r_amplitude") == r_amplitude_V) 
    #                     & (pl.col("s_amplitude") == 5.96) 
    #                     & (pl.col("s_width") == 1e-8) 
    #                     & (pl.col("row_number") < 426),
    #             tests=['non_iid', 'iid', 'restart'],
    #             #xor_method='rolling',
    #             skip_existing=True
    #         )
    
    ## Accidentally missed 7 ns data point here and had to add it
    # for r_amplitude_V in [-0.7]:
    #     pipeline.evaluate_entropy(
    #         device_id="011_08",
    #         # str -> float approach necessary to avoid least significant digits from being introduced which prevent the runs from being found.
    #         ## I encountered this problem when using np.linspace(2e-9,8e-9,1e-9) or even multiplying s_width_ns*1e-9 to define the s_width condition
    #         polars_expr=(pl.col("r_amplitude") == r_amplitude_V) 
    #                 & (pl.col("s_amplitude") == 5.96) 
    #                 & (pl.col("s_width") == 6e-9) 
    #                 & (pl.col("row_number") < 426)
    #                 & (pl.col("repeats") == 1_000_000),
    #         tests=['non_iid', 'iid', 'restart'],
    #         skip_existing=True
    #     )

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

    # for r_amplitude_V in [-0.9]:
    #     for s_width_ns in np.arange(2,7,1):
    #         pipeline.evaluate_entropy(
    #             device_id="07_08",
    #             # str -> float approach necessary to avoid least significant digits from being introduced which prevent the runs from being found.
    #             ## I encountered this problem when using np.linspace(2e-9,8e-9,1e-9) or even multiplying s_width_ns*1e-9 to define the s_width condition
    #             polars_expr=(pl.col("r_amplitude") == r_amplitude_V) 
    #                     & (pl.col("s_amplitude") == 4.73) 
    #                     & (pl.col("s_width") == float(f"{s_width_ns}e-9")) 
    #                     & (pl.col("row_number") < 555)
    #                     & (pl.col("repeats") == 1_000_000),
    #             tests=['non_iid', 'iid', 'restart'],
    #             skip_existing=True
    #         )

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

    ###############################################################
    ### Run entropy evaluation for D023_07

    pipeline.evaluate_entropy(
        device_id="023_07",
        #First run of 10 million repeats only 
        polars_expr=(pl.col("row_number") == 290),
        xor_levels=[2,4,8],
        tests=['non_iid', 'iid', 'restart'],
        skip_existing=False
    )

    # pipeline.evaluate_entropy(
    #     device_id="023_07",
    #     #All 10 million repeats runs, with drift ~50% -> 30%
    #     polars_expr=(pl.col("row_number").is_between(290,305)),
    #     tests=['non_iid', 'iid', 'restart'],
    #     #xor_method='rolling',
    #     skip_existing=True
    # )
    # pipeline.evaluate_entropy(
    #     device_id="023_07",
    #     #Late September data 
    #     polars_expr=(pl.col("row_number").is_between(6690, 6709)),
    #     tests=['non_iid', 'iid', 'restart'],
    #     skip_existing=True
    # )

    ###############################################################
    ### Run entropy evaluation for D027_12

    # pipeline.evaluate_entropy(
    #     device_id="027_12",
    #     #First run of 10 million repeats only 
    #     polars_expr=(pl.col("row_number").is_between(2226, 2245)),
    #     tests=['non_iid', 'iid', 'restart'],
    #     skip_existing=True
    # )

    # pipeline.evaluate_entropy(
    #     device_id="027_12",
    #     #First run of 10 million repeats only 
    #     polars_expr=(pl.col("row_number").is_between(1992, 2011)),
    #     tests=['non_iid', 'iid', 'restart'],
    #     skip_existing=True
    # )

    # pipeline.evaluate_entropy(
    #     device_id="027_12",
    #     #First run of 10 million repeats only 
    #     polars_expr=(pl.col("row_number").is_between(1932, 1951)),
    #     tests=['non_iid', 'iid', 'restart'],
    #     skip_existing=True
    # )

