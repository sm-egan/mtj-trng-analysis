# src/analysis/entropy_evaluation.py
import subprocess
import hashlib
import numpy as np
from pathlib import Path
import polars as pl
from datetime import datetime
import logging
from typing import List, Optional, Tuple

from src.analysis.processing import xorn_whitening