import pandas as pd
import numpy as np
import pandas_profiling as pp
from data import get_dataset

def generate_report(path):
    profile = pp.ProfileReport(get_dataset(csv_path=path), title = 'Forest prfiling')
    profile.to_file('Forest report.html')
    