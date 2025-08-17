# Original code: https://www.kaggle.com/gemartin/load-data-reduce-memory-usage by @gemartin
# Modified: help with categorical type identification
# Modified: help reduce the datasets to lower memeory requirement.

import numpy as np
import pandas as pd
from pandas.api.types import CategoricalDtype, is_datetime64_any_dtype as is_datetime


def optimize_memory(data, use_float16: bool = False) -> pd.DataFrame:


    start_mem = data.memory_usage().sum() / 1024**2
    print(f"Memory usage of dataframe is {start_mem:.2f} MB")

    for col in data.columns:
        col_type = data[col].dtype

        # Skip datetime and categorical columns
        if is_datetime(data[col]) or isinstance(col_type, CategoricalDtype):
            continue

        if col_type != object:
            c_min = data[col].min()
            c_max = data[col].max()

            if str(col_type)[:3] == "int":
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    data[col] = data[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    data[col] = data[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    data[col] = data[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    data[col] = data[col].astype(np.int64)

            else:  # floats
                if use_float16 and c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    data[col] = data[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    data[col] = data[col].astype(np.float32)
                else:
                    data[col] = data[col].astype(np.float64)

        else:
            data[col] = data[col].astype("object")

    end_mem = data.memory_usage().sum() / 1024**2
    print(f"New memory usage: {end_mem:.2f} MB")
    print(f"Memory reduction: {100 * (start_mem - end_mem) / start_mem:.1f}%")

    return data