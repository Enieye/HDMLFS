import sys
import pandas as pd

def memory_report():
    variables = globals()
    memory_info = [
        {"Variable": var, "Size (MB)": sys.getsizeof(value) / (1024**2)}
        for var, value in variables.items()
    ]
    df = pd.DataFrame(memory_info).sort_values(by="Size (MB)", ascending=False)
    return df