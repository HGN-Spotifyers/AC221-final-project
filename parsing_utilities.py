import numpy as np
import pandas as pd


def unpack_multi_index(data,padding='bottom'):
    """
        Convert columns to multi-index.
        Assumes first row is the name of source table (e.g. S0101_C01_001M) and ignores it.
        Interprets the first table row as column names encoded as a hierarchy (joined by !!).
        padding==top: Pads (with empty strings) at the top of the hierarchy.
        padding==bottom: Pads (with empty strings) at the bottom of the hierarchy.
    """
    new_data = data.copy()
    new_colnames = []
    # Process header rows:
    for col in new_data.columns:
        colname_values = []
        for val in str(new_data[col][0]).strip().split('!!'):
            colname_values.append(val)
        new_colnames.append(colname_values)
    # Remove header rows:
    new_data = new_data.iloc[1:]
    # Pad column hierarchy so all lists have the same depth:
    max_length = 0
    for colname_values in new_colnames:
        max_length = max(max_length,len(colname_values))
    for i,colname_values in enumerate(new_colnames):
        pad = ""
        npads = max_length - len(colname_values)
        for _ in range(npads):
            if padding=='top':
                new_colnames[i].insert(0,pad)
            elif padding=='bottom':
                new_colnames[i].append(pad)
            else:
                raise ValueError("Padding should be 'top' or 'borrom'.")
    # Transpose and add columns level by level:
    new_data = new_data.transpose()
    level_names = ['level_'+str(i+1) for i in range(max_length)]
    for i,level_name in enumerate(level_names):
        new_data[level_name] = [colname_values[i] for colname_values in new_colnames]
    new_data = new_data.reset_index(drop=True).set_index(level_names)
    new_data = new_data.transpose()
    # Return coopy of original data with new column index:
    #   (to avoid changes in data type that sometimes happen when transposing)
    new_index = new_data.columns
    new_data = data.iloc[1:].copy()
    new_data.columns = new_index
    return new_data
    