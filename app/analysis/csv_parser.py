# Author: Junbong Jang
# Date: 12/3/2018
# app/csv_parser.py

import pandas as pd
def read_uploaded_csv(filename):
    uploaded_df = pd.read_csv("%s%s" % ('app/uploads/', filename))
    uploaded_df.dropna(inplace=True)
    return [uploaded_df.columns.tolist()] + uploaded_df.values.tolist()