# Author: Junbong Jang
# Date: 1/15/2018
# app/categorical_analysis.py

import pandas as pd
import numpy as np
from scipy.stats import chisquare

class Categorical_Analysis(object):

    def __init__(self):
        print('initialized')
        self.caclulate_chisquare()

    def caclulate_chisquare(self):
        print(chisquare([16, 18, 16, 14, 12, 30]))


if __name__ == "__main__":

    categorical_analysis_obj = Categorical_Analysis()
    # categorical_analysis_obj.calculate_chisquare()
