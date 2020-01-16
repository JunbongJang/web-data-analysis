# Author: Junbong Jang
# Date: 1/12/2018
# app/anova_analysis.py
from __future__ import print_function
from statsmodels.formula.api import ols
from statsmodels.multivariate.manova import MANOVA
from scipy.stats import bartlett
from scipy.stats import levene
from analysis.data_preprocessor import Data_Preprocessor


class Anova_Analysis(object):

    def __init__(self, data_preprocessor):
        self.data = data_preprocessor

    def run_anova(self):
        formula = 'subt1 ~ C(a01) + C(a08) + dept1'
        lm = ols(formula, self.data.feature_df).fit()
        print(lm.summary())

        # https://www.statsmodels.org/dev/examples/notebooks/generated/interactions_anova.html
        # from statsmodels.compat import urlopen
        # import numpy as np
        # import pandas as pd
        # import matplotlib.pyplot as plt

        # plt.figure(figsize=(6, 6))
        # symbols = ['D', '^']
        # colors = ['r', 'g', 'blue']
        # factor_groups = self.data.feature_df.groupby(['a01', 'a08'])
        # for values, group in factor_groups:
        #     i, j = values
        #     plt.scatter(group['dept1'], group['subt1'], marker=symbols[j], color=colors[i - 1],
        #                 s=144)
        # plt.xlabel('Experience')
        # plt.ylabel('Salary')
        # plt.show()

    def run_manova(self):
        # https://stackoverflow.com/questions/51553355/how-to-get-pvalue-from-statsmodels-manova
        formula = 'cpt1 + dept1 + jelt1 ~ C(a01) + C(a08) + C(a01) * C(a08)'
        manova = MANOVA.from_formula(formula, self.data.feature_df)
        manova_model = manova.mv_test()
        print(type(manova_model))
        print(manova_model.summary())

    def calc_bartlett(self, column_names):
        bartlett_args = []
        for column_name in column_names:
            bartlett_args.append(self.data.feature_df[column_name].values)
        stat, pvalue = bartlett(*bartlett_args)
        print(stat, pvalue)

    def calc_levene(self, column_names):
        levene_args = []
        for column_name in column_names:
            levene_args.append(self.data.feature_df[column_name].values)
        stat, pvalue = levene(*levene_args)
        print(stat, pvalue)


if __name__ == "__main__":
    # homework 4
    data_preprocessor = Data_Preprocessor(missing_data='drop',
                                          filename="Dataset2.csv",
                                          predict_column=['subt1'],
                                          x_columns=['a01', 'a08', 'dept1'],
                                          categorical_columns=['a01', 'a08'])
    anova_analysis_obj = Anova_Analysis(data_preprocessor)
    anova_analysis_obj.run_anova()
    anova_analysis_obj.calc_bartlett(['a01', 'a08'])
    anova_analysis_obj.calc_levene(['a01', 'a08'])

    # homework 5
    data_preprocessor = Data_Preprocessor(missing_data='drop',
                                          filename="Dataset2.csv",
                                          predict_column=['a01', 'a08'],
                                          x_columns=['cpt1', 'dept1', 'jelt1'],
                                          categorical_columns=['a01', 'a08'])
    anova_analysis_obj = Anova_Analysis(data_preprocessor)
    anova_analysis_obj.run_manova()
