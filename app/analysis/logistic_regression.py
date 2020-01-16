# Author: Junbong Jang
# Date: 1/12/2018
# app/logistic_regression.py

import numpy as np
from analysis.data_preprocessor import Data_Preprocessor


class Logistic_Regression(object):

    def __init__(self, data_preprocessor):
        self.data = data_preprocessor

    # LogitResults object's attribute information obtained from
    # https://github.com/statsmodels/statsmodels/blob/master/statsmodels/discrete/discrete_model.py
    def calc_logistic_regression_stat(self):
        import statsmodels.api as sm
        predictor_with_constant = sm.add_constant(self.data.X)

        import statsmodels.discrete.discrete_model as sm
        logit = sm.Logit(self.data.y, predictor_with_constant)
        logit_model = logit.fit()
        print(type(logit_model))
        print(logit_model.summary())
        print(logit_model.params)
        print(logit_model.conf_int())
        print(np.exp(logit_model.params))
        print(logit_model.llf)
        print(logit_model.llnull)
        print(logit_model.llr)
        print(logit_model.llr_pvalue)
        print(logit_model.df_model)
        print(logit_model.df_resid)
        print(logit_model.prsquared)
        print(logit_model.aic)
        print(logit_model.bic)
        print(logit_model.bse)
        #     McFadden's pseudo-R-squared. `1 - (llf / llnull)`

    def calc_freq_table(self, column_name):
        no_counter = 0
        yes_counter = 0
        missing_counter = 0
        for index, row in self.data.y.iterrows():
            column_value = row[column_name]
            if column_value == 0:
                no_counter += 1
            elif column_value == 1:
                yes_counter += 1
            elif np.isnan(column_value):
                missing_counter += 1

        valid_total = no_counter + yes_counter
        total_rows = valid_total + missing_counter

        no_percentage = no_counter / total_rows
        yes_percentage = yes_counter / total_rows
        missing_percentage = missing_counter / total_rows

        valid_no_percentage = no_counter / valid_total
        valid_yes_percentage = yes_counter / valid_total

        print(no_counter)
        print(yes_counter)
        print(missing_counter)
        print(valid_total)
        print(total_rows)

        print(no_percentage)
        print(yes_percentage)
        print(missing_percentage)

        print(valid_no_percentage)
        print(valid_yes_percentage)


if __name__ == "__main__":
    data_preprocessor = Data_Preprocessor(missing_data='drop')
    logistic_regression_obj = Logistic_Regression(data_preprocessor)
    logistic_regression_obj.calc_freq_table('a14')
    logistic_regression_obj.calc_logistic_regression_stat()
