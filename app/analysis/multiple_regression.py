# Author: Junbong Jang
# Date: 11/6/2018
# app/multiple_regression.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math

from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.api as sm  # import statsmodels

from scipy import stats, linalg
from scipy.stats.mstats import zscore

from analysis.data_preprocessor import Data_Preprocessor


class Multiple_Regression(object):

    # when not running flask, change the path to 'uploads/Dataset1.csv'
    def __init__(self, data_preprocessor):
        self.data = data_preprocessor

    def calc_multiple_regression_stat(self):
        print('---------------Correlation Matrix----------------')
        partial_corr = self.calc_partial_corr(self.data.feature_df)[1]
        partial_corr.pop(0)
        # print('----------- Multicollinearity Test -------------')
        vif_df = self.calc_vif(self.data.X)
        # tolerance_df = vif_df['VIF Factor'].rdiv(1)
        multicol_list = []
        param_names = []
        for index, row in vif_df.iterrows():
            param_names.append(row['features'])
            multicol_list.append(("{0:.3f}".format(row['VIF Factor']), "{0:.3f}".format(1 / row['VIF Factor'])))

        # print('----------- Linear Regression ------------')
        # self.scikit_linear_regression(X, y)
        ols_model, model_ss, residual_ss, total_ss = self.ols_linear_regression(self.data.X, self.data.y)
        standardized_model, model_ss2, residual_ss2, total_ss2 = self.ols_linear_regression(zscore(self.data.X),
                                                                                            zscore(self.data.y))

        model_stat_dict = {
            'model_ss': model_ss.round(3),
            'residual_ss': residual_ss.round(3),
            'total_ss': total_ss.round(3),
            'model_mse': ols_model.mse_model.round(3),
            'residual_mse': ols_model.mse_resid.round(3),
            'total_mse': ols_model.mse_total.round(3),
            'fvalue': ols_model.fvalue.round(3),  # F-statistic of the fully specified model
            'pvalue': "{0:.3f}".format(ols_model.f_pvalue),  # p-value of the F-statistic
            'model_df': "{0:.0f}".format(ols_model.df_model),
            'residual_df': "{0:.0f}".format(ols_model.df_resid),
            'total_df': "{0:.0f}".format(ols_model.df_model + ols_model.df_resid),
            'rvalue': "{0:.3f}".format(math.sqrt(ols_model.rsquared)),
            'rsquared': ols_model.rsquared.round(3),
            'rsquared_adj': ols_model.rsquared_adj.round(3),
        }
        coefficients_dict = {
            'param_names': param_names,
            'unstandardized_beta': Multiple_Regression.round_list(ols_model.params.tolist()),
            'bse': Multiple_Regression.round_list( ols_model.bse.tolist()),
            'standardized_beta': Multiple_Regression.round_list(standardized_model.params.tolist()),
            'tvalues': Multiple_Regression.round_list(ols_model.tvalues.tolist()),
            'pvalues': Multiple_Regression.round_list(ols_model.pvalues.tolist()),
            'zero_order_corr': self.calc_zero_order_corr()[0][0],
            'partial_corr': Multiple_Regression.round_list(partial_corr),
            'semi_partial_corr': '',
            'multicol_list': multicol_list
        }
        return model_stat_dict, coefficients_dict

    @staticmethod
    def print_statistics(ols_model, standardized_model):
        # https://www.statsmodels.org/dev/generated/statsmodels.regression.linear_model.RegressionResults.html
        print('---------------------OLS Statistics-------------------')
        print(ols_model.summary())
        print(ols_model.mse_model)
        print(ols_model.mse_resid)
        print(ols_model.mse_total)
        print(ols_model.fvalue)
        print(ols_model.f_pvalue)
        print('--------------- Coefficients ----------------')
        print(ols_model.params)  # coefficients
        print(ols_model.bse)  # The standard errors of the parameter estimates.
        print(standardized_model.params)  # standardized coefficients
        print(ols_model.tvalues)  # t-test statistics on coefficients
        print(ols_model.pvalues)  # coefficient's p-value

    @staticmethod
    def round_list(a_list):
        new_list = []
        for value in a_list:
            new_list.append("{0:.3f}".format(value))
        return new_list

    def ols_linear_regression(self, X, y):
        X = sm.add_constant(X)
        ols_model = sm.OLS(y, X).fit()
        model_ss, residual_ss, total_ss = self.calc_ss(ols_model, X, self.data.y)

        return ols_model, model_ss, residual_ss, total_ss

    def scikit_linear_regression(self, X, y):
        from sklearn import linear_model
        lm = linear_model.LinearRegression()
        scikit_model = lm.fit(X, y)
        scikit_predictions = lm.predict(X)

    def calc_ss(self, ols_model, X, y):
        ols_predictions = np.array(ols_model.predict(X).tolist())
        predictions = np.reshape(ols_predictions, (-1, 1))

        # sum of squared
        model_ss = sum(np.array
                       ([(prediction - y.values.mean()) ** 2 for prediction in predictions]))
        residual_ss = sum((y.values - predictions) ** 2)
        total_ss = model_ss + residual_ss

        model_mse = model_ss / ols_model.df_model
        residual_mse = residual_ss / ols_model.df_resid

        # F is the ratio of the Model Mean Square to the Error Mean Square
        f_value = model_mse / residual_mse

        return model_ss[0], residual_ss[0], total_ss[0]

    def calc_vif(self, given_df):
        """
        https://etav.github.io/python/vif_factor_python.html
        The Variance Inflation Factor (VIF) is a measure of colinearity
        among predictor variables within a multiple regression.

        :param given_df:
        :return:
        """
        from statsmodels.tools.tools import add_constant
        given_df = add_constant(given_df)
        vif = pd.DataFrame()
        vif["features"] = given_df.columns
        vif["VIF Factor"] = [variance_inflation_factor(given_df.values, i) for i in range(given_df.shape[1])]
        return vif

    def calc_partial_corr(self, C):
        """
        https://stats.stackexchange.com/questions/288273/partial-correlation-in-panda-dataframe-python
        Returns the sample linear partial correlation coefficients between pairs of variables in C, controlling
        for the remaining variables in C.
        Parameters
        ----------
        C : array-like, shape (n, p)
            Array with the different variables. Each column of C is taken as a variable
        Returns
        -------
        P : array-like, shape (p, p)
            P[i, j] contains the partial correlation of C[:, i] and C[:, j] controlling
            for the remaining variables in C.
        """
        C = sm.add_constant(C)
        C = np.asarray(C)
        p = C.shape[1]
        P_corr = np.zeros((p, p), dtype=np.float)
        for i in range(p):
            P_corr[i, i] = 1
            for j in range(i + 1, p):
                idx = np.ones(p, dtype=np.bool)
                idx[i] = False
                idx[j] = False
                beta_i = linalg.lstsq(C[:, idx], C[:, j])[0]
                beta_j = linalg.lstsq(C[:, idx], C[:, i])[0]

                res_j = C[:, j] - C[:, idx].dot(beta_i)
                res_i = C[:, i] - C[:, idx].dot(beta_j)

                corr = stats.pearsonr(res_i, res_j)[0]
                P_corr[i, j] = corr
                P_corr[j, i] = corr

        return P_corr.tolist()

    def draw_correlation_matrix(self, filename):
        """

        :return:
        """
        # get pearson's r and p values
        corr_matrix = self.data.feature_df.corr()

        # Plot Correlational Matrix heat map table
        fig, ax = plt.subplots()
        # Using matshow here just because it sets the ticks up nicely. imshow is faster.
        ax.matshow(corr_matrix, cmap='seismic')
        plt.xticks(range(len(self.data.feature_df.columns)), self.data.feature_df.columns)
        plt.yticks(range(len(self.data.feature_df.columns)), self.data.feature_df.columns)
        plt.title('Correlational Matrix')

        for (i, j), z in np.ndenumerate(corr_matrix):
            ax.text(j, i, '{:0.2f}'.format(z), ha='center', va='center',
                    bbox=dict(boxstyle='round', facecolor='white', edgecolor='0.3'))
        # plt.show()
        fig.savefig('app/static/dynamic_img/correlation_'+filename+'.png')  # save the figure to file
        plt.close(fig)

    def calc_zero_order_corr(self):
        feature_df_rows, feature_df_cols = self.data.feature_df.shape
        import scipy.stats as ss

        corr_list = [ss.pearsonr(self.data.feature_df.values[:, i], self.data.feature_df.values[:, j])
                     for i in range(feature_df_cols)
                     for j in range(feature_df_cols)]
        correlation_values = np.transpose(np.array(corr_list))[0].round(3)
        correlation_p = np.transpose(np.array(corr_list))[1].round(5)

        rows = correlation_values.shape
        # I want correlation values in 2d array
        corr_values_2d = correlation_values.reshape(int(math.sqrt(rows[0])), int(math.sqrt(rows[0])))
        corr_p_2d = correlation_p.reshape(int(math.sqrt(rows[0])), int(math.sqrt(rows[0])))

        return corr_values_2d.tolist(), corr_p_2d.tolist()  # converted to list so that jinja2 iterate it in for loop

    def calc_descriptive_dict(self):
        descriptive_dict = [{"name": column_name,
                             'mean': round(self.data.feature_df.mean().loc[column_name], 3),
                             'std': round(self.data.feature_df.std().loc[column_name], 3),
                             'count': round(self.data.feature_df.count().loc[column_name])} for column_name in
                            self.data.all_columns]

        return descriptive_dict


if __name__ == "__main__":
    data_preprocessor = Data_Preprocessor(missing_data='drop',
        x_columns = ['amt1', 'jelt1', 'subt1', 'cpt1', 'rdt1', 'a01'],
        y_columns = ['cont1'], categorical_columns=['a01'])
    multiple_regression_obj = Multiple_Regression(data_preprocessor)
    multiple_regression_obj.calc_multiple_regression_stat()
