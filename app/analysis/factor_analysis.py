# Author: Junbong Jang
# Date: 2/6/2018
# app/factor_analysis.py
from analysis.data_preprocessor import Data_Preprocessor
import statsmodels.multivariate.factor as sm
import numpy as np
import pandas as pd


class Factor_Analysis(object):

    def __init__(self, data_preprocessor_in):
        self.data = data_preprocessor_in

    def run_factor(self, rotate_option='varimax'):
        factor = sm.Factor(self.data.X, 35)
        factor_model = factor.fit()
        factor_model.rotate(rotate_option)
        print(factor_model.summary())
        factor_model.plot_scree()
        import matplotlib.pyplot as plt
        plt.show()


    # https://factor-analyzer.readthedocs.io/en/latest/_modules/factor_analyzer/factor_analyzer.html
    def covariance_to_correlation(self, m):
        """
        This is a port of the R `cov2cor` function.

        Parameters
        ----------
        m : numpy array
            The covariance matrix.

        Returns
        -------
        retval : numpy array
            The cross-correlation matrix.

        Raises
        ------
        ValueError
            If the input matrix is not square.
        """

        # make sure the matrix is square
        numrows, numcols = m.shape
        if not numrows == numcols:
            raise ValueError('Input matrix must be square')

        Is = np.sqrt(1 / np.diag(m))
        retval = Is * m * np.repeat(Is, numrows).reshape(numrows, numrows)
        np.fill_diagonal(retval, 1.0)
        return retval

    def partial_correlations(self, data):
        """
        This is a python port of the `pcor` function implemented in
        the `ppcor` R package, which computes partial correlations
        of each pair of variables in the given data frame `data`,
        excluding all other variables.

        Parameters
        ----------
        data : pd.DataFrame
            Data frame containing the feature values.

        Returns
        -------
        df_pcor : pd.DataFrame
            Data frame containing the partial correlations of of each
            pair of variables in the given data frame `df`,
            excluding all other variables.
        """
        numrows, numcols = data.shape
        df_cov = data.cov()
        columns = df_cov.columns

        # return a matrix of nans if the number of columns is
        # greater than the number of rows. When the ncol == nrows
        # we get the degenerate matrix with 1 only. It is not meaningful
        # to compute partial correlations when ncol > nrows.

        # create empty array for when we cannot compute the
        # matrix inversion
        empty_array = np.empty((len(columns), len(columns)))
        empty_array[:] = np.nan
        if numcols > numrows:
            icvx = empty_array
        else:
            # we also return nans if there is singularity in the data
            # (e.g. all human scores are the same)
            try:
                icvx = np.linalg.inv(df_cov)
            except np.linalg.LinAlgError:
                icvx = empty_array
        pcor = -1 * self.covariance_to_correlation(icvx)
        np.fill_diagonal(pcor, 1.0)
        df_pcor = pd.DataFrame(pcor, columns=columns, index=columns)
        return df_pcor

    def calculate_kmo(self, data):
        """
        Calculate the Kaiser-Meyer-Olkin criterion
        for items and overall. This statistic represents
        the degree to which each observed variable is
        predicted, without error, by the other variables
        in the dataset. In general, a KMO < 0.6 is considered
        inadequate.

        Parameters
        ----------
        data : pd.DataFrame
            The data frame from which to calculate KMOs.

        Returns
        -------
        kmo_per_variable : pd.DataFrame
            The KMO score per item.
        kmo_total : float
            The KMO score overall.
        """

        # calculate the partial correlations
        partial_corr = self.partial_correlations(data)
        partial_corr = partial_corr.values

        # calcualte the pair-wise correlations
        corr = data.corr()
        corr = corr.values

        # fill matrix diagonals with zeros
        # and square all elements
        np.fill_diagonal(corr, 0)
        np.fill_diagonal(partial_corr, 0)

        partial_corr = partial_corr ** 2
        corr = corr ** 2

        # calculate KMO per item
        partial_corr_sum = partial_corr.sum(0)
        corr_sum = corr.sum(0)
        kmo_per_item = corr_sum / (corr_sum + partial_corr_sum)
        kmo_per_item = pd.DataFrame(kmo_per_item,
                                    index=data.columns,
                                    columns=['KMO'])

        # calculate KMO overall
        corr_sum_total = corr.sum()
        partial_corr_sum_total = partial_corr.sum()
        kmo_total = corr_sum_total / (corr_sum_total + partial_corr_sum_total)
        return kmo_per_item, kmo_total


if __name__ == "__main__":
    # homework 6
    data_preprocessor = Data_Preprocessor(missing_data='drop',
                                          filename="Dataset2.csv",
                                          predict_column=['subt1'],
                                          x_columns=['am01', 'am02', 'am03', 'am04', 'am05', 'am06', 'am07', 'am08', 'am09',
                                                     'con01', 'con03', 'con05', 'con09', 'con12', 'con13', 'con14', 'con15', 'con17',
                                                     'cp10', 'cp11', 'cp12', 'cp13', 'cp15', 'cp02', 'cp07', 'cp08',
                                                     'dom01', 'dom02', 'dom03', 'dom04', 'dom05', 'dom06', 'dom07', 'dom08', 'dom09'],
                                          categorical_columns=['a01', 'a08'])
    factor_analysis_obj = Factor_Analysis(data_preprocessor)
    print(factor_analysis_obj.calculate_kmo(factor_analysis_obj.data.X))
    factor_analysis_obj.run_factor('varimax')
