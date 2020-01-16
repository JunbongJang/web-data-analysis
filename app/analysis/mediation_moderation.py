# Author: Junbong Jang
# Date: 1/23/2018
# mediation_moderation.py

import statsmodels.api as sm
import statsmodels.genmod.families.links as links
from statsmodels.stats.mediation import Mediation
import pandas as pd
from analysis.data_preprocessor import Data_Preprocessor


class Mediation_Moderation(object):

    def __init__(self, data_preprocessor_in):
        self.data = data_preprocessor_in

    def run_mediation(self, predict_col_categorical=False):
        if predict_col_categorical:
            probit = links.probit
            outcome_model = sm.GLM.from_formula("cont1 ~ rdt1 + jelt1", self.data.X, family=sm.families.Binomial(link=probit()))
        else:
            outcome_model = sm.GLM.from_formula("cont1 ~ rdt1 + jelt1", self.data.X)

        mediator_model = sm.OLS.from_formula("rdt1 ~ jelt1", self.data.X)
        med = Mediation(outcome_model, mediator_model, "jelt1", mediator="rdt1").fit()
        with pd.option_context('display.max_rows', None, 'display.max_columns', None):
            print(med.summary())

if __name__ == "__main__":
    # homework 3
    data_preprocessor = Data_Preprocessor(missing_data='drop',
                                          filename="Dataset1.csv",
                                          predict_column=['rdt1'],
                                          x_columns=['rdt1', 'jelt1', 'cont1'])
    mediation_obj = Mediation_Moderation(data_preprocessor)
    mediation_obj.run_mediation()
