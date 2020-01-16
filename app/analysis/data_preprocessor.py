# Author: Junbong Jang
# Date: 1/23/2019
# app/data_preprocessor.py

import pandas as pd

# note that when connected with ALI, columns of the data frame will be 'object'
class Data_Preprocessor(object):
    def __init__(self,
             filename="../uploads/Dataset1.csv",
             covariate_columns=[],
             x_columns=['rct1', 'rdt1', 'a11'],
             y_columns=['a14'],
             categorical_columns=['a01'],
             missing_data='drop'):

        self.cov_columns = covariate_columns
        self.x_columns = x_columns
        self.y_columns = y_columns
        self.categorical_columns = categorical_columns
        self.all_columns = covariate_columns + x_columns + y_columns
        # ------------- Data Preprocessing -------------------

        self.input_df = pd.read_csv(filename)

#        self.handle_missing_data()

        self.drop_missing_data(missing_data)
        # self.convert_to_correct_type()
        self.handle_categorical_columns(categorical_columns)
        self.drop_missing_data(missing_data)
        # ------------- Variables Defined -------------------
        self.X = self.input_df[self.x_columns]
        self.y = self.input_df[self.y_columns]
        self.feature_df = self.input_df[self.x_columns + self.y_columns]

    def convert_to_correct_type(self):
        # all the items that are not categorical
        for column in [item for item in self.all_columns if item not in self.categorical_columns]:
            self.input_df[column] = pd.to_numeric(self.input_df[column])

    def handle_categorical_columns(self, categorical_columns):
        for column_name in categorical_columns:
            self.input_df.loc[:, column_name] = self.input_df.loc[:, column_name].map({'3': 3, '2': 2, '1': 1, '0': 0})


    def handle_missing_data(self):
        # retain only alphanumeric characters
        for a_column in self.all_columns:
            self.input_df[a_column].str.replace('\W', '')
            self.input_df = self.input_df[self.input_df[a_column] != '']

        # ALI doc report marks null value with -
        for a_column in self.all_columns:
            self.input_df = self.input_df[self.input_df[a_column] != '-']

    def drop_missing_data(self, missing_data):
        # print(input_df.isnull().any()) # check for missing values
        if missing_data == 'drop':
            self.input_df.dropna(inplace=True)
        elif missing_data == 'fill':
            print('fill')

