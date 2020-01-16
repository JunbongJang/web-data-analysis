# Author: Junbong Jang
# Date: 11/29/2018
# app/apa_formater.py


def corr_matrix_apa(multi_reg):
    corr_values_2d, corr_p_2d = multi_reg.calc_zero_order_corr()
    for row_index, row in enumerate(corr_values_2d):
        for col_index, elem in enumerate(row):
            if row_index == col_index and row_index != 0:
                corr_values_2d[row_index][col_index] = '--'
            elif col_index > row_index and row_index != 0:
                corr_values_2d[row_index][col_index] = ''
            else:
                if corr_p_2d[row_index][col_index] < 0.001:
                    corr_values_2d[row_index][col_index] = str(corr_values_2d[row_index][col_index]) + '**'
                elif corr_p_2d[row_index][col_index] < 0.05:
                    corr_values_2d[row_index][col_index] = str(corr_values_2d[row_index][col_index]) + '*'

    return corr_values_2d