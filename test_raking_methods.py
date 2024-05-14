"""
This Python script is to test the raking methods
"""

import numpy as np
import pandas as pd

from raking_methods import get_margin_matrix_vector
from raking_methods import raking_chi2_distance
from raking_methods import raking_l2_distance
from raking_methods import raking_entropic_distance
from raking_methods import raking_general_distance
from raking_methods import raking_logit

pd.options.mode.chained_assignment = None

# Read dataset
df = pd.read_excel('../test_raking_data/2D_raking_example.xlsx', nrows=16)
df.drop(columns=['Unnamed: 0'], inplace=True)

# Format dataset
df.rename(columns={'acause': 'cause', \
                   'parent_value': 'all_cause_value', \
                   'mcnty_value': 'all_race_value'}, \
    inplace=True)
df['value'] = df['value'] * df['pop']
df['all_cause_value'] = df['all_cause_value'] * df['pop']
total_pop = np.sum(df['pop'].unique())
df['all_race_value'] = df['all_race_value'] * total_pop
df.drop(columns=['parent', 'wt', 'level', 'pop_weight'], inplace=True)

# Test chi2 distance
# ------------------

# Rake by race and cause using both functions
I = len(df['cause'].unique())
J = len(df['race'].unique())
mu_i = df['all_race_value'].unique()
mu_j = df['all_cause_value'].unique()
v_i = np.ones(J)
v_j = np.ones(I)
x = df['value'].to_numpy()
q = np.ones(len(x))
(A, y) = get_margin_matrix_vector(v_i, v_j, mu_i, mu_j)
(result_direct, lambda_direct) = raking_chi2_distance(x, q, A, y, True)
(result_full, lambda_full) = raking_chi2_distance(x, q, A, y, False)

# Compare values between two raking methods
print('Chi2 distance, diff mu = ', np.sum(np.abs(result_direct - result_full)))
print('Chi2 distance, diff lambda = ', np.sum(np.abs(lambda_direct - lambda_full)))

# Test l2 distance
# ------------------

# Rake by race and cause using both functions
I = len(df['cause'].unique())
J = len(df['race'].unique())
mu_i = df['all_race_value'].unique()
mu_j = df['all_cause_value'].unique()
v_i = np.ones(J)
v_j = np.ones(I)
x = df['value'].to_numpy()
q = np.ones(len(x))
(A, y) = get_margin_matrix_vector(v_i, v_j, mu_i, mu_j)
(result_direct, lambda_direct) = raking_l2_distance(x, q, A, y, True)
(result_full, lambda_full) = raking_l2_distance(x, q, A, y, False)

# Compare values between two raking methods
print('L2 distance, diff mu = ', np.sum(np.abs(result_direct - result_full)))
print('L2 distance, diff lambda = ', np.sum(np.abs(lambda_direct - lambda_full)))

