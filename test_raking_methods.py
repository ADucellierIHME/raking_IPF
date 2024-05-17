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

# Test entropic distance
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
(result_direct, lambda_direct) = raking_entropic_distance(x, q, A, y, 1.0, 500, False, True)
(result_full, lambda_full) = raking_entropic_distance(x, q, A, y, 1.0, 500, False, False)

# Compare values between two raking methods
print('Entropic distance, diff mu = ', np.sum(np.abs(result_direct - result_full)))
print('Entropic distance, diff lambda = ', np.sum(np.abs(lambda_direct - lambda_full)))

# Test general distance
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
(result_direct, lambda_direct) = raking_general_distance(-0.5, x, q, A, y, 1.0, 500, False, True)
(result_full, lambda_full) = raking_general_distance(-0.5, x, q, A, y, 1.0, 500, False, False)

# Compare values between two raking methods
print('General distance, diff mu = ', np.sum(np.abs(result_direct - result_full)))
print('General distance, diff lambda = ', np.sum(np.abs(lambda_direct - lambda_full)))

# Test logit raking
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
l = np.zeros(len(x))
h = df['pop'].to_numpy()
(A, y) = get_margin_matrix_vector(v_i, v_j, mu_i, mu_j)
(result_direct, lambda_direct) = raking_logit(x, l, h, q, A, y, 1.0, 500, False, True)
(result_full, lambda_full) = raking_logit(x, l, h, q, A, y, 1.0, 500, False, False)

# Compare values between two raking methods
print('Logit raking, diff mu = ', np.sum(np.abs(result_direct - result_full)))
print('Logit raking, diff lambda = ', np.sum(np.abs(lambda_direct - lambda_full)))

