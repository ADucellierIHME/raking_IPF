"""
This Python script is to test the 
"""

import numpy as np
import pandas as pd

from raking_methods import get_margin_matrix_vector
from raking_methods import raking_l2_distance, raking_vectorized_l2_distance
from raking_methods import raking_general_distance, raking_vectorized_IPF

pd.options.mode.chained_assignment = None

# Read dataset
df = pd.read_excel('../data/2D_raking_example.xlsx', nrows=16)
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

# Copy dataset
df1 = df.copy()
df1['year'] = 2016
df1['value'] = 1.1 * df1['value']
df1['all_cause_value'] = 1.1 * df1['all_cause_value']
df1['all_race_value'] = 1.1 * df1['all_race_value']
df2 = df.copy()
df2['age'] = 25
df2['value'] = 1.2 * df2['value']
df2['all_cause_value'] = 1.2 * df2['all_cause_value']
df2['all_race_value'] = 1.2 * df2['all_race_value']
df = pd.concat([df, df1, df2])

constant_vars = ['mcnty', 'year', 'age', 'sex', 'sim']
agg_vars = ['race', 'cause']

# Test l2 distance
# For each year, rake by race and cause using initial function
years = df['year'].unique().tolist()
ages = df['age'].unique().tolist()
df_raked = []
for year in years:
    for age in ages:
        df_sub = df.loc[(df['year'] == year) & (df['age'] == age)]
        if len(df_sub) > 0:
            I = len(df_sub['cause'].unique())
            J = len(df_sub['race'].unique())
            mu_i = df_sub['all_race_value'].unique()
            mu_j = df_sub['all_cause_value'].unique()
            v_i = np.ones(J)
            v_j = np.ones(I)
            x = df_sub['value'].to_numpy()
            q = np.ones(len(x))
            (A, y) = get_margin_matrix_vector(v_i, v_j, mu_i, mu_j)
            mu = raking_l2_distance(x, q, A, y)
            df_sub['value_raked'] = mu
            df_raked.append(df_sub)
df_raked = pd.concat(df_raked)
df_raked.sort_values(by=constant_vars + agg_vars, inplace=True)

# Rake by race and cause using vectorized function
df_raked_vector = raking_vectorized_l2_distance(df, agg_vars, constant_vars)
df_raked_vector.sort_values(by=constant_vars + agg_vars, inplace=True)

# Compare values between two raking methods
print('L2 distance, rake by race and cause, diff = ', \
    np.sum(np.abs(df_raked['value_raked'].to_numpy() - df_raked_vector['value_raked'].to_numpy())))

# Test entropic distance
# For each year, rake by race and cause using initial function
years = df['year'].unique().tolist()
ages = df['age'].unique().tolist()
df_raked = []
for year in years:
    for age in ages:
        df_sub = df.loc[(df['year'] == year) & (df['age'] == age)]
        if len(df_sub) > 0:
            I = len(df_sub['cause'].unique())
            J = len(df_sub['race'].unique())
            mu_i = df_sub['all_race_value'].unique()
            mu_j = df_sub['all_cause_value'].unique()
            v_i = np.ones(J)
            v_j = np.ones(I)
            x = df_sub['value'].to_numpy()
            q = np.ones(len(x))
            (A, y) = get_margin_matrix_vector(v_i, v_j, mu_i, mu_j)
            mu = raking_general_distance(0, x, q, A, y)
            df_sub['value_raked'] = mu
            df_raked.append(df_sub)
df_raked = pd.concat(df_raked)
df_raked.sort_values(by=constant_vars + agg_vars, inplace=True)

# Rake by race and cause using vectorized function
df_raked_vector = raking_vectorized_IPF(df, agg_vars, constant_vars)
df_raked_vector.sort_values(by=constant_vars + agg_vars, inplace=True)

# Compare values between two raking methods
print('Entropic distance, rake by race and cause, diff = ', \
    np.sum(np.abs(df_raked['value_raked'].to_numpy() - df_raked_vector['value_raked'].to_numpy())))
