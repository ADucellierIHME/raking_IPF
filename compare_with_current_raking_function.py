"""
This Python script is to compare the current raking function from
the LSAE Engineering team with our own raking code
"""

import numpy as np
import pandas as pd

from run_experiment import IPF

# Read dataset
df = pd.read_excel('../data/2D_raking_example.xlsx', nrows=16)
df.drop(columns=['Unnamed: 0'], inplace=True)

# Get raked values by race and cause
df_race_cause = pd.read_csv('../data/raked_race_cause.csv')

# Mutiply by population to get total number of cases
df['total_value'] = df['value'] * df['pop']
df['total_parent_value'] = df['parent_value'] * df['pop']
total_pop = np.sum(df['pop'].unique())
df['total_mcnty_value'] = df['mcnty_value'] * total_pop

# Verify that margins add up to the same value
print('Sum over all races of all causes deaths: ', \
      np.sum(df['total_parent_value'].unique()))
print('Sum over all causes of all races deaths: ', \
      np.sum(df['total_mcnty_value'].unique()))

# Rake by cause and race
n = len(df['acause'].unique())
m = len(df['race'].unique())
x = np.reshape(df['total_value'].to_numpy(), (n, m), order='F')
mu_i = df['total_mcnty_value'].unique()
mu_j = df['total_parent_value'].unique()
x_raked = IPF(x, mu_i, mu_j)[0]
x_raked = np.reshape(x_raked, (n * m, 1), order='F')

# Divide by population to get prevalence
df['raked_values'] = x_raked[:, 0] / df['pop']

# Check if the raked values add up to the margin for each race
for race in df['race'].unique().tolist():
    df_sub = df.loc[df['race'] == race]
    print('race ', race, ' - difference = ', \
        abs(np.sum(df_sub['raked_values'].to_numpy())- \
        df_sub['parent_value'].iloc[0]))

# Check if the raked values add up to the margin for each cause
for cause in df['acause'].unique().tolist():
    df_sub = df.loc[df['acause'] == cause]
    print('cause ', cause, ' - difference = ', \
        abs(np.sum(df_sub['raked_values'].to_numpy() * \
        df_sub['pop'].to_numpy()) - df_sub['total_mcnty_value'].iloc[0]))

# Check if the difference between the current raking function
# and our own raking code is 0
diff = np.sum(np.abs(df_race_cause['value'].to_numpy() - \
                     df['raked_values'].to_numpy()))
if diff < 1.0e-10:
    print('We get the same results when raking by race and cause.')
else:
    print('We get different results when raking by race and cause.')

