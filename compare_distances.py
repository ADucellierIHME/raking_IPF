"""
Python scripts to comapre the raking methods
with different distances
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from raking_methods import get_margin_matrix_vector, raking_chi2_distance, raking_general_distance
from run_experiment import IPF

# Read dataset
df = pd.read_excel('../data/2D_raking_example.xlsx', nrows=16)
df.drop(columns=['Unnamed: 0'], inplace=True)

# Mutiply by population to get total number of cases
df['total_value'] = df['value'] * df['pop']
df['total_parent_value'] = df['parent_value'] * df['pop']
total_pop = np.sum(df['pop'].unique())
df['total_mcnty_value'] = df['mcnty_value'] * total_pop

# Prepare linear constraints to rake by cause and race
I = len(df['acause'].unique())
J = len(df['race'].unique())
mu_i = df['total_mcnty_value'].unique()
mu_j = df['total_parent_value'].unique()
v_i = np.ones(J)
v_j = np.ones(I)

# Prepare data and weights for the raking
x = df['total_value'].to_numpy()
#q = np.ones(len(x))
q = 1.0 / x

# Raking using IPF
x_raked = np.reshape(df['total_value'].to_numpy(), (I, J), order='F')
x_raked = IPF(x_raked, mu_i, mu_j)[0]
x_raked = np.reshape(x_raked, (I * J, 1), order='F')

# Divide by population to get prevalence
df['raked_IPF'] = x_raked[:, 0] / df['pop']

# Choose values for alpha
alphas = [1, -0.5, -1, -2]

# Define names of raking methods
names = ['alpha = 1',
#         'alpha = 0',
         'alpha = -1/2',
         'alpha = -1',
         'alpha = -2']

# Rake using different distances
(A, y) = get_margin_matrix_vector(v_i, v_j, mu_i, mu_j)
mu = np.zeros((len(x), len(alphas)))
for index, alpha in enumerate(alphas):
    print('alpha = ', alpha)
    mu[:, index] = raking_general_distance(alpha, x, q, A, y)

# Divide by population to get prevalence
for index, name in enumerate(names):
    df[name] = mu[:, index] / df['pop']

# Check if the raked values add up to the margin for each race
#for race in df['race'].unique().tolist():
#    df_sub = df.loc[df['race'] == race]
#    print('race ', race, ' - difference = ', \
#        abs(np.sum(df_sub['raked_general'].to_numpy())- \
#        df_sub['parent_value'].iloc[0]))

# Check if the raked values add up to the margin for each cause
#for cause in df['acause'].unique().tolist():
#    df_sub = df.loc[df['acause'] == cause]
#    print('cause ', cause, ' - difference = ', \
#        abs(np.sum(df_sub['raked_general'].to_numpy() * \
#        df_sub['pop'].to_numpy()) - df_sub['total_mcnty_value'].iloc[0]))

# Plot
plt.figure(figsize=(12, 6))
plt.scatter(np.arange(0, I * J), df['raked_IPF'].to_numpy(), color='black', marker='o', label='IPF')
plt.scatter(np.arange(0, I * J), df['alpha = 1'].to_numpy(), color='blue', marker='o', label='alpha = 1')
#plt.scatter(np.arange(0, I * J), df['alpha = 0'].to_numpy(), color='green', marker='o', label='alpha = 0')
plt.scatter(np.arange(0, I * J), df['alpha = -1/2'].to_numpy(), color='yellow', marker='o', label='alpha = -1/2')
plt.scatter(np.arange(0, I * J), df['alpha = -1'].to_numpy(), color='orange', marker='o', label='alpha = -1')
plt.scatter(np.arange(0, I * J), df['alpha = -2'].to_numpy(), color='red', marker='o', label='alpha = -2')
ticks = np.arange(0, I * J).tolist()
tick_labels_race = []
before = int((I - 1) / 2)
after = I - 1 - before
for race in df['race'].unique():
    label = ['\n\n'] * before + ['\n\n ' + str(race)] + ['\n\n'] * after
    tick_labels_race = tick_labels_race + label
tick_labels_cause = []
for cause in df['acause'].tolist():
    tick_labels_cause.append(cause[1:])
tick_labels = new_labels = [''.join(x) for x in zip(tick_labels_cause, tick_labels_race)]
plt.xticks(ticks, tick_labels, fontsize=16)
plt.yticks(fontsize=16)
plt.title('Comparison of raked observations with different distances', fontsize=24)
plt.xlabel('Cause and race', fontsize=20)
plt.ylabel('Mortality rate', fontsize=20)
plt.legend(fontsize=16, frameon=False)
plt.tight_layout()
plt.savefig('compare_distances.png')

