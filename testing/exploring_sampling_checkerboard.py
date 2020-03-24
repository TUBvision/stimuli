#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 11:55:20 2020

@author: guille
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_context('talk')
sns.set_style('ticks')

# %% sampling scheme simulation

reflectances = np.arange(13).astype(int)
n = 9
sample_repeat = 9


def algo1(reflectances, n, sample_repeat):
    # algorithm as it is now implemented by Max
    # build n x n matrix with values drawn from reflectances with at most >sample_repeat< repeats of each value
    draw_set = np.repeat(reflectances, sample_repeat)
    board = np.random.choice(draw_set, (n, n), replace=False)
    return board


def algo2(reflectances, n):
    # simply sampling with replacement
    board = np.random.choice(reflectances, n*n, replace=True).reshape((n,n))
    return board


nsim = 10000

sim1 = np.zeros((nsim, n*n))
sim2 = np.zeros((nsim, n*n))

for i in range(nsim):

    sim1[i,:] = algo1(reflectances, n, sample_repeat=sample_repeat).flatten()
    sim2[i,:] = algo2(reflectances, n).flatten()


sim1 = sim1.astype(int)
sim2 = sim2.astype(int)
    
# %%

def calculate_frequency(sim):
    freq = np.zeros((nsim, len(reflectances)))
    for i in range(nsim):
        for j, v in enumerate(reflectances):
            freq[i, j] = np.sum(sim[i,:]==v)
    return freq
        
freq1 = calculate_frequency(sim1)
freq2 = calculate_frequency(sim2)


# %%
expectedmeanvalue = n*n / len(reflectances)

plt.figure()
plt.plot(freq1.mean(axis=0), 'o-', label='algo1')
plt.plot(freq2.mean(axis=0), 'o-', label='algo2')
plt.hlines(y=expectedmeanvalue, xmin=0, xmax=len(reflectances))
plt.legend(frameon=False)
plt.ylim(0.9*expectedmeanvalue, 1.1*expectedmeanvalue) # +- 10%
plt.xlabel('value')
plt.ylabel('mean frequency')
plt.title('frequency of each value, mean across sim')
sns.despine()


plt.figure()
plt.plot(freq1.std(axis=0), 'o-', label='algo1')
plt.plot(freq2.std(axis=0), 'o-', label='algo2')
plt.legend(frameon=False)
plt.xlabel('value')
plt.ylabel('standard deviation')
plt.title('frequency of each value, std across sim')
sns.despine()


# it seems that algorithm1 (using a given draw set that is a multiple of the 
# vector, and sampling from it without replacement) gives a smaller variability 
# in the frequency of selected values than
# algorithm 2, which is simply drawing with replacement). I remember I explored
# the reason for this years ago, but now I can't remember the theoretical reason
# why it is like this.

# So, better stay with algorithm 1