#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  4 13:37:25 2024

@author: janecohen

1D Ising model to simulate the evolution of a chain of spins.
Includes the comparaison of the numerical simulation to analytical results
using a Monte Carlo average of energy, magnetization and entropy equilbirum values. 
"""

#%% imports and functions

import numpy as np
from numpy import random
import matplotlib.pyplot as plt
from numba import jit
import timeit
import scipy.constants as constants


"Set initial spins for a N long chain using parameter p"
@jit(nopython=True)
def set_microstate_1d(N, p):
    spin = np.ones(N) # lattice of all spin up
    E = 0. 
    for i in range(0,N):
        if np.random.rand(1) < p:
            spin[i] = -1 # flip spin
        E = E-spin[i-1] * spin[i] # energy

    E = E - spin[N-1] * spin[0] # periodic bc inclusion
    return spin, E 

"Metropolis algorithm to flip a random spin"
@jit(nopython=True)
def metropolis_algorithm_1d(spin, N, kBT, E):
    # generate a random number from length of spin chain
    flip_index = random.randint(0, N-1)
    
    #calculate the energy difference (with periodic boundary condition)
    delta_E = 2 * ep * spin[flip_index] * (spin[flip_index-1] +  spin[(flip_index+1)%N])
    
    # check if the spin gets flipped
    if (delta_E < 0) or (random.rand(1)< np.exp(-delta_E/kBT)):
        spin[flip_index] = -spin[flip_index] # flip spin
        E += delta_E # update energy
    return E


"1D Ising model"
@jit(nopython=True)
def Ising_model_1d(N, p, iterations, kBT):

    # set initial microstate {si}, using some order parameter 
    spin_chain, E = set_microstate_1d(N, p)
    
    # set up empty arrays
    spin_chain_steps = np.zeros((iterations, N))
    E_steps = np.zeros((iterations))
    
    # iterate
    for step in np.arange(0, iterations, 1):
        E = metropolis_algorithm_1d(spin_chain, N, kBT, E)
        spin_chain_steps[step,:] = spin_chain
        E_steps[step] = E
        
    return spin_chain_steps, E_steps

"Compute analytic energy value"
@jit(nopython=True)
def energy_analytic(N, kBT):
    return -N*ep*np.tanh(ep*1/kBT)

"Compute numerical magnetization value"
@jit(nopython=True)
def magnetization_numerical(spin_list, Nmc, N):
    M_sum = 0
    for c in range(1,Nmc): # iterate through chains
        M_sum += np.sum(spin_list[c])
    return M_sum / Nmc
    
"Compute analytic entropy value"
@jit(nopython=True)
def entropy_analytical(N, kBT):
    kB = constants.k
    S = N*(np.log(2*np.cosh(1/kBT*ep)) - 1/kBT*ep*np.tanh(1/kBT*ep))
    return S/N

"Calculate Monte Carlo equilibrium values for a range of KTs"
@jit(nopython=True)
def equilbirum_values(test_num, KT_max, KT_min, N, p, iterations, num_equi, num_mc):
    
    # empty arrays
    energies, energies_an  = np.zeros((test_num)), np.zeros((test_num))
    magnetizations, magnetizations_an = np.zeros((test_num)), np.zeros((test_num))
    entropies, entropies_an = np.zeros((test_num+1)), np.zeros((test_num))
    
    # range of KT values
    KT_values = KT_min + (KT_max - KT_min)*np.arange(test_num)/(test_num-1)
    
    # iterate through KT values
    for i in range(0,test_num):
        spin_chain_steps, E_steps = Ising_model_1d(N, p, iterations, KT_values[i]) # run Ising model
        energies[i] = (np.sum(E_steps[num_equi:]) / num_equi) / N # equilbirum energy 
        energies_an[i] = energy_analytic(N, KT_values[i]) / N # analytical energy
        magnetizations[i] = magnetization_numerical(spin_chain_steps[num_equi:], num_mc, N) / N # equilbirum magnetization 
        magnetizations_an[i] = 0 # analytical magnetization
        if (i != 0):
            entropies[i] = entropies[i-1] + (energies[i] - energies[i-1]) / KT_values[i] # equilbirum entropy
        entropies_an[i] = entropy_analytical(N, KT_values[i]) # analytical entropy
    
    return KT_values, energies, energies_an, magnetizations, magnetizations_an, entropies, entropies_an

"Plot results for spins and energy of a N long chain"
def plot_1d_results(spins, energy, avg_energy, N, kBT):
    
    iters = np.arange(0,iterations,1)
    
    plt.rcParams.update({'font.size': 20})
    plt.rcParams['figure.dpi']= 120
    fig, axs = plt.subplots(nrows=2, sharex=True, figsize=(7, 6))

    # spin colormesh
    axs[0].pcolormesh(spins)
    axs[0].set_ylabel('Spin N') 
    axs[0].set_ylim(0,49)
    
    # energy plot
    iters_scaled = iters
    energy_scaled = energy/N/ep
    axs[1].plot(iters_scaled, energy_scaled, label='E')
    axs[1].set_ylabel('E/$\epsilon$N') 
    axs[1].axhline(avg_energy/N, color='r', label=r'$\langle$E$\rangle$ an')
    
    # x ticks
    x_tick_locs = np.arange(iters.min(), iters.max() + 1, 20*N)  # generate tick locations
    x_tick_labels = (x_tick_locs / N).astype(int) # calculate tick labels
    axs[1].set_xticks(x_tick_locs)
    axs[1].set_xticklabels(x_tick_labels)
 
    # common x-label for the bottom subplot
    axs[1].set_xlabel('Iteration/N')
    
    # common title for the entire figure
    axs[0].set_title(f'Spin evolution with KT = {kBT}')
    
    axs[1].legend()
    plt.tight_layout()
    plt.show()

"General plotting function"
def plot_general(x, y_num, y_an, title, xlabel, ylabel):
    fig, ax = plt.subplots(figsize = (8,6))
    plt.scatter(x, y_num, label='Numerical')
    plt.plot(x, y_an, color='orange', label='Analytical')
    plt.xlabel(f"{xlabel}")
    plt.ylabel(f"{ylabel}")
    plt.title(f"{title}")
    plt.tight_layout()
    plt.show()
            
    
#%% Q1(a) 1d Ising model

"Comparing bath temperatures"

ep = 1 # epsilon
p = 0.6 # spin down probability 
N = 50 # number of spins in spin chain
KT_List = [0.1, 0.5, 1] # kBT values to test
iterations = N*100 # number of iterations

# KT loop
for kBT in KT_List:
    spin_chain_steps, E_steps = Ising_model_1d(N, p, iterations, kBT) # run 1D Ising model
    avg_energy = np.sum(E_steps) / iterations # average energy 
    plot_1d_results(spin_chain_steps.transpose(), E_steps, avg_energy, N, kBT) # plot results

print("A warm start (most spins start up) takes longer to reach equilibrium than a cold start. For a warm bath temperature (high KT), it takes longer to evolve to equilibirum, but is more stable once it reaches it.")

#%% Q1(b) 

"Monte Carlo equilbirum values"

show_plots = True

# evolved to equilibrium using 400N iterations.
# averaged again over another 400N iterations.

N = 100 # number of spins in spin chain
num_equi  = N*400 # number of iterations to evolve to equilibrium
num_mc = N*400 # number of Monte Carlo samplings
iterations = num_equi + num_mc # total number of iterations

test_num = 200 # number of test values
KT_min = 0.1 # minimum KT value
KT_max = 6 # maximum KT value

# calculate equilibirum values
KTs, E, E_an, M, M_an, S, S_an = equilbirum_values(test_num, KT_max, KT_min, N, p, iterations, num_equi, num_mc) 

if (show_plots == 1):
    plot_general(KTs, E, E_an, 'Energy vs. temperature', 'k$_B$T/$\epsilon$', r'$\langle E \rangle$/N$\epsilon$') # plot energies
    plot_general(KTs, M, M_an, 'Magnetization vs. temperature', 'k$_B$T/$\epsilon$', r'$\langle M \rangle$/N') # plot magnetization
    plot_general(KTs, S[:-1], S_an, 'Entropy vs. temperature', 'k$_B$T/$\epsilon$', r'$\langle S \rangle$/Nk') # plot entropy
    
print("In general, the numerical simulations agree with the analytical solutions. The entropy has a random vertical offset from the analytical curve, which the cause of could not be determined")

        