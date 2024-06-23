#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 13 19:27:22 2024

@author: janecohen

2D Ising model to simulate the evolution of a 2D lattice of spins.
Includes the comparaison of the numerical simulation to analytical results
using a Monte Carlo average of energy, magnetization and entropy equilbirum values. 
"""

#%% imports and functions

import numpy as np
from numpy import random
import matplotlib.pyplot as plt
from numba import jit, prange
import timeit
import scipy.constants as constants

ep = 1. # epsilon

"Set initial spins for a NxN lattice using parameter p"
@jit(nopython=True)
def set_microstate_2d(N, p):
    spin = np.ones((N,N)) # initial lattice
    E = 0.
    
    for x in range(N):
        for y in range(N):
            if random.rand(1) < p:
                spin[x,y] = -1
                E -= spin[x,y] * (spin[x,y-1] + spin[x,(y+1)%N] + spin[x-1,y] + spin[(x+1)%N,y]) # energy update
    return spin, E

"Metropolis algorithm to flip a random spin"
@jit(nopython=True)
def metropolis_algorithm_2d(spin, N, kBT, E):
    # generate a random number from length of spin chain
    flip_x, flip_y = random.randint(0, N), random.randint(0, N)
    
    #calculate the energy difference (with periodic boundary condition)
    delta_E = 2 * ep * spin[flip_x,flip_y] * (spin[flip_x-1,flip_y] +  spin[(flip_x+1)%N,flip_y] + spin[flip_x,flip_y-1] + spin[flip_x, (flip_y+1)%N])
    
    # decide if spin is flipped
    if (delta_E < 0. or random.rand() < np.exp(-delta_E/kBT)):
        spin[flip_x,flip_y] = -spin[flip_x,flip_y]
        E += delta_E
    return spin,E

"2D Ising model"
@jit(nopython=True)
def ising_model_2D(iterations, kBT, N, p):
    # empty arrays
    snapshots = np.zeros((iterations,N,N), dtype=np.int8)
    energies = np.zeros(iterations)

    # set initial microstate using order parameter p
    lattice, E = set_microstate_2d(N,p)
    
    # iterate 
    for i in range(iterations):
        lattice, E = metropolis_algorithm_2d(lattice, N, kBT, E)
        snapshots[i] = lattice
        energies[i] = E
        
    return snapshots, energies

"Compute numerical magnetization"
@jit(nopython=True)
def magnetization_numerical(spin_list, Nmc, N):
    M_sum = 0
    for c in range(1,Nmc): # iterate through lattices
        M_sum += np.sum(spin_list[c])
    return M_sum / Nmc

"Calculate Monte Carlo equilibrium values for a range of KTs"
@jit(nopython=True, parallel=True)
def equilbirum_values(test_num, KT_max, KT_min, N, p, iterations, num_equi, num_mc):
    
    # empty arrays
    energies = np.zeros((test_num))
    magnetizations =  np.zeros((test_num))
    
    # range of KT values
    KT_values = KT_min + (KT_max - KT_min)*np.arange(test_num)/(test_num-1)
    
    # iterate through range of KT values
    for i in prange(0,test_num):
        lattices, energy = ising_model_2D(iterations, KT_values[i], N, p)
        energies[i] = (np.sum(energy[num_equi:]) / num_equi) / N**2 # equilbirum energy
        magnetizations[i] = (np.sum(lattices[num_equi:]) / num_mc ) / N**2 # equilibirum magnetization
    
    return KT_values, energies, magnetizations

"Plot results for of 2D spin lattice for four steps"
def plot_2d_results(spin_list, energy, N, kBT, snaps, title):
    
    plt.rcParams.update({'font.size': 16})
    plt.rcParams['figure.dpi']= 120
    fig, axs = plt.subplots(nrows=2, ncols=2, sharex=True, figsize=(6, 8))

    # plot spins on each subplot
    axs[0,0].pcolormesh(spin_list[snaps[0]])
    axs[0,0].set_ylim(0,N)
    
    axs[0,1].pcolormesh(spin_list[snaps[1]])
    axs[0,1].set_ylim(0,N)
    
    axs[1,0].pcolormesh(spin_list[snaps[2]])
    axs[1,0].set_ylim(0,N)

    axs[1,1].pcolormesh(spin_list[snaps[3]])
    axs[1,1].set_ylim(0,N)
    
    # x and y labels
    axs[0,0].set_ylabel('Cells (y)')
    axs[1,0].set_ylabel('Cells (y)')
    axs[1,0].set_xlabel('Cells (x)')
    axs[1,1].set_xlabel('Cells (x)')
    
    # titles for each subplot
    axs[0,0].set_title(f'Iter = {snaps[0]}')
    axs[0,1].set_title(f'Iter = {snaps[1]}')
    axs[1,0].set_title(f'Iter = {snaps[2]}')
    axs[1,1].set_title(f'Iter = {snaps[3]}')
    
    plt.suptitle(f"KT = {title}")
    plt.tight_layout() 
    plt.show()
    
"General plotting function"
def plot_general(x, y_num, title, xlabel, ylabel):
    fig, ax = plt.subplots(figsize = (8,6))
    plt.scatter(x, y_num, label='Numerical')
    plt.xlabel(f"{xlabel}")
    plt.ylabel(f"{ylabel}")
    plt.title(f"{title}")
    plt.tight_layout()
    plt.show()



#%%

"Evolution to equilbirum -  change kT to observe the effect of different temperatures"
N = 20 # size of lattice (N x N)
p = .6 # probability of flip 
kBT = 2 # KT value to test

iterations = N**4

start = timeit.default_timer()
lattices, energies = ising_model_2D(iterations, kBT, N, p)
end = timeit.default_timer()
print("Timer:", end-start)

snaps = [0,int(iterations/50),int(iterations/25),int(iterations/5)] # iterations to plot
plot_2d_results(lattices, energies, N, kBT, snaps, kBT)

#%%

"Monte Carlo average of equilibrirum energy and magnetization"
N = 20 # size of lattice (N x N)
p = .6
test_num = 70 # number of KT test values
num_equi = N**4 # number of iterations to evolve to equilibrium
num_mc = N**4 # number of Monte Carlo samplings
iterations = num_equi+num_mc
KT_min = 0.1 # minimum KT value
KT_max = 3 # maximum KT value
KTc = 2*ep / (np.log(1+np.sqrt(2))) # critical temperature

KTs, E, M = equilbirum_values(test_num, KT_max, KT_min, N, p, iterations, num_equi, num_mc)
    
# plot equilbirum values
plot_general(KTs / KTc, E,'Energy vs. temperature', 'k$_B$T/$\epsilon$T$_c$', r'$\langle E \rangle$/N$^2$$\epsilon$') # plot energies
plot_general(KTs / KTc, np.abs(M),  'Magnetization vs. temperature', 'k$_B$T/$\epsilon$T$_c$', r'|$\langle M \rangle$|/N$^2$$\mu$') # plot magnetization

print("The plots show consistency with the normalized critical temperature of 1")



#%%



    
    
    