import numpy as np
from matplotlib import pyplot as plt
import matplotlib.ticker as ticker
import math as m
import scipy.constants as constants
from numpy import random
import timeit
from numba import jit
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

"Set initial spins for NxN lattice"
@jit(nopython=True)
def set_microstate_2d(N, p, spin):
    E = 0.
    for x in range(num_rows+2):
        for y in range(N):
            if random.rand(1) < p:
                spin[x,y] = -1
                E -= spin[x,y] * (spin[x,y-1] + spin[x,(y+1)%N] + spin[x-1,y] + spin[(x+1)%xMax,y]) # energy update
    return E, spin

"Metropolis algorithm to flip random spins"
@jit(nopython=True)
def metropolis_algorithm_2d(spin, N, kBT, E):
    # generate a random number from length of spin chain
    flip_x, flip_y = random.randint(0, num_rows+2), random.randint(0, N)
    
    # calculate the energy difference (with periodic boundary condition)
    delta_E = 2 * ep * spin[flip_x,flip_y] * (spin[flip_x-1,flip_y] +  spin[(flip_x+1)%xMax,flip_y] + spin[flip_x,flip_y-1] + spin[flip_x, (flip_y+1)%N])
    
    # decide if spin gets flipped
    if (delta_E < 0. or random.rand() < np.exp(-delta_E/kBT)):
        spin[flip_x,flip_y] = -spin[flip_x,flip_y]
        E += delta_E
    return E, spin


"Boundary communication between processes"
def boundary_communication(array):
    requests = []
    if rank > 0:
        # top boundary
        req = comm.Isend(array[1, :], dest=rank-1, tag=1)
        requests.append(req)
        req = comm.Irecv(array[0,:], source=rank-1, tag=2)
        requests.append(req)
    
    if rank < size - 1:
        # bottom boundary
        req = comm.Irecv(array[-1,:], source=rank+1, tag=1)
        requests.append(req)
        req = comm.Isend(array[-2, :], dest=rank+1, tag=2)
        requests.append(req)
        
    MPI.Request.Waitall(requests)


"2D Ising model with plotting"
def Ising_model_2d(N, p, iterations, kBT):

    # set initial microstate {si}, using some order parameter 
    E, spin_lattice = set_microstate_2d(N, p, local_lattice)
    
    # set up empty arrays
    spin_lattice_steps = np.zeros((iterations, num_rows+2, N))
    E_steps = np.zeros((iterations))
    
    # iterate and plot
    for step in range(iterations):
        boundary_communication(spin_lattice) # boundary communication
        
        E, spin_lattice = metropolis_algorithm_2d(spin_lattice, N, kBT, E) # call metropolis algorithm
        
        spin_lattice_steps[step,:,:] = spin_lattice 
        E_steps[step] = E
        
        # plot if iteration is in stamp_list
        if (step in stamp_list):
            final_lattice = np.empty((N,N), dtype=np.float64)  # full array to gather into for one iteration
            comm.Gather(spin_lattice[1:-1,:], final_lattice, root=0) # gather all processes  
            
            if (rank==0):
                # plotting if rank is zero
                fig, ax = plt.subplots(figsize = (6,4))
                plt.pcolormesh(final_lattice)
                ax.set_title(f'Step {step}')
                ax.set_xlabel('Cells (x)')
                ax.set_ylabel('Cells (y)')
                ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%d'))
                plt.gca().set_aspect('equal')
                plt.tight_layout()
                plt.savefig(f"./iter_{step}.pdf", dpi=300)
  
    return spin_lattice_steps, E_steps


ep = 1 # epsilon
kBT = 2
N = 20 # number of spins in spin chain
p = 0.6 # spin up probability 

# split along y axis
num_rows = N // size
iterations = N**4
xMax = num_rows+2

"2d Array"
full_lattice = np.ones((N+2*size,N), dtype=np.float64)

"Local Array for Each Process"
local_lattice = np.empty((num_rows+2,N), dtype=np.float64) # array for lattice in each process

comm.Scatter(full_lattice, local_lattice, root=0) # scatter lattice to all processes

stamp_list = [0,int(iterations/50),int(iterations/25),int(iterations/5)] # iterations to plot

start = timeit.default_timer()
spin_chain_steps, E_steps = Ising_model_2d(N, p, iterations, kBT) # call Ising model
end = timeit.default_timer()
if (rank == 0):
    print("Timer (including plotting):", end-start)
    


    
    