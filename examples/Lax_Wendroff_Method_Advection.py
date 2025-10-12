import os
import numpy as np

from matplotlib import pyplot as plt
import matplotlib.animation as animation

from finitedelta.grid_handler1d import grid_handler1d

# Set up spatial grid and sampling
Length_total = 12
Freq_spatial = 20
h_spatial = 1.0/Freq_spatial
x_grid = np.arange(-(Length_total//2)*Freq_spatial,(Length_total//2)*Freq_spatial+1)*h_spatial

# Set up temporal grid and sampling
Time_total = 4
Freq_sample = 1000
h_sample = 1.0/Freq_sample
t_grid = np.arange(0,Time_total*Freq_sample+1)*h_sample

# Parameters for system: u_t - alpha * u_x = 0
params = {'alpha': -1.5, 'mu': -3, 'width': 2}

# Set up initial conditions
U = np.exp(-(params['width']*(x_grid - params['mu']))**2)

# Build first and second derivative finite difference coefficients in matrix form
D_x,D_xx = grid_handler1d([1,2], accuracy=4, num_samples=len(x_grid), 
                          h=h_spatial, mode='uniform')

# Put step in sparse matrix form using Lax-Wendroff method
Lax_Wendroff_Step = np.eye(len(x_grid)) ## 1. Start with previous state
Lax_Wendroff_Step += params['alpha']*h_sample*D_x ## 2. Add first-order term
Lax_Wendroff_Step += (params['alpha']**2)*(h_sample**2)*D_xx ## 3. Add second-order term

fig, ax = plt.subplots()

ax.set(
    xlim=[min(x_grid), max(x_grid)], 
    ylim = [-0.1,1.1], 
    xlabel = 'Length', 
    ylabel = 'Density', 
    title = 't = '+f"{0.0:.3f}"+" sec."
)

line = ax.plot(x_grid, U, c='blue', lw=1)[0]