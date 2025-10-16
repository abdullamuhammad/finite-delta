#!/usr/bin/env python3
"""
Simulates advection PDE via Lax-Wendroff method.
Uses custom parameters and finitedelta to calculate finite difference coefficients.
Simulation is saved as ./tmp/Lax_Wendroff_Method_Advection.mp4
"""

import os
import sys

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation

from finitedelta import grid_handler1d

def main():
    # Set up spatial grid and sampling
    Length_total = 12
    Freq_spatial = 20
    h_spatial = 1.0/Freq_spatial
    x_grid = np.arange(-(Length_total//2)*Freq_spatial,(Length_total//2)*Freq_spatial+1)*h_spatial
    
    # Set up temporal grid and sampling
    Time_total = 4
    Frames_per_second = 100
    Freq_sample_minimum = 1000
    
    Iterations_per_frame = int(np.ceil(Freq_sample_minimum/Frames_per_second))
    Freq_sample = Frames_per_second*Iterations_per_frame
    h_sample = 1.0/Freq_sample
    
    # Parameters for system: u_t - alpha * u_x = 0
    params = {'alpha': -1.5, 'mu': -3, 'width': 2}
    
    # Build first and second derivative finite difference coefficients in matrix form
    D_x, D_xx = grid_handler1d(
        (1,2), 
        accuracy=4, 
        num_samples=len(x_grid), 
        h=h_spatial, 
        mode='uniform'
    )
    
    # Put step in sparse matrix form using Lax-Wendroff method
    Lax_Wendroff_Step = np.eye(len(x_grid)) ## 1. Start with previous state
    Lax_Wendroff_Step += params['alpha']*h_sample*D_x ## 2. Add first-order term
    Lax_Wendroff_Step += (params['alpha']**2)*(h_sample**2)*D_xx ## 3. Add second-order term
    
    # Simulate system at Freq_spatial and save frames at Frames_per_second
    fig, ax = plt.subplots()
    
    # Set up and plot initial conditions
    U = np.exp(-(params['width']*(x_grid - params['mu']))**2)
    ax.set(
        xlim=[min(x_grid), max(x_grid)], 
        ylim = [-0.1,1.1], 
        xlabel = 'Length', 
        ylabel = 'Density', 
        title = 't = '+f"{0:.2f}"+" sec."
    )
    
    line = ax.plot(x_grid, U, lw=1, c='b')[0]
    
    # Define frame update function
    def update(frame):
        for step in range(Iterations_per_frame):
            U[:] = Lax_Wendroff_Step@U
        
        line.set_ydata(U)
        ax.set_title('t = '+f"{Iterations_per_frame*frame*h_sample:.2f}"+" sec.")
        
        return line, ax
    
    # Build animation
    os.system("mkdir -p tmp")
    anim = FuncAnimation(
        fig=fig, 
        func=update, 
        frames=Time_total*Frames_per_second+1, 
        interval=1000/Frames_per_second
    )
    
    anim.save(
        filename="tmp/Lax_Wendroff_Method_Advection.mp4",
        writer="ffmpeg",
        fps=Frames_per_second
    )

    return 0

if __name__ == "__main__":
    sys.exit(main())
