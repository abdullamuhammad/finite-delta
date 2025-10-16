#!/usr/bin/env python3
"""
Generates random function u(x,y,z) with SymPy and approximated partial derivatives.
Comparison of true and calculated values is saved as ./tmp/3D_Random_Function.png
"""

import os
import sys
import math
import itertools

import numpy as np
import sympy as sp
from matplotlib import pyplot as plt

from finitedelta.grid_handlernd import grid_handlernd
from finitedelta.get_partials import get_partials

# Custom function to take in a tuple of SymPy symbols and a desired order, 
# and generate a random polynomial.
def generate_poly(x_sym, order):
    polynomial = np.random.uniform(-5,5)
    for k in range(1,order+1):
        terms = get_partials(len(x_sym), k)
        for term in terms:
            new_poly = np.random.uniform(-1,1)
            for index in range(len(x_sym)):
                power = term[index]
                for iteration in range(power):
                    new_poly *= x_sym[index]

            polynomial += new_poly

    return polynomial

def main():
    dim = 3 # Number of variables of function u(x_1, ..., x_dim)
    h = np.random.uniform(0.005,0.04,dim) # Step-size for each axis
    num_samples_half = np.random.randint(15,30,dim) # Number of samples for each half-axis
    num_neighbors = 3 # Number of neighbors to include in each direction for stencil
    
    # For each dimension i, sample uniformly at distance h[i]
    X_samples = [np.arange(-num_samples_half[i], num_samples_half[i]+1)*h[i] for i in range(dim)]
    
    # Samples at every combination of variables from X_samples and stores in (N,dim) array
    X_list = np.array([samples_per_dim for samples_per_dim in itertools.product(*X_samples)])
    X_mesh = [X_list[:,i].reshape(*[len(samples) for samples in X_samples]) for i in range(dim)]

    # Takes index of meshgrid and maps to index of list
    mesh_to_index = np.arange(len(X_list)).reshape(X_mesh[0].shape)
    
    # Takes indes of list and maps to index of meshgrid
    index_to_mesh = {
        int(mesh_to_index[i]): i for i in itertools.product(
            *[range(axis_length) for axis_length in mesh_to_index.shape]
        )
    }

    # Builds stencil object that meets specifications. The total number of points is N, where
    # (N,dim) is the shape of X_list. The stencil variable must be a list of length N, where 
    # the i^th entry is a list of the indices of neighbors of point i used to estimate partial
    # derivative at i.
    # In this case, we simply account for the nearest neighbors across each axis.

    stencils = []
    for i_list in range(len(index_to_mesh)):
        i_mesh = index_to_mesh[i_list]
    
        stencil_new = [mesh_to_index[n] for n in itertools.product(*[range(
            max(0,i_mesh[i] - num_neighbors),
            min(mesh_to_index.shape[i], i_mesh[i] + num_neighbors + 1)
        ) for i in range(dim)])]
    
        stencils.append(stencil_new)

    # Generate all partial derivatives of order 4, and randomly choose 3.
    all_partials = get_partials(dim, order = 4)
    partials = [all_partials[i] for i in np.random.choice(len(all_partials),3,False)]

    # Use grid_handlernd to generate sparse matrices to estimate respective partials using custom stencil.
    D = grid_handlernd(
        dim=dim, 
        order = None,
        partials = partials,
        samples = X_list,
        stencil = stencils,
        tol = 1e-8
    )

    # Use SymPy to generate 3 variables in 'x_sym' and a custom function u(x_sym[0], ..., x_sym[dim-1]).
    x_sym = sp.symbols(f'x0:{dim}')
    u = generate_poly(x_sym, 5) 
    u += np.random.normal(0,1)*sp.cos(generate_poly(x_sym, 2))
    u += np.random.normal(0,1)*sp.sin(generate_poly(x_sym, 2))

    # Use SymPy's lambdify function to convert symbolic function into NumPy function and apply to samples.
    u_np = sp.lambdify(x_sym, u, 'numpy')
    u_true = u_np(*[X_list[:,i] for i in range(dim)])

    # For each partial derivative, generate symbolic functions 'du', NumPy functions 'du_np', and apply 
    # functions to samples. This is the True partial derivative.
    du = {}; du_np = {}; du_true = {}
    for partial in partials:
        partial_sym = []
        for i in range(dim):
            for iteration in range(partial[i]):
                partial_sym.append(x_sym[i])
    
        du[partial] = sp.diff(u, *partial_sym)
        du_np[partial] = sp.lambdify(x_sym, du[partial], 'numpy')
        du_true[partial] = du_np[partial](*[X_list[:,i] for i in range(dim)])

    # Use finite difference coefficients in 'D' to approximate partial derivatives numerically.
    du_guess = {}
    for partial in partials:
        du_guess[partial] = D[partial]@u_true

    # Generate figure to compare true and numerical results
    fig = plt.figure(figsize=(9.5,3))
    for ind,partial in enumerate(partials):
        ax = fig.add_subplot(1,3,ind+1)
        ax.scatter(du_true[partial], du_guess[partial], s=0.25, c='r', alpha=0.05)
        ax.plot(
            [min(min(du_true[partial]),min(du_guess[partial])),
             max(max(du_true[partial]),max(du_guess[partial]))],
            [min(min(du_true[partial]),min(du_guess[partial])),
             max(max(du_true[partial]),max(du_guess[partial]))],
            lw = 1, c='k', ls='--'
        )
        title_str = "($\\partial^4 u$) / ("
        ind_to_letter = {0: 'x', 1: 'y', 2: 'z'}
        for ind,partial_order in enumerate(partial):
            if partial_order > 1:
                title_str += "$\\partial " + ind_to_letter[ind] + "^" + str(partial_order) + "$"
            elif partial_order == 1:
                title_str += "$\\partial " + ind_to_letter[ind] + "$"
    
        title_str += ")"
        ax.set_title(title_str)
        ax.set_xlabel("True Value")
        ax.set_ylabel("Calculated Value")
        ax.set_aspect('equal', 'box')
        R2 = 1 - np.std(du_guess[partial] - du_true[partial])/np.std(du_true[partial])
        R2_text = f"{R2:.3f}"
        ax.text(0.95, 0.05,
                '$R^2 ='+R2_text+'$',
                transform=ax.transAxes, 
                va='bottom', ha='right')
    
    
    plt.tight_layout()
    os.system("mkdir -p tmp")
    fig.savefig("tmp/3D_Random_Function.png", dpi=400)

    return 0

if __name__ == "__main__":
    sys.exit(main())
