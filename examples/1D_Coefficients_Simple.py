#!/usr/bin/env python3
"""
A simple example of extracting only a simple set of 1D finite difference coefficients.
Applied to u=sin(x) with true and calculated derivatives are summarized in 1D_Coefficients_Simple.png
"""

import os
import sys

import numpy as np
from matplotlib import pyplot as plt

from finitedelta import get_coef1d

def main():
    # How many derivatives to calculate
    max_order = 4

    # Sample grid and calculate u(x) = sin(x)
    h = np.pi/50
    x = np.arange(-200,201)*h
    u = np.sin(x)

    # Get finite difference coefficients based on uniform 5-point stencil for each order
    D = {}
    for k in range(max_order):
        D[k] = get_coef1d([-2*h,-h,0,h,2*h], k+1)

    
    # For each order 1<=k<=4, plot true k^th derivative and finite difference approximation vs. t
    fig = plt.figure(figsize=(8,6))
    for k in range(max_order):
        ax = fig.add_subplot(3,2,k+1)
        du_true = ((1j**(k+1)).real)*np.sin(x) + (-(1j**(k+2)).real)*np.cos(x)
        ax.plot(x,du_true,lw=0.75,c='k')
    
        du_guess = np.correlate(u,D[k],'same')
        ax.scatter(x[2:-2],du_guess[2:-2], s=3, c='r')
        if k==0:
            ax.set_title("$du$ / $dx$")
        else:
            ax.set_title("$d^{"+str(k+1)+"}u$ / $dx^{"+str(k+1)+"}$")
    
        ax.set_xlim([-4.2*np.pi,4.2*np.pi]); ax.set_ylim([-1.2,1.2])
        ax.set_xticks([-4*np.pi,-2*np.pi,0,2*np.pi,4*np.pi])
        ax.set_xticklabels(["$-4\\pi$", "$-2\\pi$", "$0$", "$2\\pi$", "$4\\pi$"])
    
    plt.tight_layout()

    os.system("mkdir -p tmp")
    fig.savefig("tmp/1D_Coefficients_Simple.png", dpi=400)

    return 0

if __name__ == "__main__":
    sys.exit(main())
