# finite-delta

## About this library

The goal of this repository is to provide a simple Python framework for generating finite difference coefficients. 
During my research, when needing to generate matrices that would apply the finite difference coefficients as a linear operator onto data, I found myself frequently using this wonderful [website](https://web.media.mit.edu/~crtaylor/calculator.html) \[1\].
I figured developing a Python framework to programmatically generate coefficients would help me learn a lot about both numerical PDEs and the best practices for developing code.
After starting this library, I discovered a [similar one](https://github.com/bjodah/finitediff) \[2\], which is much more computationally optimized than this one.
Nonetheless, I thought this would be a fun personal project, where I could throw in a few extra features and demos I personally found interesting.

## How to use this package

0. **Prerequisites**

    1. (Optional) Set up Python Virtual Environment

       The "best practice" is to set up virtual environment built specifically for numerical methods usecases. On most Linux systems, this is done via terminal:

       ```
       python3 -m venv ./my_virtual_env
       ```
    
       To enter the virtual environment (which will install Python packages seperately from your main system), use:
       ```
       source ./my_virtual_env/bin/activate
       ```

    2. (Optional) Install required packages
  
       The only prerequisite package for using the most basic form of finitedelta is NumPy. However, SciPy is used for representing the linear transformations corresponding to finite difference coefficient approximations as sparse matrices. Furthermore, installation is easiest when cloning directly from github, which requires git. Additionally, SymPy and Matplotlib are used in some of the examples. While not explicitly required to install these, I recommend it for certain usecases.

       ```
       pip install git scipy sympy matplotlib
       ```

1. **Installation**

    Currently, the easiest way to install the package is to clone the repository and use pip. In the future, the package will be made available on PyPI to make this process even simpler.

    1. Clone the repository

       ```
       git clone https://github.com/abdullamuhammad/finite-delta.git
       ```

   2. Go into directory and pip install.

      I recommend installing in editable mode, which allows the source code (in /finite-delta/src/finitedelta/) to be edited without requiring pip installation every time to update functions.

      ```
      cd finite-delta
      pip install -e .
      ```

2. **Usage**

    1. _Ex. Getting finite difference coefficients for simple 1D stencil_
    
       Set a step size and sample locations relative to a central point; **finitedelta.get_coef1d** will give back the finite difference coefficients!
  
       ```
       import numpy as np
       from finitedelta import get_coef1d


       # Get 3rd derivative coefficients with a 5-point stencil sampled with step size 0.1
       h = 0.1
       stencil = np.array([-2*h, -1*h, 0, h, 2*h])
       coef = get_coef1d(stencil, 3)
       print(coef)
       ```

   2. _Ex. Identify linear operator of finite difference coefficients to solve PDE numerically_

      See /examples/Lax_Wendroff_Method_Advection.py for a simulation of the 1D Advection PDE: $\frac{\partial u}{\partial t} = \alpha \frac{\partial u}{\partial x}$. Solving for $u(x,t)$ while maintaining numerical stability requires estimations of both $\frac{\partial u}{\partial x}, \frac{\partial^2 u(x,t)}{\partial x^2}$ (i.e. the [Lax-Wendroff Scheme](https://en.wikipedia.org/wiki/Lax%E2%80%93Wendroff_method) \[3\].) . Using **finitedelta.grid_handler1d** returns linear operators $D_x, D_{xx}$ which approximate these partial derivatives using the method of finite differences. Running the code:
  
      ```
      ./examples/Lax_Wendroff_Method_Advection.py
      ```
      
      Generates the file /examples/tmp/Lax_Wendroff_Method_Advection.mp4 which shows a video of the solution numerical solution evolving over time.

   2. _Ex. Getting finite difference coefficients for mixed partial derivatives_
  
      Estimating general mixed partial derivatives order $k$ for a function $u(t,x_1,\ldots,x_n)$ is possible using **finitedelta.get_coefnd** (for coefficients corresponding to a stencil) or **finitedelta.grid_handlernd** (for generating a linear operator as a scipy.sparse.csr_matrix object.) In general, these methods become computationally intractable as $n$ increases; however, some use-cases exist. To my knowledge, this is the only Python package that generates finite differences coefficients for estimating of high-order mixed partial derivatives numerically. For linear PDEs, this is irrelevant since coordinate changes can be used to factor out mixed partials. However, there are usecases for certain non-linear systems.

      As a proof-of-concept, see the code in /examples/3D_Random_Function.py. This generates a random function $u(x,y,z)$ using SymPy, estimates three randomly selected mixed partials, and gives back a plot comparing numerically estimated vs. true values in  /examples/tmp/3D_Random_Function.png
      

## References

1. Taylor, C.R. (2016). *Finite Difference Coefficients Calculator* [https://web.media.mit.edu/~crtaylor/calculator.html](https://web.media.mit.edu/~crtaylor/calculator.html)
2. Dahlgren, B.I. (2021). *finitediff* [https://github.com/bjodah/finitediff](https://github.com/bjodah/finitediff).
3. [https://en.wikipedia.org/wiki/Lax-Wendroff_method](https://en.wikipedia.org/wiki/Lax%E2%80%93Wendroff_method)
