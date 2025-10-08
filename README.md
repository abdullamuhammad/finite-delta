# finite-delta

### About this library

The goal of this repository is to provide a simple Python framework for generating finite difference coefficients. 
The finite difference method is particularly useful for numerically solving problems related to PDEs, optimization, and control.
I frequently use the coefficients to numerically approximate kinematics from raw experimental data, and to perform regularization in optimization and control problems where the cost depends on derivatives.
The idea for the Python library was inspired by the [website](https://web.media.mit.edu/~crtaylor/calculator.html) developed by Dr. Chris Taylor \[1\], which I used quite a bit in my research.
Shortly after starting this library, I discovered a [similar one](https://github.com/bjodah/finitediff) developed by Björn Dahlgren \[2\], which is much more computationally optimized than this one.
Nonetheless, I thought this was a fun personal project, where I could throw in a few extra features and demos I personally thought were interesting to include.

### References

1. Taylor, C.R. (2016). *Finite Difference Coefficients Calculator* [https://web.media.mit.edu/~crtaylor/calculator.html](https://web.media.mit.edu/~crtaylor/calculator.html)
2. Dahlgren, B.I. (2021). *finitediff* [https://github.com/bjodah/finitediff](https://github.com/bjodah/finitediff).
