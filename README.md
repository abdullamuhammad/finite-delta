# finite-delta

### About this library

The goal of this repository is to provide a simple Python framework for generating finite difference coefficients. 
During my research, when needing to generate matrices that would apply the finite difference coefficients as a linear operator onto data, I found myself frequently using this wonderful [website](https://web.media.mit.edu/~crtaylor/calculator.html) \[1\].
I figured developing a Python framework to programmatically generate coefficients would help me learn a lot about both numerical PDEs and the best practices for developing code.
After starting this library, I discovered a [similar one](https://github.com/bjodah/finitediff) \[2\], which is much more computationally optimized than this one.
Nonetheless, I thought this would be a fun personal project, where I could throw in a few extra features and demos I personally found interesting.

### References

1. Taylor, C.R. (2016). *Finite Difference Coefficients Calculator* [https://web.media.mit.edu/~crtaylor/calculator.html](https://web.media.mit.edu/~crtaylor/calculator.html)
2. Dahlgren, B.I. (2021). *finitediff* [https://github.com/bjodah/finitediff](https://github.com/bjodah/finitediff).
