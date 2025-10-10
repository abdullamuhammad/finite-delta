import numpy as np

def get_coef1d(stencil, order, return_all = False):
    """
    Inputs:
    stencil -   a list describing unique locations of neighboring samples used in the 
                finite differences scheme, relative to location at which derivative is
                being estimated (i.e. an entry of 0 indicates where the estimation 
                takes place
    order -     a non-negative integer describing which order derivative to estimate
    
    return_all - boolean that determines whether matrix of all finite difference
                 coefficients is returned

    Outputs:
    coef -      a numpy array the same size as stencil, where the element coef[i] indicates
                the weight applied to a sample at relative location stencil[i] to estimate
                the derivative
    """

    # Raise exceptions if...

    ## 'order' is not integer
    if not isinstance(order, (int, np.integer)):
        raise TypeError(f"Input 'order' must be of integer type, not {type(order).__name__}.")

    ## 'order' is negative
    if order<0:
        raise ValueError("Input 'order' must be non-negative integer.")

    ## 'stencil' does not specify enough data points to estimate order.
    if len(stencil)<order+1:
        raise ValueError(f"Estimating derivative of order {order} requires at least {order+1} data points, but 'stencil' only specified {len(stencil)}.")

    # Case 1: Estimating a 0th order derivative
    
    if order==0:
        ## Estimating 0th order derivative requires sample at 0.
        if 0 not in stencil:
            raise ValueError("If 'order'=0, 'stencil' must contain 0.")
            
        else:
            coef = np.zeros(len(stencil))
            for index,location in enumerate(stencil):
                if location==0:
                    coef[index] = 1
                    break

            return coef

    # Case 2: Estimating derivative of any order

    ## Simple implementation of factorial coefficients for Taylor series
    factorial = np.arange(len(stencil))
    factorial[0] = 1
    for index in range(len(stencil)-1):
        factorial[index+1] = factorial[index]*factorial[index+1]

    ## Begin building matrix of Taylor series coefficients at distances specified by stencil
    taylor_coef = np.array(stencil,dtype='float').reshape(-1,1)**np.arange(len(stencil))
    taylor_coef *= 1.0/factorial

    # Return finite difference coefficients
    finite_diff = np.linalg.inv(taylor_coef)
    coef = finite_diff[order,:]

    if return_all:
        return coef, finite_diff
    else:
        return coef
