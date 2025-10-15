import numpy as np
import math
from finitedelta.get_partials import get_partials

def get_coefnd(stencil, dim, partial, order = None, return_all = False):
    """
    Inputs:
    stencil -   a list describing unique locations of neighboring samples used in the 
                finite differences scheme, relative to location at which derivative is
                being estimated (i.e. an entry of 0 indicates where the estimation 
                takes place
    dim -       number of variables of function f(x_1, ..., x_dim) being sampled
    partial -   an array-like type of non-negative integers that sums to the desired
                order of the derivative to be estimated
    order -     order of Taylor series approximation.
                    If None: order will be set to match desired partial
                    If 'max': order will be set to the largest possible for the stencil size
                    If type is int: order will be set to desired int
    return_all - boolean that determines whether matrix of all finite difference
                 coefficients is returned
    

    Outputs:
    coef -      a numpy array the same size as stencil, where the element coef[i] indicates
                the weight applied to a sample at relative location stencil[i] to estimate
                the derivative
    """
    # Raise Errors
    
    ## Number of variables ('dim')>1
    if not isinstance(dim, (int, np.integer)):
        raise TypeError(f"Input 'dim' must be of integer type, not {type(dim).__name__}.")
    elif dim<1:
        raise ValueError(f"Input 'dim' must be 1 or greater, not {dim}.")
    elif dim==1:
        raise ValueError("Specified method for estimating coefficients for multivariable functions, but listed dim=1.")
    
    ## 'partial' must be a list of non-negative integers
    if isinstance(partial, (list,tuple)):
        if len(partial) != dim:
            raise ValueError(f"Input 'partial' must be of length dim to indicate how many partials to take of eachv variable, not {len(partial)}")
        elif not all(isinstance(partial_order, (int,np.integer)) and not isinstance(partial_order, bool) for partial_order in partial):
            raise TypeError("Input 'partial' must consist of solely integers.")
        elif not all(partial_order>=0 for partial_order in partial):
            raise ValueError("All entries of input 'partial' must be non-negative.")

    elif isinstance(partial, np.ndarray):
        if len(partial.shape) != 1:
            raise ValueError(f"If 'partial' is NumPy array, it must be 1-dimensional, not {len(partial.shape)}.")
        elif partial.size != dim:
            ValueError(f"Input 'partial' must be of length dim to indicate how many partials to take of eachv variable, not {partial.size}")
        elif not np.issubdtype(partial.dtype, np.integer):
            raise TypeError(f"If 'partial' is NumPy array, it must contain integers, not {partial.dtype}.")
        
    else:
        raise TypeError(f"Input 'partial' must be integer or list/tuple/array of integers, not {type(partial).__name__}.")

    ## Order must be one of desired options:
    if order is None:
        order = sum(partial)
        if not order > 0:
            raise ValueError(f"Order of partial derivative to be calculated must be > 0, not {order}.")
            
    elif order == 'max':
        order = 0
        num_samples_needed = 1+math.comb(order+dim,dim-1)
        while num_samples_needed<=len(stencil):
            order += 1
            num_samples_needed += math.comb(order+dim,dim-1)

    elif not isinstance(order, (int, np.integer)):
        raise TypeError(f"Input 'dim' must be of integer type, not {type(order).__name__}.")

    elif order < sum(partial):
        raise ValueError(f"Input 'order' was {order}, which was less than order of desired partial derivative: {sum(partial)}.")

    elif not order > 0:
        raise ValueError(f"Input 'order' must be >0, was {order}.")
    
    ## Generate all partial derivatives up to 'order'
    partials_all = get_partials(dim, order, True)

    ## Assign index to each partial derivative and count how many of each order
    partials_all_ordered = []
    partials_all_reverse = {}
    row_ind = 0
    for k in range(order+1):
        partials_all_ordered += partials_all[k]
        for partial_deriv in partials_all[k]:
            partials_all_reverse[partial_deriv] = row_ind
            row_ind += 1
    
    ## 'stencil' does not specify enough data points to estimate order.
    if len(stencil)<len(partials_all_ordered)+1:
        raise ValueError(f"Estimating partial derivative of order {order} requires at least {len(partials_all_ordered)+1} data points, but 'stencil' only specified {len(stencil)}.")
    
    ## Calculate permutations of multiset normalized by Taylor series coefficients
    factorials = np.ones(len(partials_all_ordered),dtype='int')
    for partial_deriv_ind, partial_deriv in enumerate(partials_all_ordered):
        hash_key = ([0]*(order+1))[:]
        for partial_order in partial_deriv:
            if partial_order>1:
                factorials[partial_deriv_ind] *= math.factorial(partial_order)

    ## Begin building matrix of Taylor series coefficients at distances specified by stencil
    taylor_coef = np.ones((len(stencil), len(partials_all_ordered)))

    ### Iterate over samples in stencil (rows)
    for row_ind in range(len(stencil)):
        ### Iterate over mixed partial derivatives <= order (columns)
        for col_ind in range(1,len(partials_all_ordered)):
            ### Iterate over number of partials taken at each index
            for dim_ind in range(dim):
                ### Multiply respective index in Taylor coefficient matrix by h^n
                for iteration in range(partials_all_ordered[col_ind][dim_ind]):
                    taylor_coef[row_ind,col_ind] *= stencil[row_ind][dim_ind]

    taylor_coef *= 1.0/factorials

    finite_diff = np.linalg.pinv(taylor_coef)
    coef = finite_diff[partials_all_reverse[partial],:]
    if return_all:
        return coef, finite_diff
    else:
        return coef