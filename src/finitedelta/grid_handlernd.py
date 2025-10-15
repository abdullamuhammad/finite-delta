import math
import numpy as np

import hashlib
from .get_coefnd import get_coefnd
from .get_partials import get_partials

def grid_handlernd(dim, partials,
                   samples, stencil,
                   order = 2,
                   output = 'sparse',
                   use_cache = True,
                   tol = 1e-8):
    """
    Inputs:
    dim -           Number of variables for multivariate function f(x_1, ..., x_dim) whose partial derivatives
                    we are approximating with finite difference coefficients.
                    Must be integer dim > 1.
                    
                    If dim = 1, f is univariate and grid_handler1d should be used.
    order -         Order of Taylor series approximation of f to use at each sampled value for determining
                    finite-difference coefficients. 
                    Default: order == 2
                    Recommended: order <= 4
                    Higher order means calculated derivatives will converge to true value with greater precision
                    as f is more densely sampled.
                    Lower order means calculated derivatives are less sensitive to noise.
                    
    partials -      List of partial derivatives to calculate finite difference coefficients for.
                    Each partial derivative is represented as a tuple of non-negative integers of length dim,
                    indicating how many times each variable is differentiated to achieve desired partial.
                    
                    Assumes that Fubini's Theorem applies, i.e. f is smooth.
                    
                    E.g. 
                    - If f(x,y,z), then the partial derivative (d^3 f)/(dx^2 dz) would be represented as (2,0,1).
                    - If f(x,y) and we wanted to calculate all partials up to order 2, then partials would 
                      be given by the list:
                          [(1,0), (0,1), (2,0), (1,1), (0,2)], corresponding to the partials:
                          [df/dx, df/dy, (d^2 f)/(dx^2), (d^2 f)/(dx dy), (d^2 f)/(dy^2)]
                    - To generate all partials of order=order for f(x_1, ..., x_dim), use:
                          get_partials(dim, order)
                    
    samples -       List-like object of all points where the function f is sampled.
                    If NumPy array, samples will have shape (N,dim)
                    If list or tuple, each entry of samples will be of length dim
                    See examples for how to convert meshgrid into desired 'samples' object and track indices.
                    
    stencil -       List of same length as 'samples', where each entry is a list of integers. 
                    The i^th entry of stencil is a list of indices for points in 'samples' that will be used
                    to calculate the requested partial derivative at the point samples[i].
                    Typically, stencil[i]  will contain the index i.
                    
    tol -           Precision of finite difference coefficients (used for caching to speed up computation when 
                    number of samples is large.)
                    Default: tol == 1e-8

    use_cache -     If True: cache is used to speed up computation. (Recommended for uniformly sampled grids.)
                    If False: new coefficients are calculated every time (much slower.)

    Outputs:
    output -        If 'list': output will be a dictionary {'rows': [...], 'cols': [...], 'coef': [...]}, such
                    that for each index i, the matrix A whose entries are A[rows[i], cols[i]] = coef[i] is a 
                    linear operator that transforms a function f sampled at N points to the corresponding 
                    partial derivative sampled at the same points.
                    Each index i describes the finite difference coefficient coef[i] applied to samples[cols[i]]
                    to approximate the desired partial derivative at samples[rows[i]].

                    If 'sparse': same as above, but output is sparse matrix of type scipy.sparse.csr_matrix.
                    Recommended: output == 'sparse' requires scipy.sparse.
    """

    #########################
    # Handling input errors #
    #########################

    # Makes sure output is either 'sparse' (requires scipy.sparse) or 'list'
    if output == 'sparse':
        try:
            from scipy.sparse import csr_matrix
        except ImportError as e:
            raise ImportError("Sparse output requires scipy.") from e
    elif output != 'list':
        raise ValueError("Input 'output' must be either 'sparse' or 'list'.")

    # Checks that dim is integer
    if (not isinstance(dim, (int, np.integer))) or isinstance(dim, bool):
        raise TypeError(f"Input 'dim' must be of integer type, not {type(dim).__name__}.")
    elif dim < 1:
        raise ValueError("Input 'dim' (number of vars for multivariate function) must be positive integer.")
    elif dim < 2:
        raise ValueError("Use grid_handler1d for better performance in univariate case.")

    # Checks that partials is either tuple of non-negative integers, or list-like type of such tuples
    if isinstance(partials, (list,tuple)):
        if all(isinstance(partial, (list,tuple)) for partial in partials):
            if all(len(partial) != dim for partial in partials):
                raise ValueError("Each partial derivative in 'partials' must be of length 'dim'.")
            elif not all(isinstance(partial_order, (int, np.integer)) and partial_order >= 0 for partial_order in partial for partial in partials):
                raise ValueError("Each partial derivative in 'partials' must consist of non-negative integers, describing how many times each variable is differentiated.")
            
                
        elif all(isinstance(partial, np.ndarray) for partial in partials):
            if not all(len(partial.shape)==1 and partial.size == dim for partial in partials):
                raise ValueError("Each partial derivative in 'partials' must be represented by a shape (dim,) array.")
            elif not (np.issubdtype(partials.dtype, np.integer) and np.all(partials>=0)):
                raise TypeError("Each partial derivative in 'partials' must consist of non-negative integers.")
                
        elif all(isinstance(partial, (int, np.integer)) and not isinstance(partial, bool) for partial in partials):
            if len(partials) != dim:
                raise ValueError(f"Input 'partials' must be of length 'dim', not {len(partials)}.")
            elif not all(partial >= 0 for partial in partials):
                raise ValueError("Input 'partials' must consist of non-negative integers, describing how many times each variable is differentiated.")
            else:
                partials = [partials]
                
        else:
            raise TypeError("Input 'partials' must either be length 'dim' tuple of non-negative integers, or list-like type of such tuples.")
            
    elif isinstance(partials, np.ndarray):
        ### If 'partials' is given as a NumPy array:
        if len(partials.shape) == 1:
            partials = partials.reshape(1,-1)

        ### It must be shaped (N,d)
        if len(partials.shape) != 2:
            raise ValueError("If 'partials' is NumPy array, it must be of shape (N,dim) where each of N rows is a partial derivative of f(x_1, ..., x_dim) to estimate. Partial derivatives are represented as multisets of dim non-negative integers that sum to the order of the desired partial derivative.")
            
        elif partials.shape[1] != dim:
            raise ValueError(f"Each partial derivative in 'partials' must be a tuple of length 'dim' (currently 'dim' = {dim}, but length of 'partials' = {partials.shape[1]}), with each index describing how many times a partial derivative of a multivariable function f(x_1, ..., x_dim) is taken with respect to the corresponding variable.")

        ### And must consist of non-negative integers
        elif not np.issubdtype(partials.dtype, np.integer):
            raise ValueError(f"Each partial derivative in 'partials' must be a tuple of length 'dim' (currently 'dim' = {dim}, but length of 'partials' = {partials.shape[1]}), with each index describing how many times a partial derivative of a multivariable function f(x_1, ..., x_dim) is taken with respect to the corresponding variable.")
            
        elif not np.all(partials>=0):
            raise ValueError("Input 'partials' must consist of non-negative integers, describing how many times a partial derivative is taken with respect to each index to achieve desired partial derivative.")

    # Checks that 'samples' is either (N_samples,dim) NumPy array or list of (dim,) arrays.
    if isinstance(samples, nd.array):
        if not (len(samples.shape) == 2 and samples.shape[1] == dim):
            raise ValueError("If 'samples' is NumPy array, it must have shape (N_samples,dim).")

    elif isinstance(samples, (tuple,list)):
        if all(isinstance(sample_axis, (tuple,list)) for sample_axis in samples):
            if not all(len(sample_axis) == dim for sample_axis in samples):
                raise ValueError("Input 'samples' must be a list of points where a function is sampled. Each sample must be of length 'dim'.")
            if not all(isinstance(sample_coordinate, (int, float, np.number)) for sample_coordinate in sample_axis for sample_axis in samples):
                raise TypeError("Input 'samples' must be a list of sample points, where each entry is of numeric type.")
    
    else:
        raise TypeError("Input 'samples' must be list-like object (list/tuple/array) of length-dim sample points.")

    # Number of samples in 'samples', each one is length-dim 
    num_samples_total = len(samples)

    # Make sure stencil is a list of length num_samples_total, indicating which samples are 
    # used to approximate finite difference coefficients.
    if isinstance(stencil, np.ndarray):
        if not (len(stencil.shape)==2 and stancil.shape[0] == num_samples_total):
            raise ValueError("If 'stencil' is NumPy array, it must be of shape (N_samples, N_approx), where N_samples is the total number of samples and N_approx is number of neighbors used to approximate derivatives at each given sample.")
        elif not (np.all(stencil>=0) and np.all(stencil < num_samples_total)):
            raise ValueError("Input 'stencil' can only have entries that are indices of points in samples.")

        stencil_size_per_sample = [stencil.shape[1] for sample_index in range(num_samples_total)]

    elif isinstance(stencil, (list, tuple)):
        if not len(stencil) == num_samples_total:
            raise ValueError(f"Length of 'stencil' must be the same as 'samples', was {len(stencil)}.")
        elif not all(isinstance(stencil_per_sample, (list, tuple, np.ndarray)) for stencil_per_sample in stencil):
            raise TypeError("Each entry in 'stencil' must be a list-like type indicating indices of points used to approximate derivatives at corresponding index.")
        else:
            stencil_size_per_sample = []
            length_check = []
            integer_check = []
            for stencil_per_sample in stencil:
                if isinstance(stencil_per_sample, (list,tuple)):
                    stencil_size_per_sample.append(len(stencil_per_sample))
                    length_check.append(True)
                    integer_check.append(all(isinstance(sample_index, (int, np.integer)) and sample_index>= 0 and sample_index < num_samples_total) for sample_index in stencil_per_sample)
                else:
                    stencil_size_per_sample.append(stencil_per_sample.size)
                    length_check.append(len(stencil_per_sample.shape)==1)
                    integer_check.append(np.issubdtype(stencil_per_sample.dtype, np.integer) and np.all(stencil_per_sample>=0) and np.all(stencil_per_sample<num_samples_total))

            if not all(length_check):
                raise ValueError("Input 'stencil' must be list of list of integers.")
            elif not all(integer_check):
                raise ValueError("All entries of each stencil must be integer of index of points in 'samples'.")
                    

    # Determine what order Taylor series to use to estimate coefficients
    desired_order_per_partial = [sum(partial) for partial in partials]
    if min(desired_order_per_partial) <= 0:
        raise ValueError("Each calculated partial derivative must have positive order.")
    
    if order is None:
        # if 'order' is given as None, then it is automatically set to the lowest order partial to be calculated
        order = max(desired_order_per_partial)
    elif order == 'max':
        # if 'order' is set to max, then it calculates the largest order possible given the number of sample points
        order = 0
        num_samples_needed = 1 + math.comb(order+dim,dim-1)
        while num_samples_needed<=min(stencil_size_per_sample):
            order += 1
            num_samples_needed += math.comb(order+dim,dim-1)

    elif isinstance(order, (int, np.integer)):
        if order < 0:
            raise ValueError("Input 'order' of Taylor approximation must be positive integer.")
        else:
            num_samples_needed = 0
            for k in range(order+1):
                num_samples_needed += math.comb(order+dim-1,dim-1)

            if num_samples_needed>min(stencil_size_per_sample):
                raise ValueError(f"Estimating partial derivatives of order {order} for {dim}-dim. function requires {num_samples_needed} samples at each point, but stencil specified only {min(stencil_size_per_sample)} samples at some points.")

    else:
        raise TypeError(f"Input 'order' must be None, 'max', or of integer type, not {type(order)}.")

    if order < 1:
        raise ValueError(f"Given inputs, resulting order of Taylor series approximation was {order}, but must be >= 1.")

    # Make sure tolerance for coefficients is small, positive float
    if not isinstance(tol, (float, np.number)):
        raise TypeError("Input 'tol' must be of numeric type 0 < tol < 1.")
    elif tol<=0:
        raise ValueError("Input 'tol' must be positive.")
    elif tol>=1:
        raise ValueError("Input 'tol' must be 0 < tol < 1.")

    if not isinstance(use_cache, bool):
        raise TypeError("Input 'use_cache' must be boolean.")

    ############################
    # Calculating coefficients #
    ############################
    if use_cache:
        hash_cache = {}
    
    outputs = {partial_derivative: {key: [] for key in ['cols','rows','coef']} for partial_derivative in partials}

    # Generate all partial derivatives <= order. This ordering is uniquely determined by DFS in get_partials.
    partials_all = get_partials(dim, order, True)
    partials_all_ordered = []
    partials_all_reverse = {}
    taylor_coef_row_index = 0
    for k in range(order+1):
        partials_all_ordered += partials_all[k]
        for partial_derivative in partials_all[k]:
            partials_all_reverse[partial_derivative] = taylor_coef_row_index
            taylor_coef_row_index += 1

    # Iterate over all samples in 'samples'
    for sample_index in range(num_samples_total):
        sample_center = np.array(samples[sample_index][:])

        # Define relative stencil for each sample based on stencil list of indices
        stencil_local = np.zeros((len(stencil[sample_index]), dim))
        for stencil_index, neighbor_index in enumerate(stencil[sample_index]):
            stencil_local[stencil_index,:] = np.array(samples[neighbor_index][:])

        stencil_local -= sample_center
        stencil_hash = hashlib.sha256((np.round(stencil_local.reshape(-1) / tol)*tol).astype(np.float64).tobytes()).hexdigest()

        for partial_derivative in partials:
            unique_partial_index = partials_all_reverse[partial_derivative]
            if use_cache:
                try:
                    coef = hash_cache[stencil_hash][unique_partial_index,:]
                except:
                    coef, finite_difference_matrix = get_coefnd(
                        stencil_local, dim, partial_derivative, order=order, return_all = True
                    )
                    hash_cache[stencil_hash] = finite_difference_matrix

            else:
                coef = get_coefnd(
                    stencil_local, dim, partial_derivative, order=order, return_all = False
                )

            outputs[partial_derivative]['rows'].append(([sample_index]*len(stencil[sample_index]))[:])
            outputs[partial_derivative]['cols'].append(list(stencil[sample_index]))
            outputs[partial_derivative]['coef'].append(coef.tolist())

    if output == 'list':
        return outputs
    else:
        output_csr = {}
        for partial_derivative in partials:
            output_csr[partial_derivative] = csr_matrix(
                (outputs[partial_derivative]['coef'], (
                    outputs[partial_derivative]['rows'],
                    outputs[partial_derivative]['cols']
                )
                )
            )

        
        return output_csr