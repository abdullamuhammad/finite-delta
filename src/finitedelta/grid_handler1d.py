import numpy as np
import hashlib
from .get_coef1d import get_coef1d

def grid_handler1d(orders, accuracy = None, samples = None, 
                   mode = 'nonuniform', edge = 'order',
                   h = None, num_samples = None,
                   stencil = None,
                   output = 'matrix',
                   use_cache = True,
                   tol = 1e-8):
    """
    Inputs:
    orders -    int or list of ints, indicating which order derivatives to calculate
    accuracy -  int indicating polynomial order of estimate error. set to 2 by default
                Note: increasing accuracy also increases numerical instability
    samples -   list of all points on grid for which to compute coefficients
    mode -      if ('nonuniform' or 'nu') non-uniform spacing is assumed
                    'samples' is required, calculates at each point
                    'accuracy' is required
                if 'uniform', uniform spacing of 'samples' is assumed
                    'h' is required
                    ('samples' or 'num_samples') is required
                if 'custom', non-uniform spacing with custom stencil is used
                    'samples' is required
                    'stencil' is required
    edge -      if 'order', order of estimate error is preserved at edges
                if 'number', number of coefficients is preserved at edges
                if 'half', the number of coefficients on one side is preserved at edges
    h -         (if using grid with uniform spacing) indicates spacing between 'samples'
    num_samples - number of samples (only used in uniform sampling case)
    stencil -   list of lists of ints, same length as samples, indicates which samples
                to use to compute coefficients at each point in samples
    
    Outputs:
    if output == 'list':
        output_list -   list of finite difference coefficients for each sample
        output_locs -   relative location of samples that coefficients are applied to
                        (same as stencil, but generated as output)
    elif output == 'matrix':
        output_mat -    returns square matrix that can be applied as linear operator
                        on input function to sampled at 'samples' to approximate derivative
    elif output == 'sparse':
        output_mat -    returns sparse matrix (requires scipy)
    """
    #########################
    # Handling input errors #
    #########################

    ## Checks that orders is either integer or list of integers (excluding booleans)
    if isinstance(orders, (list,tuple)):
        if not all(isinstance(order, (int,np.integer)) and not isinstance(order, bool) for order in orders):
            raise TypeError(f"If 'orders' is a list, all entries must be of integer type.")

    elif isinstance(orders, np.ndarray):
        if len(orders.shape) != 1:
            raise ValueError(f"If 'orders' is NumPy array, it must be 1-dimensional, not {len(orders.shape)}.")
        elif not np.issubdtype(orders.dtype, np.integer):
            raise TypeError(f"If 'orders' is NumPy array, it must contain integers, not {orders.dtype}.")

    elif isinstance(orders, (int, np.integer)):
        orders = [orders]
        
    else:
        raise TypeError(f"Input 'orders' must be integer or list/tuple/array of integers, not {type(orders).__name__}.")
    
    ## Checks that 'mode' is one of required options
    if mode not in ('nonuniform','nu','uniform','u','custom','c'):
        raise ValueError(f"Input 'mode' was given as {mode}. Must be 'nonuniform' (Default), 'uniform', or 'custom'.")

    ## Checks that 'accuracy' is positive integer >= 2
    if (not isinstance(accuracy, (int, np.integer, type(None)))):
        raise TypeError(f"Input 'accuracy' must be of integer type, not {type(accuracy).__name__}.")

    ## Checks that 'edge' is one of required options
    if edge not in ('number', 'num', 'n', 'order', 'o', 'accuracy', 'acc','a', 'half', 'h'):
        raise ValueError(f"Input 'edge' was given as {edge}. Must be 'order' (Default) or 'number'.")

    ## Imports scipy if sparse output is desired
    if output not in ('matrix','list','sparse'):
        raise ValueError(f"Input 'output' was given as {output}. Must be 'matrix' (Default), 'list', or 'sparse'.")
    elif output == 'sparse':
        try:
            from scipy.sparse import csr_matrix
        except ImportError as e:
            raise ImportError("Sparse outputs require scipy.") from e

    # Case 1: Mode is set to uniform 
    if mode in ('uniform','u'):
        ## Set 'accuracy' default to 2
        if accuracy is None:
            accuracy = 2
        elif accuracy<2:
            raise ValueError(f"Input 'accuracy' was {accuracy}. Must be at least 2.")
        
        ## Require 'h' and the number of samples to be specified
        if h is None:
            raise ValueError("Input 'h' must be specified when mode is set to uniform.")
        elif not isinstance(h, (int, float, np.number)) or isinstance(h, (bool, np.bool_)):
            raise TypeError(f"Input 'h' must be a number, not {type(h).__name__}.")
        elif h<=0:
            raise ValueError("Input 'h' must be positive.")

        ## Require num_samples and samples to be of correct type or match
        if samples is None:
            if num_samples is None:
                raise ValueError("Uniform mode requires either 'num_samples' or 'samples'.")
            elif not (isinstance(num_samples, (int, np.integer)) and num_samples>0):
                raise ValueError("Input 'num_samples' must be positive integer.")

        elif isinstance(samples, (list, tuple, np.ndarray)):
            samples = np.array(samples)
            if num_samples is None:
                num_samples = len(samples)
            elif num_samples != len(samples):
                raise ValueError("Input 'num_samples' must match length of samples.")
            
        else:
            raise TypeError(f"Input 'samples' must be list or NumPy array type, not {type(samples).__name__}.")

        if stencil != None:
            raise TypeError("Input 'stencil' can't be specified for uniform mode.")

        # Iterate over orders
        output_per_order = []
        for order in orders:
            ## Determine how many coefficients are needed for central differences at desired accuracy
            num_coef_half = 0
            while 2*num_coef_half + 1 < order + accuracy:
                num_coef_half += 1

            num_coef = 2*num_coef_half + 1

            ## Get coefficients at center
            stencil = np.arange(-num_coef_half,num_coef_half+1)*h
            center_coef = get_coef1d(stencil, order)

            if output == 'list':
                ### If output is desired in list format
                output_locs_center = [stencil.tolist()]
                output_list_center = [center_coef.tolist()]
                edge_list = {'left': [], 'right': []}
                edge_locs = {'left': [], 'right': []}
            else:
                ### If output is desired in matrix format
                output_mat = np.zeros((num_samples,num_samples))
                for sample_index in range(num_samples - 2*num_coef_half):
                    output_mat[sample_index+num_coef_half,sample_index:sample_index+num_coef] = center_coef

            ## Get coefficients at edges
            for edge_ind in range(num_coef_half):
                if edge in ('order','o','accuracy','acc','a'):
                    stencil = np.arange(-edge_ind,num_coef+1-edge_ind)*h
                elif edge in ('number','n','num'):
                    stencil = np.arange(-edge_ind,num_coef-edge_ind)*h
                else:
                    stencil = np.arange(-edge_ind,num_coef_half+1)*h

                edge_coef = get_coef1d(stencil, order)

                # Combine coefficients and relative locations on grid into list
                if output == 'list':
                    edge_locs['left'] = edge_locs['left'] + [stencil.tolist()]
                    edge_locs['right'] = [stencil.tolist()] + edge_locs['right']

                    edge_list['left'] = edge_list['left'] + [edge_coef.tolist()]
                    if order % 2 == 0:
                        edge_list['right'] = [np.flip(edge_coef).tolist()] + edge_list['right']
                    else:
                        edge_list['right'] = [(-np.flip(edge_coef)).tolist()] + edge_list['right']
                        
                # Add coefficients into matrix
                else:
                    output_mat[edge_ind,0:len(stencil)] = edge_coef
                    if order % 2 == 0:
                        output_mat[-edge_ind-1,-len(stencil)::] = np.flip(edge_coef)
                    else:
                        output_mat[-edge_ind-1,-len(stencil)::] = -np.flip(edge_coef)

            if len(orders) == 1:
                if output == 'list':
                    output_locs = edge_locs['left'] + output_locs_center + edge_locs['right']
                    output_list = edge_list['left'] + output_list_center + edge_list['right']
                    return output_list, output_locs
                elif output == 'matrix':
                    return output_mat
                else:
                    return csr_matrix(output_mat)
            else:
                if output == 'list':
                    output_locs = edge_locs['left'] + output_locs_center + edge_locs['right']
                    output_list = edge_list['left'] + output_list_center + edge_list['right']
                    output_per_order += [(output_list, output_locs)]
                elif output == 'matrix':
                    output_per_order += [output_mat]
                else:
                    output_per_order += [csr_matrix(output_mat)]

        return tuple(output_per_order)
    
    elif mode in ('nonuniform','nu'):
        if use_cache not in (True,False,None):
            raise TypeError("Input 'use_cache' must be of boolean type.")

        ## Build cache of coefficients according to specified tolerance
        if use_cache:
            hash_cache = {}
        
        if samples is None:
            raise ValueError("Input 'samples' must be specified for non-uniform grid.")
        elif not isinstance(samples, (list, tuple, np.ndarray)):
            raise TypeError("Input 'samples' must be of list-like type.")
        else:
            num_samples = len(samples)
            samples = np.array(samples)

        if accuracy is None:
            if stencil is None:
                raise ValueError("Either 'accuracy' xor 'stencil' must be specified for non-uniform grid.")
            else:
                ## Iterate over all provides derivative orders
                output_per_order = []
                for order in orders:
                    ## Determine how many coefficients are needed for central differences at desired accuracy
                    if output == 'list':
                        output_list = []
                    else:
                        output_mat = np.zeros((num_samples,num_samples))

                    for sample_index in range(num_samples):
                        ### Get indices of stencil relative to input 'samples'
                        stencil_inds = np.array(stencil[sample_index])

                        ### Get new stencil input into get_coef1d function
                        stencil_input = samples[stencil_inds] - samples[sample_index]
                        if use_cache:
                            stencil_hash = hashlib.sha256((np.round(stencil_input / tol)*tol).astype(np.float64).tobytes()).hexdigest()
                            try:
                                new_coef = hash_cache[stencil_hash][order,:]
                            except KeyError or IndexError:
                                new_coef, inverse_taylor_matrix = get_coef1d(stencil_input, order, return_all = True)
                                hash_cache[stencil_hash] = inverse_taylor_matrix
                            
                        else:
                            new_coef = get_coef1d(stencil_input, order)

                        if output == 'list':
                            output_list += [new_coef.tolist()]
                        else:
                            output_mat[sample_index,stencil_inds] = new_coef

                    if len(orders) == 1:
                        if output == 'list':
                            return output_list
                        elif output == 'matrix':
                            return output_mat
                        else:
                            return csr_matrix(output_mat)
                    else:
                        if output == 'list':
                            output_per_order += [(output_list, output_locs)]
                        elif output == 'matrix':
                            output_per_order += [output_mat]
                        else:
                            output_per_order += [csr_matrix(output_mat)]

                return tuple(output_per_order)
                    
                

        else:
            ### Case where accuracy is specified, rather than a provided stencil
            if stencil is None:
                ## Iterate over all provides derivative orders
                output_per_order = []
                for order in orders:
                    ## Determine how many coefficients are needed for central differences at desired accuracy
                    num_coef_half = 0
                    while 2*num_coef_half + 1 < order + accuracy:
                        num_coef_half += 1
        
                    num_coef = 2*num_coef_half + 1

                    if output == 'list':
                        output_list = []
                    else:
                        output_mat = np.zeros((num_samples,num_samples))
    
                    ### Iterate over samples where derivative is calculated
                    for sample_index in range(num_samples):
                        #### Determine stencil
                        if sample_index < num_coef_half:
                            #### Case where stencil is close to left edge
                            sample_left = 0
                            if edge in ('order','o','accuracy','acc','a'):
                                stencil_inds = np.arange(num_coef+1)
                            elif edge in ('number','n','num'):
                                stencil_inds = np.arange(num_coef)
                            else:
                                stencil_inds = np.arange(sample_index+num_coef_half+1)
                                
                        elif sample_index >= num_samples - num_coef_half:
                            #### Case where stencil is close to right edge
                            if edge in ('order','o','accuracy','acc','a'):
                                stencil_inds = np.arange(num_samples - num_coef, num_samples)
                            elif edge in ('number','n','num'):
                                stencil_inds = np.arange(num_samples - num_coef + 1, num_samples)
                            else:
                                stencil_inds = np.arange(num_samples - (num_samples-sample_index+num_coef_half), num_samples)
                        else:
                            #### Case where stencil is in center
                            stencil_inds = np.arange(sample_index - num_coef_half,sample_index + num_coef_half + 1)

                        stencil = samples[stencil_inds] - samples[sample_index]
                        if use_cache:
                            stencil_hash = hashlib.sha256((np.round(stencil / tol)*tol).astype(np.float64).tobytes()).hexdigest()
                            try:
                                new_coef = hash_cache[stencil_hash][order,:]
                            except KeyError or IndexError:
                                new_coef, inverse_taylor_matrix = get_coef1d(stencil, order, return_all = True)
                                hash_cache[stencil_hash] = inverse_taylor_matrix
                            
                        else:
                            new_coef = get_coef1d(stencil, order)

                        if output == 'list':
                            output_list += [new_coef.tolist()]
                        else:
                            output_mat[sample_index,stencil_inds] = new_coef

                    if len(orders) == 1:
                        if output == 'list':
                            return output_list
                        elif output == 'matrix':
                            return output_mat
                        else:
                            return csr_matrix(output_mat)
                    else:
                        if output == 'list':
                            output_per_order += [(output_list, output_locs)]
                        elif output == 'matrix':
                            output_per_order += [output_mat]
                        else:
                            output_per_order += [csr_matrix(output_mat)]
                    
                return tuple(output_per_order)
            else:
                raise ValueError("Inputs 'accuracy' and 'stencil' can not both be specified for non-uniform grid.")


