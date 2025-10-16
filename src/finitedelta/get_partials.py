import numpy as np

def get_partials(dim, order, return_all = False):
    """
    This function returns all partial derivatives of a desired order, e.g. the partial
    derivatives of f(x,y) of order 2 are given by (dx,dx), (dx,dy), (dy,dy). 
    Inputs:
    dim -       How many variables the function f(x_1, ..., x_dim) takes in
    order -     What order partial derivatives we are interested in
    return_all - 
        if True, then all partials <= order are returned
        if False, only partials == order are returned

    Outputs - 
    multisets - A list of all (dim)-tuples of non-negative integers that sum to order

    Note: This is equivalent to the problem of choosing all multisets of length order
    that consists of dim unique entries. Using the "stars and bars" method, it can be
    shown that this is equivalent to (order+dim-1) choose (dim-1). See:
    - https://discrete.openmathbooks.org/dmoi2/sec_stars-and-bars.html
    - https://en.wikipedia.org/wiki/Stars_and_bars_(combinatorics)
    """
    # Force dim and order to be positive integers
    if not isinstance(dim, (int, np.integer)):
        raise TypeError(f"Input 'order' must be of integer type, not {type(dim).__name__}.")
        
    if not isinstance(order, (int, np.integer)):
        raise TypeError(f"Input 'order' must be of integer type, not {type(order).__name__}.")

    if not dim>0:
        raise ValueError(f"Input 'dim' must be positive integer.")

    if not order>0:
        raise ValueError(f"Input 'order' must be positive integer.")

    if not isinstance(return_all, bool):
        raise TypeError(f"Input 'order' must be of boolean type, not {type(return_all).__name__}.")

    # Define a tree-like structure that keeps track of multisets
    multiset_tree = {depth: {} for depth in range(order+1)}
    multiset_tree[0][tuple([0]*dim)[:]] = 0
    
    # The root node is (0,0,...,0) at depth 0
    depth = 0
    path = [list(list(multiset_tree[0])[0])]

    # Implementation of DFS algorithm using iteration for speed.
    # Continue until 'path' converges to root node.
    while len(path)>0:
        # The index variable keeps track of which nodes have been visited
        # from each node.
        index = multiset_tree[depth][tuple(path[-1])]
        
        if depth < order and index < dim:
            # Case 1: Go forward by adding 1 to the entry at position index
            
            multiset_tree[depth][tuple(path[-1])] += 1
            path.append(path[-1][:])
            path[-1][index] += 1
            depth += 1
        
            try:
                multiset_tree[depth][tuple(path[-1])]
            except:
                multiset_tree[depth][tuple(path[-1])] = 0
        else:
            # Case 2: Go backward in path
            
            path = path[0:-1]
            depth -= 1

    # Output either == order or <= order cases.
    if return_all:
        return {depth: list(multiset_tree[depth]) for depth in range(order+1)}
    else:
        return list(multiset_tree[order])

    
    
    