# -*- coding: utf-8 -*-
from math import log10


def get_block_exp_nodes(wall_dist, exp_factor, block_height, denominator):
    """Returns the number of nodes and the exact expansion factor for a
    wall distance, initial expansion factor and block height. The number of
    nodes can be divided by denominator without a remainder."""

    d = float(wall_dist)
    g = float(exp_factor)
    h = float(block_height)
    denominator = int(denominator)
    # Get initial guess for n (extrusion steps = n+1; nodes = n+2 )
    n = (log10(h*(g - 1.0)/d + 1.0)/log10(g)) - 1.0
    print(n)
    n = int(n) + 1
    num_ext_steps = n + 1
    num_nodes = n + 2
    # Increment numExtsteps until it can be divided by denominator without
    # remainder
    while (num_ext_steps % denominator) != 0:
        n += 1
        num_ext_steps += 1
        num_nodes += 1

    # Function for expansion factor
    def f(g):
        return g**(n + 1.0) - h/d*g + h/d - 1.0

    # First derivative of function above
    def df(g):
        return (n + 1.0)*g**n - h/d

    # Solve using newtons method
    num_iter = 10
    for i in range(num_iter):
        if df(g) == 0:
            return g
        g = g - f(g)/df(g)
        print(i, ' : ', g)
    # Get exact block height
    block_height = d * ((g**(n + 1.0) - 1.0)/(g - 1.0))
    exp_factor = g

    return num_nodes, block_height, exp_factor
