from scipy.optimize import linprog
import numpy as np
import sys
import pickle
# from time import time
import time
import matplotlib.pyplot as plt
import os

# from volume import calculate_polyhedron_volume

def compute_min_and_max_dot(inequalities, b, direction):
    """
    returns (min_val, max_val) of w^T direction subject to w in 'poly' by
    solving two linear programs using the polyhedron's H-representation (inequalities)
    """
    # H = poly.get_inequalities()
    # A_ub = []
    # b_ub = []
    # for row in H:
    #     b_i = float(row[0])
    #     A_i = -np.array([float(x) for x in row[1:]], dtype=float)
    #     A_ub.append(A_i)
    #     b_ub.append(b_i)

    A_ub = np.array(inequalities, dtype=float) #Not sure why we need the negative sign
    b_ub = np.array(b, dtype=float)
    c = np.array(direction, dtype=float)

    # solve for max (c^T w) => min (-(c^T) w).
    res_max = linprog(c=-c, A_ub=A_ub, b_ub=b_ub, bounds=(None, None))
    max_val = float(c.dot(res_max.x)) if res_max.success else float('-inf')     

    # solve for min (c^T w)
    res_min = linprog(c=c, A_ub=A_ub, b_ub=b_ub, bounds=(None, None))
    min_val = float(c.dot(res_min.x)) if res_min.success else float('-inf')

    return min_val, max_val

