
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial import ConvexHull
from itertools import combinations

def find_feasible_weights(pairs, preferences):
    '''
    Representing each preference as an inequality constraint and using scipy linprog implementation to find a set of 
    feasible weights (or identify if there are no feasible weights)
    '''
    A_neq = []
    b_neq = []
    A_eq = []
    b_eq = []

    epsilon = 1e-5  # small margin to have < instead of <= ; this is kinda iffy though so will look for better options

    for (features_0, features_1), pref in zip(pairs, preferences):
        delta_f = features_1 - features_0
        
        if pref == 1:
            A_neq.append(delta_f)
            b_neq.append(-epsilon)
        elif pref == -1:
            A_neq.append(-delta_f)
            b_neq.append(-epsilon)
        elif pref == 0:
            A_eq.append(delta_f)
            b_eq.append(0)
    
    # convert lists to arrays for linprog
    A_neq = np.array(A_neq) if A_neq else None
    b_neq = np.array(b_neq) if b_neq else None
    A_eq = np.array(A_eq) if A_eq else None
    b_eq = np.array(b_eq) if b_eq else None
    
    # solve using linprog
    n_features = len(pairs[0][0])  # number of features
    result = linprog(c=np.zeros(n_features), A_ub=A_neq, b_ub=b_neq, A_eq=A_eq, b_eq=b_eq, bounds=(None, None))

    return result

def find_feasible_weight_space(pairs, preferences, test_weight=None):
    '''
    * Returns boolean for if feasible weight space exists *
    '''

    num_features = len(pairs[0][0])
    pref_matrix = []
    epsilon = 1e-5

    # constructing inequality matrix
    for (features_0, features_1), pref in zip(pairs, preferences):
        delta_f = features_1 - features_0
        
        if pref == 1: 
            # w · (features_1 - features_0) >= epsilon 
            row = [-epsilon] + list(delta_f)
            pref_matrix.append(row)
        elif pref == -1:  
            # w · (features_1 - features_0) <= epsilon
            row = [-epsilon] + list(-delta_f)
            pref_matrix.append(row)
        elif pref == 0:  # strict equality
            row1 = [0] + list(delta_f)
            row2 = [0] + list(-delta_f)
            pref_matrix.extend([row1, row2])

    mat = cdd.Matrix(pref_matrix, number_type='fraction')
    mat.rep_type = cdd.RepType.INEQUALITY

    # convert to polyhedron representation
    poly = cdd.Polyhedron(mat)
    if poly.get_generators().row_size == 0:
        # print("no feasible space!")
        return False  
    
    # check if test weights r in polyhedron - this is just for testing
    if test_weight is not None:
        inequalities = poly.get_inequalities()
        
        # each row of inequalities is in the form [b, -a1, -a2, ..., -an]
        # Representing the inequality: a1*x1 + a2*x2 + ... + an*xn <= b
        for inequality in inequalities:
            b = inequality[0]
            a = np.array(inequality[1:])
            
            # check if the inequality holds for test_weight
            if np.dot(a, test_weight) > b:
                raise ValueError("the feasible weight is NOT within the polyhedron defined by the preference constraints")
                # return False

        # print("the test weight lies within the feasible space.")
        return True
    
    # print("polyhedron of feasible weightspace is", poly.get_generators())
    return True

def find_full_weight_space(pairs, preferences, basic_bounds=None, test_weight=None, num_features = 2):
    '''
    * Returns the whole polyhedron of feasible weight space *
    '''
    pref_matrix = []
    epsilon = 1e-5
    
    if(len(pairs) != 0):
        assert num_features == len(pairs[0][0])
        for (f0, f1), pref in zip(pairs, preferences):
            delta_f = f1 - f0
            if pref == -1:
                pref_matrix.append([-epsilon] + list(-delta_f))
            elif pref == 1:
                pref_matrix.append([-epsilon] + list(delta_f))
            elif pref == 0:
                pref_matrix.extend([[0] + list(delta_f), [0] + list(-delta_f)])

    if basic_bounds is not None:
        if len(basic_bounds) != num_features:
            raise ValueError("basic_bounds length must match number of features")
        for i, (L, U) in enumerate(basic_bounds):
            row_lb = [-L] + [1 if j == i else 0 for j in range(num_features)]
            row_ub = [U] + [-1 if j == i else 0 for j in range(num_features)]
            pref_matrix.extend([row_lb, row_ub])

    mat = cdd.Matrix(pref_matrix, number_type='fraction')

    # volume = calculate_polyhedron_volume(mat)
    # print(volume)
    mat.rep_type = cdd.RepType.INEQUALITY
    # print("matrix issss" , mat)
    return cdd.Polyhedron(mat)


def find_conflicts(pairs, preferences):
    ''' just returns all subsets of 1,2,3 pairs that are creating conflicts '''
    weights = find_feasible_weights(pairs, preferences)
    weights = weights.x
    result = find_feasible_weight_space(pairs, preferences)
    conflicts = []

    # if result.success:
    if result:
        print("no conflicts yay")
        return None
        
    # if conflict exists, we'll try removing subsets of 1, 2, or 3 preferences
    for r in range(1, min(len(preferences), 4)):
        indices = range(len(preferences))
        for to_remove in combinations(indices, r):

            # dont reuse conflicts we alr found
            if any(set(conflict).issubset(to_remove) for conflict in conflicts):
                continue
            
            reduced_pairs = [pair for i, pair in enumerate(pairs) if i not in to_remove]
            reduced_preferences = [pref for i, pref in enumerate(preferences) if i not in to_remove]

            result = find_feasible_weight_space(reduced_pairs, reduced_preferences)
            
            # if result.success:
            if result:
                print(f"No feasible solution for the given preferences. Feasible solution can be found after removing preferences at indices {to_remove}.")
                conflicts.append(to_remove)

        if len(conflicts) == 0:
            print(f"No feasible solution exists even after removing all possible subsets of {r} preference(s).")

    print(f"Conflict can be resolved by removing any of the following subsets: {conflicts}")
    return conflicts
