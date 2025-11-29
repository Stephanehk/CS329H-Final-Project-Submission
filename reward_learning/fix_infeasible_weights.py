import numpy as np
from scipy.optimize import linprog
import sys
import pickle
from pathlib import Path
from itertools import combinations
from reward_learning.active_pref_learning import generate_inequalities, load_reward_ranges


def load_infeasible_results(
    env_name: str,
    model_name: str,
    direct_llm_preference: bool,
    extra_details: str = "",
):
    """
    Load the results saved when result.success was False.
    
    Args:
        env_name: Name of the environment
        model_name: Name of the model used
        direct_llm_preference: Whether direct LLM preferences were used
        extra_details: Additional details in filenames
        
    Returns:
        Dictionary containing all loaded data
    """
    base_path = f"active_learning_res/{env_name}{extra_details}_no_feasible_solution_{model_name}_{direct_llm_preference}"
    
    print(f"Loading infeasible results from: {base_path}")
    
    # Load numpy arrays
    feasible_w = np.load(f"{base_path}_prefs_feasible_weights.npy")
    A_ub = np.load(f"{base_path}_prefs_A_ub.npy")
    b_ub = np.load(f"{base_path}_prefs_b_ub.npy")
    
    # Load pickle files
    with open(f"{base_path}_preferences.pkl", 'rb') as f:
        preferences = pickle.load(f)
    with open(f"{base_path}_pairs.pkl", 'rb') as f:
        all_pairs = pickle.load(f)
    with open(f"{base_path}_pref_justifications.pkl", 'rb') as f:
        pref_justifications = pickle.load(f)
    
    # Try to load guiding principles if they exist
    guiding_principles = None
    direction_pref_guidance = None
    direction_prefs = None
    
    gp_path = Path(f"{base_path}_guiding_principles.txt")
    if gp_path.exists():
        with open(gp_path, 'r') as f:
            guiding_principles = f.read()
        with open(f"{base_path}_direction_pref_guidance.txt", 'r') as f:
            direction_pref_guidance = f.read()
        with open(f"{base_path}_dominating_features.pkl", 'rb') as f:
            direction_prefs = pickle.load(f)
    
    return {
        'feasible_w': feasible_w,
        'A_ub': A_ub,
        'b_ub': b_ub,
        'preferences': preferences,
        'all_pairs': all_pairs,
        'pref_justifications': pref_justifications,
        'guiding_principles': guiding_principles,
        'direction_pref_guidance': direction_pref_guidance,
        'direction_prefs': direction_prefs,
    }


def analyze_infeasibility(A_ub, b_ub, preferences, all_pairs):
    """
    Analyze why the linear program became infeasible.
    """
    print("\n" + "="*80)
    print("INFEASIBILITY ANALYSIS")
    print("="*80)
    
    print(f"\nConstraint matrix shape: {A_ub.shape}")
    print(f"Number of constraints: {len(b_ub)}")
    print(f"Number of variables (features): {A_ub.shape[1]}")
    print(f"Number of preference pairs: {len(preferences)}")
    
    # Try to solve the LP
    dim_w = A_ub.shape[1]
    result = linprog(c=[0]*dim_w, A_ub=A_ub, b_ub=b_ub, bounds=(None, None))
    
    print(f"\nLP Status: {result.message}")
    print(f"Success: {result.success}")


    #Lets make sure that removing the last preference makes the LP feasible
    reduced_pairs = all_pairs[:-1]
    reduced_preferences = preferences[:-1]
    inequalities, b = generate_inequalities(reduced_pairs, reduced_preferences, dim=dim_w)
    reduced_A_ub = np.array(inequalities, dtype=float) 
    reduced_b_ub = np.array(b, dtype=float)
    result_sanity_check = linprog(c=[0]*dim_w, A_ub=reduced_A_ub, b_ub=reduced_b_ub, bounds=(None, None))
    assert result_sanity_check.success, "LP should be feasible if we remove the last preference"

    
    if not result.success:
        print("\nThe constraint set is infeasible.")
        print("This indicates conflicting preferences have been provided.")

        # if conflict exists, we'll try removing subsets of 1, 2, or 3 preferences
        for subset_size in range(1, 4):
            #remove the last indice; we always want to keep the last preference which causes the infeasibility
            indices = range(len(preferences))[:-1]

            #TODO: remove later
            indices = indices[50:]
            
            for to_remove in combinations(indices, subset_size):

                

                reduced_pairs = [pair for i, pair in enumerate(all_pairs) if i not in to_remove]
                reduced_preferences = [pref for i, pref in enumerate(preferences) if i not in to_remove]

                inequalities, b = generate_inequalities(reduced_pairs, reduced_preferences, dim=dim_w)
                reduced_A_ub = np.array(inequalities, dtype=float) 
                reduced_b_ub = np.array(b, dtype=float)

                
                result = linprog(c=[0]*dim_w, A_ub=reduced_A_ub, b_ub=reduced_b_ub, bounds=(None, None))

                if result.success:
                    print(f"Successfully found a feasible solution after removing {subset_size} preferences")
                    print(f"Feasible weights: {result.x}")

                    #right now we will exit upon finding the first thing to remove that makes the LP feasible
                    #TODO: potentially keep track of all the things that make the LP feasible and return all of them
                    return list(to_remove)
                else:
                    print(f"Failed to find a feasible solution after removing {subset_size} preferences")

def print_feature_pair(f1, f2, feature_names):
    text_to_print = ""
    text_to_print += "\nOutcome 1:\n"
    for i, feature_name in enumerate(feature_names):
        text_to_print += f"- {feature_name}: {f1[i]}\n"

    # Present outcome 2
    text_to_print += "\nOutcome 2:\n"
    for i, feature_name in enumerate(feature_names):
        text_to_print += f"- {feature_name}: {f2[i]}\n"
    print (text_to_print) 

def main():
    # Parse command line arguments (same as active_pref_learning.py)
    if len(sys.argv) != 8:
        print("Usage: python fix_infeasible_weights.py <environment_name> <pref_gen> <model_name> <resume> <direct_llm_preference> <should_elicit_guiding_principles> <stakeholder_type>")
        print("Example: python fix_infeasible_weights.py pandemic llm o4-mini false true true default")
        sys.exit(1)
    
    env_name = sys.argv[1]
    pref_gen = sys.argv[2]
    model_name = sys.argv[3]
    resume = sys.argv[4]
    direct_llm_preference = sys.argv[5]
    should_elicit_guiding_principles = sys.argv[6]
    stakeholder_type = sys.argv[7]

    #python3 -m reward_learning.fix_infeasible_weights traffic llm o4-mini false true true default
    
    # Convert string args to proper types
    assert resume == "true" or resume == "false"
    resume = True if resume == "true" else False
    
    assert direct_llm_preference == "true" or direct_llm_preference == "false"
    direct_llm_preference = True if direct_llm_preference == "true" else False
    
    should_elicit_guiding_principles = True if should_elicit_guiding_principles == "true" else False
    
    # Build extra_details string
    extra_details = ""
    if stakeholder_type != "default":
        extra_details += f"_{stakeholder_type}"
    
    # Load environment configuration
    if env_name == "traffic":
        horizon = 300
    elif env_name == "pandemic":
        horizon = 192
    elif env_name == "glucose":
        horizon = 5760
    elif env_name == "mujoco":
        horizon = 1000
    else:
        raise ValueError(f"Environment {env_name} not supported")
    
    binary_feature_ranges, continious_feature_ranges, binary_features, feature_names, feature_rewards, reward_ranges_raw = load_reward_ranges(env_name, range_ceiling=float('inf'), horizon=horizon)
    print(f"Feature names: {feature_names}")
    
    print("="*80)
    print("LOADING INFEASIBLE WEIGHTS")
    print("="*80)
    print(f"Environment: {env_name}")
    print(f"Preference Generation: {pref_gen}")
    print(f"Model: {model_name}")
    print(f"Resume: {resume}")
    print(f"Direct LLM Preference: {direct_llm_preference}")
    print(f"Elicit Guiding Principles: {should_elicit_guiding_principles}")
    print(f"Stakeholder Type: {stakeholder_type}")
    print(f"Extra Details: {extra_details}")
    
    # Load the infeasible results
    try:
        results = load_infeasible_results(
            env_name=env_name,
            model_name=model_name,
            direct_llm_preference=direct_llm_preference,
            extra_details=extra_details
        )

        if len(results['pref_justifications']) != len(results['preferences']):
            assert len(results['pref_justifications']) + 50 == len(results['preferences'])
            #fix small bug where the pref_justifications list is not the same length as the preferences list because of dominating feature preferences
            results['pref_justifications'] = [None] * 50 + results['pref_justifications'] 
        
        print("\nSuccessfully loaded infeasible results!")
        print(f"Number of preferences: {len(results['preferences'])}")
        print(f"Number of pairs: {len(results['all_pairs'])}")
        print(f"Number of pref_justifications: {len(results['pref_justifications'])}")
        print(f"Feasible weights shape: {results['feasible_w'].shape}")
        print(f"A_ub shape: {results['A_ub'].shape}")
        print(f"b_ub shape: {results['b_ub'].shape}")
        
    
        # Analyze the infeasibility
        to_remove = analyze_infeasibility(
            results['A_ub'],
            results['b_ub'],
            results['preferences'],
            results['all_pairs']
        )
        
        # TODO: Implement fixing logic here
        print (f"To remove: {to_remove}")
        for i in to_remove:
            print (f"Preferences: {results['preferences'][i]}")
            print (f"All pairs: {results['all_pairs'][i]}")
            print (f"Preference justifications: {results['pref_justifications'][i]}")
            print_feature_pair(results['all_pairs'][i][0], results['all_pairs'][i][1], feature_names)

        print ("--------------------------------")
        conflicting_pref_i = -1
        print (f"Conflicting preference: {results['preferences'][conflicting_pref_i]}")
        print (f"Conflicting pair: {results['all_pairs'][conflicting_pref_i]}")
        print (f"Conflicting preference justification: {results['pref_justifications'][conflicting_pref_i]}")
        print_feature_pair(results['all_pairs'][conflicting_pref_i][0], results['all_pairs'][conflicting_pref_i][1], feature_names)




        #just needed for this analysis but not in general ofc
        
    except FileNotFoundError as e:
        print(f"\nError: Could not find infeasible results files.")
        print(f"Make sure the files exist at the expected path.")
        print(f"Error details: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\nError loading or analyzing results: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

