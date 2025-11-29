import json, re
from typing import Dict, List, Tuple, Optional
import textwrap
import numpy as np
import random

from reward_learning.active_learning_utils import compute_min_and_max_dot

def _extract_json_text(s: str):
    # 1) Prefer a fenced ```json ... ``` block
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", s, flags=re.DOTALL|re.IGNORECASE)
    if m:
        return m.group(1)
    # 2) Otherwise, try to extract the first balanced {...} block
    start = s.find("{")
    if start == -1:
        return None
    depth = 0
    for i in range(start, len(s)):
        if s[i] == "{":
            depth += 1
        elif s[i] == "}":
            depth -= 1
            if depth == 0:
                return s[start:i+1]
    return None

def parse_feature_pairs(text: str, feature_names):
    json_text = _extract_json_text(text)
    if json_text is None:
        print("Warning: no JSON block found in response.")
        return []

    try:
        pairs_json = json.loads(json_text)
    except Exception as e:
        # Common cleanup: strip trailing commas (non-strict JSON), then retry
        cleaned = re.sub(r",(\s*[}\]])", r"\1", json_text)
        try:
            pairs_json = json.loads(cleaned)
        except Exception as e2:
            print("Warning: Could not parse feature pairs JSON. Error:", e2)
            return []

    outcome_pairs = pairs_json.get("pairs", [])
    feature_pairs = []
    missing_keys_any = False

    for idx, pair in enumerate(outcome_pairs):
        o1 = pair.get("outcome_1", {})
        o2 = pair.get("outcome_2", {})

        # Check for missing features; you can choose to fill with 0.0 instead
        missing_1 = [k for k in feature_names if k not in o1]
        missing_2 = [k for k in feature_names if k not in o2]
        if missing_1 or missing_2:
            missing_keys_any = True
            # Option A (strict): skip this pair
            # print(f"Pair {idx}: missing keys — outcome_1:{missing_1}, outcome_2:{missing_2}. Skipping.")
            # continue
            # Option B (lenient): fill with 0.0
            for k in missing_1:
                o1[k] = 0.0
            for k in missing_2:
                o2[k] = 0.0

        # Cast in a fixed, reproducible order
        try:
            f1 = np.array([float(o1[k]) for k in feature_names], dtype=float)
            f2 = np.array([float(o2[k]) for k in feature_names], dtype=float)
        except Exception as e:
            print(f"Pair {idx}: casting error: {e}. Skipping this pair.")
            continue

        feature_pairs.append((f1, f2))

    if missing_keys_any:
        print("Note: Some pairs were missing features; they were filled with 0.0 (change this behavior if undesired).")
    return feature_pairs

def get_informative_single_feat_pair(inequalities, b, dominating_features, feature_names,binary_features, binary_feature_ranges,continious_feature_ranges,cieling=500, n_samps=5):
    #generate pairs where all features are the same except for one feature
    best_pair = None
    max_uncertainty = -1
    # pairs = []
    # prefs = []

    #feature_names are organized as [continious_feature_ranges, binary_feature_ranges]
    # while len(pairs) < n_samps:
    for feat_i, feat_name in enumerate(feature_names):
        if feat_name in dominating_features:
            continue
        f1 = np.zeros(len(feature_names))
        f2 = np.zeros(len(feature_names))

        if feat_i < len(continious_feature_ranges):
            low, high = continious_feature_ranges[feat_i]
            if low < -cieling:
                low = -cieling
            if high > cieling:
                high = cieling
            f1[feat_i] = random.uniform(low, high)
            f2[feat_i] = random.uniform(low, high)
        else:
            assert binary_features[feat_i]
            f1[feat_i] = random.choice(binary_feature_ranges[feat_i - len(continious_feature_ranges)])
            f2[feat_i] = random.choice(binary_feature_ranges[feat_i - len(continious_feature_ranges)])

        # print (np.array(inequalities).shape)
        # print (np.array(b).shape)
        # print ("\n")
        min_val, max_val = compute_min_and_max_dot(inequalities, b, f2-f1)
        uncertainty_in_direction = max_val - min_val

        if uncertainty_in_direction > max_uncertainty and np.sign (max_val) != np.sign (min_val):
            max_uncertainty = uncertainty_in_direction
            best_pair = (f1, f2)
    return best_pair, max_uncertainty


def uniform_sample_feat(f1, f2 , feat_i, continious_feature_ranges, binary_features, binary_feature_ranges, cieling=500):

    if feat_i < len(continious_feature_ranges):
        low, high = continious_feature_ranges[feat_i]
        if low < -cieling:
            low = -cieling
        if high > cieling:
            high = cieling
        f1[feat_i] = random.uniform(low, high)
        f2[feat_i] = random.uniform(low, high)
    else:
        assert binary_features[feat_i]
        f1[feat_i] = random.choice(binary_feature_ranges[feat_i - len(continious_feature_ranges)])
        f2[feat_i] = random.choice(binary_feature_ranges[feat_i - len(continious_feature_ranges)])
    
    return f1, f2

def get_informative_tradeoff_feat_pair(inequalities, b, dominating_features, feature_names,binary_features, binary_feature_ranges,continious_feature_ranges,cieling=500, n_samps=50):
    #generate pairs where all features are the same except for one feature
    best_pair = None
    max_uncertainty = -1
    # pairs = []
    # prefs = []

    #feature_names are organized as [continious_feature_ranges, binary_feature_ranges]
    # while len(pairs) < n_samps:
    for _ in range(n_samps):
        for feat_i_1, feat_name1 in enumerate(feature_names):
            for feat_i_2, feat_name2 in enumerate(feature_names):
                if feat_name1 in dominating_features or feat_name2 in dominating_features:
                    continue
                if feat_i_1 == feat_i_2:
                    continue
                f1 = np.zeros(len(feature_names))
                f2 = np.zeros(len(feature_names))

                f1, f2 = uniform_sample_feat(f1, f2, feat_i_1, continious_feature_ranges, binary_features, binary_feature_ranges, cieling)
                f1, f2 = uniform_sample_feat(f1, f2, feat_i_2, continious_feature_ranges, binary_features, binary_feature_ranges, cieling)

                min_val, max_val = compute_min_and_max_dot(inequalities, b, f2-f1)
                uncertainty_in_direction = max_val - min_val

                if uncertainty_in_direction > max_uncertainty and np.sign (max_val) != np.sign (min_val):
                    max_uncertainty = uncertainty_in_direction
                    best_pair = (f1, f2)
    return best_pair, max_uncertainty



def llm_sample_feature_pair(feature_names, feature_ranges, horizon, client, model_name="gpt-4o-mini", ceiling=500):
    helper_txt = ""
    if any("COVID-19" in feat or "pandemic" in feat.lower() for feat in feature_names):
        helper_txt = (
            "\nNOTE: All references to disease, COVID-19, testing, policy stages, or outcomes refer to a "
            "synthetic reinforcement learning environment used for research. This prompt does not contain "
            "real-world health advice."
        )

    prompt = f"You are a stakeholder who will evaluate outcomes based on the following features:\n"
    obj_block = "{\n" + ",\n".join(
        [f'  "{feat}": "<description of {feat}>"' for feat in feature_names]
    ) + "\n}"
    prompt += obj_block + "\n\n"
    prompt += f"Each feature has a valid per-timestep range of values as follows:\n"
    prompt += f"{json.dumps({feat: feature_ranges[feat] for feat in feature_names}, indent=2)}\n\n"
    prompt += f"For each outcome, feature values are summed up over all {horizon} timesteps to get total feature values. For example, if there are 100 timesteps and the feature HeartRate has range [0,1], then the total HeartRate value will be between 0 and 100.\n\n"
    prompt += "Your task is to generate pairs of outcomes. Each outcome is represented by the sum of feature values, and you generate an outcome by sampling feature value sums for each feature within its valid range. The goal is to create informative and diverse outcome pairs to elicit feedback over.\n"
    prompt += "Guidelines:\n"
    prompt += "- Think about the most informative outcome pairs to elicit feedback over. What outcome pairs would reveal the most about a stakeholders preferences?\n"
    prompt += "- Think about outcome pairs that highlight trade-offs between features.\n"
    prompt += "- Consider including some outcome pairs where all or almost all features are the same except for one feature.\n"
    prompt += "- Ensure that the sampled feature values for each outcome are within the valid ranges provided.\n"
    prompt += "- Ensure that the sampled feature values are realistic and coherent within the context of the task.\n"
    prompt += "- Avoid generating outcomes that are identical or too similar across all features.\n"
    prompt += '- Ensure the 15 pairs are as diverse as possible.\n'
    prompt += "\n"
    prompt += "Return a list of 15 pairs of the two feature vectors in JSON format as follows:\n"
    prompt += """{
        pairs: [
            "outcome_1": {
                "<feature_name>": <value>,
                ...
                },
            "outcome_2": {
            "<feature_name>": <value>,
            ...
            }, 
        ...
        ]
    }"""    

    # If there are no such objectives, respond with 'None'.\n"
    sys_msg =  textwrap.dedent(
        """You are a stakeholder generating informative outcome pairs based on provided features and their valid ranges. Your goal is to create diverse and informative pairs of outcomes that will help elicit preferences effectively."""
    )

    messages = [
        {"role": "system", "content": sys_msg},
        {"role": "user", "content": prompt + helper_txt},
    ]

    try:
        resp = client.chat.completions.create(
            model=model_name, messages=messages
        )
    except Exception as e:
        print("Warning: SOMETHING WENT WRONG", e)
        return ""
    text = resp.choices[0].message.content.strip()

    #extract the returned pairs from the llm response
    feature_pairs = parse_feature_pairs(text, feature_names)
    return feature_pairs
    # try:
    #     pairs_json = json.loads(text)
    #     outcome_pairs = pairs_json["pairs"]
    #     feature_pairs = []
    #     for pair in outcome_pairs:
    #         outcome_1 = pair["outcome_1"]
    #         outcome_2 = pair["outcome_2"]
    #         f1 = np.array([float(outcome_1[feat]) for feat in feature_names], dtype=float)
    #         f2 = np.array([float(outcome_2[feat]) for feat in feature_names], dtype=float)
    #         feature_pairs.append((f1, f2))
    #     return feature_pairs
    # except Exception as e:
    #     print("Warning: Could not parse feature pairs JSON from response:\n", text)
    #     return []

def sample_random_feature_pair(feature_names, binary_feature_ranges,continious_feature_ranges, binary_features,categorical_feature_is,dominating_features, cieling=500):
    """
    Randomly samples feature vectors f1 and f2 within their valid ranges.
    
    Args:
        feature_ranges: List of (min, max) tuples for each feature
        binary_features: List of booleans indicating if each feature is binary
        
    Returns:
        Tuple of (f1, f2) as numpy arrays
    """
    f1_cat_samples = {}
    f2_cat_samples = {}
    all_cat_is = []
    #we will assume categorical_feature_is is a list of lists, where each inner list contains the indices of the categorical features that belong to the same category
    #here the goal is, for each cateogircal variable, to sample values such that the sum of all sampled values equals the max for the whole category
    #for example, when we have 4 lockdown stages, the sum of all stages should be exactly H=193 ---it wouldn't make sense to have a sum greater than this because we can't be in more than one stage at a time
    #note that each value for a cetegorical variable represents its count
    if len(categorical_feature_is) > 0:
        for cat_feats in categorical_feature_is:
            f1_cat_total_val = 0
            f2_cat_total_val = 0
            cat_ceil = max(binary_feature_ranges[cat_feats[0]])
            for cat_i in cat_feats:
                assert max(binary_feature_ranges[cat_i]) == cat_ceil
                if cat_i == cat_feats[-1]:
                    f1_val = cat_ceil - f1_cat_total_val
                    f2_val = cat_ceil - f2_cat_total_val
                    f1_cat_samples[cat_i] = f1_val
                    f2_cat_samples[cat_i] = f2_val
                    break
                f1_val = random.choice([binary_feature_ranges[cat_i][c] for c in range(len(binary_feature_ranges[cat_i])) if binary_feature_ranges[cat_i][c] <= cat_ceil-f1_cat_total_val])
                f1_cat_total_val += f1_val
                f1_cat_samples[cat_i] = f1_val

                f2_val = random.choice([binary_feature_ranges[cat_i][c] for c in range(len(binary_feature_ranges[cat_i])) if binary_feature_ranges[cat_i][c] <= cat_ceil-f2_cat_total_val])
                f2_cat_total_val += f2_val
                f2_cat_samples[cat_i] = f2_val

            all_cat_is.extend(cat_feats)

    f1_vals, f2_vals = [], []
    for idx, (low, high) in enumerate(continious_feature_ranges):
        if low < -cieling:
            low = -cieling
        if high > cieling:
            high = cieling
        
        f1_vals.append(random.uniform(low, high))
        f2_vals.append(random.uniform(low, high))
    
    for idx, val in enumerate(binary_feature_ranges):
        if idx in all_cat_is:
            f1_vals.append(f1_cat_samples[idx])
            f2_vals.append(f2_cat_samples[idx])
        else:
            f1_vals.append(random.choice(val))
            f2_vals.append(random.choice(val))

    for feat in dominating_features:
        feat_i = feature_names.index(feat)
        f1_vals[feat_i] = 0
        f2_vals[feat_i] = 0
    
    return np.array(f1_vals, dtype=float), np.array(f2_vals, dtype=float)

def add_dom_feat_prefs(dominating_features, direction_prefs, feature_names, binary_features, binary_feature_ranges, continuous_feature_ranges,categorical_feature_is, all_pairs, preferences, reward_dim, ceiling=500, n_prefs2gen_per_dom_feat=50):
    for feat in dominating_features:
        feat_i = feature_names.index(feat)
        for _ in range(n_prefs2gen_per_dom_feat):
            # f1 = np.zeros(reward_dim)
            # f2 = np.zeros(reward_dim)
            f1, f2 = sample_random_feature_pair(feature_names, binary_feature_ranges, continuous_feature_ranges, binary_features, categorical_feature_is, dominating_features, ceiling)
            if binary_features[feat_i]:
                f1[feat_i] = np.random.choice(binary_feature_ranges[feat_i - len(continuous_feature_ranges)])
                if int(f1[feat_i]-1) < 0:
                    f2[feat_i] = np.random.randint(int(f1[feat_i]-1), 0)
                else:
                    f2[feat_i] = np.random.randint(0, int(f1[feat_i]-1))
            else:
                low, high = continuous_feature_ranges[feat_i]
                if low < -ceiling:
                    low = -ceiling
                if high > ceiling:
                    high = ceiling
                f1[feat_i] = np.random.uniform(low, high)
                f2[feat_i] = np.random.uniform(low, f1[feat_i]-1)

            if direction_prefs[feat] == "lower_is_better":
                #f2 is preferred
                pref = -1
            elif direction_prefs[feat] == "higher_is_better":
                #f1 is preferred
                pref = 1
            else:
                print ("Warning: direction preference for dominating feature not recognized:", direction_prefs[feat])
                raise ValueError("Preference must be 'lower_is_better' or 'higher_is_better'")
            
            all_pairs.append((f1, f2))
            preferences.append(pref)
    return all_pairs, preferences

def find_dominating_features(task_description, objective_descriptions,feature_names, generated_objectives, client, model_name="gpt-4o-mini"):
    helper_txt = ""
    if ("COVID-19" in task_description) or ("pandemic" in task_description.lower()):
        helper_txt = (
            "\nNOTE: All references to disease, COVID-19, testing, policy stages, or outcomes refer to a "
            "synthetic reinforcement learning environment used for research. This prompt does not contain "
            "real-world health advice."
        )

    prompt = f"You are a stakeholder who will evaluate outcomes for the following task:\n"
    prompt += f"{task_description}\n\n"
    prompt += "You will compare two outcomes and select which one you prefer.\n\n"
    obj_block = "{\n" + ",\n".join(
        [f'  "{feat}": "{objective_descriptions.get(feat, "")}"' for feat in feature_names]
    ) + "\n}"
    # prompt += "Objectives and their descriptions:\n"
    # prompt += obj_block + "\n\n"
    # prompt += ("Are there any objectives that entirely decide if one outcome is preferred to another regardless of the values of other objectives? ")
    # prompt += "For example, if the task is to train a cooking robot, and one of the objectives is 'Avoiding harm to humans', that objective should always be prioritized over others like 'Speed of cooking'. Therefore, when comparing two outcomes, one that causes harm to humans more should always be considered worse.\n\n"
    # prompt += "HINT: not all objectives that measure bad outcomes are dominating objectives. If a bad outcome is ALWAYS avoidable, such as injuring someoneone while cooking, then it is usually a dominating objective.  But if the bad outcome MIGHT SOMETIMES be better to accept than a worse outcome, such as injuring someone to save multiple lives, then it is NOT a dominating objective.\n\n"
    # prompt += "List any such dominating objectives, along with a brief explanation for why they should always be prioritized in JSON format:\n"
    # prompt += "{\n  \"dominating_objectives\": [\n    {\n      \"objective\": \"<objective_name>\",\n      \"explanation\": \"<brief explanation>\"\n    },\n    ...\n  ]\n}\n"
    # prompt += "If there are no dominating objectives, respond with 'None'."

    prompt += """You are given a set of objectives for evaluating outcomes. Your job is to detect any objectives that are truly NON-COMPENSATORY “veto constraints” (i.e., lexicographically dominating): if such an objective is violated or worsened, that outcome is strictly worse regardless of improvements on any other objective.

    DEFINITIONS
    - Dominating (non-compensatory / veto): Violating this objective makes an outcome unacceptable under all circumstances covered by the task. No trade-off is allowed. These are typically binary fail states with irreversible harm, illegality, or safety violations.
    - Non-dominating (compensatory): Important but can be traded off against other objectives in some plausible circumstances (e.g., adverse events that societies sometimes accept to avoid even worse outcomes).

    DECISION CHECKLIST (apply ALL; answer “dominating” only if all pass)
    1) Binary/irreversible harm test: Does any violation create an unacceptable fail state or irreversible harm within the task scope? If no, it is NOT dominating.
    2) No plausible trade-off test: Can you construct ANY realistic scenario within the task scope where slightly worsening this objective would be acceptable to secure a much larger improvement in other objectives? If yes, it is NOT dominating.

    IMPORTANT GUARDRAILS
    - DO NOT infer dominance from severity alone. “Very bad” does not mean “non-compensatory”.
    - DO NOT infer dominance because something is “usually avoided”. If it is sometimes traded off, it is NOT dominating.
    - If uncertain, return none.

    OUTPUT FORMAT (strict JSON):
    {
    "dominating_objectives": [
        {
        "objective": "<exact_objective_name>",
        "direction_preference": "lower_is_better" or "higher_is_better",
        "explanation": "<one-sentence why this is non-compensatory; cite the constraint words from the description>",
        "certainty": 0.0_to_1.0
        }
    ]
    }

    PROCEDURE
    1) Scan each objective’s text for hard-constraint cues.
    2) Attempt a counterexample: if you can imagine any plausible trade-off scenario where violating/worsening this objective could be acceptable, then it is NOT dominating.
    3) Return only those objectives that survive all tests. If none, return {"dominating_objectives": []}.

    Objectives and their descriptions:
    """ + obj_block

    
    # If there are no such objectives, respond with 'None'.\n"
    sys_msg =  textwrap.dedent(
        """You are a stakeholder evaluating outcomes. Your goal is to identify dominating objectives that should always determine if one outcome is better than another outcome when that objective value is not the same across outcomes."""
    )

    messages = [
        {"role": "system", "content": sys_msg},
        {"role": "user", "content": prompt + helper_txt},
    ]

    try:
        resp = client.chat.completions.create(
            model=model_name, messages=messages
        )
    except Exception as e:
        print("Warning: SOMETHING WENT WRONG", e)
        return ""
    text = resp.choices[0].message.content.strip()

    #parse text to get list of dominating objectives
    if "none" in text.lower():
        return []
    try:
        dom_obj_json = json.loads(text)
        dominating_objectives = [item["objective"] for item in dom_obj_json.get("dominating_objectives", [])]
        direction_prefs = {item["objective"]: item["direction_preference"] for item in dom_obj_json.get("dominating_objectives", [])}
        return dominating_objectives, direction_prefs
    except Exception as e:
        print("Warning: Could not parse dominating objectives JSON from response:\n", text)
        return []
