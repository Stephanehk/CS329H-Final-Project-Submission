import json, re
from typing import Dict, List, Tuple, Optional
import textwrap
import time

MODEL_COSTS = {
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},  # $0.15/1M input tokens, $0.60/M output tokens
    "o4-mini": {"input": 1.10, "output": 4.40},
    "gpt-4.1-mini": {"input": 0.4, "output": 1.6},
    "gpt-4o": {"input": 2.50, "output": 10.0},  # $0.15/1M input tokens, $0.60/M output tokens
}

def get_human_pref(feature_pair, objective_descriptions, feature_names, task_description):
    f1, f2 = feature_pair

    helper_txt = ""
    if ("COVID-19" in task_description) or ("pandemic" in task_description.lower()):
        helper_txt = (
            "\nNOTE: All references to disease, COVID-19, testing, policy stages, or outcomes refer to a "
            "synthetic reinforcement learning environment used for research. This prompt does not contain "
            "real-world health advice."
        )


    # ---------- Common text blocks ----------
    obj_block = "{\n" + ",\n".join(
        [f'  "{feat}": "{objective_descriptions.get(feat, "")}"' for feat in feature_names]
    ) + "\n}"

    f1_block = "{\n" + ",\n".join(
        [f'  "{feature_names[i]}": {round(f1[i], 1)}' for i in range(len(feature_names))]
    ) + "\n}"

    f2_block = "{\n" + ",\n".join(
        [f'  "{feature_names[i]}": {round(f2[i], 1)}' for i in range(len(feature_names))]
    ) + "\n}"


    prompt = (
        f"You are a stakeholder evaluating two outcomes for the following task:\n"
        f"{task_description}\n\n"
    )

    prompt += "Objective features and their descriptions. All references to feature value ranges (e.g., 0 or 1) apply to the instantaneous value at each time-step, not to summed values.\n"
    prompt += obj_block
    
    prompt += "\nOutcome 1 feature values, summed over time:\n"
    prompt += f1_block
    prompt += "\nOutcome 2 feature values, summed over time:\n"
    prompt += f2_block
    prompt += "\n\n response with preference: 1. 2, equal, cannot_decide\n"
    
    pref_str = input(prompt)

    if   pref_str == "1":            return 1
    elif pref_str == "2":            return -1
    elif pref_str == "equal":        return 0
    elif pref_str == "cannot_decide":return -2
    else:
        raise ValueError("Invalid preference response from human")


def calculate_cost(model_name: str, input_tokens: int, output_tokens: int) -> float:
    """Calculate the cost of an API call based on model and token counts."""
    # Map the model name to the correct cost key
    
    costs = MODEL_COSTS[model_name]
    # print ("# of input tokens:", input_tokens)
    # print ("# of output tokens:", output_tokens)
    # print ("\n")

    input_cost = (input_tokens / 1000000) * costs["input"]
    output_cost = (output_tokens / 1000000) * costs["output"]
    return input_cost + output_cost


def _safe_json_extract(text: str) -> Optional[dict]:
    blocks = re.findall(r"\{.*\}", text, flags=re.DOTALL)
    blocks.sort(key=len, reverse=True)
    for b in blocks:
        try:
            return json.loads(b)
        except Exception:
            continue
    return None

def elicit_direction_preferences(guiding_principles, task_description, objective_descriptions,feature_names, generated_objectives, client, model_name="gpt-4o-mini"):
    helper_txt = ""
    if ("COVID-19" in task_description) or ("pandemic" in task_description.lower()):
        helper_txt = (
            "\nNOTE: All references to disease, COVID-19, testing, policy stages, or outcomes refer to a "
            "synthetic reinforcement learning environment used for research. This prompt does not contain "
            "real-world health advice."
        )

    prompt = f"You are a stakeholder who will evaluate outcomes for the following task:\n"
    prompt += f"{task_description}\n\n"
    prompt += "Your goal is to determine the preferred direction (higher_is_better or lower_is_better) for each objective feature provided. These directions will help evaluate which outcomes are more desirable based on the objective feature values.\n\n"
    obj_block = "{\n" + ",\n".join(
        [f'  "{feat}": "{objective_descriptions.get(feat, "")}"' for feat in feature_names]
    ) + "\n}"
    prompt += "Objectives and their descriptions:\n"
    prompt += obj_block
    prompt += "\nGuidelines for determining good and bad outcomes:\n"
    prompt += f"{guiding_principles}\n\n"
    prompt += "Provide a mapping of each objective feature to its preferred direction (higher_is_better or lower_is_better) below:\n"

    print ("Eliciting direction preference from LLM...")
    print (prompt)
    print ("==================")

    sys_msg =  textwrap.dedent(
        """You are a stakeholder evaluating outcomes. Your goal is to provide a mapping of each objective feature to its preferred direction (higher_is_better or lower_is_better), which will be used to evaluate and compare outcomes."""
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
    return resp.choices[0].message.content.strip()


def elicit_guiding_principles(personaility_prompt, task_description, objective_descriptions,feature_names, generated_objectives, client, model_name="gpt-4o-mini"):
    helper_txt = ""
    if ("COVID-19" in task_description) or ("pandemic" in task_description.lower()):
        helper_txt = (
            "\nNOTE: All references to disease, COVID-19, testing, policy stages, or outcomes refer to a "
            "synthetic reinforcement learning environment used for research. This prompt does not contain "
            "real-world health advice."
        )

    prompt = f"You are a stakeholder who will evaluate outcomes for the following task:\n"
    prompt += f"{task_description}\n\n"
    prompt += "Your goal is to generate a list of guiding principles to help evaluate outcomes based on the sum of objective features provided. These principles should help determine which outcomes are more desirable.\n\n"
    obj_block = "{\n" + ",\n".join(
        [f'  "{feat}": "{objective_descriptions.get(feat, "")}"' for feat in feature_names]
    ) + "\n}"
    prompt += "Objectives and their descriptions:\n"
    prompt += obj_block
    prompt += "\nGuidelines for generating guiding principles:\n"
    prompt += "1. Generate principles to evaluate outcomes based on the objective feature values provided.\n"
    prompt += "2. Outcomes will be presented to you as the sum of each objective feature over time. You will not have access to information about individual time steps or a sequence of time steps.\n"
    prompt += "3. Consider the trade-offs between different objectives when formulating principles.\n"
    prompt += "4. List if and when higher and lower values of each objective feature are preferable.\n"
    prompt += "Example: If an objective feature is 'StepCount', and the task is to 'promote a healthy lifestyle' then a higher value is generally better as it indicates more physical activity. But in the presence of high counts for another objective such as 'DaysOfInjury', a lower value may be preferable to avoid injury.\n\n"
    prompt += "5. Aim for principles that are clear, actionable, and aligned with the overall goals of the task. You will later use these principles to guide your evaluation of task outcomes.\n"
    prompt += "6. If an objective feature is not relevant to the task or is unlikely to be useful when evaluating outcomes, state that it should not be considered when evaluating outcomes and why.\n"
    prompt += "Use the implementation of each objective feature below to help you understand what each objective measures. Each objective feature is implemented as a RewardFunction subclass.\n"
    prompt += "\nImplemented Objective Features:\n"
    prompt += f"{generated_objectives}\n\n"
    if personaility_prompt != None:
        prompt += "IMPORTANT - Your guiding principles must be aligned with the stakeholder goals and principles outlined below:\n"
        prompt += f"{personaility_prompt}\n"
        prompt += f"Restate the stakeholder goals and principles above, and ensure all guiding principles are aligned with the stakeholder goals and principles.\n"
    # if personaility_prompt != None:
    #     prompt += "IMPORTANT - Your guiding principles must be aligned with the stakeholder goals and principles outlined below:\n"
    #     prompt += f"{personaility_prompt}\n"
    #     # prompt += f"Restate the stakeholder goals and principles above, and ensure all guiding principles are aligned with the stakeholder goals and principles.\n"
    #     prompt += "In addition to generating a list of guiding principles, identify which patterns of objective values are desirable and undesirable given the stakeholder goals and principles above. For example, if the stakeholder goal is to 'promote a healthy lifestyle', then a higher value of 'StepCount' is desirable, but a higher value of 'DaysOfInjury' is undesirable. Also identify which patterns are not desirable given the stakeholder goals and principles above.\n"
    #     prompt += "For each pattern, provide a name, a description of the pattern, and the objective features involved. For example, if the pattern is 'a higher value of 'StepCount' is desirable, but a higher value of 'DaysOfInjury' is undesirable, then the name is 'Healthy Lifestyle', the description is 'a higher value of 'StepCount' is desirable, but a higher value of 'DaysOfInjury' is undesirable', and the objective features involved are 'StepCount' and 'DaysOfInjury'.\n"
    #     prompt += "Please provide a list of desirable and undesirable patterns below:\n"
    #     prompt += "Name: [name]\n"
    #     prompt += "Description: [description]\n"
    #     prompt += "Objective Features Involved: [list of objective features]\n"
    #     prompt += "Is this pattern desirable or undesirable? [desirable/undesirable]\n"

        prompt += "Please provide a list of patterns to watch out for below, and a list of guiding principles to help you evaluate outcomes based on the objective features provided:\n"
    else:
        prompt += "Please provide a list of guiding principles below:\n"

    print ("Eliciting guiding principles from LLM...")
    print (prompt)
    print ("==================")

    sys_msg =  textwrap.dedent(
        """You are a stakeholder evaluating outcomes. Your goal is to generate guiding principles to help evaluate outcomes based on multiple objectives."""
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
    return resp.choices[0].message.content.strip()

def _weighted_margin(
    f1: List[float],
    f2: List[float],
    feature_names: List[str],
    weights: Dict[str, float],
) -> float:
    """
    Positive -> prefer f1; negative -> prefer f2.
    RAW feature differences (no normalization).
    """
    margin = 0.0
    wsum = 0.0
    f1_score = 0
    f2_score =0
    for i, feat in enumerate(feature_names):
        if feat not in weights:
            continue
        w = float(weights[feat])
        if w <= 0:
            continue
        diff = f1[i] - f2[i]  # positive favors f1 when higher_is_better
        # if directions[feat] == "lower_is_better":
        #     diff = -diff
        margin += w * diff
        wsum += w

        f1_score += w * f1[i]
        f2_score += w * f2[i]
    if wsum > 0:
        margin /= wsum
    return margin,f1_score, f2_score


def elicit_LLM_pref(
    feature_pair: Tuple[List[float], List[float]],
    objective_descriptions: Dict[str, str],
    dominating_features: List[str],
    feature_names: List[str],
    task_description: str,
    client,
    total_cost,
    model_name: str = "gpt-4o-mini",
    *,
    direct_llm_preference: bool = True,
    min_confidence: float = 0.6,
    abstain_on_contradictions: bool = True,
    categorical_features: Optional[List[str]] = None,
    guiding_principles = None,
    direction_pref_guidance = None,
    max_retries=5):

    n_retires = 0
    is_valid=False
    advice = ""

    # Mask out dominating features
    feature_pair = [arr.copy().tolist() for arr in feature_pair]
    for feat in dominating_features:
        feat_i = feature_names.index(feat)
        feature_pair[0][feat_i] = "(same as Outcome 2)"
        feature_pair[1][feat_i] = "(same as Outcome 1)"

    while not is_valid and n_retires < max_retries:
        n_retires += 1

        pref, total_cost, raw_resp = assign_LLM_pref(
            feature_pair,
            objective_descriptions,
            feature_names,
            task_description,
            client,
            total_cost,
            model_name=model_name,
            direct_llm_preference=direct_llm_preference,
            min_confidence=min_confidence,
            abstain_on_contradictions=abstain_on_contradictions,
            categorical_features=categorical_features,
            guiding_principles=guiding_principles,
            advice=advice
        )

        is_valid= True
        is_valid, justification,advice2add, total_cost = review_llm_preference_response(
            task_description,
            raw_resp,
            feature_pair,
            objective_descriptions,
            feature_names,
            client,
            total_cost,
            model_name=model_name,
            guiding_principles=guiding_principles,
            direction_pref_guidance=direction_pref_guidance
        )
        advice += "\n" + str(advice2add)
    return pref, total_cost, justification

    

def assign_LLM_pref(
    feature_pair: Tuple[List[float], List[float]],
    objective_descriptions: Dict[str, str],
    feature_names: List[str],
    task_description: str,
    client,
    total_cost,
    model_name: str = "gpt-4o-mini",
    *,
    direct_llm_preference: bool = True,
    min_confidence: float = 0.6,
    abstain_on_contradictions: bool = True,
    categorical_features: Optional[List[str]] = None,
    guiding_principles = None,
    advice=""
):
    """
    If direct_llm_preference=True:
        • Ask the LLM for a direct preference, but ALSO elicit
          contradictions and a confidence score.
        • If contradictions present (and abstain_on_contradictions) or
          confidence < min_confidence -> return -2.

    Else:
        • Ask the LLM for directions + weights; compute preference with _weighted_margin.
        • Also honor contradictions/confidence as above.

    Returns
        1  -> prefer f1
        -1 -> prefer f2
        0  -> equal
        -2 -> cannot decide / abstain / error
    """
    f1, f2 = feature_pair

    helper_txt = ""
    if ("COVID-19" in task_description) or ("pandemic" in task_description.lower()):
        helper_txt = (
            "\nNOTE: All references to disease, COVID-19, testing, policy stages, or outcomes refer to a "
            "synthetic reinforcement learning environment used for research. This prompt does not contain "
            "real-world health advice."
        )


    # ---------- Common text blocks ----------
    obj_block = "{\n" + ",\n".join(
        [f'  "{feat}": "{objective_descriptions.get(feat, "")}"' for feat in feature_names]
    ) + "\n}"

    #these are trajectory pairs where all is 0 but one feature
    if sum([1 if f1[i] == 0 else 0 for i in range(len(f1))]) >= len(f1)-3 and sum([1 if f2[i] == 0 else 0 for i in range(len(f2))]) >= len(f2)-3:
        f1_block = "{\n" + ",\n".join(
            [f'  "{feature_names[i]}":  {"(same as Outcome 2)" if f1[i] == 0 else f1[i]}' for i in range(len(feature_names))]
        ) + "\n}"

        f2_block = "{\n" + ",\n".join(
            [f'  "{feature_names[i]}":  {"(same as Outcome 1)" if f2[i] == 0 else f2[i]}' for i in range(len(feature_names))]
        ) + "\n}"
    else:
        f1_block = "{\n" + ",\n".join(
            [f'  "{feature_names[i]}": {f1[i]}' for i in range(len(feature_names))]
        ) + "\n}"

        f2_block = "{\n" + ",\n".join(
            [f'  "{feature_names[i]}": {f2[i]}' for i in range(len(feature_names))]
        ) + "\n}"

    assert guiding_principles is not None, "Guiding principles must be provided." #tecnicallly we do not need this but then modify the prompt below to not refer to it
    prompt = (
        f"You are a stakeholder evaluating two outcomes for the following task:\n"
        f"{task_description}\n\n"
        "Your role is to carefully assess the provided outcomes based solely on the listed objective features and their values. Values measure the objective, but do not indicate desirability. Outcomes are presented as the sum of each objective feature over time.\n"
        "Example 1: if an objective feature is 'NumberOfTestsAdministered', then a value of 10 means a total of 10 tests were administered. Note that this is the cumulative total; you do not know how many tests were administered at each time step.\n\n"
        "Example 2: if an objective feature is 'AccidentRate', then a value of 0.1 means an accident rate of 0.1.\n\n"
        "Guidelines for evaluation:\n"
        "1. Evaluate each outcome based on the sum of objective feature values provided.\n"
        "2. Critically consider how and why each objective feature value and their combinations might indicate the desirability of an outcome. For each objective:\n"
        "   - Think explicitly about whether higher or lower for each objective feature are beneficial or harmful, and how they interact to create an overall picture of desirability. USE THE Guiding Principles below to inform your evaluation about the desirable direction of each feature.\n"
        "   - Usually, for non-categorical objective features, a more negative objective feature value is less desirable, and a more positive value is more desirable. But this is not always the case. Rely on the objective feature descriptions provided below.\n"
        "   - Keep in mind that not all objective features may be relevant or informative; there may be some objective features that do not impact how desirable an outcome is.\n"
        "   - NOT all objective features are equally important or relevant; some are more critical than others. The relative weighting of objective features is for you to decide.\n"
        "   - Do NOT simply count the number of favorable objective feature values for each trajectory. A single or small number of highly important counts may outweigh a larger number of less important ones.\n"
        "   - Consider interactions between objective features: some combinations may be logically impossible or internally inconsistent.\n"
        "3. Use a step-by-step Chain of Thought (CoT) reasoning to justify your preference. Clearly articulate your thought process, considering each objective feature individually and collectively, and explicitly weigh more important objective features more heavily in your reasoning.\n"
        "5. Prefer the outcome that you deem more desirable.\n"
        "6. If the outcomes have equivalent value across all credible objective features, respond with 'equal'.\n"
        "7. If the outcomes have the same value for a specific objective feature, consider that objective feature as not informative for your preference.\n"
        "8. If an outcome includes logically impossible or internally contradictory feature (e.g., distance = 0 and speed = 60), respond with 'cannot decide'.\n"
        "9. Provide a confidence score in [0,1] reflecting reliability of your choice.\n"
    )

    if categorical_features is not None and len(categorical_features) > 0:
        prompt += "IMPORTANT - Carefully consider the categorical features below:\n"
        prompt += f"{categorical_features}\n"
        prompt += "Categorical feature values represent counts of discrete categories (e.g., number of items in category A, B, C).\n"
        prompt += "More of each categorical feature is NOT always better; these are not quantities to maximize. In fact, more of one categorical feature means less of another. You must decide which categorical feature(s) are most desirable and weigh them accordingly.\n"
        prompt += "For example, if a categorical feature represents a user satisfaction level where 0 is dissatisfied, 1 is neutral, and 2 is satisfied, then higher counts of 2 is best, and lower counts of 0 are best.\n"
        prompt += "Carefully consider what each categorical feature represents---outlined in the feature descriptions above---and how it impacts the desirability of an outcome.\n"

    if guiding_principles is not None:
        prompt += "\nIMPORTANT - Here are the guiding principles you MUST USE to evaluate the outcomes based on the objective features provided:\n"
        prompt += f"{guiding_principles}\n"

    if advice != "":
        prompt += "\nIMPORTANT - Here is advice you MUST FOLLOW when evaluating outcomes. This advice is based on things you previously got wrong:\n"
        prompt += f"{advice}\n"

    prompt += "Objective features and their descriptions. All references to feature value ranges (e.g., 0 or 1) apply to the instantaneous value at each time-step, not to summed values.\n"
    prompt += obj_block
    # sys_msg = (
    #     "You are a stakeholder evaluating two outcomes based on multiple objectives. "
    #     "Features can be noisy, redundant, or contradictory. "
    #     "Reason step-by-step, then output STRICT JSON containing contradictions, confidence, and preference."
    # )
    prompt += "\nOutcome 1 feature values, summed over time:\n"
    prompt += f1_block
    prompt += "\nOutcome 2 feature values, summed over time:\n"
    prompt += f2_block

    print ("=====ELICITING PREFERENCES OVER FEATURE PAIR=====")
    print ("\nOutcome 1 feature values:\n", f1_block)
    print ("\nOutcome 2 feature values:\n", f2_block)
    print ("=============================================")
    # =====================================================================
    #  PATH A: Direct preference with contradictions + confidence
    # =====================================================================
    if direct_llm_preference:
    
        prompt +=  textwrap.dedent(
        """
        Using Chain of Thought reasoning, carefully explain your reasoning step-by-step.
        Then, OUTPUT STRICT JSON ONLY with exactly these keys:
        {
            "rationale_brief": "<text>",
            "contradictions": [ "<text>", ... ],
            "confidence": <float>,
            "preference": "1" | "2" | "equal" | "cannot_decide"
        }
        Output guidelines:
        1. Provide a brief rationale in 'rationale_brief' explaining your reasoning, including your Chain of Thought (CoT).
        2. Provide contradictions or impossible combinations of objective values in 'contradictions' only if they exist.
        3. Provide a confidence score in [0,1] reflecting reliability of your choice in 'confidence'. Only use confidence > 0.9 if you are extremely confident in your choice of the best outcome. Use < 0.1 if you are completely unsure.
        4. Set 'preference' to '1' if you prefer Outcome 1, '2' for Outcome 2, 'equal' if they are equivalent, or 'cannot_decide' if you cannot reliably choose.
        """
        )

        sys_msg =  textwrap.dedent(
            """You are a stakeholder evaluating system outcomes based on multiple objectives. 
            You will be given two outcomes and asked to choose which one you prefer. 
            First think step-by-step, then output STRICT JSON in the form:
            {
                "rationale_brief": "<text>",
                "contradictions": [ "<text>", ... ],
                "confidence": <float>,
                "preference": "1" | "2" | "equal" | "cannot_decide"
            }"""
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
            return -2,total_cost, ""

        # cost
        try:
            in_tok = resp.usage.prompt_tokens
            out_tok = resp.usage.completion_tokens
        except Exception:
            in_tok = out_tok = 0
        cost = calculate_cost(model_name, in_tok, out_tok)
        total_cost += cost
        print(f"\nTotal cost of preference queries : ${total_cost:}")

        text = resp.choices[0].message.content.strip()
        print("------------------")
        print("LLM full response:\n", text)
        print("------------------")

        data = _safe_json_extract(text)

        if data is None:
            # # Fallback to PREFERENCE line if JSON missing; no contradictions/confidence available.
            # m = re.findall(r"PREFERENCE:\s*(1|2|equal|cannot\s*decide)", text, flags=re.I)
            # if not m:
            #     return -2
            # pref = m[-1].lower().replace(" ", "")
            # # Without JSON we cannot reliably enforce contradictions/confidence; abstain to be safe.
            print ("WARNING, DATA COULD NOT BE EXTRACTED FROM LLM RESPONSE")
            return -2, total_cost, ""

        # Validate required keys
        for k in ["contradictions", "confidence", "preference"]:
            if k not in data:
                print("Missing key in JSON:", k)
                return -2, total_cost, ""

        contradictions = data.get("contradictions", []) or []
        confidence = float(data.get("confidence", 0.0))
        pref_str = str(data.get("preference", "")).lower()

        # print ("----Extracted data from LLM response----")
        # print("Contradictions:", contradictions)
        # print("Confidence:", confidence)
        # print("Preference:", pref_str)
        # print("-----------------------------------------")
        # print ("(comment out time.sleep later...)")
        # time.sleep(30)

        # Apply abstention rules
        if abstain_on_contradictions and len(contradictions) > 0:
            return -2, total_cost, text
        if confidence < min_confidence:
            return -2, total_cost, text

        # Map preference
        if   pref_str == "1":           return 1,total_cost, text
        elif pref_str == "2":           return -1,total_cost, text
        elif pref_str == "equal":       return 0,total_cost, text
        elif pref_str == "cannot_decide": return -2,total_cost, text
        else:                           return -2,total_cost, text

    # =====================================================================
    #  PATH B: Weight elicitation + numeric aggregation (RAW features)
    # =====================================================================
    prompt += textwrap.dedent(
        """
        Using Chain of Thought reasoning, carefully explain your reasoning step-by-step.
        Then, OUTPUT STRICT JSON ONLY with exactly these keys:
        {
            "rationale_brief": "<text>",
            "contradictions": [ "<text>", ... ],
            "confidence": <float>,
            "preference": "1" | "2" | "equal" | "cannot_decide"
            "weights": { "<objective>": <float> },
        }\n
        """
        )
    prompt += (
            "Output guidelines:\n"
            "1. Provide a brief rationale in 'rationale_brief' explaining your reasoning, including your Chain of Thought (CoT).\n"
            "2. Provide contradictions or impossible combinations of objective values in 'contradictions' only if they exist.\n"
            "3. Provide a confidence score in [0,1] reflecting reliability of your choice in 'confidence'. Only use confidence > 0.9 if you are extremely confident in your choice of the best outcome. Use < 0.1 if you are completely unsure.\n"
            "4. Set 'preference' to '1' if Outcome 1 is preferred to Outcome 2, '2' if Outcome 2 is preferred to Outcome 1, 'equal' if Outcome 1 is equal to Outcome 1, or 'cannot_decide' if you cannot reliably choose.\n"
            "5. To explain your preference, for each objective, assign a weight in [-1, 1] indicating its importance in 'weights'. The desirability of each outcome should entirely be determined by the linear combination of the outcome's objective values and the weights you assign to each objective, where higher is better."
            "IMPORTANT considerations for assigning weights:"
            "- the weights you assign must explain your preference between outcomes.\n"
            "- you will rarely want to assign a weight < 0. The objective descriptions already indicate whether higher or lower values are better, so a negative weight would mean you prefer the opposite of what the objective indicates. Your preference may still be correct even if you need to assign a negative weight, but think carefully about doing so as it is often not needed.\n"
        )

    sys_msg = textwrap.dedent(
        """You are a stakeholder evaluating system outcomes based on multiple objectives. 
        You will be given two outcomes and asked to choose which one you prefer based on the weights that you assign to each objective used to describe the outcomes.
        First think step-by-step, then output STRICT JSON in the form:
        {
            "rationale_brief": "<text>",
            "contradictions": [ "<text>", ... ],
            "confidence": <float>,
            "preference": "1" | "2" | "equal" | "cannot_decide"
            "weights": { "<objective>": <float> },
        }
        """
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
        return -2, total_cost, ""

    # cost
    try:
        in_tok = resp.usage.prompt_tokens
        out_tok = resp.usage.completion_tokens
    except Exception:
        in_tok = out_tok = 0
    cost = calculate_cost(model_name, in_tok, out_tok)
    total_cost += cost
    print(f"\nTotal cost of preference queries : ${total_cost:.4f}")

    text = resp.choices[0].message.content.strip()
    print("------------------")
    print("LLM full response:\n", text)
    print("------------------")

    data = _safe_json_extract(text)
    if data is None:
        print ("WARNING, DATA COULD NOT BE EXTRACTED FROM LLM RESPONSE")
        return -2,total_cost, ""

    required = [
        "weights",
        "confidence", "preference", "contradictions"
    ]
    for k in required:
        if k not in data:
            print("Missing key in JSON:", k)
            return -2,total_cost, ""

    weights = data["weights"]
    confidence = float(data.get("confidence", 0.0))
    contradictions = data.get("contradictions", []) or []
    pref_str = str(data.get("preference", "")).lower()

    # Our numeric margin using RAW features
    our_margin,f1_score,f2_score = _weighted_margin(f1, f2, feature_names, weights)

    # print ("----Extracted data from LLM response----")
    # print("Weights:", weights)
    # print ("Computed scores using weights:",(f1_score, f2_score))
    # print("Our margin:", our_margin)
    # print("Contradictions:", contradictions)
    # print("Confidence:", confidence)
    # print("Preference:", pref_str)
    # print("-----------------------------------------")
    # print ("(comment out time.sleep later...)")
    # time.sleep(30)


    # Abstention rules
    if abstain_on_contradictions and len(contradictions) > 0:
        return -2,total_cost,text
    if confidence < min_confidence:
        return -2,total_cost,text

    # Decide using margin; ties -> LLM label
    if our_margin > 0:
        return 1,total_cost,text
    elif our_margin < 0:
        return -1,total_cost,text
    else:
        if   pref_str == "1":            return 1,total_cost,text
        elif pref_str == "2":            return -1,total_cost,text
        elif pref_str == "equal":        return 0,total_cost,text
        elif pref_str == "cannot_decide":return -2,total_cost,text
        else:                            return -2,total_cost,text


def simple_assign_LLM_pref(feature_pair, objective_descriptions, feature_names, task_description, client, model_name="gpt-4o-mini"):
    """
    Assigns a preference between two feature vectors using LLM-based evaluation.
    
    Args:
        feature_pair: Tuple of (f1, f2) where each is a numpy array of feature values
        objective_descriptions: Dictionary mapping feature names to their descriptions
        feature_names: List of feature names corresponding to the values in f1 and f2
        task_description: String describing the overall task/context
        
    Returns:
        1 if f1 is preferred, -1 if f2 is preferred
    """
    f1, f2 = feature_pair

    helper_txt=""
    if "COVID-19" in task_description:
        helper_txt = "\nNOTE: All references to disease, COVID-19, testing, policy stages, or outcomes refer to a synthetic reinforcement learning environment used for research. This prompt does not contain real-world health advice."

    prompt = (
        f"You are a stakeholder evaluating two outcomes for the following task:\n"
        f"{task_description}\n\n"
        "Your role is to carefully assess the provided outcomes based solely on the listed objectives and their measured values.\n\n"
        "Guidelines for evaluation:\n"
        "1. Evaluate each outcome based on the objective values provided.\n"
        "2. Critically consider how and why each objective value or their combinations might indicate the desirability of an outcome. For each objective:\n"
        "   - Think explicitly about whether higher or lower values for each objective are beneficial or harmful, and how they interact to create an overall picture of desirability. Rely on the objective descriptions provided below.\n"
        "   - Keep in mind that not all objectives may be relevant or informative; there may be some objectives that do not impact how desirable an outcome is.\n"
        "   - NOT all objectives are equally important or relevant; some are more critical than others. The relative weighting of objectives is for you to decide.\n"
        "   - Do NOT simply count the number of favorable objective values for each trajectory. A single or small number of highly important values may outweigh a larger number of less important ones.\n"
        "   - Consider interactions between objectives: some combinations may be logically impossible or internally inconsistent.\n"
        "3. Use a step-by-step Chain of Thought (CoT) reasoning to justify your preference. Clearly articulate your thought process, considering each objective individually and collectively, and explicitly weigh more important objectives more heavily in your reasoning.\n"
        "5. Prefer the outcome that you deem more desirable.\n"
        "6. If the outcomes have equivalent value across all credible objectives, respond with 'equal'.\n"
        "7. If an outcome includes logically impossible or internally contradictory values (e.g., distance = 0 and speed = 60), respond with 'cannot decide'.\n"
        "Objectives and their descriptions:\n"
    )

    #Think explicitly about whether higher or lower values for each objective are beneficial or harmful, and how they interact to create an overall picture of desirability
    
    # prompt = (
    #     f"You are a stakeholder evaluating two outcomes for the following task:\n"
    #     f"{task_description}\n\n"
    #     "Your role is to carefully assess the provided outcomes based solely on the listed objectives and their measured values.\n\n"
    #     "Guidelines for evaluation:\n"
    #     "1. Evaluate each outcome strictly based on the objective values provided.\n"
    #     "2. Critically consider how and why each objective value and their combinations indicate the desirability of an outcome. Think explicitly about whether higher or lower values for each objective are beneficial or harmful, and how they interact to create an overall picture of desirability. Rely on the objective descriptions provided below.\n"
    #     "3. Keep in mind that not all objectives are equally important or relevant; there may be some objectives that do not impact how desirable an outcome is.\n"
    #     "4. Use a step-by-step Chain of Thought (CoT) reasoning to carefully justify your preference. Clearly articulate your thought process, considering each objective individually and collectively.\n"
    #     "5. Prefer the outcome that better achieves your interests as indicated by the objective values.\n"
    #     "6. If outcomes have equivalent value across all objectives, respond with 'equal'.\n"
    #     "7. If an outcome includes logically impossible or contradictory values (e.g., distance traveled = 0 and average speed = 60), respond with 'cannot decide'.\n"
    #     "8. Do NOT infer or assume information not explicitly provided in the objectives.\n\n"
    #     "Objectives and their descriptions:\n"
    # )

    # Add objective descriptions
    for feature_name in feature_names:
        prompt += f"- {feature_name}: {objective_descriptions[feature_name]}\n"

    # Present outcome 1
    prompt += "\nOutcome 1:\n"
    for i, feature_name in enumerate(feature_names):
        prompt += f"- {feature_name}: {f1[i]}\n"

    # Present outcome 2
    prompt += "\nOutcome 2:\n"
    for i, feature_name in enumerate(feature_names):
        prompt += f"- {feature_name}: {f2[i]}\n"

    # Final instruction
    prompt += (
        "\nUsing Chain of Thought reasoning, carefully explain your reasoning step-by-step. After your reasoning, clearly indicate your preference with exactly one of the following responses:\n"
        "PREFERENCE: 1\n"
        "PREFERENCE: 2\n"
        "PREFERENCE: equal\n"
        "PREFERENCE: cannot decide\n"
        "Respond ONLY with your detailed reasoning followed by the 'PREFERENCE' line."
    )

    messages = [
        {
            "role": "system",
            "content": (
                "You are a stakeholder evaluating system outcomes based on multiple objectives. "
                "You will be given two outcomes and asked to choose which one you prefer. "
                "First think step-by-step, then finish with exactly one line of the form "
                "'PREFERENCE: 1', 'PREFERENCE: 2', 'PREFERENCE: equal', or 'PREFERENCE: cannot decide'."
            ),
        },
        {"role": "user", "content": prompt + helper_txt},
    ]

    # Make the API call
    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            # temperature=0.1,          # Low temperature for consistency
        )
    except Exception as e:
        print(f"Warning: SOMETHING WENT WRONG")
        print (e)
        return -2

    input_tokens = response.usage.prompt_tokens
    output_tokens = response.usage.completion_tokens
    cost = calculate_cost(model_name, input_tokens, output_tokens)
    global total_cost
    total_cost += cost
    print(f"\nTotal cost of preference queries : ${total_cost:.4f}")


    # --- Extract preference ---
    response_text = response.choices[0].message.content.strip()
    # Find ALL occurrences, take the last one
    matches = re.findall(
        r"PREFERENCE:\s*(1|2|equal|cannot decide)", response_text, flags=re.IGNORECASE
    )

    print("------------------")
    print("LLM full response:\n", response_text)
    print("Matches:", matches)
    print("------------------")

    if matches:
        # Take last match in case the model repeated the pattern earlier
        preference = matches[-1].lower()

        if preference == "1":
            return 1
        if preference == "2":
            return -1
        if preference == "equal":
            return 0
        if preference == "cannot decide":
            return -2

    # # Fallback: could not parse preference
    # print(f"Warning: Could not parse preference from response:\n{response_text}")
    # return random.choice([-1, 1])



def review_llm_preference_response(
    task_description: str,
    response_text: str,
    feature_pair,
    objective_descriptions,
    feature_names,
    client,
    total_cost: float,
    model_name: str = "gpt-4o-mini",
    guiding_principles = None,
    direction_pref_guidance = None
):
    
    
    # ---------- Reviewer prompt ----------
    helper_txt = ""
    if ("COVID-19" in task_description) or ("pandemic" in task_description.lower()):
        helper_txt = (
            "\nNOTE: All references to disease, COVID-19, testing, policy stages, or outcomes refer to a "
            "synthetic reinforcement learning environment used for research. This prompt does not contain "
            "real-world health advice."
        )

    f1, f2 = feature_pair
    obj_block = "{\n" + ",\n".join(
        [f'  "{feat}": "{objective_descriptions.get(feat, "")}"' for feat in feature_names]
    ) + "\n}"

    f1_block = "{\n" + ",\n".join(
        [f'  "{feature_names[i]}": {f1[i]}' for i in range(len(feature_names))]
    ) + "\n}"

    f2_block = "{\n" + ",\n".join(
        [f'  "{feature_names[i]}": {f2[i]}' for i in range(len(feature_names))]
    ) + "\n}"

    prompt = (
        f"You are auditing an earlier model's evaluation for the task:\n"
        f"{task_description}\n\n"
        "Only answer the following question: did the response correctly interpret the directionality and magnitude of each objective (i.e., whether higher/lower values are better or worse) based on the objective features and their mapping below? For example, if Outcome 1's SuccessRate is higher than Outcome 2's, as long as the response to audit does not say otherwise Outcome 1 or Outcome 2 are both valid preferences. But if Outcome 1's DeathToll is greater than Outcome 2's, and the response text incorrectly indicates that higher DeathToll is better, than the preference is invalid.\n"
        # "HINT: watch our for a rationale that claims 'extreme positive or negative values are [better, worse]' or 'near-zero values are better'. The directional desirability of each objective is fixed and does not change based on magnitude.\n\n" 
        "Mapping from objective features to preferred direction:\n"
        f"{direction_pref_guidance}\n\n"
        "Outcome 1 feature values:\n"
        f"{f1_block}\n\n"
        "Outcome 2 feature values:\n"
        f"{f2_block}\n"
    )

    # if guiding_principles is not None:
    #     prompt += "\nWere the following guiding principles respected:\n" + guiding_principles + "\n"
    prompt += f"\nBelow is the response to audit:\n{response_text}\n\n"
    prompt += "Now, OUTPUT STRICT JSON ONLY with exactly these keys:\n"
    prompt += (
        "{\n"
        '  "is_valid": true | false,\n'
        '  "justification": ["<explanation for is_valid value>", ...],\n'
        '  "advice": ["<if is_valid== false, list the objective feature that were misinterpreted and why>", ...],\n'
    )

    
    sys_msg = textwrap.dedent(
        """You are a strict, detail-oriented auditor. Check directionality/magnitude interpretation. Output STRICT JSON only."""
    )

    messages = [
        {"role": "system", "content": sys_msg},
        {"role": "user", "content": prompt + helper_txt},
    ]

    # ---------- Call reviewer LLM ----------
    try:
        resp = client.chat.completions.create(model=model_name, messages=messages)
    except Exception as e:
        print("Warning: SOMETHING WENT WRONG during audit:", e)
        return -2, {"error": str(e)}, total_cost

    # cost accounting
    try:
        in_tok = resp.usage.prompt_tokens
        out_tok = resp.usage.completion_tokens
    except Exception:
        in_tok = out_tok = 0
    cost = calculate_cost(model_name, in_tok, out_tok)
    total_cost += cost
    print(f"\nTotal cost after audit : ${total_cost:.4f}")

    try:
        audit_text = resp.choices[0].message.content.strip()
        audit = _safe_json_extract(audit_text)
    except Exception as e:
        audit = None
    if audit is None:
        print("Warning: Could not parse audit JSON from response:\n", audit_text)
        # If reviewer failed to output JSON, abstain
        return -2, {"parse_error": "audit JSON not found", "raw": audit_text}, total_cost
    

    is_valid = bool(audit.get("is_valid", False))
    justification = audit.get("justification", "")
    advice = audit.get("advice", "")

    # print ("====================PROMPT FOR ADUITING=======================")
    # print (prompt)
    print ("==================AUDITING OF PREFERENCE RESPONSE==================")
    print ("is_valid:", is_valid)
    print ("justification:", justification)
    print ("advice:", advice)

    print ("===================================================================")

    return is_valid, justification,advice, total_cost