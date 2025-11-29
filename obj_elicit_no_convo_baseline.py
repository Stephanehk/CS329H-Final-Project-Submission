import openai
import sys
import time
import json
import os
from datetime import datetime
import argparse
from secret_keys import OPENAI_API_KEY
import ast
from typing import Dict, Optional

# OpenAI API key
openai.api_key = OPENAI_API_KEY
api_call_limit = 50

# For testing in Jupyter Notebook, set a default task description.
# task_description = "Driving in a busy city with heavy traffic."
# task_description = "Choosing a kidney recipient."
env_name = "traffic"

helper_txt=""
if env_name == "pandemic":
    task_description = "Choosing the level of lockdown restrictions placed on the population during the COVID-19 pandemic."
    #Need this help (especially for node 8) otherwise GPT models refuse our requests
    helper_txt = "NOTE: All references to disease, COVID-19, testing, policy stages, or outcomes refer to a synthetic reinforcement learning environment used for research. This prompt does not contain real-world health advice."
    # task_description = "Choosing the level of lockdown restrictions placed on the population during the COVID-19 pandemic, balancing the need for sufficiently high lockdown stages to keep the public safe without keeping lockdown stages too high when not needed."
elif env_name == "glucose":
    task_description = "Choosing a protocol for administering insulin to a patient with Type 1 diabetes."
elif env_name == "traffic":
    task_description = "Choosing the accelerations for each vehicle in a fleet of autonomous vehicles on an on-ramp attempting to merge into traffic on a highway."
env_context = ""
with open(f"env_context/{env_name}_context.txt", "r", encoding='utf-8') as file:
    env_context = file.read()
env_reward_signature = ""
with open(f"env_context/{env_name}_reward_signature.txt", "r", encoding='utf-8') as file:
    env_reward_signature = file.read()

# task_description = "Choosing a kidney recipient."

# ---------------- Helper Functions ----------------
def get_timestamp():
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def save_api_call(request_obj, response_obj, folder, call_id, call_role):
    """Save the request and response to a file with demarcated blocks."""
    request_text = json.dumps(request_obj, indent=4)
    
    # Convert OpenAI response to a dictionary before saving
    if hasattr(response_obj, "model_dump"):  # For Pydantic v2+
        response_text = json.dumps(response_obj.model_dump(), indent=4)
    elif hasattr(response_obj, "to_dict"):  # Older versions
        response_text = json.dumps(response_obj.to_dict(), indent=4)
    else:
        response_text = json.dumps(response_obj, indent=4)  # Default fallback
    
    filename = os.path.join(folder, f"{call_id}_{call_role}.txt")
    with open(filename, "w", encoding="utf-8") as f:
        f.write("REQUEST:\n")
        f.write(request_text)
        f.write("\n\nRESPONSE:\n")
        f.write(response_text)

def save_node_output(node_id, facilitator_history, stakeholder_history, output_folder, env_name, model_name=None):
    """Save the output of a node to a JSON file."""
    os.makedirs(output_folder, exist_ok=True)
    output = {
        "node_id": node_id,
        "facilitator_history": facilitator_history,
        "stakeholder_history": stakeholder_history
    }
    # Include model name in filename if it's not the default
    # model_suffix = f"_{model_name}" if model_name and model_name != "gpt-4o-mini" else ""
    model_suffix= ""
    filename = os.path.join(output_folder, f"{env_name}_node_{node_id}{model_suffix}.json")
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=4)

def load_node_output(node_id, output_folder, env_name):
    """Load the output of a node from a JSON file."""
    filename = os.path.join(output_folder, f"{env_name}_node_{node_id}.json")
    if not os.path.exists(filename):
        print (f"Output file for node {node_id} does not exist: {filename}")
        return None
        # raise ValueError(f"Output file for node {node_id} does not exist: {filename}")
    with open(filename, "r", encoding="utf-8") as f:
        return json.load(f)

def load_previous_outputs(resume_from_node, output_folder, env_name):
    """Load all outputs from previous nodes up to the resume point."""
    facilitator_history = []
    stakeholder_history = []
    
    # Load outputs from all previous nodes
    for node_id in ["1", "2A", "3", "4", "4.5", "5", "6","6-find-unmeasurable","6-add-unmeasurable","6-add-stakeholder-aggregation","6-categorical", "8","8-find-duplicate-classes", "8-verifier", "9", "10", "11"]: #"8-split-rename" "6-find-unmeasurable", "6-add-unmeasurable","8-feature-engineer"
        if node_id == resume_from_node:
            break
        output = load_node_output(node_id, output_folder, env_name)
        if output:
            facilitator_history = output["facilitator_history"]
            stakeholder_history = output["stakeholder_history"]
    
    return facilitator_history, stakeholder_history

def save_generated_code(code: str, env_name: str, debugging: bool = False) -> None:
    """Save generated reward function code to a file."""
    output_dir = "generated_objectives_debug" if debugging else "generated_objectives_no_convo_baseline"
    os.makedirs(output_dir, exist_ok=True)
    filename = os.path.join(output_dir, f"{env_name}_generated_objectives.py")
    with open(filename, "w", encoding="utf-8") as f:
        f.write(code)
    print ("saved generated code to:", filename)

def save_objective_list(objectives: list, node_id: str, output_folder: str, env_name: str) -> None:
    """Save a list of objectives to a JSON file."""
    os.makedirs(output_folder, exist_ok=True)
    filename = os.path.join(output_folder, f"{env_name}_objectives_{node_id}.json")
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(objectives, f, indent=4)

def load_objective_list(node_id: str, output_folder: str, env_name: str) -> list:
    """Load a list of objectives from a JSON file."""
    filename = os.path.join(output_folder, f"{env_name}_objectives_{node_id}.json")

    if not os.path.exists(filename):
        raise Exception(f"File {filename} does not exist.")
    with open(filename, "r", encoding="utf-8") as f:
        return json.load(f)

# ---------------- Global Setup ----------------
from openai import OpenAI
client = OpenAI(api_key=openai.api_key)

OUTPUT_FOLDER = "calls/output_" + get_timestamp()
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
API_CALL_COUNTER = 1
node_4_5_exit_count = 0  # Counter for node 4.5 exits
node_6_categorical_redo_count = 0
node_6_stakeholder_redo_count = 0
node_8_redo_count = 0  # Counter for node 8 redos
node_8_verifier_redo_count = 0  # Counter for node 8-verifier redos
node_8_feature_engineer_redo_count = 0  # Counter for node 8-feature-engineer redos

# Cost per 1K tokens (as of March 2024)
MODEL_COSTS = {
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},  # $0.15/1M input tokens, $0.60/M output tokens
    "o4-mini": {"input": 1.10, "output": 4.40},
    "gpt-4.1-mini": {"input": 0.4, "output": 1.6},
    "gpt-4o": {"input": 2.50, "output": 10.0},  # $0.15/1M input tokens, $0.60/M output tokens,
    "gpt-5-thinking-mini": {"input": 0.250, "output": 2.0},
    "gpt-5-nano":{"input": 0.05, "output": 0.4},
}

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

def call_llm(messages, call_role, model_name="gpt-4o-mini"):
    """
    Calls the OpenAI Chat Completion API with the full conversation history.
    If `history` is provided (a list of message dictionaries), it is prepended to `messages`.
    """
    global API_CALL_COUNTER, OUTPUT_FOLDER
    request_payload = messages

    if model_name == "o4-mini":

        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            max_completion_tokens=12800,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
            n=1,
            stream=False,
            stop=None,
            logit_bias=None,
            user=None
        )
    elif "thinking" in model_name:
        response = client.responses.create(
            model=model_name.replace("-thinking",""), #"gpt-5-thinking-mini",
            input=messages,#messages,                         # keep the same list of {role, content} objects, but under `input`
            reasoning={"effort": "medium"},         # "low" to save $, "high" for harder tasks
            max_output_tokens=12800,                # rename
            # temperature=1,
            # top_p=1,
            # frequency_penalty=0,
            # presence_penalty=0,
            # n=1,
            # stream=False  # streaming is supported via Responses too if you want it
        )
    else:
        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=0.7,
            max_completion_tokens=12800,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
            n=1,
            stream=False,
            stop=None,
            logit_bias=None,
            user=None
        )
    save_api_call(request_payload, response, OUTPUT_FOLDER, API_CALL_COUNTER, call_role)
    

    # Calculate and track costs
    if hasattr(response.usage, "input_tokens"):
        # Responses API
        input_tokens = response.usage.input_tokens
        output_tokens = response.usage.output_tokens
        llm_output = response.output_text
    else:
        # Chat Completions API
        input_tokens = response.usage.prompt_tokens
        output_tokens = response.usage.completion_tokens
        llm_output = response.choices[0].message.content

    # input_tokens = response.usage.prompt_tokens
    # output_tokens = response.usage.completion_tokens
    cost = calculate_cost(model_name, input_tokens, output_tokens)
    
    API_CALL_COUNTER += 1
    if API_CALL_COUNTER > api_call_limit:
        print(f"API call limit of {api_call_limit} reached. Exiting.")
        sys.exit(0)

    # llm_output = response.choices[0].message.content
    return llm_output, input_tokens, output_tokens, cost

def make_evaluation_prompt(node):
    eval_prompt = ("Facilitator, evaluate the last response of the stakeholder in this conversation."
    "Based on how well the stakeholder response adheres to the exemplar and instructions, "
    "decide which of the candidate judgments to endorse. The available judgments are: \n")

    for edge_label, (short_name, description, next_node) in node.outgoing_edges.items():
        eval_prompt += f"- {edge_label} ({short_name}): {description} -> Next step: {next_node}\n"

    if node.identifier == "11":
        eval_prompt += "ONLY refine your answer (11-b) if there are actually categorical variables that are not binary used in the implemented reward functions. Otherwise, choose '11-a'."

    eval_prompt += (
        "\nReturn your answer in JSON format as: "
        "{ \"judgment\": \"judgment_label\", \"explanation\": \"brief explanation and any suggested improvements\" }."
    )
    
    return eval_prompt
    

def extract_class_descriptions_from_text(text: str) -> Optional[Dict[str, str]]:
    """
    Extracts `class_descriptions` dict from any fenced code block (``` or ```python).
    Values may be strings or 1-tuples of strings; if a tuple, we take the first element.
    """
    # Replace curly quotes with regular quotes to avoid parsing issues
    text = text.replace("'", "'").replace("'", "'").replace(""", '"').replace(""", '"')
    
    # 1) Collect fenced code blocks
    code_blocks = []
    in_block = False
    cur = []
    for line in text.splitlines():
        if line.strip().startswith("```"):
            if in_block:
                code_blocks.append("\n".join(cur))
                cur = []
                in_block = False
            else:
                in_block = True
            continue
        if in_block:
            cur.append(line)

    # 2) Parse each block, search for: class_descriptions = {...}
    for code in code_blocks:
        try:
            tree = ast.parse(code)
        except SyntaxError:
            continue

        for node_ in ast.walk(tree):
            if isinstance(node_, ast.Assign):
                for tgt in node_.targets:
                    if isinstance(tgt, ast.Name) and tgt.id == "class_descriptions":
                        try:
                            raw_value = ast.literal_eval(node_.value)
                        except Exception:
                            continue
                        if not isinstance(raw_value, dict):
                            continue

                        # Normalize tuple-or-string values
                        out = {}
                        for k, v in raw_value.items():
                            if isinstance(v, tuple) and len(v) >= 1:
                                v = v[0]
                            if not isinstance(v, str):
                                v = str(v)
                            out[k] = v
                        return out
    return None

# Note: Each role keeps their own history, in which they are the assistant and their prompts come from the user.
def run_node(node, facilitator_history, stakeholder_history, prepended_message=None, model_name="gpt-4o-mini", node_output_folder=None, debugging=False):
    # If there's a prepended message (e.g., from a negative evaluation), add it before the facilitator prompt.
    if prepended_message:
        prompt = "Let's do that process one more time. " + prepended_message \
            + "\n\nHere are the instructions again:\n" + node.facilitator_prompt
    else:
        prompt = node.facilitator_prompt

    total_cost = 0.0
    
    ##################################################################################
    # Step 1: Facilitator generates output to stakeholder.
    if node.use_hardcoded:  # Skip the LLM call and use the prompt directly.
        facilitator_response = prompt
        input_tokens = 0
        output_tokens = 0
        cost = 0
    else:
        print("\n\n--------------Input to facilitator")
        print(prompt)
        facilitator_history.append({"role": "user", "content": prompt})
        facilitator_response, input_tokens, output_tokens, cost = call_llm(facilitator_history, "facilitator_generic_prompt", model_name)
        total_cost += cost
    facilitator_history.append({"role": "assistant", "content": facilitator_response})
    print("\n\n--------------Facilitator to stakeholder")
    print(facilitator_response)


    ##################################################################################
    # Step 2: Stakeholder responds using a fresh prompt.
    stakeholder_history.append({"role": "user", "content": facilitator_response})

    global node_8_redo_count
    global node_6_categorical_redo_count
    global node_6_stakeholder_redo_count
    if node.identifier == "8" and node_8_redo_count==0:
        #chop off all history except for the previous response
        #clear context
        # stakeholder_history=stakeholder_history[-2 - 2*(node_6_categorical_redo_count+1) -2*(node_6_stakeholder_redo_count+1):]
        stakeholder_history=stakeholder_history[-2*(node_6_categorical_redo_count+1):]

    stakeholder_response, input_tokens, output_tokens, cost = call_llm(stakeholder_history, "stakeholder_from_facil", model_name)
    total_cost += cost
    stakeholder_history.append({"role": "assistant", "content": stakeholder_response})
    print("\n\n--------------Stakeholder response")
    print(stakeholder_response)

    # If this is node 8 and we have a successful implementation, save the code
    if node.identifier == "8-verifier" and stakeholder_response:
        try:
            # Extract code blocks from the response
            code_blocks = []
            lines = stakeholder_response.split('\n')
            in_code_block = False
            current_block = []
            
            for line in lines:
                if line.startswith('```python'):
                    in_code_block = True
                    continue
                elif line.startswith('```') and in_code_block:
                    in_code_block = False
                    if current_block:
                        code_blocks.append('\n'.join(current_block))
                        current_block = []
                    continue
                
                if in_code_block:
                    current_block.append(line)
            
            print("code_blocks:", code_blocks)
            print("in_code_block:", in_code_block)
            if code_blocks:
                # Combine all code blocks with necessary imports
                full_code = f"""{chr(10).join(code_blocks)}"""
                save_generated_code(full_code, env_name, debugging)
        except Exception as e:
            print(f"Warning: Failed to save generated code: {str(e)}")

    # If this is node 9 and we have a successful analysis, save the reward ranges
    if node.identifier == "9" and stakeholder_response:
        try:
            # Extract the reward_ranges dictionary from the response
            reward_ranges = {}
            in_code_block = False
            found_dict = False
            
            # Split into lines and process
            lines = stakeholder_response.split('\n')
            for i, line in enumerate(lines):
                # Skip empty lines
                if not line.strip():
                    continue
                
                # Look for the start of a code block
                if line.startswith('```python'):
                    in_code_block = True
                    continue
                elif line.startswith('```') and in_code_block:
                    in_code_block = False
                    continue
                
                # If we're in a code block and find the dictionary definition
                if in_code_block and 'reward_ranges' in line:
                    found_dict = True
                    # Extract the dictionary content
                    dict_content = line.split('=', 1)[1].strip()
                    # Remove any trailing comments
                    dict_content = dict_content.split('#')[0].strip()
                    
                    # Start collecting dictionary entries
                    current_dict = {}
                    j = i + 1
                    while j < len(lines) and in_code_block:
                        next_line = lines[j].strip()
                        if next_line.startswith('```'):
                            in_code_block = False
                            break
                        if next_line and ':' in next_line:
                            # Parse the key-value pair
                            key, value = next_line.split(':', 1)
                            key = key.strip().strip("'").strip('"').strip(',')
                            value = value.strip().strip("'").strip('"').strip(',')
                            if key and value:
                                current_dict[key] = value
                        j += 1
                    
                    if current_dict:
                        reward_ranges = current_dict
                        break
            
            if reward_ranges:
                # Save to generated_objectives folder
                output_dir = "generated_objectives_debug" if debugging else "generated_objectives_no_convo_baseline"
                os.makedirs(output_dir, exist_ok=True)
                filename = os.path.join(output_dir, f"{env_name}_reward_ranges.json")
                with open(filename, "w", encoding="utf-8") as f:
                    json.dump(reward_ranges, f, indent=4)
                print(f"Saved reward ranges to: {filename}")
            else:
                print("Warning: No reward ranges dictionary found in node 9 output")
                if found_dict:
                    print("Found dictionary definition but failed to parse entries")
        except Exception as e:
            print(f"Warning: Failed to save reward ranges from node 9: {str(e)}")

    # If this is node 10 and we have a successful analysis, save the class descriptions
    if node.identifier == "10" and stakeholder_response:
        
        class_descriptions = extract_class_descriptions_from_text(stakeholder_response)

        if class_descriptions:
            # Save to generated_objectives folder
            output_dir = "generated_objectives_debug" if debugging else "generated_objectives_no_convo_baseline"
            os.makedirs(output_dir, exist_ok=True)
            filename = os.path.join(output_dir, f"{env_name}_objective_descriptions.json")
            with open(filename, "w", encoding="utf-8") as f:
                json.dump(class_descriptions, f, indent=4)
            print(f"Saved class descriptions to: {filename}")
        else:
            print("Warning: No class descriptions dictionary found in node 10 output")
            if found_dict:
                print("Found dictionary definition but failed to parse entries")
     
    
    if node.identifier == "11" and stakeholder_response:
        try:
            # Try to extract JSON (allow raw JSON or fenced ```json blocks)
            raw = stakeholder_response.strip()
            if raw.startswith("```"):
                # handle fenced blocks
                lines = raw.splitlines()
                json_lines = []
                in_json = False
                for ln in lines:
                    if ln.strip().lower().startswith("```json"):
                        in_json = True
                        continue
                    if ln.strip().startswith("```") and in_json:
                        break
                    if in_json:
                        json_lines.append(ln)
                if json_lines:
                    raw = "\n".join(json_lines).strip()

            # Parse JSON
            categorical_map = json.loads(raw)

            # Basic sanity check: dict[str, list[str]]
            if not isinstance(categorical_map, dict):
                raise ValueError("Expected a JSON object (dict).")
            for k, v in categorical_map.items():
                if not isinstance(k, str) or not isinstance(v, list) or not all(isinstance(x, str) for x in v):
                    raise ValueError("Expected mapping of str -> list[str].")

            # Save to generated_objectives folder
            output_dir = "generated_objectives_debug" if debugging else "generated_objectives_no_convo_baseline"
            os.makedirs(output_dir, exist_ok=True)
            filename = os.path.join(output_dir, f"{env_name}_categorical_usage.json")
            with open(filename, "w", encoding="utf-8") as f:
                json.dump(categorical_map, f, indent=4)
            print(f"Saved categorical usage map to: {filename}")
        except Exception as e:
            print(f"Warning: Failed to parse/save categorical usage JSON in node 11: {str(e)}")


    
    prompt = node.facilitator_prompt

    ##################################################################################
    # Step 3: Facilitator evaluates the stakeholder response (internal, hidden process).
    facilitator_history.append({"role": "user", "content": stakeholder_response})
    facil_eval_prompt = make_evaluation_prompt(node)
    facilitator_history.append({"role": "system", "content": facil_eval_prompt})
    facil_eval, input_tokens, output_tokens, cost = call_llm(facilitator_history, "facilitator_evaluates", model_name)
    total_cost += cost
    facilitator_history.append({"role": "assistant", "content": facil_eval})
    print("\n\nFacilitator evaluation of stakeholder response")
    print("----------------------------------------------")
    print(f"Evaluation prompt: {facil_eval_prompt}")
    print(f"Evaluation: {facil_eval}")

    # Print total cost for this node
    print(f"\nTotal cost for node {node.identifier}: ${total_cost:.4f}")

    # Parse evaluation to select the edge and next facilitator step (a node).
    try:
        eval_json = json.loads(facil_eval.replace("'\'",""))  # parse facilitator's response as JSON
        chosen_edge, explanation = eval_json.get("judgment"), eval_json.get("explanation")
        edge_tuple = node.outgoing_edges.get(chosen_edge)
        if edge_tuple is None:
            raise Exception(f"Invalid chosen edge: {chosen_edge}. No matching outgoing edge found.")
    except Exception as e:  # If JSON parsing failed, terminate.
        raise Exception(
            f"Failed to parse evaluator response as JSON or invalid chosen edge. Error: {str(e)}. "
            f"Raw response: {facil_eval}"
        )
    edge_short_name, edge_description, next_node_id = edge_tuple

    # If the evaluation indicates a negative outcome (e.g., a judgment code ending with '-b'),
    # capture the explanation to prepend in the next node.
    prep_msg = explanation if (chosen_edge.endswith("-b") and node.identifier != "6-check-for-stakeholders") else None
    if node.identifier == "8" and prep_msg is not None:
        prep_msg += " CRITICAL: Re-implement all classes that you previously implemented, not just the objectives are are missing or the classes that need fixing.\n"

    if node.identifier == "6-categorical":
        # global node_6_categorical_redo_count
        if chosen_edge == "6-categorical-b":
            node_6_categorical_redo_count += 1

    if node.identifier == "8":
        # global node_8_redo_count
        if chosen_edge == "8-b":
            node_8_redo_count += 1
            if node_8_redo_count >= 5:
                # next_node_id = "8-verifier"
                next_node_id = "8-find-duplicate-classes"
                print(f"Node 8 exceeded 5 redos, forcing transition to node 8-rename")
    # Check if we're in node 8-verifier and handle redo counter
    if node.identifier == "8-verifier":
        global node_8_verifier_redo_count
        if chosen_edge == "8-verifier-b":
            node_8_verifier_redo_count += 1
            if node_8_verifier_redo_count >= 3:
                next_node_id = "9"  # Force transition to node 9 after 3 redos
                print(f"Node 8-verifier exceeded 3 redos, forcing transition to node 9")
   
   
    return next_node_id, facilitator_history, stakeholder_history, prep_msg, total_cost

# ---------------- Node, call_llm, evaluate_response ----------------
class Node:
    def __init__(self, identifier, facilitator_prompt, outgoing_edges, 
                 use_hardcoded=False):
        self.identifier = identifier
        self.facilitator_prompt = facilitator_prompt
        # outgoing_edges: dict with keys as edge codes and values as triples:
        # (short_name, description, next_node_id)
        self.outgoing_edges = outgoing_edges
        self.use_hardcoded = use_hardcoded  # Skip calling LLM for facilitator output?

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run the objective elicitation process')
    parser.add_argument('--resume', action='store_true', help='Resume from a previous run')
    parser.add_argument('--resume-from', type=str, help='Node ID to resume from (e.g., "4", "5")')
    parser.add_argument('--skip-reward-hacking', action='store_true', help='Skip the reward hacking prevention node')
    parser.add_argument('--model', type=str, default='gpt-4o-mini', help='GPT model to use (default: gpt-4o-mini)')
    parser.add_argument('--powerful-nodes', type=str, help='Comma-separated list of node numbers to use o4-mini model (e.g., "1,2,3")')
    parser.add_argument('--construct-causal-graph', action='store_true', help='Construct a causal graph of objectives after node 7.5')
    parser.add_argument('--debugging', action='store_true', help='Save generated files to generated_objectives_debug directory instead of generated_objectives')
    args = parser.parse_args()

    #python3 obj_elicit_no_convo_baseline.py --powerful-nodes 6,6-add-unmeasurable,8,10
    #--resume --resume-from 10

    # Parse powerful nodes if provided
    powerful_nodes = set()
    if args.powerful_nodes:
        try:
            powerful_nodes = {node.strip() for node in args.powerful_nodes.split(',')}
        except Exception as e:
            print(f"Error parsing powerful-nodes argument: {e}")
            print("Continuing with default model for all nodes")

    # Initialize histories
    facilitator_history = []   
    stakeholder_history = []   
    prepended_message = None   # To store explanation from negative evaluations
    loop_count = 0
    node_4_5_exit_count = 0  # Counter for node 4.5 exits
    node_8_verifier_redo_count = 0  # Counter for node 8-verifier redos
    node_8_feature_engineer_redo_count = 0  # Counter for node 8-feature-engineer redos
    total_cost = 0.0  # Track total cost across all nodes

    # Create output folder for this run
    OUTPUT_FOLDER = "calls/output_" + get_timestamp()
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    
    # Create folder for node outputs
    NODE_OUTPUT_FOLDER = "last_run_data_no_convo_baseline"
    os.makedirs(NODE_OUTPUT_FOLDER, exist_ok=True)

    # If resuming, load previous outputs
    if args.resume and args.resume_from:
        print(f"Resuming from node {args.resume_from}")
        facilitator_history, stakeholder_history = load_previous_outputs(args.resume_from, NODE_OUTPUT_FOLDER, env_name)
        current_node_id = args.resume_from
    else:
        current_node_id = "1"

   
    # Define nodes with outgoing_edges as triples: (short_name, description, next_node_id)
    nodes = {}
    nodes["1"] = Node(
        identifier="1",
        use_hardcoded=True,
        facilitator_prompt = (
            f"We will focus on the following task: {task_description}\n\n"
            "You are the facilitator guiding a stakeholder through a structured, multi-stage objective elicitation and refinement process. "
            "The goal is to produce a set of well-specified, measurable objectives and patterns that can later be implemented as a reward function to evaluate an outcome's success or failure.\n\n"
            "Please confirm that you understand by saying 'I understand.'"
        ),

        outgoing_edges={
            "1-a": ("Confirmation", "Stakeholder confirms understanding", "6-categorical"),
        }
    )

    nodes["6-categorical"] = Node(
        identifier="6-categorical",
        use_hardcoded=True,
        facilitator_prompt=(
            f"Task Description: {task_description}\n"
            "Using ONLY the FINAL SET OF OBJECTIVES and the environment's observation-space definition below, identify every VARIABLE that is both CATEGORICAL and not binary—i.e., it takes values from a finite, integer-coded set such as 0..7.\n\n"
            "Environment observation-space source (for evidence about whether a VARIABLE is categorical or no):\n"
            f"{env_context}\n\n"
            "Instructions:\n"
            "1) Parse the variables referenced in the list from the previous step (look under each item's 'VARIABLES:' field). Given the environment code/comments, consider if each variable is both categorical and not binary.\n"
            "2) For EACH not-binary categorical variable that appears in ANY objective, extract its discrete set of possible values. Output exact enumerations (e.g., {0,1,2,3,4,5,6,7}).\n"
            "3) For EACH objective that uses the categorical variable, propose a new AGGREGATION method to account for the variable's one-hot encoding.\n"
            "4) For EACH such variable, report: the name, the sorted list of discrete values, the min, the max, which objectives (by name/id from the previous list) use it, and a new AGGREGATION method for the objective(s) that use the categorical variable.\n"
            "OUTPUT FORMAT: Return ONLY a JSON object with this exact schema, no extra commentary:\n"
            "{\n"
            '  \"categorical_variables\": [\n'
            "    {\n"
            '      \"name\": \"<variable_name>\",\n'
            '      \"values\": [<int>, <int>, ...] | {\"range\": [<min_int>, <max_int>]} | \"Unknown\",\n'
            '      \"min\": <int|null>,\n'
            '      \"max\": <int|null>,\n'
            '      \"used_by_objectives\": [\"objective_0\", \"objective_3\", ...],\n'
            '      \"AGGREGATION\": \"<new_aggregation_method>\"\n'
            "    }\n"
            "  ]\n"
            "}\n\n"
            "Only include variables that (a) are categorical per the environment, (b) not-binary and (c) are actually used by at least one measurable objective from the SMALL SET OF OBJECTIVES.\n"
            f"{helper_txt}"
        ),
        outgoing_edges={
            "6-categorical-a": ("Categoricals Identified", "All categorical variables used by the measurable objectives are listed with min/max/values, or no categorical variables are present.", "8"),
            "6-categorical-b": ("Refine Categoricals", "The categorical-variable list needs refinement or clarification.", "6-categorical"),
        }
    )

    nodes["8"] = Node(
        identifier="8",
        use_hardcoded=True,
        facilitator_prompt=(
            f"Task Description: {task_description}\n"
            "Design a set of objectives that can be combined to evaluate an outcome's success or failure. Implement a Python class for each objective that inherits from the RewardFunction abstract base class.\n\n"
            "Below is the source code for the observation space of the environment that implements this task:\n\n"
            f"{env_context}\n\n"
            "The RewardFunction interface is defined as:\n"
            f"{env_reward_signature}"
            "For each objective you design:\n"
            "1. Create at least 1 new class that inherits from RewardFunction. You may need to create multiple new classes per objective (e.g., see consideration 11).\n"
            "2. Name the class to indicate exactly what variables are being used. Class names should ONLY refer to the specific variables and patterns being used, not the objective being implemented. For example, a class named DailyPhysicalActivity should be renamed to NumberOfStepsPerDay.\n\n"
            "3. Implement the calculate_reward method using only the variables specified in the list\n"
            "4. Use the variables and aggregation/penalty methods specified\n"
            "5. Return a float value representing the reward (positive for desirable objective outcomes and negative for undesirable objective outcomes)\n\n"
            "Important considerations:\n"
            "1. Each class you implement must be executable python code. Define all variables and functions you need within the class.\n"
            "2. Only use variables that are available in the observation space of the environment. DO NOT assume any additional values are available, such as by being passed in at initialization.\n"
            "3. Assume that nothing outside of RewardFunction is available to you. You cannot use any other variables, functions, or libraries, even if they are available in the environment source code. \n"
            "4. DO NOT USE any hard-coded or arbitrary thresholds or values unless they are explicitly defined in the observation space. Instead, use relative changes between variables if needed.\n"
            "5. Implement the exact measurement/detection and aggregation/penalty methods as specified\n"
            "6. Do not set arbitrary thresholds or values, or assume that such values exist unless they are exactly given already. Consider using the relative change between variables instead.\n"
            "7. Add comments explaining the calculation logic\n"
            "8. Handle edge cases appropriately\n"
            "9. Avoid binary (0/1) or ternary (-1/0/1) rewards where possible. Instead, use continuous values that reflect the degree of achievement or violation.\n"
            "10. Make sure you do not encounter divide by zero errors. If you are dividing by a variable, make sure it is never zero.\n"
            "11. For objectives, if possible, use a continuous scale that reflects how well the objective is being achieved (e.g., instead of just checking if a condition is met, measure how far from optimal the current state is).\n"
            "12. For undesirable objective outcomes, if possible, make the penalty proportional to the severity of the violation (e.g., instead of just penalizing when a threshold is crossed, make the penalty proportional to how far the threshold is exceeded).\n"
            "13. Objectives that count the “Number of x” should usually be defined as the change in x between consecutive observations, i.e., obs.x - prev_obs.x. For example, “Number of steps” = obs.steps - prev_obs.steps."
            "14. DO NOT return a sum or linear combination of multiple measures in any class. If a class returns a sum or linear combination of multiple measures (e.g., return f(x) + g(x)), it MUST be split into separate RewardFunction classes. Each new class should return ONLY a single, distinct measure.\n"
            "15. DO NOT implement different classes for the same objective. Each implemented class should be distinct in what it measures.\n\n"
            "\n\n"
            "STRICT HANDLING FOR CATEGORICAL VARIABLES:\n"
            "You must treat any categorical variables previously identified via one-hot expansion.\n"
            "Rules:\n"
            "A) If an objective depends on a categorical variable C with discrete values V = {{v0, v1, ..., vK-1}}, then you NEED to split that single objective into K separate objectives/classes—one per value.\n"
            "B) Treat the categorical variable as one-hot encoded. For example, if C has values {{0,1,2}}, then create 3 classes: one that returns a reward based on whether C==0, one for C==1, and one for C==2. Do not aggregate over the numerical differences between categories; the scalar value assigned to each category holds no meaning other than to differentiate between categories.\n"
            "C) Implement ONE RewardFunction subclass per categorical value.\n"
            "D) This one-hot exception is allowed even though binary outputs were previously discouraged. For categorical indicators ONLY, returning discrete value {{-1, 0, 1}} is acceptable.\n"
            "E) For all objectives that use the not-binary categorical variables identified, use the newly suggested aggregation method(s) not the original one(s).\n"
            "F) Continue to respect all other constraints: use ONLY observation-space variables; handle edge cases; keep each class to a single, distinct measurement; NO linear combinations within a class.\n"
            "G) When splitting an original objective due to a categorical dependency, ensure the union of the new classes covers all values in the variable's discrete set. Do not invent extra values.\n\n"
            "Please implement a class for each item in the FINAL SET SET OF OBJECTIVES, following the important considerations and strict handling of categorical variables above, ensuring the rewards are as expressive as possible."
            f"{helper_txt}"
        ),
        outgoing_edges={
            "8-a": ("Implementation Complete", "All measurable objectives and patterns have been implemented as reward function classes", "8-verifier"),
            "8-b": ("Refine Implementation", "Further refinement of reward function implementations needed", "8")
        }
    )

    nodes["8-verifier"] = Node(
        identifier="8-verifier",
        use_hardcoded=True,
        facilitator_prompt=(
            f"Task Description: {task_description}\n"
            "You are a code verifier. Your task is to analyze the Python code generated in the previous steps and identify any issues that would prevent it from running correctly.\n\n"
            "Only consider the generated code from the previous step and check for the following issues:\n"
            "1. Undefined variables within classes\n"
            "2. Use of functions or libraries that are not available\n"
            "3. Missing imports or dependencies\n"
            "4. Syntax errors\n"
            "5. Logical errors in the reward calculations\n\n"
            "6. Duplicate classes with the same calculate_reward(.) implementation\n\n"
            "CRITICAL: Check for and remove any arbitrary thresholds or values, including predefined values not explicitly defined on the observation space. Common examples to look for:\n"
            "1. Hard-coded numbers (e.g., if x > 0.5, if distance < 10)\n"
            "2. Fixed thresholds for penalties or rewards\n"
            "3. Arbitrary scaling factors\n"
            "4. Binary conditions that could be continuous\n"
            "5. Arbitrary milestones or targets that are not explicitly defined on the observation space\n\n"
            "For each issue found:\n"
            "1. Identify the specific class and line number where the issue occurs\n"
            "2. Explain what the issue is\n"
            "Important guidelines for corrections:\n"
            "1. Only use variables that are available in the observation space\n"
            "2. Do not introduce any external dependencies\n"
            "3. Keep the reward functions as expressive as possible\n"
            "4. Maintain the original intent of the reward function\n"
            "CRITICAL: remove duplicate classes that have the same calculate_reward(.) implementation as another class already present in the code, EVEN IF the comments or class name is different.\n"
            "CRITICAL: Remove hardcoded values used as thresholds (e.g., if x > 2) and re-implement the objective without them. For example, applying a penalty if x exceeds a threshold (e.g., x > 2) should be replaced with a continuous penalty that increases as x increases (e.g., -x), without any fixed cutoff.\n"
            "CRITICAL: ensure each class name explicitly indicates the specific measurement or calculation it performs, not to the objective it implements. Rename classes that do not follow this convention.\n\n"
            "After verifying and correcting the code and removing any duplicate classes that have the exact same implementation, output the complete corrected code in a Python code block. The output should be in this exact format:\n\n"
            "```python\n"
            "[Complete corrected code here]\n"
            "```\n\n"
            "The code block should contain all the corrected classes WITHOUT DUPLICATE IMPLEMENTATIONS, properly formatted and ready to be saved. Include detailed comments in the code block. Do not include any other text or explanations outside the code block."
        ),
        outgoing_edges={
            "8-verifier-a": ("Verification Complete", "Code has been verified and corrected. All duplicate class implementations are removed.", "9"),
            "8-verifier-b": ("Verification Failed", "Code verification failed and needs to be regenerated", "8-verifier")
        }
    )

    nodes["9"] = Node(
        identifier="9",
        use_hardcoded=True,
        facilitator_prompt=(
            f"Task Description: {task_description}\n"
            "Analyze each reward function class implemented in the previous step and determine its output range. "
            "For each class, determine whether it produces:\n"
            "1. Discrete values (e.g., binary {0,1} or ternary {-1,0,1})\n"
            "2. Continuous bounded values (e.g., [0,1] or [-1,1])\n"
            "3. Continuous semi-bounded values (e.g., [0,inf) or (-inf,0])\n"
            "4. Continuous unbounded values (e.g., (-inf,inf))\n\n"
            "For each class, analyze:\n"
            "1. The mathematical operations performed\n"
            "2. Any normalization or scaling applied\n"
            "3. Any clamping or thresholding operations\n"
            "4. The nature of the input variables and their ranges\n\n"
            "IMPORTANT: If a negative multiplier is applied to the output, adjust the reward ranges accordingly."
            "Create a dictionary mapping each class name to its output range in the following format:\n"
            "```python\n"
            "reward_ranges = {\n"
            "    'ClassName1': 'Range: [0, 1] or [0, -1]',  # For bounded continuous\n"
            "    'ClassName2': 'Range: {0, 1} or {0, -1}',  # For discrete\n"
            "    'ClassName3': 'Range: [0, inf) or (-inf, 0)',  # For semi-bounded\n"
            "    'ClassName4': 'Range: (-inf, inf)',  # For unbounded\n"
            "}\n"
            "```\n\n"
            "Please analyze each class and provide the complete dictionary of ranges."
        ),
        outgoing_edges={
            "9-a": ("Analysis Complete", "All reward function ranges have been analyzed and documented", "10"),
            "9-b": ("Refine Analysis", "Further refinement of range analysis needed", "9")
        }
    )

    nodes["10"] = Node(
        identifier="10",
        use_hardcoded=True,
        facilitator_prompt=(
            f"Task Description: {task_description}\n"
            "For each reward function class implemented in the previous steps, provide a brief 1-2 sentence description "
            "that explains what objective or pattern the class is designed to measure or detect.\n"
            "IMPORTANT: add an additional sentence explaining what the direction of the reward value implies. For objectives that only output a discrete binary value (e.g., {0,1} or {-1,0}), explain what each discrete value indicates. For all other objectives, STRICTLY phrase the sentence as 'The more [positive/negative] the value, the more [<explanation>]'. Note that a more negative value generally indicates a more undesirable outcome, but cross-reference your implementation above to ensure this is the case.\n"
            "IMPORTANT: when explaining what the direction of the reward value implies, check your implementation to ensure you correctly interpret any negative multipliers applied to the output. For example, if a class is called DeathProportion, but returns the negative proportion of deaths, than higher values are better.\n"
            "IMPORTANT: do not describe an objective feature in terms of causality or correlation with other variables. Focus on the direct measurement or detection performed by the class.\n\n"
            "Create a dictionary mapping each class name to its description in the following format:\n"
            "```python\n"
            "class_descriptions = {\n"
            "    'ClassName1': 'Description of what this class measures or detects. Description of what the direction of the reward value implies (e.g., The more [positive/negative] the value, the more [<explanation>])',\n"
            "    'ClassName2': 'Description of what this class measures or detects. Description of what the direction of the reward value implies (e.g., The more [positive/negative] the value, the more [<explanation>])',\n"
            "    'BinaryClass1': 'Description of what this class measures or detects. Explanation of what each discrete value indicates (e.g., 0 indicates [<explanation>], 1 indicates [<explanation>])',\n"
            "}\n"
            "```\n\n"
            "Please provide a complete dictionary with descriptions for all implemented classes. "
            "Keep each description concise but informative, focusing on the objective or pattern being measured/detected."
        ),
        outgoing_edges={
            "10-a": ("Descriptions Complete", "All class descriptions have been provided", "11"),
            "10-b": ("Refine Descriptions", "Further refinement of descriptions needed", "10")
        }
    )

    nodes["11"] = Node(
        identifier="11",
        use_hardcoded=True,
        facilitator_prompt=(
            f"Task Description: {task_description}\n"
            "You will analyze the most recent, corrected Python code for RewardFunction subclasses that was produced in prior steps of this conversation.\n\n"
            "Goal: Identify every RewardFunction subclass that uses a CATEGORICAL variable. "
            "A variable should be treated as categorical if the code branches on equality against multiple discrete integer-coded values (≥ 3 values), "
            "uses per-category indicators, or if the class name/value structure indicates a per-category split.\n\n"
            "Instructions for identification:\n"
            "1) Inspect the latest code block of RewardFunction subclasses in the conversation history.\n"
            "2) Consider a subclass as implementing a categorical variable if it:\n"
            "   - Checks patterns like `if var == k` for multiple discrete k values; OR\n"
            "   - Implements separate per-category classes (e.g., \"<VarName>Equals3\"); OR\n"
            "   - Uses one-hot style logic tied to specific discrete values.\n"
            "3) Group subclasses by the categorical variable they depend on.\n\n"
            "OUTPUT FORMAT (STRICT):\n"
            "Return ONLY a JSON object (no extra commentary). Keys are the categorical variable names (strings). "
            "Values are lists of subclass names (strings) that use that categorical variable.\n"
            "Example:\n"
            "{\n"
            "  \"test_type\": [\"TestTypeEquals0\", \"TestTypeEquals1\", \"TestTypeEquals2\"],\n"
            "  \"pain_level\": [\"PainLevelIs0\", \"PainLevelIs1\", \"PainLevelIs2\", \"PainLevelIs3\"]\n"
            "}\n"
            f"{helper_txt}"
        ),
        outgoing_edges={
            "11-a": ("Categorical Map Complete", "Categorical variable to subclass-name mapping emitted", None),
            "11-b": ("Refine Map", "Mapping needs refinement or clarification", "11")
        }
    )

    
    # Main loop: process nodes until termination.
    while current_node_id is not None:
        loop_count += 1
        current_node = nodes[current_node_id]

        if current_node_id == "8":
            # Only format the prompt if it hasn't been formatted yet
            if "{final_objectives}" in current_node.facilitator_prompt:
                #load in the final objectives to format into the prompt
                output_dir = "generated_objectives_debug" if args.debugging else "generated_objectives_no_convo_baseline"
                output_file = os.path.join(output_dir, f"{env_name}_final_objectives.txt")
                with open(output_file, 'r') as f:
                    final_objectives = f.read()

                current_node.facilitator_prompt = current_node.facilitator_prompt.format(
                    final_objectives=final_objectives
                )
        
        # Format the prompt for causal_construction node before running it
        if current_node_id == "causal_construction":
            initial_objectives = load_objective_list("5", NODE_OUTPUT_FOLDER, env_name)
            measurable_objectives = load_objective_list("7.5", NODE_OUTPUT_FOLDER, env_name)
            
            # print("initial_objectives:", initial_objectives)
            # print("measurable_objectives:", measurable_objectives)
            
            # Format the objectives as readable strings
            initial_obj_str = json.dumps(initial_objectives, indent=2)
            measurable_obj_str = json.dumps(measurable_objectives, indent=2)
            
            current_node.facilitator_prompt = current_node.facilitator_prompt.format(
                initial_objectives=initial_obj_str,
                measurable_objectives=measurable_obj_str
            )
        
        # Format the prompt for 8-verifier node with correct output directory
        if current_node_id == "8-verifier":
            output_dir = "generated_objectives_debug" if args.debugging else "generated_objectives_no_convo_baseline"
            current_node.facilitator_prompt = current_node.facilitator_prompt.format(
                output_dir=output_dir
            )
        
        print("\n\n" + "=" * 40)
        print(f"Running Node {current_node_id}...\n")
        print(f"(Loop # {loop_count})")
        
        # Skip node 7.6 if --skip-reward-hacking is set
        if current_node_id == "7.6" and args.skip_reward_hacking:
            print("Skipping reward hacking prevention node as requested")
            current_node_id = "8"
            continue
        
        # Determine which model to use for this node
        #"o4-mini"
        node_model ="gpt-4o" if current_node_id in powerful_nodes else args.model
        print(f"Using model: {node_model}")
        
        current_node_id, facilitator_history, stakeholder_history, prepended_message, node_cost = run_node(
            current_node,
            facilitator_history=facilitator_history,
            stakeholder_history=stakeholder_history,
            prepended_message=prepended_message,
            model_name=node_model,
            node_output_folder=NODE_OUTPUT_FOLDER,
            debugging=args.debugging
        )
        
        total_cost += node_cost
        print(f"\nRunning total cost: ${total_cost:.4f}")
        
        # Save node output after each run
        if current_node_id is not None:  # Don't save if we're terminating
            save_node_output(current_node.identifier, facilitator_history, stakeholder_history, NODE_OUTPUT_FOLDER, env_name, node_model)
        
        time.sleep(2.5)  # Optional delay for rate limiting.

    print("\n\nProcess complete.")
    print(f"Total cost for entire run: ${total_cost:.4f}\n")

if __name__ == "__main__":
    main()
