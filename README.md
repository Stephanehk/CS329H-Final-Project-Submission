# LLM Reward Design

This project implements a Deliberative-Reward-Design approach for generating reward functions using LLMs. Below are the instructions for running the various components of the pipeline.

## Setup

Before running any scripts, ensure you have your OpenAI API key configured.
1. Open `secret_keys.py`.
2. Add your API key:
   ```python
   OPENAI_API_KEY = "your-api-key-here"
   ```

## 1. Generate Reward Features

### Deliberative-Reward-Design (Our Method)
To generate a set of reward features using our Deliberative-Reward-Design approach:

1. Edit `obj_elicit.py` to set the desired task description/name.
2. Run the following command:
   ```bash
   python3 obj_elicit.py --powerful-nodes 6,6-add-unmeasurable,8,10
   ```

The generated features can be found in the `generated_objectives/` folder. Note that we have already run this script, so you can view those generated objectives without needing to re-run.

### Non-Deliberative-Reward-Design (Baseline)
To generate a set of reward features using the baseline Non-Deliberative approach:

```bash
python3 obj_elicit_no_convo_baseline.py --powerful-nodes 6,6-add-unmeasurable,8,10
```

The generated features can be found in the `generated_objectives_no_convo_baseline/` folder.

## 2. Learn Reward Weights

To learn the weights over the generated features via our facilitator-guided preference elicitation procedure, run the command corresponding to your environment. Note that we have already run this script, so you can view those generated objectives without needing to re-run.

**For Pandemic:**
```bash
python3 -m reward_learning.active_pref_learning pandemic llm o4-mini false true true default false
```

**For Traffic:**
```bash
python3 -m reward_learning.active_pref_learning traffic llm o4-mini false true true default false
```

*Note: To elicit weights for the **Non-Deliberative-Reward-Design baseline**, change the last boolean argument in the commands above from `false` to `true`.*

## 3. Evaluation

You can evaluate the generated reward functions for expressivity and alignment. 
For all evaluation commands below, you can add the `--no-convo-base-line` flag to switch the analysis to the reward function designed by Non-Deliberative-Reward-Design instead of Deliberative-Reward-Design.

### Evaluate Expressivity
To evaluate the expressivity of the generated reward features:

**For Pandemic:**
```bash
python3 -m reward_learning.learn_reward_weights --env_name pandemic
```

**For Traffic:**
```bash
python3 -m reward_learning.learn_reward_weights --env_name traffic
```

### Evaluate Alignment
To evaluate the alignment of the generated reward function against the ground truth:

**For Pandemic:**
```bash
python3 -m reward_learning.rewards_dictionary_eval --env_name pandemic --weights_name pandemic_o4-mini_True_prefs
```

**For Traffic:**
```bash
python3 -m reward_learning.rewards_dictionary_eval --env_name traffic --weights_name traffic_o4-mini_True_prefs
```

