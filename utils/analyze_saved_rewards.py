import numpy as np
import matplotlib.pyplot as plt
from reward_learning.learn_reward_weights import load_rollout_data
from utils.glucose_rollout_and_save import TrajectoryStep  # noqa: F401 – type hints

POLICY_ALIASES = {
    0: "BC-policy",
    1: "environment-reward-opt-policy",
    2: "alt-environment-reward-opt-policy",
    3: "slightly-worse-than-BC-policy",
    4: "slightly-better-than-BC-policy",
    5: "uniform-policy",
}

def alias_policy_name(name: str, policy_names) -> str:
    try:
        idx = policy_names.index(name)
        return POLICY_ALIASES.get(idx, name)
    except ValueError:
        return name

env_policy_mapping = {
    "traffic": [
        "traffic_base_policy",
        "2025-06-24_13-51-42",
        "2025-06-17_16-14-06",
        "2025-07-10_13-33-33",
        "2025-07-09_16-57-36",
        "uniform-policy"
    ],
    "glucose": [
        "glucose_base_policy",
        "2025-06-24_13-53-32",
        "2025-05-12_14-12-46",
        "2025-07-09_16-56-49_checkpoint_000025",
        "2025-07-09_16-56-49_checkpoint_000050",
        "uniform-policy"
    ],
    "pandemic": [
        "pandemic_base_policy",
        "2025-06-24_13-49-08",
        "2025-05-05_21-29-00",
        "2025-07-10_11-40-34",
        "2025-07-09_16-58-20",
        "uniform-policy"
    ]
}

for env_name, policy_names in env_policy_mapping.items():
    print(f"\nEnvironment: {env_name}")
    all_policy_returns = []
    display_names = []

    for p in policy_names:
        display_name = alias_policy_name(p, policy_names)
        print(f"Policy: {display_name}")

        if p == "uniform-policy":
            all_rets = np.load(f"uniform_policy_returns/{env_name}_returns.npy")
        else:
            rollout_data = load_rollout_data(env_name, "rollout_data/", [p], 50)
            ret = 0
            all_rets = []
            for transition in rollout_data:
                reward = transition["magni_rew"] if env_name == "glucose" else transition["true_reward"]
                ret += reward
                if transition["done"]:
                    all_rets.append(ret)
                    ret = 0

        print("mean return: ", np.mean(all_rets))
        print("std return: ", np.std(all_rets))
        all_policy_returns.append(all_rets)
        display_names.append(display_name)

    # Plot full policy set
    plt.figure(figsize=(10, 6))
    plt.boxplot(all_policy_returns, labels=display_names, showmeans=True)
    plt.title(f"Policy Returns in {env_name.capitalize()} Environment")
    plt.ylabel("Return")
    plt.xlabel("Policy")
    plt.xticks(rotation=15)
    plt.grid(axis='y')
    plt.tight_layout()
    plt.savefig(f"plots/policy_returns_{env_name}.png")
    plt.close()

    # Plot subset of 3 policies
    subset_names = {"BC-policy", "environment-reward-opt-policy", "slightly-better-than-BC-policy"}
    subset_returns = [r for r, name in zip(all_policy_returns, display_names) if name in subset_names]
    subset_labels = [name for name in display_names if name in subset_names]

    plt.figure(figsize=(8, 5))
    plt.boxplot(subset_returns, labels=subset_labels, showmeans=True)
    plt.title(f"Subset Policy Returns in {env_name.capitalize()}")
    plt.ylabel("Return")
    plt.xlabel("Policy")
    plt.grid(axis='y')
    plt.tight_layout()
    plt.savefig(f"plots/policy_returns_{env_name}_subset.png")
    plt.close()
