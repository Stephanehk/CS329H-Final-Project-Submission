import numpy as np
from ray.rllib.policy.policy import Policy
from ray.rllib.utils.annotations import override
from ray.rllib.utils.typing import ModelWeights, PolicyID, SampleBatchType

class RandomActionPolicy(Policy):
    """A policy wrapper that makes the original policy take random actions 50% of the time."""
    
    def __init__(self, policy: Policy, random_prob: float = 1.0):
        """Initialize the random action policy wrapper.
        
        Args:
            policy: The original policy to wrap
            random_prob: Probability of taking a random action (default: 0.5)
        """
        super().__init__(policy.observation_space, policy.action_space, policy.config)
        self.policy = policy
        self.random_prob = random_prob
        
    @override(Policy)
    def compute_actions(
        self,
        observations,
        state=None,
        prev_action=None,
        prev_reward=None,
        info=None,
        policy_id=None,
        full_fetch=False,
        explore=None,
    ):
        # Get actions from original policy
        actions, state_out, info_out = self.policy.compute_actions(
            observations, state, prev_action, prev_reward, info, policy_id, full_fetch, explore
        )
        
        # Randomly replace actions with random ones based on probability
        if isinstance(actions, np.ndarray):
            mask = np.random.random(len(actions)) < self.random_prob
            if mask.any():
                if self.action_space.is_discrete():
                    random_actions = np.array([
                        self.action_space.sample() for _ in range(mask.sum())
                    ])
                else:
                    random_actions = np.array([
                        self.action_space.sample() for _ in range(mask.sum())
                    ])
                actions[mask] = random_actions
        else:
            # Handle single action case
            if np.random.random() < self.random_prob:
                actions = self.action_space.sample()
        # print ("randomly choosing action...")
        # assert False
        return actions, state_out, info_out
    
    @override(Policy)
    def compute_single_action(
        self,
        observation,
        state=None,
        prev_action=None,
        prev_reward=None,
        info=None,
        policy_id=None,
        full_fetch=False,
        explore=None,
    ):
        # Get action from original policy
        action, state_out, info_out = self.policy.compute_single_action(
            observation, state, prev_action, prev_reward, info, policy_id, full_fetch, explore
        )
        
        # Randomly replace with random action based on probability
        if np.random.random() < self.random_prob:
            action = self.action_space.sample()
        # print ("randomly choosing action...")
        # assert False
        return action, state_out, info_out
    
    @override(Policy)
    def get_weights(self) -> ModelWeights:
        return self.policy.get_weights()
    
    @override(Policy)
    def set_weights(self, weights: ModelWeights) -> None:
        self.policy.set_weights(weights) 
