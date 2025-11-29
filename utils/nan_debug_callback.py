"""
Callback for debugging NaN values in PPO training.
Checks observations, actions, rewards, advantages, and other training components.
"""
import numpy as np
import torch
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.typing import PolicyID
from typing import Dict, Optional


class NaNDebugCallback(DefaultCallbacks):
    """
    Callback that detects NaN values at various points in the training pipeline.
    Useful for debugging numerical instability issues.
    """
    
    def __init__(self):
        super().__init__()
        self.nan_detected = False
        self.check_count = 0
        
    def on_episode_step(self, *, worker, base_env, policies, episode, **kwargs):
        """Check for NaNs after each environment step."""
        # Get the last observation, action, and reward
        last_obs = episode.last_observation_for()
        last_action = episode.last_action_for()
        last_reward = episode.last_reward_for()
        
        # Check observation
        if last_obs is not None:
            if np.any(np.isnan(last_obs)):
                print(f"\n{'='*80}")
                print(f"[NaN DETECTED] Episode {episode.episode_id}, Step {episode.length}")
                print(f"Location: OBSERVATION")
                print(f"Observation shape: {last_obs.shape if hasattr(last_obs, 'shape') else 'N/A'}")
                print(f"NaN indices: {np.where(np.isnan(last_obs))}")
                print(f"Observation sample: {last_obs[:10] if len(last_obs) > 10 else last_obs}")
                print(f"{'='*80}\n")
                self.nan_detected = True
        
        # Check action
        if last_action is not None:
            action_array = np.asarray(last_action)
            if np.any(np.isnan(action_array)):
                print(f"\n{'='*80}")
                print(f"[NaN DETECTED] Episode {episode.episode_id}, Step {episode.length}")
                print(f"Location: ACTION")
                print(f"Action: {last_action}")
                print(f"{'='*80}\n")
                self.nan_detected = True
        
        # Check reward
        if last_reward is not None and np.isnan(last_reward):
            print(f"\n{'='*80}")
            print(f"[NaN DETECTED] Episode {episode.episode_id}, Step {episode.length}")
            print(f"Location: REWARD")
            print(f"Reward: {last_reward}")
            print(f"{'='*80}\n")
            self.nan_detected = True
    
    def on_sample_end(self, *, worker, samples, **kwargs):
        """Check for NaNs in collected samples before training."""
        self.check_count += 1
        
        # Handle both MultiAgentBatch and single SampleBatch
        if hasattr(samples, 'policy_batches'):
            # Multi-agent case
            policy_batches = samples.policy_batches.items()
        else:
            # Single policy case - wrap in dict format
            policy_batches = [("default_policy", samples)]
        
        # Check every sample batch
        for policy_id, sample_batch in policy_batches:
            batch_size = len(sample_batch)
            
            # Check observations
            if SampleBatch.OBS in sample_batch:
                obs = sample_batch[SampleBatch.OBS]
                if np.any(np.isnan(obs)):
                    nan_count = np.sum(np.isnan(obs))
                    print(f"\n{'='*80}")
                    print(f"[NaN DETECTED] on_sample_end (check #{self.check_count})")
                    print(f"Policy: {policy_id}, Batch size: {batch_size}")
                    print(f"Location: OBSERVATIONS in sample batch")
                    print(f"NaN count: {nan_count} / {obs.size}")
                    print(f"Obs shape: {obs.shape}")
                    print(f"NaN locations (first 5): {np.argwhere(np.isnan(obs))[:5]}")
                    print(f"Non-NaN obs stats: min={np.nanmin(obs):.4f}, max={np.nanmax(obs):.4f}, mean={np.nanmean(obs):.4f}")
                    print(f"{'='*80}\n")
                    self.nan_detected = True
            
            # Check actions
            if SampleBatch.ACTIONS in sample_batch:
                actions = sample_batch[SampleBatch.ACTIONS]
                if np.any(np.isnan(actions)):
                    nan_count = np.sum(np.isnan(actions))
                    print(f"\n{'='*80}")
                    print(f"[NaN DETECTED] on_sample_end (check #{self.check_count})")
                    print(f"Policy: {policy_id}, Batch size: {batch_size}")
                    print(f"Location: ACTIONS in sample batch")
                    print(f"NaN count: {nan_count} / {actions.size}")
                    print(f"Actions shape: {actions.shape}")
                    print(f"{'='*80}\n")
                    self.nan_detected = True
            
            # Check rewards
            if SampleBatch.REWARDS in sample_batch:
                rewards = sample_batch[SampleBatch.REWARDS]
                if np.any(np.isnan(rewards)):
                    nan_count = np.sum(np.isnan(rewards))
                    print(f"\n{'='*80}")
                    print(f"[NaN DETECTED] on_sample_end (check #{self.check_count})")
                    print(f"Policy: {policy_id}, Batch size: {batch_size}")
                    print(f"Location: REWARDS in sample batch")
                    print(f"NaN count: {nan_count} / {len(rewards)}")
                    print(f"Non-NaN reward stats: min={np.nanmin(rewards):.4f}, max={np.nanmax(rewards):.4f}, mean={np.nanmean(rewards):.4f}")
                    print(f"Reward sample (first 20): {rewards[:20]}")
                    print(f"{'='*80}\n")
                    self.nan_detected = True
            
            # Check value predictions (if available)
            if SampleBatch.VF_PREDS in sample_batch:
                vf_preds = sample_batch[SampleBatch.VF_PREDS]
                if np.any(np.isnan(vf_preds)):
                    nan_count = np.sum(np.isnan(vf_preds))
                    print(f"\n{'='*80}")
                    print(f"[NaN DETECTED] on_sample_end (check #{self.check_count})")
                    print(f"Policy: {policy_id}, Batch size: {batch_size}")
                    print(f"Location: VALUE PREDICTIONS in sample batch")
                    print(f"NaN count: {nan_count} / {len(vf_preds)}")
                    print(f"Non-NaN VF stats: min={np.nanmin(vf_preds):.4f}, max={np.nanmax(vf_preds):.4f}, mean={np.nanmean(vf_preds):.4f}")
                    print(f"{'='*80}\n")
                    self.nan_detected = True
            
            # Check action logits/log probs (if available)
            if SampleBatch.ACTION_LOGP in sample_batch:
                action_logp = sample_batch[SampleBatch.ACTION_LOGP]
                if np.any(np.isnan(action_logp)):
                    nan_count = np.sum(np.isnan(action_logp))
                    print(f"\n{'='*80}")
                    print(f"[NaN DETECTED] on_sample_end (check #{self.check_count})")
                    print(f"Policy: {policy_id}, Batch size: {batch_size}")
                    print(f"Location: ACTION LOG PROBS in sample batch")
                    print(f"NaN count: {nan_count} / {len(action_logp)}")
                    print(f"{'='*80}\n")
                    self.nan_detected = True

            # Check action distribution inputs (logits) - CRITICAL for Gaussian stability
            if SampleBatch.ACTION_DIST_INPUTS in sample_batch:
                logits = sample_batch[SampleBatch.ACTION_DIST_INPUTS]
                if np.any(np.isnan(logits)):
                    nan_count = np.sum(np.isnan(logits))
                    print(f"\n{'='*80}")
                    print(f"[NaN DETECTED] on_sample_end (check #{self.check_count})")
                    print(f"Policy: {policy_id}, Batch size: {batch_size}")
                    print(f"Location: ACTION DIST INPUTS (Logits) in sample batch")
                    print(f"NaN count: {nan_count} / {logits.size}")
                    print(f"Logits stats: min={np.nanmin(logits):.4f}, max={np.nanmax(logits):.4f}, mean={np.nanmean(logits):.4f}")
                    print(f"{'='*80}\n")
                    self.nan_detected = True
                
                # Also check for extreme values that cause instability
                if np.any(np.abs(logits) > 100):
                    print(f"\n{'='*80}")
                    print(f"[INSTABILITY WARNING] on_sample_end (check #{self.check_count})")
                    print(f"Location: ACTION DIST INPUTS (Logits) has large values > 100")
                    print(f"Logits stats: min={np.min(logits):.4f}, max={np.max(logits):.4f}")
                    print(f"This can cause NaN gradients in log_std -> std calculation")
                    print(f"{'='*80}\n")
    
    def on_learn_on_batch(self, *, policy, train_batch, result, **kwargs):
        """Check for NaNs in processed batch before learning."""
        
        # Check advantages
        if "advantages" in train_batch:
            advantages = train_batch["advantages"]
            if isinstance(advantages, torch.Tensor):
                advantages = advantages.cpu().numpy()
            if np.any(np.isnan(advantages)):
                nan_count = np.sum(np.isnan(advantages))
                print(f"\n{'='*80}")
                print(f"[NaN DETECTED] on_learn_on_batch (check #{self.check_count})")
                print(f"Location: ADVANTAGES")
                print(f"NaN count: {nan_count} / {advantages.size}")
                print(f"Non-NaN advantage stats: min={np.nanmin(advantages):.4f}, max={np.nanmax(advantages):.4f}, mean={np.nanmean(advantages):.4f}, std={np.nanstd(advantages):.4f}")
                print(f"{'='*80}\n")
                self.nan_detected = True
        
        # Check value targets
        if "value_targets" in train_batch:
            value_targets = train_batch["value_targets"]
            if isinstance(value_targets, torch.Tensor):
                value_targets = value_targets.cpu().numpy()
            if np.any(np.isnan(value_targets)):
                nan_count = np.sum(np.isnan(value_targets))
                print(f"\n{'='*80}")
                print(f"[NaN DETECTED] on_learn_on_batch (check #{self.check_count})")
                print(f"Location: VALUE TARGETS")
                print(f"NaN count: {nan_count} / {value_targets.size}")
                print(f"Non-NaN VT stats: min={np.nanmin(value_targets):.4f}, max={np.nanmax(value_targets):.4f}, mean={np.nanmean(value_targets):.4f}")
                print(f"{'='*80}\n")
                self.nan_detected = True
        
        # Check learning results for NaN losses
        if "learner_stats" in result:
            for key, value in result["learner_stats"].items():
                if isinstance(value, (int, float)) and np.isnan(value):
                    print(f"\n{'='*80}")
                    print(f"[NaN DETECTED] on_learn_on_batch (check #{self.check_count})")
                    print(f"Location: LEARNER STATS - {key}")
                    print(f"Value: {value}")
                    print(f"{'='*80}\n")
                    self.nan_detected = True
        
        # Check model parameters for NaNs
        if hasattr(policy, 'model'):
            for name, param in policy.model.named_parameters():
                if param is not None and torch.any(torch.isnan(param)):
                    nan_count = torch.sum(torch.isnan(param)).item()
                    print(f"\n{'='*80}")
                    print(f"[NaN DETECTED] on_learn_on_batch (check #{self.check_count})")
                    print(f"Location: MODEL PARAMETER - {name}")
                    print(f"NaN count: {nan_count} / {param.numel()}")
                    print(f"Param shape: {param.shape}")
                    print(f"{'='*80}\n")
                    self.nan_detected = True
                
                # Check gradients
                if param.grad is not None and torch.any(torch.isnan(param.grad)):
                    nan_count = torch.sum(torch.isnan(param.grad)).item()
                    print(f"\n{'='*80}")
                    print(f"[NaN DETECTED] on_learn_on_batch (check #{self.check_count})")
                    print(f"Location: GRADIENT - {name}")
                    print(f"NaN count: {nan_count} / {param.grad.numel()}")
                    print(f"Gradient shape: {param.grad.shape}")
                    # print(f"Non-NaN grad stats: min={torch.nanmin(param.grad).item():.6f}, max={torch.nanmax(param.grad).item():.6f}, mean={torch.nanmean(param.grad).item():.6f}")
                    print(f"{'='*80}\n")
                    self.nan_detected = True
    
    def on_train_result(self, *, algorithm, result, **kwargs):
        """Check training metrics for NaNs."""
        
        # Check if any training metric contains NaN
        if "info" in result and "learner" in result["info"]:
            for policy_id, policy_stats in result["info"]["learner"].items():
                if "learner_stats" in policy_stats:
                    for stat_name, stat_value in policy_stats["learner_stats"].items():
                        if isinstance(stat_value, (int, float)) and np.isnan(stat_value):
                            print(f"\n{'='*80}")
                            print(f"[NaN DETECTED] on_train_result")
                            print(f"Training iteration: {result.get('training_iteration', 'N/A')}")
                            print(f"Location: Training metric - {policy_id}/{stat_name}")
                            print(f"Value: {stat_value}")
                            print(f"{'='*80}\n")
                            self.nan_detected = True
        
        # Print summary if NaNs were detected
        if self.nan_detected:
            print(f"\n{'#'*80}")
            print(f"NaN SUMMARY for iteration {result.get('training_iteration', 'N/A')}")
            print(f"Total checks performed: {self.check_count}")
            print(f"NaNs have been detected - see detailed logs above")
            print(f"{'#'*80}\n")
            # Reset flag for next iteration
            self.nan_detected = False

