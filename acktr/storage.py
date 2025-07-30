import torch
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
import time

def _flatten_helper(T, N, _tensor):
    return _tensor.view(T * N, *_tensor.size()[2:])


# --- START OF MODIFICATION ---
# The logic from `compute_returns` is extracted into a standalone function.
# We apply the torch.compile decorator here for optimization.
# The mode 'reduce-overhead' is ideal for this kind of loop-heavy function.
@torch.compile(mode="reduce-overhead")
def _compute_returns_compiled(
    rewards: torch.Tensor,
    value_preds: torch.Tensor,
    masks: torch.Tensor,
    bad_masks: torch.Tensor,
    gamma: float,
    gae_lambda: float,
) -> torch.Tensor:
    """
    A compiled, optimized, and unified version of the GAE and return calculation logic.

    To replicate the behavior of the original function's flags:
    - For standard GAE (originally `use_gae=True`), pass your `gae_lambda` value.
    - To disable GAE (originally `use_gae=False`), set `gae_lambda=1.0`.
    - To use proper time limits (originally `use_proper_time_limits=True`), pass the correct `bad_masks`.
    - To disable proper time limits (originally `use_proper_time_limits=False`), pass a tensor of ones for `bad_masks`.
    """
    num_steps = rewards.size(0)
    returns = torch.zeros_like(value_preds)
    gae = 0.0

    # Precompute deltas for the entire sequence to simplify the loop body
    deltas = rewards + gamma * value_preds[1:] * masks[1:] - value_preds[:-1]

    # Iterate backwards to compute GAE and returns
    for step in reversed(range(num_steps)):
        # This is the unified GAE update rule.
        # The `bad_masks` correctly resets GAE at timeout-terminated episode boundaries.
        # `gae` at the start of this line is the advantage from the next step (t+1).
        gae = deltas[step] + gamma * gae_lambda * masks[step + 1] * gae
        gae = gae * bad_masks[step + 1]
        returns[step] = gae + value_preds[step]

    return returns


# the shape of observation: batch * cpu * length
class RolloutStorage(object):
    def __init__(self, num_steps, num_processes, obs_shape, action_space,
                 recurrent_hidden_state_size,can_give_up, enable_rotation, pallet_size):
        self.obs = torch.zeros(num_steps + 1, num_processes, *obs_shape)
        self.recurrent_hidden_states = torch.zeros(
            num_steps + 1, num_processes, recurrent_hidden_state_size)
        self.rewards = torch.zeros(num_steps, num_processes, 1)
        self.value_preds = torch.zeros(num_steps + 1, num_processes, 1)
        self.returns = torch.zeros(num_steps + 1, num_processes, 1)
        self.action_log_probs = torch.zeros(num_steps, num_processes, 1)

        # --- ACTION TENSOR MODIFICATION ---
        # Determine the shape of a single action
        if action_space.__class__.__name__ == 'Discrete':
            action_shape = 1
        elif action_space.__class__.__name__ == 'MultiDiscrete':
            # For MultiDiscrete, the shape is the number of discrete action parts.
            # e.g., for [L, W, 2], this is 3.
            action_shape = action_space.nvec.shape[0]
        else:
            # Fallback for continuous spaces like Box
            action_shape = action_space.shape[0]
        
        # Initialize the actions tensor with the correct shape
        self.actions = torch.zeros(num_steps, num_processes, action_shape)
        
        # Both Discrete and MultiDiscrete actions are represented as integers
        if action_space.__class__.__name__ in ['Discrete', 'MultiDiscrete']:
            self.actions = self.actions.long()
        
        self.masks = torch.ones(num_steps + 1, num_processes, 1)
        if enable_rotation:
            self.location_masks = torch.zeros(num_steps+1, num_processes, 2 * pallet_size**2)
        elif can_give_up:
            self.location_masks = torch.zeros(num_steps+1, num_processes, pallet_size**2 +1)
        else:
            self.location_masks = torch.zeros(num_steps+1, num_processes, pallet_size**2)

        # Masks that indicate whether it's a true terminal state
        # or time limit end state
        self.bad_masks = torch.ones(num_steps + 1, num_processes, 1)
        self.num_steps = num_steps
        self.step = 0

    def to(self, device):
        self.obs = self.obs.to(device)
        self.recurrent_hidden_states = self.recurrent_hidden_states.to(device)
        self.rewards = self.rewards.to(device)
        self.value_preds = self.value_preds.to(device)
        self.returns = self.returns.to(device)
        self.action_log_probs = self.action_log_probs.to(device)
        self.actions = self.actions.to(device)
        self.masks = self.masks.to(device)
        self.bad_masks = self.bad_masks.to(device)
        self.location_masks = self.location_masks.to(device)

    def insert(self, obs, recurrent_hidden_states, actions, action_log_probs,
               value_preds, rewards, masks, bad_masks, location_masks):
        self.obs[self.step + 1].copy_(obs)
        self.recurrent_hidden_states[self.step + 1].copy_(recurrent_hidden_states)
        self.actions[self.step].copy_(actions)
        self.action_log_probs[self.step].copy_(action_log_probs)
        self.value_preds[self.step].copy_(value_preds)
        self.rewards[self.step].copy_(rewards)
        self.masks[self.step + 1].copy_(masks)
        self.bad_masks[self.step + 1].copy_(bad_masks)
        self.location_masks[self.step + 1].copy_(location_masks)
        self.step = (self.step + 1) % self.num_steps

    def after_update(self):
        self.obs[0].copy_(self.obs[-1])
        self.recurrent_hidden_states[0].copy_(self.recurrent_hidden_states[-1])
        self.masks[0].copy_(self.masks[-1])
        self.bad_masks[0].copy_(self.bad_masks[-1])
        self.location_masks[0].copy_(self.location_masks[-1])

    # --- START OF MODIFICATION ---
    def compute_returns(self, next_value, use_gae, gamma, gae_lambda, use_proper_time_limits):
        """
        Updates the logic to correctly call the new optimized function.
        """
        # The last value prediction is the value of the state after the final action
        self.value_preds[-1] = next_value

        # --- START OF MODIFICATION ---

        # If GAE is disabled, it's equivalent to setting lambda to 1.0.
        current_gae_lambda = gae_lambda if use_gae else 1.0

        # If not using proper time limits, we pass a tensor of all ones.
        # This makes the bad_mask have no effect, as intended.
        if use_proper_time_limits:
            current_bad_masks = self.bad_masks
        else:
            current_bad_masks = torch.ones_like(self.bad_masks)
        
        # Call the compiled function with the correct 6 arguments
        #start_time = time.perf_counter()
        self.returns = _compute_returns_compiled(
            rewards=self.rewards,
            value_preds=self.value_preds,
            masks=self.masks,
            bad_masks=current_bad_masks,
            gamma=gamma,
            gae_lambda=current_gae_lambda
        )
        # --- END OF MODIFICATION ---
        #end_time = time.perf_counter()
        #elapsed_time = end_time - start_time
        #print(f"Time taken for compute returns: {elapsed_time} seconds")


    def feed_forward_generator(self,
                               advantages,
                               num_mini_batch=None,
                               mini_batch_size=None):
        num_steps, num_processes = self.rewards.size()[0:2]
        batch_size = num_processes * num_steps

        if mini_batch_size is None:
            assert batch_size >= num_mini_batch, (
                "PPO requires the number of processes ({}) "
                "* number of steps ({}) = {} "
                "to be greater than or equal to the number of PPO mini batches ({})."
                "".format(num_processes, num_steps, num_processes * num_steps,
                          num_mini_batch))
            mini_batch_size = batch_size // num_mini_batch
        sampler = BatchSampler(
            SubsetRandomSampler(range(batch_size)),
            mini_batch_size,
            drop_last=True)
        for indices in sampler:
            obs_batch = self.obs[:-1].view(-1, *self.obs.size()[2:])[indices]
            recurrent_hidden_states_batch = self.recurrent_hidden_states[:-1].view(
                -1, self.recurrent_hidden_states.size(-1))[indices]
            actions_batch = self.actions.view(-1,
                                              self.actions.size(-1))[indices]
            value_preds_batch = self.value_preds[:-1].view(-1, 1)[indices]
            return_batch = self.returns[:-1].view(-1, 1)[indices]
            masks_batch = self.masks[:-1].view(-1, 1)[indices]
            old_action_log_probs_batch = self.action_log_probs.view(-1,
                                                                    1)[indices]
            if advantages is None:
                adv_targ = None
            else:
                adv_targ = advantages.view(-1, 1)[indices]

            yield obs_batch, recurrent_hidden_states_batch, actions_batch, \
                value_preds_batch, return_batch, masks_batch, old_action_log_probs_batch, adv_targ

    def recurrent_generator(self, advantages, num_mini_batch):
        num_processes = self.rewards.size(1)
        assert num_processes >= num_mini_batch, (
            "PPO requires the number of processes ({}) "
            "to be greater than or equal to the number of "
            "PPO mini batches ({}).".format(num_processes, num_mini_batch))
        num_envs_per_batch = num_processes // num_mini_batch
        perm = torch.randperm(num_processes)
        for start_ind in range(0, num_processes, num_envs_per_batch):
            obs_batch = []
            recurrent_hidden_states_batch = []
            actions_batch = []
            value_preds_batch = []
            return_batch = []
            masks_batch = []
            old_action_log_probs_batch = []
            adv_targ = []

            for offset in range(num_envs_per_batch):
                ind = perm[start_ind + offset]
                obs_batch.append(self.obs[:-1, ind])
                recurrent_hidden_states_batch.append(
                    self.recurrent_hidden_states[0:1, ind])
                actions_batch.append(self.actions[:, ind])
                value_preds_batch.append(self.value_preds[:-1, ind])
                return_batch.append(self.returns[:-1, ind])
                masks_batch.append(self.masks[:-1, ind])
                old_action_log_probs_batch.append(
                    self.action_log_probs[:, ind])
                adv_targ.append(advantages[:, ind])

            T, N = self.num_steps, num_envs_per_batch
            # These are all tensors of size (T, N, -1)
            obs_batch = torch.stack(obs_batch, 1)
            actions_batch = torch.stack(actions_batch, 1)
            value_preds_batch = torch.stack(value_preds_batch, 1)
            return_batch = torch.stack(return_batch, 1)
            masks_batch = torch.stack(masks_batch, 1)
            old_action_log_probs_batch = torch.stack(
                old_action_log_probs_batch, 1)
            adv_targ = torch.stack(adv_targ, 1)

            # States is just a (N, -1) tensor
            recurrent_hidden_states_batch = torch.stack(
                recurrent_hidden_states_batch, 1).view(N, -1)

            # Flatten the (T, N, ...) tensors to (T * N, ...)
            obs_batch = _flatten_helper(T, N, obs_batch)
            actions_batch = _flatten_helper(T, N, actions_batch)
            value_preds_batch = _flatten_helper(T, N, value_preds_batch)
            return_batch = _flatten_helper(T, N, return_batch)
            masks_batch = _flatten_helper(T, N, masks_batch)
            old_action_log_probs_batch = _flatten_helper(T, N, \
                        old_action_log_probs_batch)
            adv_targ = _flatten_helper(T, N, adv_targ)

            yield obs_batch, recurrent_hidden_states_batch, actions_batch, \
                value_preds_batch, return_batch, masks_batch, old_action_log_probs_batch, adv_targ