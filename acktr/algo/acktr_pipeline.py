import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from acktr.algo.kfac import KFACOptimizer
import sys

class ACKTR():
    def __init__(self,
                 actor_critic,
                 value_loss_coef,
                 entropy_coef,
                 invaild_coef, # This will become unused
                 lr=None,
                 eps=None,
                 alpha=None,
                 max_grad_norm=None,
                 acktr=False,
                 args=None):

        self.actor_critic = actor_critic
        self.acktr = acktr

        self.value_loss_coef = value_loss_coef
        self.invaild_coef = invaild_coef
        self.max_grad_norm = max_grad_norm
        self.entropy_coef = entropy_coef
        self.args = args

        if acktr:
            self.optimizer = KFACOptimizer(actor_critic)
        else:
            self.optimizer = optim.RMSprop(
                actor_critic.parameters(), lr, eps=eps, alpha=alpha)

    def update(self, rollouts):
        obs_shape = rollouts.obs.size()[2:]
        action_shape = rollouts.actions.size()[-1]
        num_steps, num_processes, _ = rollouts.rewards.size()

        # --- NEW: Extract ground-truth masks from the observation ---
        # The observation is a flat vector of 6 channels (hmap, 3*dims, mask_o0, mask_o1)
        obs_tensor = rollouts.obs[:-1].view(-1, 6, self.args.container_size[0], self.args.container_size[1])
        # The last two channels are the masks
        gt_masks = obs_tensor[:, 4:6, :, :] # Shape: (batch, 2, width, length)

        ### MODIFIED: Call new evaluate_actions with gt_masks, which now returns 5 values
        values, action_log_probs, dist_entropy, _, infeasibility_loss = self.actor_critic.evaluate_actions(
            rollouts.obs[:-1].view(-1, *obs_shape),
            rollouts.recurrent_hidden_states[0].view(
                -1, self.actor_critic.recurrent_hidden_state_size),
            rollouts.masks[:-1].view(-1, 1),
            rollouts.actions.view(-1, action_shape),
            gt_masks) # Pass the ground-truth masks here

        values = values.view(num_steps, num_processes, 1)
        action_log_probs = action_log_probs.view(num_steps, num_processes, 1)

        advantages = rollouts.returns[:-1] - values
        value_loss = advantages.pow(2).mean()
        action_loss = -(advantages.detach() * action_log_probs).mean()

        if self.acktr and self.optimizer.steps % self.optimizer.Ts == 0:
            self.actor_critic.zero_grad()
            pg_fisher_loss = -action_log_probs.mean()
            value_noise = torch.randn(values.size())
            if values.is_cuda:
                value_noise = value_noise.cuda()
            sample_values = values + value_noise
            vf_fisher_loss = -(values - sample_values.detach()).pow(2).mean()
            fisher_loss = pg_fisher_loss + vf_fisher_loss
            self.optimizer.acc_stats = True
            fisher_loss.backward(retain_graph=True)
            self.optimizer.acc_stats = False

        self.optimizer.zero_grad()
        
        ### MODIFIED: Added infeasibility_loss (E_inf) to the final loss calculation
        loss = (value_loss * self.value_loss_coef 
                + action_loss 
                - dist_entropy * self.entropy_coef
                + infeasibility_loss * self.invaild_coef) # 'invaild_coef' is the 'ω' from the paper
        
        loss.backward()

        if self.acktr == False:
            nn.utils.clip_grad_norm_(
                self.actor_critic.parameters(), self.max_grad_norm)

        self.optimizer.step()

        # Update the return signature
        return value_loss.item(), action_loss.item(), dist_entropy.item(), infeasibility_loss.item()

def check_nan(model,index):
    for p in model.parameters():
        if np.isnan(p.grad.data.mean().item()):
            print('index '+ str(index) +' happened an error!')