# acktr/model.py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# The custom Categorical class is a wrapper that includes the final linear layer
from acktr.distributions import Categorical
# The actual distribution object returned by the wrapper
from torch.distributions.categorical import Categorical as TorchCategorical
from acktr.utils import init

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class Policy(nn.Module):
    def __init__(self, obs_shape, action_space, base=None, base_kwargs=None):
        super(Policy, self).__init__()
        if base_kwargs is None:
            base_kwargs = {}
        
        base = CNNPro
        
        # --- MODIFICATION ---
        # Get width and length from the new action space indices
        width = action_space.nvec[1]
        length = action_space.nvec[2]
        
        self.base = base(num_inputs=6, width=width, length=length, **base_kwargs)

        hidden_size = self.base.output_size

        # The order of these definitions doesn't change
        self.dist_o = Categorical(hidden_size, 2)
        self.dist_x = Categorical(hidden_size + 2, width)
        self.dist_y = Categorical(hidden_size + 2 + width, length)
        
    @property
    def is_recurrent(self):
        return self.base.is_recurrent

    @property
    def recurrent_hidden_state_size(self):
        return self.base.recurrent_hidden_state_size

    def forward(self, inputs, rnn_hxs, masks):
        raise NotImplementedError

    def get_value(self, inputs, rnn_hxs, masks):
        value, _, _ = self.base(inputs, rnn_hxs, masks)
        return value

    def act(self, inputs, rnn_hxs, masks, deterministic=False):
        value, actor_features, rnn_hxs = self.base(inputs, rnn_hxs, masks)
        
        batch_size = actor_features.shape[0]
        device = actor_features.device

        # 1. Get Orientation distribution and sample
        mask_o = torch.ones(batch_size, 2).to(device)
        dist_o, _, _ = self.dist_o(actor_features, mask_o)
        
        # --- THIS IS THE FIX ---
        # Both .mode and .sample must be CALLED as methods with parentheses ()
        action_o = dist_o.mode() if deterministic else dist_o.sample()

        # 2. Get X-position distribution and sample
        o_one_hot = F.one_hot(action_o.squeeze(-1), num_classes=2).float()
        x_input = torch.cat([actor_features, o_one_hot], dim=1)
        mask_x = torch.ones(batch_size, self.base.width).to(device)
        dist_x, _, _ = self.dist_x(x_input, mask_x)
        
        action_x = dist_x.mode() if deterministic else dist_x.sample()

        # 3. Get Y-position distribution and sample
        x_one_hot = F.one_hot(action_x.squeeze(-1), num_classes=self.base.width).float()
        y_input = torch.cat([actor_features, o_one_hot, x_one_hot], dim=1)
        mask_y = torch.ones(batch_size, self.base.length).to(device)
        dist_y, _, _ = self.dist_y(y_input, mask_y)
        
        action_y = dist_y.mode() if deterministic else dist_y.sample()

        # 4. Combine actions and log probabilities
        log_prob_o = dist_o.log_prob(action_o.squeeze(-1))
        log_prob_x = dist_x.log_prob(action_x.squeeze(-1))
        log_prob_y = dist_y.log_prob(action_y.squeeze(-1))
        action_log_probs = (log_prob_o + log_prob_x + log_prob_y).unsqueeze(-1)
        
        action = torch.cat([action_o, action_x, action_y], dim=1)

        return value, action, action_log_probs, rnn_hxs

    def evaluate_actions(self, inputs, rnn_hxs, masks, action, gt_masks):
        value, actor_features, rnn_hxs = self.base(inputs, rnn_hxs, masks)
        
        action_o, action_x, action_y = action[:, 0], action[:, 1], action[:, 2]
        
        batch_size = actor_features.shape[0]
        device = actor_features.device

        # --- Distributions and Probabilities ---
        mask_o = torch.ones(batch_size, 2).to(device)
        dist_o, _, _ = self.dist_o(actor_features, mask_o)

        o_one_hot = F.one_hot(action_o, num_classes=2).float()
        x_input = torch.cat([actor_features, o_one_hot], dim=1)
        mask_x = torch.ones(batch_size, self.base.width).to(device)
        dist_x, _, _ = self.dist_x(x_input, mask_x)

        x_one_hot = F.one_hot(action_x, num_classes=self.base.width).float()
        y_input = torch.cat([actor_features, o_one_hot, x_one_hot], dim=1)
        mask_y = torch.ones(batch_size, self.base.length).to(device)
        dist_y, _, _ = self.dist_y(y_input, mask_y)

        # --- E_inf CALCULATION ---
        # Get the full probability distributions for x and y
        probs_x = dist_x.probs
        probs_y = dist_y.probs
        
        # Approximate the joint probability P(x, y | o) as P(x|o) * P(y|o,x) ~= P(x) * P(y)
        # This gives a probability map for every (x,y) location
        prob_map = probs_x.unsqueeze(2) * probs_y.unsqueeze(1) # Shape: (batch, width, length)

        # Select the ground-truth feasibility mask for the chosen orientation
        mask_for_chosen_o = torch.gather(gt_masks, 1, action_o.view(-1, 1, 1, 1).expand(-1, 1, self.base.width, self.base.length)).squeeze(1)
        
        # The infeasibility mask is the inverse of the feasibility mask
        infeasibility_mask = 1.0 - mask_for_chosen_o
        
        # E_inf is the sum of probabilities at all infeasible locations
        infeasibility_loss = (prob_map * infeasibility_mask).sum() / batch_size

        # --- Original Calculations ---
        action_log_probs = dist_o.log_prob(action_o) + dist_x.log_prob(action_x) + dist_y.log_prob(action_y)
        dist_entropy = dist_o.entropy().mean() + dist_x.entropy().mean() + dist_y.entropy().mean()

        # Return the new loss term
        return value, action_log_probs, dist_entropy, rnn_hxs, infeasibility_loss

class NNBase(nn.Module):
    # ... (This class is unchanged) ...
    def __init__(self, recurrent, recurrent_input_size, hidden_size):
        super(NNBase, self).__init__()
        self._hidden_size = hidden_size
        self._recurrent = recurrent
    @property
    def is_recurrent(self):
        return self._recurrent
    @property
    def recurrent_hidden_state_size(self):
        if self._recurrent:
            return self._hidden_size
        return 1
    @property
    def output_size(self):
        return self._hidden_size

class CNNPro(NNBase):
    def __init__(self, num_inputs, recurrent=False, hidden_size=512, width=10, length=10):
        super(CNNPro, self).__init__(recurrent, num_inputs, hidden_size)
        
        self.width = width
        self.length = length

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0), nn.init.calculate_gain('leaky_relu'))
        
        # Shared CNN feature extractor
        self.main = nn.Sequential(
            init_(nn.Conv2d(num_inputs, 64, 3, stride=1, padding=1)),
            nn.LeakyReLU(),
            init_(nn.Conv2d(64, 64, 3, stride=1, padding=1)),
            nn.LeakyReLU(),
            init_(nn.Conv2d(64, 64, 3, stride=1, padding=1)),
            nn.LeakyReLU(),
            init_(nn.Conv2d(64, 64, 3, stride=1, padding=1)),
            nn.LeakyReLU(),
            init_(nn.Conv2d(64, 64, 3, stride=1, padding=1)),
            nn.LeakyReLU(),
            Flatten(),
            init_(nn.Linear(64 * width * length, hidden_size)),
            nn.LeakyReLU()
        )

        # Critic head
        self.critic_linear = init_(nn.Linear(hidden_size, 1))
        
        # ### --- REMOVED --- ###
        # The actor heads are now in the Policy class.
        
        self.train()

    def forward(self, inputs, rnn_hxs, masks):
        x = inputs.view(-1, 6, self.width, self.length)
        
        # Get shared features from the main body
        actor_features = self.main(x)
        
        # Compute critic value
        value = self.critic_linear(actor_features)

        # Return value and shared features for the actor
        return value, actor_features, rnn_hxs