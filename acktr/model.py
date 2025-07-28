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

    def act(self, inputs, rnn_hs, masks, deterministic=False):
        value, actor_features, rnn_hs = self.base(inputs, rnn_hs, masks)
        
        obs_image = inputs.view(-1, 6, self.base.width, self.base.length)
        mask_o0, mask_o1 = obs_image[:, 4, :, :], obs_image[:, 5, :, :]
        
        o_mask_0_valid = mask_o0.any(dim=(-1,-2)).float()
        o_mask_1_valid = mask_o1.any(dim=(-1,-2)).float()
        o_mask = torch.stack([o_mask_0_valid, o_mask_1_valid], dim=1)
        
        dist_o, _, _ = self.dist_o(actor_features, o_mask)
        action_o = dist_o.mode() if deterministic else dist_o.sample()

        o_one_hot = F.one_hot(action_o.squeeze(-1), num_classes=2).float()
        x_input = torch.cat([actor_features, o_one_hot], dim=1)
        
        condition = (action_o == 0).view(-1, 1, 1)
        mask_for_o = torch.where(condition, mask_o0, mask_o1)

        x_mask = mask_for_o.any(dim=-1).float()
        dist_x, _, _ = self.dist_x(x_input, x_mask)
        action_x = dist_x.mode() if deterministic else dist_x.sample()
        
        x_one_hot = F.one_hot(action_x.squeeze(-1), num_classes=self.base.width).float()
        y_input = torch.cat([actor_features, o_one_hot, x_one_hot], dim=1)
        
        # --- THIS IS THE FIX ---
        # Expand action_x to a 3D tensor to be used as an index for the 3D mask_for_o
        index = action_x.unsqueeze(2).expand(-1, -1, self.base.length)
        y_mask = torch.gather(mask_for_o, 1, index).squeeze(1).float()
        # --- END FIX ---

        dist_y, _, _ = self.dist_y(y_input, y_mask)
        action_y = dist_y.mode() if deterministic else dist_y.sample()

        log_prob_o = dist_o.log_prob(action_o.squeeze(-1))
        log_prob_x = dist_x.log_prob(action_x.squeeze(-1))
        log_prob_y = dist_y.log_prob(action_y.squeeze(-1))
        action_log_probs = (log_prob_o + log_prob_x + log_prob_y).unsqueeze(-1)
        
        action = torch.cat([action_o, action_x, action_y], dim=1)
        return value, action, action_log_probs, rnn_hs

    def evaluate_actions(self, inputs, rnn_hs, masks, action, gt_masks):
        value, actor_features, rnn_hs = self.base(inputs, rnn_hs, masks)
        
        action_o, action_x, action_y = action[:, 0], action[:, 1], action[:, 2]
        
        mask_o0 = gt_masks[:, 0, :, :]
        mask_o1 = gt_masks[:, 1, :, :]
        o_mask_0_valid = mask_o0.any(dim=(-1,-2)).float()
        o_mask_1_valid = mask_o1.any(dim=(-1,-2)).float()
        o_mask = torch.stack([o_mask_0_valid, o_mask_1_valid], dim=1)
        dist_o, _, _ = self.dist_o(actor_features, o_mask)

        o_one_hot = F.one_hot(action_o, num_classes=2).float()
        x_input = torch.cat([actor_features, o_one_hot], dim=1)
        
        condition = (action_o == 0).view(-1, 1, 1)
        mask_for_o = torch.where(condition, mask_o0, mask_o1)
        x_mask = mask_for_o.any(dim=-1).float()
        dist_x, _, _ = self.dist_x(x_input, x_mask)

        x_one_hot = F.one_hot(action_x, num_classes=self.base.width).float()
        y_input = torch.cat([actor_features, o_one_hot, x_one_hot], dim=1)
        
        # --- THIS IS THE FIX ---
        # Reshape the 1D action_x tensor to 3D to be used as an index
        index = action_x.view(-1, 1, 1).expand(-1, 1, self.base.length)
        y_mask = torch.gather(mask_for_o, 1, index).squeeze(1).float()
        # --- END FIX ---
        
        dist_y, _, _ = self.dist_y(y_input, y_mask)
        
        probs_x, probs_y = dist_x.probs, dist_y.probs
        prob_map = probs_x.unsqueeze(2) * probs_y.unsqueeze(1)
        mask_for_chosen_o = torch.gather(gt_masks, 1, action_o.view(-1, 1, 1, 1).expand(-1, 1, self.base.width, self.base.length)).squeeze(1)
        infeasibility_mask = 1.0 - mask_for_chosen_o
        infeasibility_loss = (prob_map * infeasibility_mask).sum() / actor_features.size(0)

        action_log_probs = dist_o.log_prob(action_o) + dist_x.log_prob(action_x) + dist_y.log_prob(action_y)
        dist_entropy = dist_o.entropy().mean() + dist_x.entropy().mean() + dist_y.entropy().mean()
        
        return value, action_log_probs, dist_entropy, rnn_hs, infeasibility_loss

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
        
        # 1. Shared convolutional feature extractor (no dense layers)
        self.shared_conv = nn.Sequential(
            init_(nn.Conv2d(num_inputs, 64, 3, stride=1, padding=1)),
            nn.LeakyReLU(),
            init_(nn.Conv2d(64, 64, 3, stride=1, padding=1)),
            nn.LeakyReLU(),
            init_(nn.Conv2d(64, 64, 3, stride=1, padding=1)),
            nn.LeakyReLU(),
            init_(nn.Conv2d(64, 64, 3, stride=1, padding=1)),
            nn.LeakyReLU(),
            init_(nn.Conv2d(64, 64, 3, stride=1, padding=1)),
            nn.LeakyReLU()
        )

        # 2. Actor head with its own dimensionality reduction
        self.actor_head = nn.Sequential(
            # Use a 1x1 Conv to reduce channels from 64 to 8
            init_(nn.Conv2d(64, 8, 1, stride=1)),
            nn.LeakyReLU(),
            Flatten(),
            # The input dimension is now small: 8 * width * length
            init_(nn.Linear(8 * width * length, hidden_size)),
            nn.LeakyReLU()
        )

        # 3. Critic head with its own dimensionality reduction
        self.critic_head = nn.Sequential(
            # Use a 1x1 Conv to reduce channels from 64 to 4
            init_(nn.Conv2d(64, 4, 1, stride=1)),
            nn.LeakyReLU(),
            Flatten(),
            # The input dimension is now small: 4 * width * length
            init_(nn.Linear(4 * width * length, hidden_size))
        )

        # Final linear layer for the critic value
        self.critic_linear = init_(nn.Linear(hidden_size, 1))
        
        self.train()

    def forward(self, inputs, rnn_hxs, masks):
        # Reshape the input
        x = inputs.view(-1, 6, self.width, self.length)
        
        # --- New Data Flow ---
        # 1. Get features from the shared convolutional base
        shared_features = self.shared_conv(x)
        
        # 2. Process through the actor head to get features for the policy
        actor_features = self.actor_head(shared_features)
        
        # 3. Process through the critic head to get the value
        critic_features = self.critic_head(shared_features)
        value = self.critic_linear(critic_features)

        # Return the same outputs as before
        return value, actor_features, rnn_hxs
