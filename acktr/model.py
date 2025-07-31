# acktr/model.py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import time

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
        #start_time = time.perf_counter()
        value, actor_features, rnn_hs = self.base(inputs, rnn_hs, masks)
        #end_time = time.perf_counter()
        #print(f"Extractor eval: {end_time - start_time} seconds")
        
        action_o, action_x, action_y = action[:, 0], action[:, 1], action[:, 2]
        
        # --- Part 1: 'o' distribution (unchanged) ---
        mask_o0 = gt_masks[:, 0]
        mask_o1 = gt_masks[:, 1]
        o_mask_0_valid = mask_o0.any(dim=(-1, -2)).float()
        o_mask_1_valid = mask_o1.any(dim=(-1, -2)).float()
        o_mask = torch.stack([o_mask_0_valid, o_mask_1_valid], dim=1)
        dist_o, _, _ = self.dist_o(actor_features, o_mask)

        o_one_hot = F.one_hot(action_o, num_classes=2).float()
        x_input = torch.cat([actor_features, o_one_hot], dim=1)
        
        # --- OPTIMIZATION 1: Calculate the chosen mask ONCE ---
        # This mask is needed for the 'x' distribution, 'y' distribution, and the loss.
        # Let gt_masks be (B, 2, W, L). This selects a mask of shape (B, W, L) based on action_o.
        condition = (action_o == 0).view(-1, 1, 1)
        mask_for_o = torch.where(condition, mask_o0, mask_o1)
        
        # --- Part 2: 'x' distribution ---
        x_mask = mask_for_o.any(dim=-1).float()
        dist_x, _, _ = self.dist_x(x_input, x_mask)

        x_one_hot = F.one_hot(action_x, num_classes=self.base.width).float()
        y_input = torch.cat([actor_features, o_one_hot, x_one_hot], dim=1)
        
        # --- OPTIMIZATION 2: Use Advanced Indexing for `y_mask` ---
        # This is more direct and faster than the original view-expand-gather-squeeze sequence.
        # It selects the row `action_x[i]` from `mask_for_o[i]` for each item in the batch.
        batch_indices = torch.arange(mask_for_o.size(0), device=action_x.device)
        y_mask = mask_for_o[batch_indices, action_x].float()
        
        # --- Part 3: 'y' distribution ---
        dist_y, _, _ = self.dist_y(y_input, y_mask)
        
        # --- OPTIMIZATION 3: Reuse `mask_for_o` for loss calculation ---
        # The original code recalculated this exact same tensor.
        probs_x, probs_y = dist_x.probs, dist_y.probs
        prob_map = probs_x.unsqueeze(2) * probs_y.unsqueeze(1)
        infeasibility_mask = 1.0 - mask_for_o.float()
        infeasibility_loss = torch.mean(prob_map * infeasibility_mask)

        # --- Final calculations (unchanged) ---
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

def init_(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, nonlinearity='leaky_relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    return m

class CNNPro(NNBase):
    def __init__(self, num_inputs, recurrent=False, hidden_size=512, width=100, length=100):
        super(CNNPro, self).__init__(recurrent, num_inputs, hidden_size)
        
        self.width = width
        self.length = length

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0), nn.init.calculate_gain('leaky_relu'))
        
        # 1. Define the shared convolutional feature extractor first
        self.shared_conv = nn.Sequential(
            # Block 1: stride 1
            init_(nn.Conv2d(num_inputs, num_inputs, kernel_size=3, stride=1, padding=1, groups=num_inputs, bias=False)),
            init_(nn.Conv2d(num_inputs, 64, kernel_size=1, stride=1, padding=0, bias=False)),
            nn.LeakyReLU(),

            # Block 2: stride 2 (DOWNSAMPLE)
            init_(nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, groups=64, bias=False)),
            init_(nn.Conv2d(64, 64, kernel_size=1, stride=1, padding=0, bias=False)),
            nn.LeakyReLU(),

            # Block 3: stride 2 (DOWNSAMPLE)
            init_(nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, groups=64, bias=False)),
            init_(nn.Conv2d(64, 64, kernel_size=1, stride=1, padding=0, bias=False)),
            nn.LeakyReLU(),

            # Block 4: stride 2 (DOWNSAMPLE)
            init_(nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, groups=64, bias=False)),
            init_(nn.Conv2d(64, 64, kernel_size=1, stride=1, padding=0, bias=False)),
            nn.LeakyReLU(),

            # Block 5: stride 1
            init_(nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, groups=64, bias=False)),
            init_(nn.Conv2d(64, 64, kernel_size=1, stride=1, padding=0, bias=False)),
            nn.LeakyReLU()
        )

        # --- DYNAMIC CALCULATION ---
        # Create a dummy input tensor and pass it through the conv layers to find the output shape
        with torch.no_grad():
            dummy_input = torch.zeros(1, num_inputs, self.width, self.length)
            conv_output = self.shared_conv(dummy_input)
            # The shape of conv_output is (1, channels, final_length, final_width)
            final_conv_length = conv_output.shape[2]
            final_conv_width = conv_output.shape[3]
        # --- END DYNAMIC CALCULATION ---
        
        # 2. Actor head - now uses the dynamically calculated size
        actor_linear_in_features = 8 * final_conv_length * final_conv_width
        self.actor_head = nn.Sequential(
            init_(nn.Conv2d(64, 8, 1, stride=1)),
            nn.LeakyReLU(),
            Flatten(),
            init_(nn.Linear(actor_linear_in_features, hidden_size)),
            nn.LeakyReLU()
        )

        # 3. Critic head - also uses the dynamically calculated size
        critic_linear_in_features = 4 * final_conv_length * final_conv_width
        self.critic_head = nn.Sequential(
            init_(nn.Conv2d(64, 4, 1, stride=1)),
            nn.LeakyReLU(),
            Flatten(),
            init_(nn.Linear(critic_linear_in_features, hidden_size))
        )

        self.critic_linear = init_(nn.Linear(hidden_size, 1))
        self.train()
    
    # The 'forward' method does not need to be changed
    def forward(self, inputs, rnn_hxs, masks):
        #start_time = time.perf_counter()
        x = inputs.view(-1, 6, self.width, self.length)
        shared_features = self.shared_conv(x)
        actor_features = self.actor_head(shared_features)
        critic_features = self.critic_head(shared_features)
        value = self.critic_linear(critic_features)
        #end_time = time.perf_counter()
        #print(f"full forward: {end_time - start_time} seconds")
        return value, actor_features, rnn_hxs