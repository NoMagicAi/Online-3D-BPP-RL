import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# The custom Categorical class is a wrapper that includes the final linear layer
from acktr.distributions import Categorical
# The actual distribution object returned by the wrapper
from torch.distributions.categorical import Categorical as TorchCategorical
from acktr.utils import init

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

#==============================================================================
# --- Base Class and Helper Module Definitions ---
#==============================================================================

def init_(m):
    if isinstance(m, SeparableConv2d):
        nn.init.kaiming_normal_(m.depthwise.weight, nonlinearity='relu') # Note: SiLU behaves like ReLU for init
        if m.depthwise.bias is not None:
            nn.init.constant_(m.depthwise.bias, 0)
        nn.init.kaiming_normal_(m.pointwise.weight, nonlinearity='relu')
        if m.pointwise.bias is not None:
            nn.init.constant_(m.pointwise.bias, 0)
    elif isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
       nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
       if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    return m
class CategoricalWithEpsilonMask(nn.Module):
    """
    A distribution that applies a soft mask with a small epsilon value
    directly to the final probabilities.

    This implementation maintains the original epsilon-masking logic on the
    probability distribution while incorporating the initialization style and
    helper methods from the reference class for improved structure.
    """
    def __init__(self, num_inputs, num_outputs, epsilon=1e-20):
        """
        Initializes the CategoricalWithEpsilonMask module.

        Args:
            num_inputs (int): The number of input features.
            num_outputs (int): The number of output actions.
            epsilon (float): A small value to assign to masked action probabilities.
        """
        super(CategoricalWithEpsilonMask, self).__init__()

        # This will now call the externally-defined `init_` function
        # that you provided to initialize the linear layer.
        self.linear = init_(nn.Linear(num_inputs, num_outputs))
        self.epsilon = epsilon

    def forward(self, x, mask=None):
        """
        Computes the policy distribution from the network inputs.

        Args:
            x (Tensor): The input tensor from the policy network.
            mask (Tensor, optional): A binary tensor where a 0 indicates a
                                     masked (invalid) action. Defaults to None.

        Returns:
            tuple: A tuple containing:
                - final_dist (TorchCategorical): The final distribution after
                  optionally applying the mask.
                - raw_probs (Tensor): The original, unmasked probabilities,
                  often useful for auxiliary losses or analysis.
        """
        logits = self.linear(x)

        # Get the initial, unmasked "raw" probability distribution.
        raw_probs = F.softmax(logits, dim=-1)

        # If no mask is provided, the final distribution is based on the original logits.
        # We still return both the distribution and raw_probs for a consistent API.
        if mask is None:
            return TorchCategorical(logits=logits), raw_probs

        # --- Epsilon Masking Logic ---
        # The core logic of this class is to apply a "soft" mask directly to the
        # probabilities. This ensures masked actions have a tiny (epsilon)
        # probability rather than being completely excluded. This differs from the
        # common "hard mask" technique of subtracting a large number from logits.

        # Clone the probabilities to avoid in-place modification of a tensor
        # that might be used elsewhere.
        probs_clone = raw_probs.clone()

        # Apply the soft mask: set the probability of invalid actions to epsilon.
        probs_clone[mask == 0] = self.epsilon

        # Re-normalize the probabilities so they sum to 1 again.
        renormalized_probs = probs_clone / probs_clone.sum(dim=-1, keepdim=True)

        # Create the final categorical distribution from the re-normalized probabilities.
        final_dist = TorchCategorical(probs=renormalized_probs)

        # Return a tuple: the final modulated distribution AND the raw probabilities.
        return final_dist, raw_probs

    def get_logits(self, x):
        """
        A helper method to get the raw logits from the linear layer, styled after
        the `get_policy_distribution` method in the reference class.

        Args:
            x (Tensor): The input tensor.

        Returns:
            Tensor: The raw output logits from the linear layer.
        """
        return self.linear(x)

class NNBase(nn.Module):
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

class SimpleCNNBase(NNBase):
    """
    A simplified CNN base network inspired by the paper's description of a
    "State CNN 5 layers LeakyReLU".

    This version removes the AdaptiveAvgPool2d layer and operates directly on the
    flattened output of the CNN feature extractor.
    """
    def __init__(self, num_inputs, recurrent=False, hidden_size=512, width=100, length=100):
        super(SimpleCNNBase, self).__init__(recurrent, num_inputs, hidden_size)

        self.width = width
        self.length = length

        # --- Simplified 5-Layer CNN Feature Extractor ---
        self.cnn_base = nn.Sequential(
            init_(nn.Conv2d(num_inputs, 32, kernel_size=3, stride=1, padding=1)),
            nn.LeakyReLU(),
            init_(nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)), # Downsample 100->50
            nn.LeakyReLU(),
            init_(nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)), # Downsample 50->25
            nn.LeakyReLU(),
            init_(nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)), # Downsample 25->13
            nn.LeakyReLU(),
            init_(nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)),
            nn.LeakyReLU()
        )

        # --- MODIFIED: Dynamically calculate the flattened feature size ---
        # Helper function to get the output size of the conv base
        def _get_conv_output_shape(shape):
            with torch.no_grad():
                dummy_input = torch.rand(1, *shape)
                output_features = self.cnn_base(dummy_input)
                return output_features.shape[1:] # Returns (C, H, W)

        # Calculate the shape and total number of features
        conv_out_shape = _get_conv_output_shape((num_inputs, self.width, self.length))
        head_in_features = math.prod(conv_out_shape) # C * H * W

        # --- MODIFIED: Heads now operate on the pure, flattened output ---
        # REMOVED: `POOL_OUTPUT_SIZE` is no longer needed.

        # Actor head to generate features for the policy
        self.actor_head = nn.Sequential(
            # REMOVED: nn.AdaptiveAvgPool2d(...)
            Flatten(),
            init_(nn.Linear(head_in_features, hidden_size)), # MODIFIED: Using the new calculated size
            nn.LeakyReLU()
        )

        # Critic head to predict the state value
        self.critic_head_decoupled = nn.Sequential(
            # REMOVED: nn.AdaptiveAvgPool2d(...)
            Flatten(),
            init_(nn.Linear(head_in_features, hidden_size)), # MODIFIED: Using the new calculated size
            nn.LeakyReLU(),
            init_(nn.Linear(hidden_size, 1))
        )

        self.train()

    def forward(self, inputs, rnn_hxs, masks):
        x = inputs.view(-1, 6, self.width, self.length)

        # 1. Extract features with the simpler 5-layer CNN
        features = self.cnn_base(x)

        # 2. Compute value and actor features from the common base
        #    (This part of the forward pass works without any changes)
        value = self.critic_head_decoupled(features)
        actor_features = self.actor_head(features)

        # 3. Return in the same format as the original class
        return value, actor_features, rnn_hxs

class SeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, bias=False):
        super(SeparableConv2d, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, 
                                       stride=stride, padding=padding, dilation=dilation, groups=in_channels, bias=bias)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


class Policy(nn.Module):
    def __init__(self, obs_shape, action_space, base=None, base_kwargs=None):
        super(Policy, self).__init__()
        if base_kwargs is None:
            base_kwargs = {}

        # Use the NEW Upgraded FinalFusionNet_v2 as the base
        base = SimpleCNNBase

        width = action_space.nvec[1]
        length = action_space.nvec[2]

        self.base = base(num_inputs=6, width=width, length=length, **base_kwargs)

        hidden_size = self.base.output_size

        self.dist_o = CategoricalWithEpsilonMask(hidden_size, 2)
        self.dist_x = CategoricalWithEpsilonMask(hidden_size + 2, width)
        self.dist_y = CategoricalWithEpsilonMask(hidden_size + width, length)

    @property
    def is_recurrent(self):
        return self.base.is_recurrent

    @property
    def recurrent_hidden_state_size(self):
        return self.base.recurrent_hidden_state_size

    def forward(self, inputs, rnn_hxs, masks):
        raise NotImplementedError

    def get_value(self, inputs, rnn_hxs, masks):
        # CHANGED: Unpacking updated for the new return signature from self.base.
        value, _, _ = self.base(inputs, rnn_hxs, masks)
        return value

    def act(self, inputs, rnn_hs, masks, deterministic=False):
            # CHANGED: Unpacking updated for the new return signature. `shared_features` is removed.
            value, orientation_features, rnn_hs = self.base(inputs, rnn_hs, masks)
            
            obs_image = inputs.view(-1, 6, self.base.width, self.base.length)
            mask_o0, mask_o1 = obs_image[:, 4, :, :], obs_image[:, 5, :, :]
            o_mask_0_valid = mask_o0.any(dim=(-1,-2)).float()
            o_mask_1_valid = mask_o1.any(dim=(-1,-2)).float()
            o_mask = torch.stack([o_mask_0_valid, o_mask_1_valid], dim=1)

            dist_o, _ = self.dist_o(orientation_features, o_mask)
            action_o = dist_o.mode() if deterministic else dist_o.sample()
            
            #o_one_hot = F.one_hot(action_o.squeeze(-1), num_classes=2).float()
            x_input = torch.cat([orientation_features, dist_o.probs], dim=1)
            #condition = (action_o == 0).view(-1, 1, 1)
            mask_for_o = torch.logical_or(mask_o0, mask_o1)
            x_mask = mask_for_o.any(dim=-1).float()

            dist_x, _ = self.dist_x(x_input, x_mask)
            action_x = dist_x.mode() if deterministic else dist_x.sample()
            
            x_one_hot = F.one_hot(action_x.squeeze(-1), num_classes=self.base.width).float()
            y_input = torch.cat([orientation_features, dist_x.probs], dim=1)
            index = action_x.unsqueeze(2).expand(-1, -1, self.base.length)
            y_mask = mask_for_o.any(dim=-2).float()

            dist_y, _ = self.dist_y(y_input, y_mask)
            action_y = dist_y.mode() if deterministic else dist_y.sample()
            
            log_prob_o = dist_o.log_prob(action_o.squeeze(-1))
            log_prob_x = dist_x.log_prob(action_x.squeeze(-1))
            log_prob_y = dist_y.log_prob(action_y.squeeze(-1))
            
            action_log_probs = (log_prob_o + log_prob_x + log_prob_y).unsqueeze(-1)
            action = torch.cat([action_o, action_x, action_y], dim=1)
            
            return value, action, action_log_probs, rnn_hs

    def evaluate_actions(self, inputs, rnn_hs, masks, action, gt_masks):
            # CHANGED: Unpacking updated for the new return signature. `shared_features` is removed.
            value, orientation_features, rnn_hs = self.base(inputs, rnn_hs, masks)
            
            action_o, action_x, action_y = action[:, 0], action[:, 1], action[:, 2]
            
            mask_o0 = gt_masks[:, 0]
            mask_o1 = gt_masks[:, 1]
            o_mask_0_valid = mask_o0.any(dim=(-1, -2)).float()
            o_mask_1_valid = mask_o1.any(dim=(-1, -2)).float()
            o_mask = torch.stack([o_mask_0_valid, o_mask_1_valid], dim=1)
            
            dist_o, _ = self.dist_o(orientation_features, o_mask)
            
            o_one_hot = F.one_hot(action_o.long(), num_classes=2).float()
            x_input = torch.cat([orientation_features, dist_o.probs], dim=1)
            #condition = (action_o == 0).view(-1, 1, 1)
            mask_for_o = torch.logical_or(mask_o0, mask_o1)
            x_mask = mask_for_o.any(dim=-1).float()
            
            dist_x, raw_probs_x = self.dist_x(x_input, x_mask)
            
            x_one_hot = F.one_hot(action_x.long(), num_classes=self.base.width).float()
            y_input = torch.cat([orientation_features, dist_x.probs], dim=1)
            #batch_indices = torch.arange(mask_for_o.size(0), device=action_x.device)
            y_mask = mask_for_o.any(dim=-2).float()
            
            dist_y, raw_probs_y = self.dist_y(y_input, y_mask)
            
            prob_map = raw_probs_x.unsqueeze(2) * raw_probs_y.unsqueeze(1)
            infeasibility_mask = 1.0 - mask_for_o.float()
            infeasibility_loss = torch.mean(prob_map * infeasibility_mask)
            
            action_log_probs = dist_o.log_prob(action_o) + dist_x.log_prob(action_x) + dist_y.log_prob(action_y)
            dist_entropy = dist_o.entropy().mean() + dist_x.entropy().mean() + dist_y.entropy().mean()
            
            return value, action_log_probs, dist_entropy, rnn_hs, infeasibility_loss