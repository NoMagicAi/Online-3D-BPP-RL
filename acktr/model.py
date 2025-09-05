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
# --- Custom Modules and Helper Functions ---
#==============================================================================

class SeparableConv2d(nn.Module):
    """
    Depthwise separable 2D convolution.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, bias=False):
        super(SeparableConv2d, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size,
                                     stride=stride, padding=padding, dilation=dilation, groups=in_channels, bias=bias)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x

def init_(m):
    """
    Initializes weights of the given module.
    """
    if isinstance(m, SeparableConv2d):
        nn.init.kaiming_normal_(m.depthwise.weight, nonlinearity='leaky_relu')
        if m.depthwise.bias is not None:
            nn.init.constant_(m.depthwise.bias, 0)
        nn.init.kaiming_normal_(m.pointwise.weight, nonlinearity='leaky_relu')
        if m.pointwise.bias is not None:
            nn.init.constant_(m.pointwise.bias, 0)
    elif isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.kaiming_normal_(m.weight, nonlinearity='leaky_relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    return m

class AddCoords(nn.Module):
    """
    Adds coordinate channels to the input tensor.
    """
    def __init__(self, with_r=False):
        super().__init__()
        self.with_r = with_r

    def forward(self, x):
        b, _, h, w = x.size()
        xx_channel = torch.arange(w, device=x.device).float()
        yy_channel = torch.arange(h, device=x.device).float()
        xx_channel = (xx_channel.repeat(b, 1, h, 1) / (w - 1)) * 2 - 1
        yy_channel = (yy_channel.repeat(b, 1, w, 1).permute(0, 1, 3, 2) / (h - 1)) * 2 - 1
        ret = torch.cat([x, xx_channel, yy_channel], dim=1)
        if self.with_r:
            rr = torch.sqrt(torch.pow(xx_channel - 0.5, 2) + torch.pow(yy_channel - 0.5, 2))
            ret = torch.cat([ret, rr], dim=1)
        return ret

class CoordConv(nn.Module):
    """
    A separable convolutional layer that first adds coordinate channels.
    """
    def __init__(self, in_channels, out_channels, with_r=False, **kwargs):
        super().__init__()
        self.addcoords = AddCoords(with_r=with_r)
        coord_channels = 3 if with_r else 2
        # Internal init_ call on the separable conv layer
        self.conv = init_(SeparableConv2d(in_channels + coord_channels, out_channels, **kwargs))

    def forward(self, x):
        x = self.addcoords(x)
        x = self.conv(x)
        return x

class h_swish(nn.Module):
    def forward(self, x):
        return x * F.relu6(x + 3) / 6

class CoordAttn(nn.Module):
    """
    Coordinate Attention Block.
    """
    def __init__(self, inp, oup, reduction=32):
        super(CoordAttn, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        mip = max(8, inp // reduction)
        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_swish()
        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        identity = x
        n, c, h, w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)
        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)
        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()
        return identity * a_w * a_h

class ResidualCoordAttnBlock(nn.Module):
    """
    A residual block with Coordinate Attention, using separable convolution.
    """
    def __init__(self, in_channels):
        super().__init__()
        self.attn = CoordAttn(in_channels, in_channels)
        self.conv = init_(SeparableConv2d(in_channels, in_channels, kernel_size=1, stride=1))

    def forward(self, x):
        res = self.attn(x)
        res = self.conv(res)
        return x + res

class CategoricalWithEpsilonMask(nn.Module):
    """
    A distribution that applies a soft mask with a small epsilon value
    directly to the final probabilities.
    (Note: The corrected implementation masks logits for stability)
    """
    def __init__(self, num_inputs, num_outputs, epsilon=1e-20):
        super(CategoricalWithEpsilonMask, self).__init__()
        self.linear = init_(nn.Linear(num_inputs, num_outputs))
        # Epsilon is kept for API compatibility but is no longer used
        # in the numerically stable logit-masking implementation.
        self.epsilon = epsilon

    def forward(self, x, mask=None):
        logits = self.linear(x)
        
        # 1. Calculate raw_probs first to satisfy the API's return requirement.
        raw_probs = F.softmax(logits, dim=-1)

        if mask is None:
            # If no mask, the distribution is based on the original logits.
            return TorchCategorical(logits=logits), raw_probs
        
        # 2. Apply the mask to the logits for numerical stability.
        # We use a large negative number to make the probability of masked actions negligible.
        masked_logits = logits.clone() # Clone to avoid modifying the original logits
        masked_logits[mask == 0] = -1e10

        # 3. Create the final distribution from the masked logits.
        # The softmax operation is handled efficiently and safely inside the constructor.
        final_dist = TorchCategorical(logits=masked_logits)
        
        # 4. Return the final distribution and the original, unmasked probabilities.
        return final_dist, raw_probs

    def get_logits(self, x):
        return self.linear(x)

        
#==============================================================================
# --- Base Class Definitions ---
#==============================================================================

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
    A CNN base network that uses Separable Convolutions, Coordinate Convolution,
    and a final Residual Coordinate Attention Block.
    """
    def __init__(self, num_inputs, recurrent=False, hidden_size=512, width=100, length=100):
        super(SimpleCNNBase, self).__init__(recurrent, num_inputs, hidden_size)

        self.width = width
        self.length = length

        # --- MODIFIED: Upgraded 5-Layer CNN Feature Extractor ---
        self.cnn_base = nn.Sequential(
            # 1. Use Coordinate Convolution at the beginning
            CoordConv(num_inputs, 32, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            
            # 2. Use Separable Convolutions for subsequent layers
            init_(SeparableConv2d(32, 64, kernel_size=3, stride=2, padding=1)), # Downsample 100->50
            nn.LeakyReLU(),
            init_(SeparableConv2d(64, 128, kernel_size=3, stride=2, padding=1)), # Downsample 50->25
            nn.LeakyReLU(),
            init_(SeparableConv2d(128, 256, kernel_size=3, stride=2, padding=1)), # Downsample 25->13
            nn.LeakyReLU(),
            init_(SeparableConv2d(256, 256, kernel_size=3, stride=1, padding=1)),
            nn.LeakyReLU(),

            # 3. Add Residual Coordinate Attention Block at the end
            ResidualCoordAttnBlock(in_channels=256)
        )

        # --- Dynamically calculate the flattened feature size ---
        def _get_conv_output_shape(shape):
            with torch.no_grad():
                dummy_input = torch.rand(1, *shape)
                output_features = self.cnn_base(dummy_input)
                return output_features.shape[1:] # Returns (C, H, W)

        conv_out_shape = _get_conv_output_shape((num_inputs, self.width, self.length))
        head_in_features = math.prod(conv_out_shape) # C * H * W

        # --- Actor and Critic Heads ---
        self.actor_head = nn.Sequential(
            Flatten(),
            init_(nn.Linear(head_in_features, hidden_size)),
            nn.LeakyReLU()
        )

        self.critic_head_decoupled = nn.Sequential(
            Flatten(),
            init_(nn.Linear(head_in_features, hidden_size)),
            nn.LeakyReLU(),
            init_(nn.Linear(hidden_size, 1))
        )

        self.train()

    def forward(self, inputs, rnn_hxs, masks):
        x = inputs.view(-1, 6, self.width, self.length)
        features = self.cnn_base(x)
        if torch.isnan(features).any():
            print(features)
        value = self.critic_head_decoupled(features)
        actor_features = self.actor_head(features)
        return value, actor_features, rnn_hxs

#==============================================================================
# --- Policy Class Definition ---
#==============================================================================

class Policy(nn.Module):
    def __init__(self, obs_shape, action_space, base=None, base_kwargs=None):
        super(Policy, self).__init__()
        if base_kwargs is None:
            base_kwargs = {}

        base = SimpleCNNBase
        width = action_space.nvec[1]
        length = action_space.nvec[2]
        self.base = base(num_inputs=6, width=width, length=length, **base_kwargs)

        hidden_size = self.base.output_size
        self.dist_o = CategoricalWithEpsilonMask(hidden_size, 2)
        self.dist_x = CategoricalWithEpsilonMask(hidden_size + 2, width)
        self.dist_y = CategoricalWithEpsilonMask(hidden_size + width + 2, length)

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

        o_mask = torch.stack([mask_o0, mask_o1], dim=1)
        mask_o = o_mask.max(dim=-1).values.max(dim=-1).values

        dist_o, _ = self.dist_o(actor_features, mask_o)
        action_o = dist_o.mode() if deterministic else dist_o.sample()
        o_one_hot = F.one_hot(action_o.squeeze(-1), num_classes=2).float()

        batch_indices = torch.arange(o_mask.size(0), device=inputs.device)
        selected_o_mask = o_mask[batch_indices, action_o.squeeze(-1), :, :]
        
        x_mask = selected_o_mask.max(dim=-1).values

        x_input = torch.cat([actor_features, o_one_hot], dim=1)
        dist_x, _ = self.dist_x(x_input, x_mask)
        action_x = dist_x.mode() if deterministic else dist_x.sample()
        x_one_hot = F.one_hot(action_x.squeeze(-1), num_classes=self.base.width).float()

        y_mask = selected_o_mask[batch_indices, action_x.squeeze(-1), :]

        y_input = torch.cat([actor_features, o_one_hot, x_one_hot], dim=1)
        dist_y, _ = self.dist_y(y_input, y_mask)
        action_y = dist_y.mode() if deterministic else dist_y.sample()

        action_log_probs = (dist_o.log_prob(action_o.squeeze(-1)) + 
                            dist_x.log_prob(action_x.squeeze(-1)) + 
                            dist_y.log_prob(action_y.squeeze(-1))).unsqueeze(-1)
        
        action = torch.cat([action_o, action_x, action_y], dim=1)

        return value, action, action_log_probs, rnn_hs

    def evaluate_actions(self, inputs, rnn_hs, masks, action, gt_masks):
        value, actor_features, rnn_hs = self.base(inputs, rnn_hs, masks)

        action_o, action_x, action_y = action[:, 0], action[:, 1], action[:, 2]

        o_mask = gt_masks
        mask_o = o_mask.max(dim=-1).values.max(dim=-1).values

        dist_o, _ = self.dist_o(actor_features, mask_o)
        o_one_hot = F.one_hot(action_o.long(), num_classes=2).float()

        batch_indices = torch.arange(o_mask.size(0), device=inputs.device)
        selected_o_mask = o_mask[batch_indices, action_o.long()]

        x_mask = selected_o_mask.max(dim=-1).values

        x_input = torch.cat([actor_features, o_one_hot], dim=1)
        dist_x, raw_probs_x = self.dist_x(x_input, x_mask)
        x_one_hot = F.one_hot(action_x.long(), num_classes=self.base.width).float()

        y_mask = selected_o_mask[batch_indices, action_x.long()]

        y_input = torch.cat([actor_features, o_one_hot, x_one_hot], dim=1)
        dist_y, raw_probs_y = self.dist_y(y_input, y_mask)

        prob_map = raw_probs_x.unsqueeze(2) * raw_probs_y.unsqueeze(1)
        infeasibility_mask = 1.0 - selected_o_mask.float()
        infeasibility_loss = torch.mean(prob_map * infeasibility_mask)

        action_log_probs = dist_o.log_prob(action_o) + dist_x.log_prob(action_x) + dist_y.log_prob(action_y)

        dist_entropy = dist_o.entropy().mean() + dist_x.entropy().mean() + dist_y.entropy().mean()

        return value, action_log_probs.unsqueeze(-1), dist_entropy, rnn_hs, infeasibility_loss

