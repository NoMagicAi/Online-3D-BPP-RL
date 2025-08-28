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

class CategoricalWithEpsilonMask(nn.Module):
    """
    A distribution that applies a soft mask with a small epsilon value
    directly to the final probabilities, as described in the paper.
    """
    def __init__(self, num_inputs, num_outputs, epsilon=1e-20):
        super(CategoricalWithEpsilonMask, self).__init__()
        
        self.linear = init_(nn.Linear(num_inputs, num_outputs))
        self.epsilon = epsilon

    def forward(self, x, mask=None):
        logits = self.linear(x)
        # Get the initial, unmasked "raw" probability distribution
        raw_probs = F.softmax(logits, dim=-1)

        if mask is not None:
            probs_clone = raw_probs.clone()
            probs_clone[mask == 0] = self.epsilon
            renormalized_probs = probs_clone / probs_clone.sum(dim=-1, keepdim=True)
            
            final_dist = TorchCategorical(probs=renormalized_probs)
            # RETURN a tuple: the final modulated distribution AND the raw probabilities
            return final_dist, raw_probs
        
        # If no mask is provided, still return both for a consistent signature
        return TorchCategorical(logits=logits), raw_probs

class GhostModule(nn.Module):
    """
    Ghost Module for efficient feature map generation.
    This module replaces a standard convolutional layer.
    MODIFIED: Replaced BatchNorm2d with GroupNorm and ReLU with SiLU.
    """
    def __init__(self, in_channels, out_channels, kernel_size=1, ratio=2, dw_kernel_size=3, stride=1, relu=True):
        super(GhostModule, self).__init__()
        self.out_channels = out_channels
        
        init_channels = math.ceil(out_channels / ratio)
        new_channels = init_channels * (ratio - 1)

        # Using 8 groups for GroupNorm as a robust default.
        # Ensure init_channels is divisible by num_groups, or handle appropriately.
        num_groups_primary = 8 if init_channels > 0 and init_channels % 8 == 0 else (4 if init_channels > 0 and init_channels % 4 == 0 else (2 if init_channels > 0 and init_channels % 2 == 0 else 1))
        num_groups_cheap = 8 if new_channels > 0 and new_channels % 8 == 0 else (4 if new_channels > 0 and new_channels % 4 == 0 else (2 if new_channels > 0 and new_channels % 2 == 0 else 1))


        self.primary_conv = nn.Sequential(
            nn.Conv2d(in_channels, init_channels, kernel_size, stride, padding=kernel_size//2, bias=False),
            nn.GroupNorm(num_groups=num_groups_primary, num_channels=init_channels),
            nn.SiLU(inplace=True) if relu else nn.Sequential(),
        )

        self.cheap_operation = nn.Sequential(
            nn.Conv2d(init_channels, new_channels, dw_kernel_size, 1, padding=dw_kernel_size//2, groups=init_channels, bias=False),
            nn.GroupNorm(num_groups=num_groups_cheap, num_channels=new_channels) if new_channels > 0 else nn.Sequential(),
            nn.SiLU(inplace=True) if relu and new_channels > 0 else nn.Sequential(),
        )

    def forward(self, x):
        x1 = self.primary_conv(x)
        if self.cheap_operation and list(self.cheap_operation.children()): # Check if cheap_operation has layers
            x2 = self.cheap_operation(x1)
            out = torch.cat([x1, x2], dim=1)
        else:
            out = x1
        return out[:, :self.out_channels, :, :]

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

class AddCoords(nn.Module):
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
            rr = torch.sqrt(torch.pow(xx_channel - 0.0, 2) + torch.pow(yy_channel - 0.0, 2))
            ret = torch.cat([ret, rr], dim=1)
        return ret

class CoordConv(nn.Module):
    def __init__(self, in_channels, out_channels, with_r=False, **kwargs):
        super().__init__()
        self.addcoords = AddCoords(with_r=with_r)
        coord_channels = 3 if with_r else 2
        self.conv = init_(SeparableConv2d(in_channels + coord_channels, out_channels, **kwargs))

    def forward(self, x):
        x = self.addcoords(x)
        x = self.conv(x)
        return x
        
class UpgradedGhostEESP(nn.Module):
    """
    An EESP block using GhostModules, GroupNorm, and SiLU.
    """
    def __init__(self, in_channels, out_channels, stride=1, k=4, dilation_rates=[1, 2, 4, 8]):
        super(UpgradedGhostEESP, self).__init__()
        assert len(dilation_rates) == k, "Number of branches should match k"
        self.k = k
        self.split_channels = [out_channels // k] * k
        self.split_channels[0] += out_channels - sum(self.split_channels)

        self.proj = GhostModule(in_channels, out_channels, relu=False)
        
        self.branches = nn.ModuleList()
        for i in range(k):
            d = dilation_rates[i]
            self.branches.append(
                nn.Conv2d(
                    self.split_channels[i], self.split_channels[i],
                    kernel_size=3, stride=stride, padding=d, dilation=d,
                    groups=self.split_channels[i], bias=False
                )
            )

        self.pointwise = GhostModule(out_channels, out_channels, relu=False)
        self.norm = nn.GroupNorm(num_groups=8 if out_channels % 8 == 0 else 4, num_channels=out_channels)
        self.act = nn.SiLU(inplace=False)

    def forward(self, x):
        x = self.proj(x)
        splits = torch.split(x, self.split_channels, dim=1)
        outputs = []
        for idx, branch in enumerate(self.branches):
            out = branch(splits[idx])
            if idx > 0:
                out = out + outputs[idx - 1]
            outputs.append(out)

        x = torch.cat(outputs, dim=1)
        x = self.pointwise(x)
        x = self.norm(x)
        return self.act(x)
        
class CoordAttn(nn.Module):
    """
    Coordinate Attention Block.
    MODIFIED: Using GroupNorm and SiLU.
    """
    def __init__(self, inp, oup, reduction=32):
        super(CoordAttn, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        mip = max(8, inp // reduction)
        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.norm1 = nn.GroupNorm(num_groups=1, num_channels=mip) # LayerNorm equivalent for channels
        self.act = nn.SiLU()
        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        identity = x
        n, c, h, w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)
        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.norm1(y)
        y = self.act(y)
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)
        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()
        return identity * a_w * a_h

class ResidualCoordAttnBlock(nn.Module):
    """ A residual block with Coordinate Attention. """
    def __init__(self, in_channels):
        super().__init__()
        self.attn = CoordAttn(in_channels, in_channels)
        self.conv = init_(SeparableConv2d(in_channels, in_channels, kernel_size=1, stride=1))

    def forward(self, x):
        res = self.attn(x)
        res = self.conv(res)
        return x + res

#==============================================================================
# --- Main Upgraded Network ---
#==============================================================================

class FinalFusionNet_v2(NNBase):
    """
    An "attention-heavy" 4-layer architecture. This model prioritizes
    computational intelligence over brute-force depth by interleaving
    attention blocks throughout the encoder.
    """
    def __init__(self, num_inputs, recurrent=False, hidden_size=512, width=100, length=100):
        super(FinalFusionNet_v2, self).__init__(recurrent, num_inputs, hidden_size)

        self.width = width
        self.length = length

        # --- Encoder Path with Interleaved Attention ---
        self.level1_down = nn.Sequential(
            CoordConv(num_inputs, 32, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=8, num_channels=32),
            nn.SiLU(),
            UpgradedGhostEESP(32, 32, stride=2)
        )
        # ADDED: Attention block for level 1
        #self.attn1 = ResidualCoordAttnBlock(32)

        self.level2_down = UpgradedGhostEESP(32, 64, stride=2)
        # ADDED: Attention block for level 2
        #self.attn2 = ResidualCoordAttnBlock(64)

        self.level3_down = UpgradedGhostEESP(64, 128, stride=2)
        # ADDED: Attention block for level 3
        self.attn3 = ResidualCoordAttnBlock(128)

        self.level4_down = UpgradedGhostEESP(128, 256, stride=2)
        # ADDED: Attention block for level 4
        self.attn4 = ResidualCoordAttnBlock(256)

        # REMOVED: level5_down has been removed.

        # CHANGED: Bottleneck now operates on the 256-channel output from the final attention block.
        self.bottleneck = UpgradedGhostEESP(256, 256, stride=1)

        # --- Heads (Reverted to 256-channel input) ---
        POOL_OUTPUT_SIZE = 4
        # CHANGED: Reverted to 256 channels for the linear layer input size.
        critic_linear_in_features = 256 * POOL_OUTPUT_SIZE * POOL_OUTPUT_SIZE
        actor_conv_out_channels = 128 
        actor_linear_in_features = actor_conv_out_channels * POOL_OUTPUT_SIZE * POOL_OUTPUT_SIZE
        
        self.orientation_head = nn.Sequential(
            # CHANGED: Reverted input channels to 256.
            init_(SeparableConv2d(256, actor_conv_out_channels, kernel_size=1)),
            nn.GroupNorm(num_groups=8, num_channels=actor_conv_out_channels),
            nn.SiLU(inplace=False),
            nn.AdaptiveAvgPool2d((POOL_OUTPUT_SIZE, POOL_OUTPUT_SIZE)),
            Flatten(),
            init_(nn.Linear(actor_linear_in_features, hidden_size)),
            nn.LayerNorm(hidden_size),
            nn.SiLU(inplace=False)
        )
        
        self.critic_head_decoupled = nn.Sequential(
            nn.AdaptiveAvgPool2d((POOL_OUTPUT_SIZE, POOL_OUTPUT_SIZE)),
            Flatten(),
            # CHANGED: Reverted input features to the linear layer.
            init_(nn.Linear(critic_linear_in_features, hidden_size)),
            nn.SiLU(),
            init_(nn.Linear(hidden_size, 1))
        )

        self.train()

    def forward(self, inputs, rnn_hxs, masks):
        x = inputs.view(-1, 6, self.width, self.length)

        # --- Encoder with Interleaved Attention---
        l1_out = self.level1_down(x)
        #l1_attended = self.attn1(l1_out)

        l2_out = self.level2_down(l1_out)
        #l2_attended = self.attn2(l2_out)

        l3_out = self.level3_down(l2_out)
        l3_attended = self.attn3(l3_out)

        l4_out = self.level4_down(l3_attended)
        l4_attended = self.attn4(l4_out)

        bottleneck_out = self.bottleneck(l4_attended)
        
        # --- Heads ---
        value = self.critic_head_decoupled(bottleneck_out)
        orientation_features = self.orientation_head(bottleneck_out)
        
        return value, orientation_features, rnn_hxs

class Policy(nn.Module):
    def __init__(self, obs_shape, action_space, base=None, base_kwargs=None):
        super(Policy, self).__init__()
        if base_kwargs is None:
            base_kwargs = {}

        # Use the NEW Upgraded FinalFusionNet_v2 as the base
        base = FinalFusionNet_v2

        width = action_space.nvec[1]
        length = action_space.nvec[2]

        self.base = base(num_inputs=6, width=width, length=length, **base_kwargs)

        hidden_size = self.base.output_size

        self.dist_o = CategoricalWithEpsilonMask(hidden_size, 2)
        self.dist_x = CategoricalWithEpsilonMask(hidden_size + 2, width)
        self.dist_y = CategoricalWithEpsilonMask(hidden_size + 2 + width, length)

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
            value, orientation_features, rnn_hs = self.base(inputs, rnn_hs, masks)
            
            obs_image = inputs.view(-1, 6, self.base.width, self.base.length)
            mask_o0, mask_o1 = obs_image[:, 4, :, :], obs_image[:, 5, :, :]
            o_mask_0_valid = mask_o0.any(dim=(-1,-2)).float()
            o_mask_1_valid = mask_o1.any(dim=(-1,-2)).float()
            o_mask = torch.stack([o_mask_0_valid, o_mask_1_valid], dim=1)

            dist_o, _ = self.dist_o(orientation_features, o_mask)
            action_o = dist_o.mode() if deterministic else dist_o.sample()
            
            o_one_hot = F.one_hot(action_o.squeeze(-1), num_classes=2).float()
            x_input = torch.cat([orientation_features, o_one_hot], dim=1)
            condition = (action_o == 0).view(-1, 1, 1)
            mask_for_o = torch.where(condition, mask_o0, mask_o1)
            x_mask = mask_for_o.any(dim=-1).float()

            dist_x, _ = self.dist_x(x_input, x_mask)
            action_x = dist_x.mode() if deterministic else dist_x.sample()
            
            x_one_hot = F.one_hot(action_x.squeeze(-1), num_classes=self.base.width).float()
            y_input = torch.cat([orientation_features, o_one_hot, x_one_hot], dim=1)
            index = action_x.unsqueeze(2).expand(-1, -1, self.base.length)
            y_mask = torch.gather(mask_for_o, 1, index).squeeze(1).float()

            dist_y, _ = self.dist_y(y_input, y_mask)
            action_y = dist_y.mode() if deterministic else dist_y.sample()
            
            log_prob_o = dist_o.log_prob(action_o.squeeze(-1))
            log_prob_x = dist_x.log_prob(action_x.squeeze(-1))
            log_prob_y = dist_y.log_prob(action_y.squeeze(-1))
            
            action_log_probs = (log_prob_o + log_prob_x + log_prob_y).unsqueeze(-1)
            action = torch.cat([action_o, action_x, action_y], dim=1)
            
            return value, action, action_log_probs, rnn_hs

    def evaluate_actions(self, inputs, rnn_hs, masks, action, gt_masks):
            value, orientation_features, rnn_hs = self.base(inputs, rnn_hs, masks)
            
            action_o, action_x, action_y = action[:, 0], action[:, 1], action[:, 2]
            
            mask_o0 = gt_masks[:, 0]
            mask_o1 = gt_masks[:, 1]
            o_mask_0_valid = mask_o0.any(dim=(-1, -2)).float()
            o_mask_1_valid = mask_o1.any(dim=(-1, -2)).float()
            o_mask = torch.stack([o_mask_0_valid, o_mask_1_valid], dim=1)
            
            dist_o, _ = self.dist_o(orientation_features, o_mask)
            
            o_one_hot = F.one_hot(action_o.long(), num_classes=2).float()
            x_input = torch.cat([orientation_features, o_one_hot], dim=1)
            condition = (action_o == 0).view(-1, 1, 1)
            mask_for_o = torch.where(condition, mask_o0, mask_o1)
            x_mask = mask_for_o.any(dim=-1).float()
            
            dist_x, raw_probs_x = self.dist_x(x_input, x_mask)
            
            x_one_hot = F.one_hot(action_x.long(), num_classes=self.base.width).float()
            y_input = torch.cat([orientation_features, o_one_hot, x_one_hot], dim=1)
            batch_indices = torch.arange(mask_for_o.size(0), device=action_x.device)
            y_mask = mask_for_o[batch_indices, action_x.long()].float()
            
            dist_y, raw_probs_y = self.dist_y(y_input, y_mask)
            
            prob_map = raw_probs_x.unsqueeze(2) * raw_probs_y.unsqueeze(1)
            infeasibility_mask = 1.0 - mask_for_o.float()
            infeasibility_loss = torch.mean(prob_map * infeasibility_mask)
            
            action_log_probs = dist_o.log_prob(action_o) + dist_x.log_prob(action_x) + dist_y.log_prob(action_y)
            dist_entropy = dist_o.entropy().mean() + dist_x.entropy().mean() + dist_y.entropy().mean()
            
            return value, action_log_probs, dist_entropy, rnn_hs, infeasibility_loss