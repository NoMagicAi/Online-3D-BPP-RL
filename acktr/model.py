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

#==============================================================================
# --- Base Class Definition ---
#==============================================================================

class NNBase(nn.Module):
    """
    Base class for neural network modules. It defines the basic properties
    like recurrence and hidden state size.
    """
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

#==============================================================================
# --- Start of Lightweight Architecture ---
#==============================================================================

class EESP(nn.Module):
    """
    Efficient ESPNet block. This is the core building block of the new
    feature extractor. It uses group convolutions and hierarchical feature
    fusion to efficiently process features.
    """
    def __init__(self, in_channels, out_channels, stride=1, k=4, dilation_rates=[1, 2, 4, 8]):
        super(EESP, self).__init__()
        assert len(dilation_rates) == k, "Number of branches should match k"
        self.k = k
        # Calculate the number of channels for each parallel branch
        self.split_channels = [out_channels // k] * k
        self.split_channels[0] += out_channels - sum(self.split_channels)  # account for remainder

        # 1x1 projection to reduce channels before splitting
        self.proj = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.branches = nn.ModuleList()
        # Create k parallel branches with different dilation rates
        for i in range(k):
            d = dilation_rates[i]
            self.branches.append(
                nn.Conv2d(
                    self.split_channels[i], self.split_channels[i],
                    kernel_size=3, stride=stride,
                    padding=d, dilation=d,
                    groups=self.split_channels[i],      # depthwise convolution
                    bias=False
                )
            )

        # Pointwise convolution to merge features
        self.pointwise = nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        # ARCHITECTURAL IMPROVEMENT: Changed ReLU to LeakyReLU for consistency.
        self.relu = nn.LeakyReLU(inplace=False)

    def forward(self, x):
        # Initial projection
        x = self.proj(x)
        # Split tensor for parallel branches
        splits = torch.split(x, self.split_channels, dim=1)
        outputs = []
        # Process each branch
        for idx, branch in enumerate(self.branches):
            out = branch(splits[idx])
            # Hierarchical Feature Fusion (HFF)
            if idx > 0:
                out = out + outputs[idx - 1]
            outputs.append(out)
        # Concatenate and process merged features
        x = torch.cat(outputs, dim=1)
        x = self.pointwise(x)
        x = self.bn(x)
        return self.relu(x)

class SeparableConv2d(nn.Module):
    """
    Depthwise separable 2D convolution. Used in helper modules and heads.
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
        # EESP uses BatchNorm, so kaiming_normal_ is a good choice
        if not isinstance(m, (EESP)): # EESP has its own init logic implicitly
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

class UpEESP(nn.Module):
    """
    Upsampling block for the EESP-Net decoder. It uses ConvTranspose2d for
    upsampling and an EESP block to process the concatenated features from
    the skip connection and the upsampled path.
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # The ConvTranspose2d layer takes the input from the layer below (x1),
        # which has 'in_channels' channels, and it halves the channel count.
        self.up = init_(nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2))
        # The EESP block processes the concatenated tensor. The concatenated tensor
        # has 'in_channels' because it's the sum of the upsampled tensor's channels
        # (in_channels / 2) and the skip connection's channels (in_channels / 2).
        self.conv = EESP(in_channels, out_channels, stride=1)


    def forward(self, x1, x2):
        # x1 is from the lower layer, x2 is the skip connection
        x1 = self.up(x1)
        # Pad to handle potential size mismatches
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # Concatenate skip connection and upsampled features
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class BottleneckResidualBlock(nn.Module):
    """
    An optimized residual block using a bottleneck design and dilated convolutions.
    This increases the receptive field and improves efficiency.
    """
    def __init__(self, channels, bottleneck_channels, dilation=1):
        super().__init__()
        # Padding must be calculated to maintain HxW dimensions with dilation
        # For a 3x3 kernel, padding = dilation.
        padding = dilation

        self.conv_block = nn.Sequential(
            # 1x1 Conv to reduce channels (the "bottleneck")
            init_(nn.Conv2d(channels, bottleneck_channels, kernel_size=1, bias=False)),
            nn.BatchNorm2d(bottleneck_channels),
            nn.LeakyReLU(inplace=False),

            # Dilated 3x3 Separable Conv for efficient feature extraction
            init_(SeparableConv2d(bottleneck_channels, bottleneck_channels, kernel_size=3, padding=padding, dilation=dilation, bias=False)),
            nn.BatchNorm2d(bottleneck_channels),
            nn.LeakyReLU(inplace=False),

            # 1x1 Conv to restore the original channel dimension
            init_(nn.Conv2d(bottleneck_channels, channels, kernel_size=1, bias=False)),
            nn.BatchNorm2d(channels)
        )
        self.final_activation = nn.LeakyReLU(inplace=False)

    def forward(self, x):
        identity = x
        out = self.conv_block(x)
        # Add the original input (skip connection) and apply final activation
        return self.final_activation(identity + out)


class EESPNetFeatureExtractor(NNBase):
    """
    An optimized, fully-convolutional feature extractor using an encoder-decoder
    structure to produce a full-resolution feature map efficiently.
    """
    def __init__(self, num_inputs, recurrent=False, hidden_size=512, width=100, length=100):
        super(EESPNetFeatureExtractor, self).__init__(recurrent, num_inputs, hidden_size)

        self.width = width
        self.length = length

        # --- Encoder Body ---
        self.inc_conv = CoordConv(num_inputs, 64, kernel_size=3, padding=1)

        bottleneck_channels = 16
        self.body = nn.Sequential(
            BottleneckResidualBlock(64, bottleneck_channels, dilation=1),
            BottleneckResidualBlock(64, bottleneck_channels, dilation=2),
            BottleneckResidualBlock(64, bottleneck_channels, dilation=4)
        )

        # Downsampling "Encoder" path
        self.final_downsample = nn.Sequential(
            init_(SeparableConv2d(64, 64, kernel_size=3, stride=2, padding=1)),
            nn.LeakyReLU(inplace=False),
            init_(SeparableConv2d(64, 64, kernel_size=3, stride=2, padding=1)),
            nn.LeakyReLU(inplace=False),
            init_(SeparableConv2d(64, 64, kernel_size=3, stride=2, padding=1)),
            nn.LeakyReLU(inplace=False),
        )

        # --- NEW: Upsampling "Decoder" Path ---
        # This block brings the feature map back to the original resolution.
        self.upsample_head = nn.Sequential(
            # Upsample 1
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            init_(SeparableConv2d(64, 64, kernel_size=3, padding=1)),
            nn.LeakyReLU(inplace=False),
            # Upsample 2
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            init_(SeparableConv2d(64, 64, kernel_size=3, padding=1)),
            nn.LeakyReLU(inplace=False),
            # Upsample 3
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            init_(SeparableConv2d(64, 32, kernel_size=3, padding=1)), # Reduce final channels
            nn.LeakyReLU(inplace=False),
        )

        # --- MODIFIED: Actor and Critic Heads ---
        # The heads now use AdaptiveAvgPool2d to handle full-resolution input efficiently.
        # This makes them independent of the specific width/length of the input images.
        POOL_OUTPUT_SIZE = 4 # A tunable hyperparameter for the pooling layer

        # Renamed from actor_head to orientation_head for clarity
        orientation_linear_in_features = 128 * POOL_OUTPUT_SIZE * POOL_OUTPUT_SIZE
        self.orientation_head = nn.Sequential(
            init_(SeparableConv2d(32, 128, kernel_size=1)), # Takes 32 channels from upsampler
            nn.LeakyReLU(inplace=False),
            nn.AdaptiveAvgPool2d((POOL_OUTPUT_SIZE, POOL_OUTPUT_SIZE)),
            Flatten(),
            init_(nn.Linear(orientation_linear_in_features, hidden_size)),
            nn.LayerNorm(hidden_size),
            nn.LeakyReLU(inplace=False)
        )

        # **MODIFIED CRITIC HEAD**
        # The critic head is now a spatial value estimator. It uses two
        # convolutional layers to produce a value map corresponding to each
        # possible action (orientation, x, y).
        num_orientations = 2  # As defined by the action space (o, x, y)
        self.critic_head = nn.Sequential(
            # First 2D convolution layer
            init_(SeparableConv2d(32, 64, kernel_size=3, padding=1, bias=False)),
            nn.LeakyReLU(inplace=False),
            # Second 2D convolution layer outputs a map per orientation
            # MODIFICATION: Set bias=True for PopArt compatibility
            init_(SeparableConv2d(64, num_orientations, kernel_size=3, padding=1, bias=True))
        )

        # Temperature for the softmax, allowing smooth interpolation between mean and max.
        self.softmax_temp = 1.0

        # The critic_linear layer is removed as the value is computed directly
        # from the spatial value map in the forward pass.

        self.train()

    def forward(self, inputs, rnn_hxs, masks):
        x = inputs.view(-1, 6, self.width, self.length)

        # --- Full Encoder-Decoder Pass ---
        x = self.inc_conv(x)
        x = self.body(x)
        downsampled_features = self.final_downsample(x)
        # The output `shared_features` is now full-resolution
        shared_features = self.upsample_head(downsampled_features)

        # --- Heads operate on the full-resolution map ---
        orientation_features = self.orientation_head(shared_features)

        # **NEW CRITIC LOGIC**

        # **FIX:** Ensure spatial dimensions of features match the input/mask dimensions.
        # This handles cases where the encoder-decoder architecture doesn't
        # perfectly restore the original size, which caused the IndexError.
        if shared_features.shape[-2:] != (self.width, self.length):
            shared_features_for_critic = F.interpolate(
                shared_features,
                size=(self.width, self.length),
                mode='bilinear',
                align_corners=False
            )
        else:
            shared_features_for_critic = shared_features

        # 1. Estimate a value map using the correctly-sized features.
        #    value_map shape: (batch, num_orientations, self.width, self.length)
        value_map = self.critic_head(shared_features_for_critic)



        # 2. Extract action masks from the input observation.
        #    mask_o0, mask_o1 shape: (batch, self.width, self.length)
        mask_o0, mask_o1 = x[:, 4, :, :], x[:, 5, :, :]
        #    action_mask shape: (batch, num_orientations, self.width, self.length)
        action_mask = torch.stack([mask_o0, mask_o1], dim=1).bool()

        # 3. Flatten map and mask for softmax. The shapes will now match.
        batch_size = value_map.size(0)
        flattened_values = value_map.view(batch_size, -1)
        flattened_mask = action_mask.view(batch_size, -1)

        # 4. Apply mask by setting values of invalid positions to -inf.
        masked_values = flattened_values.clone()
        masked_values[~flattened_mask] = -float('inf')

        # 5. Compute softmax weights over all valid positions.
        weights = F.softmax(masked_values / self.softmax_temp, dim=1)

        # 6. Calculate the final value as the weighted average.
        value = torch.sum(weights * flattened_values, dim=1).unsqueeze(-1)

        # The output signature remains the same to preserve the API.
        return value, orientation_features, shared_features, rnn_hxs

#==============================================================================
# --- End of Lightweight Architecture ---
#==============================================================================


class Policy(nn.Module):
    def __init__(self, obs_shape, action_space, base=None, base_kwargs=None):
        super(Policy, self).__init__()
        if base_kwargs is None:
            base_kwargs = {}

        # Use the EESPNetFeatureExtractor as the base
        base = EESPNetFeatureExtractor

        width = action_space.nvec[1]
        length = action_space.nvec[2]

        self.base = base(num_inputs=6, width=width, length=length, **base_kwargs)

        hidden_size = self.base.output_size

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
        value, _, _, _ = self.base(inputs, rnn_hxs, masks)
        return value

    def act(self, inputs, rnn_hs, masks, deterministic=False):
        value, orientation_features, shared_features, rnn_hs = self.base(inputs, rnn_hs, masks)

        obs_image = inputs.view(-1, 6, self.base.width, self.base.length)
        mask_o0, mask_o1 = obs_image[:, 4, :, :], obs_image[:, 5, :, :]

        o_mask_0_valid = mask_o0.any(dim=(-1,-2)).float()
        o_mask_1_valid = mask_o1.any(dim=(-1,-2)).float()
        o_mask = torch.stack([o_mask_0_valid, o_mask_1_valid], dim=1)

        dist_o, _, _ = self.dist_o(orientation_features, o_mask)
        action_o = dist_o.mode() if deterministic else dist_o.sample()

        o_one_hot = F.one_hot(action_o.squeeze(-1), num_classes=2).float()
        x_input = torch.cat([orientation_features, o_one_hot], dim=1)

        condition = (action_o == 0).view(-1, 1, 1)
        mask_for_o = torch.where(condition, mask_o0, mask_o1)

        x_mask = mask_for_o.any(dim=-1).float()
        dist_x, _, _ = self.dist_x(x_input, x_mask)
        action_x = dist_x.mode() if deterministic else dist_x.sample()

        x_one_hot = F.one_hot(action_x.squeeze(-1), num_classes=self.base.width).float()
        y_input = torch.cat([orientation_features, o_one_hot, x_one_hot], dim=1)

        index = action_x.unsqueeze(2).expand(-1, -1, self.base.length)
        y_mask = torch.gather(mask_for_o, 1, index).squeeze(1).float()

        dist_y, _, _ = self.dist_y(y_input, y_mask)
        action_y = dist_y.mode() if deterministic else dist_y.sample()

        log_prob_o = dist_o.log_prob(action_o.squeeze(-1))
        log_prob_x = dist_x.log_prob(action_x.squeeze(-1))
        log_prob_y = dist_y.log_prob(action_y.squeeze(-1))
        action_log_probs = (log_prob_o + log_prob_x + log_prob_y).unsqueeze(-1)

        action = torch.cat([action_o, action_x, action_y], dim=1)
        return value, action, action_log_probs, rnn_hs

    def evaluate_actions(self, inputs, rnn_hs, masks, action, gt_masks):
        value, orientation_features, shared_features, rnn_hs = self.base(inputs, rnn_hs, masks)
        action_o, action_x, action_y = action[:, 0], action[:, 1], action[:, 2]

        mask_o0 = gt_masks[:, 0]
        mask_o1 = gt_masks[:, 1]
        o_mask_0_valid = mask_o0.any(dim=(-1, -2)).float()
        o_mask_1_valid = mask_o1.any(dim=(-1, -2)).float()
        o_mask = torch.stack([o_mask_0_valid, o_mask_1_valid], dim=1)
        dist_o, _, _ = self.dist_o(orientation_features, o_mask)

        o_one_hot = F.one_hot(action_o, num_classes=2).float()
        x_input = torch.cat([orientation_features, o_one_hot], dim=1)

        condition = (action_o == 0).view(-1, 1, 1)
        mask_for_o = torch.where(condition, mask_o0, mask_o1)

        x_mask = mask_for_o.any(dim=-1).float()
        dist_x, _, _ = self.dist_x(x_input, x_mask)

        x_one_hot = F.one_hot(action_x, num_classes=self.base.width).float()
        y_input = torch.cat([orientation_features, o_one_hot, x_one_hot], dim=1)

        batch_indices = torch.arange(mask_for_o.size(0), device=action_x.device)
        y_mask = mask_for_o[batch_indices, action_x].float()

        dist_y, _, _ = self.dist_y(y_input, y_mask)

        probs_x, probs_y = dist_x.probs, dist_y.probs
        prob_map = probs_x.unsqueeze(2) * probs_y.unsqueeze(1)
        infeasibility_mask = 1.0 - mask_for_o.float()
        infeasibility_loss = torch.mean(prob_map * infeasibility_mask)

        action_log_probs = dist_o.log_prob(action_o) + dist_x.log_prob(action_x) + dist_y.log_prob(action_y)
        dist_entropy = dist_o.entropy().mean() + dist_x.entropy().mean() + dist_y.entropy().mean()

        return value, action_log_probs, dist_entropy, rnn_hs, infeasibility_loss