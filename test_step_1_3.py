# test_final.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import gym.spaces
from torch.distributions.categorical import Categorical as TorchCategorical

# ==============================================================================
# --- COMPLETE, FULLY MODIFIED MODEL DEFINITIONS ---
# ==============================================================================

# --- Helper Modules (with all changes already applied) ---

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class CategoricalWithEpsilonMask(nn.Module):
    def __init__(self, num_inputs, num_outputs, epsilon=1e-20):
        super(CategoricalWithEpsilonMask, self).__init__()
        self.linear = nn.Linear(num_inputs, num_outputs)
        self.epsilon = epsilon
    def forward(self, x, mask=None):
        logits = self.linear(x)
        raw_probs = F.softmax(logits, dim=-1)
        if mask is not None:
            probs_clone = raw_probs.clone()
            probs_clone[mask == 0] = self.epsilon
            renormalized_probs = probs_clone / probs_clone.sum(dim=-1, keepdim=True)
            final_dist = TorchCategorical(probs=renormalized_probs)
            return final_dist, raw_probs
        return TorchCategorical(logits=logits), raw_probs

class NNBase(nn.Module):
    def __init__(self, recurrent, recurrent_input_size, hidden_size):
        super(NNBase, self).__init__()
        self._hidden_size = hidden_size
        self._recurrent = recurrent
    @property
    def is_recurrent(self): return self._recurrent
    @property
    def recurrent_hidden_state_size(self):
        if self._recurrent: return self._hidden_size
        return 1
    @property
    def output_size(self): return self._hidden_size

class SeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, bias=False):
        super(SeparableConv2d, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=in_channels, bias=bias)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)
    def forward(self, x):
        return self.pointwise(self.depthwise(x))

def init_(m):
    if isinstance(m, (nn.Conv2d, nn.Linear, nn.ConvTranspose2d)):
        nn.init.orthogonal_(m.weight)
        if m.bias is not None: nn.init.constant_(m.bias, 0)
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
            rr = torch.sqrt(torch.pow(xx_channel - 0.5, 2) + torch.pow(yy_channel - 0.5, 2))
            ret = torch.cat([ret, rr], dim=1)
        return ret

class CoordConv(nn.Module):
    def __init__(self, in_channels, out_channels, with_r=False, **kwargs):
        super().__init__()
        self.addcoords = AddCoords(with_r=with_r)
        coord_channels = 3 if with_r else 2
        self.conv = init_(SeparableConv2d(in_channels + coord_channels, out_channels, **kwargs))
    def forward(self, x):
        return self.conv(self.addcoords(x))

class UpgradedGhostEESP(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, k=4, dilation_rates=[1, 2, 4, 8]):
        super(UpgradedGhostEESP, self).__init__()
        assert len(dilation_rates) == k, "Number of branches should match k"
        self.k = k
        self.split_channels = [out_channels // k] * k
        self.split_channels[0] += out_channels - sum(self.split_channels)
        self.proj = init_(nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False))
        self.branches = nn.ModuleList([
            nn.Conv2d(self.split_channels[i], self.split_channels[i], kernel_size=3, stride=stride, padding=d, dilation=d, groups=self.split_channels[i], bias=False)
            for i, d in enumerate(dilation_rates)
        ])
        self.pointwise = init_(nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False))
        self.norm = nn.GroupNorm(num_groups=8 if out_channels % 8 == 0 else 4, num_channels=out_channels)
        self.act = nn.SiLU(inplace=False)
    def forward(self, x):
        x = self.proj(x)
        splits = torch.split(x, self.split_channels, dim=1)
        outputs = [branch(s) for branch, s in zip(self.branches, splits)]
        for i in range(1, len(outputs)): outputs[i] += outputs[i-1]
        x = torch.cat(outputs, dim=1)
        x = self.pointwise(x)
        return self.act(self.norm(x))

class CoordAttn(nn.Module):
    def __init__(self, inp, oup, reduction=32):
        super(CoordAttn, self).__init__()
        self.pool_h, self.pool_w = nn.AdaptiveAvgPool2d((None, 1)), nn.AdaptiveAvgPool2d((1, None))
        mip = max(8, inp // reduction)
        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.norm1 = nn.GroupNorm(num_groups=1, num_channels=mip)
        self.act = nn.SiLU()
        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
    def forward(self, x):
        identity = x
        n, c, h, w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)
        y = torch.cat([x_h, x_w], dim=2)
        y = self.act(self.norm1(self.conv1(y)))
        x_h, x_w = torch.split(y, [h, w], dim=2)
        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w.permute(0, 1, 3, 2)).sigmoid()
        return identity * a_w * a_h

class ResidualCoordAttnBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.attn = CoordAttn(in_channels, in_channels)
        self.conv = init_(SeparableConv2d(in_channels, in_channels, kernel_size=1, stride=1))
    def forward(self, x):
        return x + self.conv(self.attn(x))

class AttentionGate(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(AttentionGate, self).__init__()
        self.W_g = nn.Sequential(nn.Conv2d(F_g, F_int, 1, 1, 0, bias=True), nn.GroupNorm(8, F_int))
        self.W_x = nn.Sequential(nn.Conv2d(F_l, F_int, 1, 2, 0, bias=True), nn.GroupNorm(8, F_int))
        self.psi = nn.Sequential(nn.Conv2d(F_int, 1, 1, 1, 0, bias=True), nn.GroupNorm(1, 1), nn.Sigmoid())
        self.act = nn.SiLU()
    def forward(self, g, x):
        g1, x1 = self.W_g(g), self.W_x(x)
        if g1.shape[2:] != x1.shape[2:]: x1 = F.interpolate(x1, g1.shape[2:], mode='bilinear', align_corners=False)
        psi = self.psi(self.act(g1 + x1))
        gated_x = F.interpolate(x, psi.shape[2:], mode='bilinear', align_corners=False) * psi
        return F.interpolate(gated_x, x.shape[2:], mode='bilinear', align_corners=False)

class PixelShuffleUpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor=2):
        super().__init__()
        self.conv = init_(nn.Conv2d(in_channels, out_channels * (scale_factor ** 2), 1, 1))
        self.shuffle = nn.PixelShuffle(scale_factor)
    def forward(self, x): return self.shuffle(self.conv(x))

class UpgradedGhostUpEESP(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.up = PixelShuffleUpBlock(in_channels, in_channels // 2)
        self.attn = AttentionGate(F_g=in_channels // 2, F_l=skip_channels, F_int=skip_channels // 2)
        self.conv = UpgradedGhostEESP(in_channels // 2 + skip_channels, out_channels, stride=1)
    def forward(self, x1, x2):
        x1_up = self.up(x1)
        x2_att = self.attn(g=x1_up, x=x2)
        diffY, diffX = x2_att.size(2) - x1_up.size(2), x2_att.size(3) - x1_up.size(3)
        x1_up = F.pad(x1_up, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2_att, x1_up], dim=1)
        return self.conv(x)

# --- Main Network ---
class FinalFusionNet_v2(NNBase):
    def __init__(self, num_inputs, recurrent=False, hidden_size=512, width=100, length=100):
        super(FinalFusionNet_v2, self).__init__(recurrent, num_inputs, hidden_size)
        self.width, self.length = width, length
        self.level1_down = nn.Sequential(CoordConv(num_inputs, 32, kernel_size=3, padding=1), nn.GroupNorm(8, 32), nn.SiLU(), UpgradedGhostEESP(32, 32, stride=2))
        self.level2_down = UpgradedGhostEESP(32, 64, stride=2)
        self.level3_down = UpgradedGhostEESP(64, 128, stride=2)
        self.bottleneck = UpgradedGhostEESP(128, 128, stride=1)
        self.attn = ResidualCoordAttnBlock(128)
        self.level3_up = UpgradedGhostUpEESP(128, 64, 64)
        self.level2_up = UpgradedGhostUpEESP(64, 32, 32)
        self.final_upsample = PixelShuffleUpBlock(32, 32)
        self.output_conv = init_(nn.Conv2d(32, 32, 1, 1))
        POOL_OUTPUT_SIZE = 4
        orientation_linear_in_features = 128 * POOL_OUTPUT_SIZE * POOL_OUTPUT_SIZE
        self.orientation_head = nn.Sequential(init_(SeparableConv2d(128, 128, 1)), nn.GroupNorm(8, 128), nn.SiLU(inplace=False), nn.AdaptiveAvgPool2d((POOL_OUTPUT_SIZE, POOL_OUTPUT_SIZE)), Flatten(), init_(nn.Linear(orientation_linear_in_features, hidden_size)), nn.LayerNorm(hidden_size), nn.SiLU(inplace=False))
        self.critic_head_decoupled = nn.Sequential(nn.AdaptiveAvgPool2d((POOL_OUTPUT_SIZE, POOL_OUTPUT_SIZE)), Flatten(), init_(nn.Linear(orientation_linear_in_features, hidden_size)), nn.SiLU(), init_(nn.Linear(hidden_size, 1)))
        self.train()

    def forward(self, inputs, rnn_hxs, masks):
        x = inputs.view(-1, 6, self.width, self.length)
        skip1_out = self.level1_down(x)
        skip2_out = self.level2_down(skip1_out)
        encoded = self.level3_down(skip2_out)
        bottleneck_out = self.attn(self.bottleneck(encoded))
        value = self.critic_head_decoupled(bottleneck_out)
        orientation_features = self.orientation_head(bottleneck_out)
        up3_out = self.level3_up(bottleneck_out, skip2_out)
        up2_out = self.level2_up(up3_out, skip1_out)
        shared_features = self.output_conv(self.final_upsample(up2_out))
        return value, orientation_features, shared_features, rnn_hxs

# --- Policy (with Final Correction) ---
class Policy(nn.Module):
    def __init__(self, obs_shape, action_space, base=None, base_kwargs=None):
        super(Policy, self).__init__()
        if base_kwargs is None: base_kwargs = {}
        base = FinalFusionNet_v2
        width, length = action_space.nvec[1], action_space.nvec[2]
        self.base = base(num_inputs=6, width=width, length=length, **base_kwargs)
        hidden_size = self.base.output_size
        self.dist_o = CategoricalWithEpsilonMask(hidden_size, 2)
        self.dist_x_conv = init_(nn.Conv2d(32 + 2, 1, kernel_size=1))
        
        # ✨✨✨ THIS IS THE FIX ✨✨✨
        # The number of channels from x_one_hot is `width`, not 1.
        self.dist_y_conv = init_(nn.Conv2d(32 + 2 + width, 1, kernel_size=1))

    def get_value(self, inputs, rnn_hxs, masks):
        value, _, _, _ = self.base(inputs, rnn_hxs, masks)
        return value

    def act(self, inputs, rnn_hs, masks, deterministic=False):
        value, orientation_features, shared_features, rnn_hs = self.base(inputs, rnn_hs, masks)
        obs_image = inputs.view(-1, 6, self.base.width, self.base.length)
        mask_o0, mask_o1 = obs_image[:, 4, :, :], obs_image[:, 5, :, :]
        o_mask = torch.stack([mask_o0.any(dim=(-1,-2)).float(), mask_o1.any(dim=(-1,-2)).float()], dim=1)
        dist_o, _ = self.dist_o(orientation_features, o_mask)
        action_o = dist_o.mode() if deterministic else dist_o.sample()
        o_one_hot = F.one_hot(action_o.squeeze(-1), num_classes=2).float()
        o_one_hot_spatial = o_one_hot.view(-1, 2, 1, 1).expand(-1, -1, self.base.width, self.base.length)
        x_input_map = torch.cat([shared_features, o_one_hot_spatial], dim=1)
        x_logits_map = self.dist_x_conv(x_input_map).squeeze(1)
        mask_for_o = torch.where((action_o == 0).view(-1, 1, 1), mask_o0, mask_o1)
        x_mask = mask_for_o.any(dim=-1).float()
        x_logits = x_logits_map.mean(dim=-1)
        x_logits[x_mask == 0] = -1e8
        dist_x = TorchCategorical(logits=x_logits)
        action_x = dist_x.mode() if deterministic else dist_x.sample()
        x_one_hot = F.one_hot(action_x.squeeze(-1), num_classes=self.base.width).float()
        x_one_hot_spatial = x_one_hot.view(-1, self.base.width, 1, 1).expand(-1, -1, self.base.width, self.base.length)
        y_input_map = torch.cat([shared_features, o_one_hot_spatial, x_one_hot_spatial], dim=1)
        y_logits_map = self.dist_y_conv(y_input_map).squeeze(1)
        batch_indices = torch.arange(y_logits_map.size(0), device=action_x.device)
        y_logits = y_logits_map[batch_indices, action_x.long()]
        y_mask = mask_for_o[batch_indices, action_x.long()].float()
        y_logits[y_mask == 0] = -1e8
        dist_y = TorchCategorical(logits=y_logits)
        action_y = dist_y.mode() if deterministic else dist_y.sample()
        log_prob_o, log_prob_x, log_prob_y = dist_o.log_prob(action_o.squeeze(-1)), dist_x.log_prob(action_x.squeeze(-1)), dist_y.log_prob(action_y.squeeze(-1))
        action_log_probs = (log_prob_o + log_prob_x + log_prob_y).unsqueeze(-1)
        action = torch.cat([action_o.view(-1, 1), action_x.view(-1, 1), action_y.view(-1, 1)], dim=1)
        return value, action, action_log_probs, rnn_hs

    def evaluate_actions(self, inputs, rnn_hs, masks, action, gt_masks):
        value, orientation_features, shared_features, rnn_hs = self.base(inputs, rnn_hs, masks)
        action_o, action_x, action_y = action[:, 0], action[:, 1], action[:, 2]
        mask_o0, mask_o1 = gt_masks[:, 0], gt_masks[:, 1]
        o_mask = torch.stack([mask_o0.any(dim=(-1,-2)).float(), mask_o1.any(dim=(-1,-2)).float()], dim=1)
        dist_o, _ = self.dist_o(orientation_features, o_mask)
        o_one_hot = F.one_hot(action_o.long(), num_classes=2).float()
        o_one_hot_spatial = o_one_hot.view(-1, 2, 1, 1).expand(-1, -1, self.base.width, self.base.length)
        x_input_map = torch.cat([shared_features, o_one_hot_spatial], dim=1)
        x_logits_map = self.dist_x_conv(x_input_map).squeeze(1)
        mask_for_o = torch.where((action_o == 0).view(-1, 1, 1), mask_o0, mask_o1)
        x_mask = mask_for_o.any(dim=-1).float()
        x_logits = x_logits_map.mean(dim=-1)
        raw_probs_x = F.softmax(x_logits, dim=-1)
        x_logits[x_mask == 0] = -1e8
        dist_x = TorchCategorical(logits=x_logits)
        x_one_hot = F.one_hot(action_x.long(), num_classes=self.base.width).float()
        x_one_hot_spatial = x_one_hot.view(-1, self.base.width, 1, 1).expand(-1, -1, self.base.width, self.base.length)
        y_input_map = torch.cat([shared_features, o_one_hot_spatial, x_one_hot_spatial], dim=1)
        y_logits_map = self.dist_y_conv(y_input_map).squeeze(1)
        batch_indices = torch.arange(y_logits_map.size(0), device=action_x.device)
        y_logits = y_logits_map[batch_indices, action_x.long()]
        y_mask = mask_for_o[batch_indices, action_x.long()].float()
        raw_probs_y = F.softmax(y_logits, dim=-1)
        y_logits[y_mask == 0] = -1e8
        dist_y = TorchCategorical(logits=y_logits)
        prob_map = raw_probs_x.unsqueeze(2) * raw_probs_y.unsqueeze(1)
        infeasibility_loss = torch.mean(prob_map * (1.0 - mask_for_o.float()))
        action_log_probs = dist_o.log_prob(action_o) + dist_x.log_prob(action_x) + dist_y.log_prob(action_y)
        dist_entropy = dist_o.entropy().mean() + dist_x.entropy().mean() + dist_y.entropy().mean()
        return value, action_log_probs, dist_entropy, rnn_hs, infeasibility_loss

# ==============================================================================
# --- TEST SCRIPT ---
# ==============================================================================

def test_full_policy_forward_pass():
    print("--- Running Final Test: Decoupled Spatial Heads ---")
    
    batch_size, width, length = 4, 100, 100
    obs_shape = (6, width, length)
    action_space = gym.spaces.MultiDiscrete([2, width, length])
    
    try:
        policy = Policy(obs_shape, action_space, base_kwargs={})
        print("Policy instantiated successfully.")
    except Exception as e:
        print(f"Error during Policy instantiation: {e}")
        return

    dummy_inputs = torch.randn(batch_size, *obs_shape)
    dummy_inputs[:, 4, 10, 10], dummy_inputs[:, 5, 20, 20] = 1, 1
    dummy_rnn_hxs, dummy_masks = torch.randn(batch_size, 1), torch.ones(batch_size, 1)

    print("\nTesting act() method...")
    try:
        value, action, action_log_probs, rnn_hxs = policy.act(dummy_inputs, dummy_rnn_hxs, dummy_masks)
        print("act() forward pass completed.")
        assert value.shape == (batch_size, 1), f"Wrong value shape: {value.shape}"
        assert action.shape == (batch_size, 3), f"Wrong action shape: {action.shape}"
        assert action_log_probs.shape == (batch_size, 1), f"Wrong log_probs shape: {action_log_probs.shape}"
        print("act() output shapes are correct.")
    except Exception as e:
        print(f"Error during act() forward pass: {e}")
        import traceback
        traceback.print_exc()
        return
        
    print("\nTesting evaluate_actions() method...")
    dummy_action = torch.tensor([[0, 10, 10], [1, 20, 20], [0, 5, 5], [1, 15, 15]], dtype=torch.float32)
    dummy_gt_masks = torch.zeros(batch_size, 2, width, length)
    dummy_gt_masks[0, 0, 10, 10], dummy_gt_masks[1, 1, 20, 20], dummy_gt_masks[2, 0, 5, 5], dummy_gt_masks[3, 1, 15, 15] = 1, 1, 1, 1
    try:
        value, log_probs, entropy, _, infeasibility = policy.evaluate_actions(dummy_inputs, dummy_rnn_hxs, dummy_masks, dummy_action, dummy_gt_masks)
        print("evaluate_actions() forward pass completed.")
        assert value.shape == (batch_size, 1) and log_probs.shape == (batch_size,) and entropy.shape == () and infeasibility.shape == ()
        print("evaluate_actions() output shapes are correct.")
    except Exception as e:
        print(f"Error during evaluate_actions() forward pass: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print("\n✅ All Tests Passed: Full policy network is functional.\n")

if __name__ == "__main__":
    test_full_policy_forward_pass()