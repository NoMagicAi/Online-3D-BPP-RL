import time
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

# ==============================================================================
# KFAC Optimizer (Final, Robust Version)
# ==============================================================================

class AddBias(nn.Module):
    def __init__(self, bias):
        super(AddBias, self).__init__()
        self._bias = nn.Parameter(bias.unsqueeze(1))
    def forward(self, x):
        if x.dim() == 2: bias = self._bias.t().view(1, -1)
        else: bias = self._bias.t().view(1, -1, 1, 1)
        return x + bias

class SplitBias(nn.Module):
    def __init__(self, module):
        super(SplitBias, self).__init__()
        self.module = module
        self.add_bias = AddBias(module.bias.data)
        self.module.bias = None
    def forward(self, input):
        x = self.module(input)
        return self.add_bias(x)

@torch.compile(mode="reduce-overhead")
def _extract_patches(x, kernel_size, stride, padding):
    if padding[0] + padding[1] > 0:
        x = F.pad(x, (padding[1], padding[1], padding[0], padding[0])).data
    x = x.unfold(2, kernel_size[0], stride[0])
    x = x.unfold(3, kernel_size[1], stride[1])
    x = x.transpose_(1, 2).transpose_(2, 3).contiguous()
    return x.view(x.size(0), x.size(1), x.size(2), -1)

@torch.compile(mode="reduce-overhead")
def compute_cov_a(a, classname, layer_info, fast_cnn):
    batch_size = a.size(0)
    if classname == 'Conv2d':
        kernel_size, stride, padding, groups = layer_info
        a = _extract_patches(a, kernel_size, stride, padding)
        a = a.view(batch_size, -1, a.size(-1)).mean(1)
        if groups > 1:
            a = a.view(batch_size, groups, -1).transpose(0, 1)
            cov_a = torch.bmm(a.transpose(1, 2), a / batch_size)
        else:
            cov_a = a.t() @ (a / batch_size)
    elif classname == 'AddBias':
        a = torch.ones(a.size(0), 1, device=a.device)
        cov_a = a.t() @ (a / batch_size)
    else: # Linear
        cov_a = a.t() @ (a / batch_size)
    return cov_a

@torch.compile(mode="reduce-overhead")
def compute_cov_g(g, classname, layer_info, fast_cnn):
    batch_size = g.size(0)
    if classname == 'Conv2d':
        _, _, _, groups = layer_info
        spatial_size = g.size(2) * g.size(3)
        if fast_cnn:
             g = g.sum(dim=(2, 3))
        else:
             g = g.transpose(1, 2).transpose(2, 3).contiguous()
             g = g.view(-1, g.size(-1)) * spatial_size
        if groups > 1:
            g = g.view(-1, groups, g.size(-1) // groups).transpose(0, 1)
            g_ = g * batch_size
            cov_g = torch.bmm(g_.transpose(1, 2), g_ / g.size(1))
        else:
            g_ = g * batch_size
            cov_g = g_.t() @ (g_ / g.size(0))
    elif classname == 'AddBias':
        g = g.view(g.size(0), g.size(1), -1).sum(-1)
        g_ = g * batch_size
        cov_g = g_.t() @ (g_ / g.size(0))
    else: # Linear
        g_ = g * batch_size
        cov_g = g_.t() @ (g_ / g.size(0))
    return cov_g

def update_running_stat(tensor, running_stat, momentum):
    running_stat.mul_(momentum).add_(tensor, alpha=1 - momentum)

class KFACOptimizer(optim.Optimizer):
    def __init__(self, model, lr=0.25, momentum=0.9, stat_decay=0.99, kl_clip=0.001, damping=1e-2, weight_decay=0, fast_cnn=False, Ts=1, Tf=10):
        defaults = dict()
        def split_bias(module):
            for mname, child in module.named_children():
                if hasattr(child, 'bias') and child.bias is not None:
                    module._modules[mname] = SplitBias(child)
                else: split_bias(child)
        split_bias(model)
        super(KFACOptimizer, self).__init__(model.parameters(), defaults)
        self.known_modules = {'Linear', 'Conv2d', 'AddBias'}
        self.modules = []
        self.model = model
        self._prepare_model()
        self.steps = 0
        self.m_aa, self.m_gg = {}, {}
        self.Q_a, self.Q_g = {}, {}
        self.d_a, self.d_g = {}, {}
        self.stat_decay, self.momentum, self.lr = stat_decay, momentum, lr
        self.kl_clip, self.damping, self.weight_decay = kl_clip, damping, weight_decay
        self.fast_cnn, self.Ts, self.Tf = fast_cnn, Ts, Tf
        self.acc_stats = True
        self.optim = optim.SGD(model.parameters(), lr=self.lr * (1 - self.momentum), momentum=self.momentum)

    def _save_input(self, module, input):
        if torch.is_grad_enabled() and self.steps % self.Ts == 0:
            classname, layer_info = module.__class__.__name__, None
            if classname == 'Conv2d': layer_info = (module.kernel_size, module.stride, module.padding, module.groups)
            aa = compute_cov_a(input[0].data, classname, layer_info, self.fast_cnn)
            if self.steps == 0: self.m_aa[module] = torch.zeros_like(aa)
            update_running_stat(aa, self.m_aa[module], self.stat_decay)

    def _save_grad_output(self, module, grad_input, grad_output):
        if self.acc_stats:
            classname, layer_info = module.__class__.__name__, None
            if classname == 'Conv2d': layer_info = (module.kernel_size, module.stride, module.padding, module.groups)
            gg = compute_cov_g(grad_output[0].data, classname, layer_info, self.fast_cnn)
            if self.steps == 0: self.m_gg[module] = torch.zeros_like(gg)
            update_running_stat(gg, self.m_gg[module], self.stat_decay)

    def _prepare_model(self):
        for module in self.model.modules():
            if module.__class__.__name__ in self.known_modules:
                self.modules.append(module)
                module.register_forward_pre_hook(self._save_input)
                module.register_full_backward_hook(self._save_grad_output)

    def step(self, closure=None):
        if self.weight_decay > 0:
            for group in self.param_groups:
                for p in group['params']:
                    if p.grad is None: continue
                    p.grad.data.add_(p.data, alpha=self.weight_decay)
        
        updates = {}
        for m in self.modules:
            p = next(m.parameters())
            if p.grad is None: continue
            la = self.damping + self.weight_decay
            if self.steps % self.Tf == 0:
                self.d_a[m], self.Q_a[m] = torch.linalg.eigh(self.m_aa[m])
                self.d_g[m], self.Q_g[m] = torch.linalg.eigh(self.m_gg[m])
                self.d_a[m].mul_((self.d_a[m] > 1e-6).float())
                self.d_g[m].mul_((self.d_g[m] > 1e-6).float())
            is_grouped = self.Q_g[m].dim() == 3
            if is_grouped:
                groups = m.groups
                p_grad_mat = p.grad.data.view(groups, self.Q_g[m].size(1), -1)
                v1 = torch.bmm(self.Q_g[m].transpose(1, 2), torch.bmm(p_grad_mat, self.Q_a[m]))
                denominator = self.d_g[m].unsqueeze(2) * self.d_a[m].unsqueeze(1) + la
                v2 = v1 / denominator
                v = torch.bmm(self.Q_g[m], torch.bmm(v2, self.Q_a[m].transpose(1, 2)))
            else:
                classname = m.__class__.__name__
                p_grad_mat = p.grad.data.view(p.grad.data.size(0), -1) if 'Conv' in classname else p.grad.data
                v1 = self.Q_g[m].t() @ p_grad_mat @ self.Q_a[m]
                v2 = v1 / (self.d_g[m].unsqueeze(1) * self.d_a[m].unsqueeze(0) + la)
                v = self.Q_g[m] @ v2 @ self.Q_a[m].t()
            updates[p] = v.view(p.grad.data.size())
        
        vg_sum = sum((updates[p] * p.grad.data * self.lr * self.lr).sum() for p in updates if p.grad is not None)
        nu = min(1.0, math.sqrt(self.kl_clip / vg_sum) if vg_sum > 0 else 1.0)
        
        for p in self.model.parameters():
            if p.grad is None: continue
            if p in updates:
                p.grad.data = updates[p].mul_(nu)

        self.optim.step()
        self.steps += 1

# ==============================================================================
# ACKTR Algorithm (Modified for Compatibility)
# ==============================================================================

class ACKTR():
    def __init__(self,
                 actor_critic,
                 value_loss_coef,
                 entropy_coef,
                 invaild_coef,
                 acktr=False,
                 # KFAC-specific hyperparameters with sensible defaults
                 lr=0.25, 
                 kfac_clip=0.001,
                 kfac_damping=1e-2,
                 kfac_stat_decay=0.99,
                 # Standard hyperparameters
                 eps=None,
                 alpha=None,
                 max_grad_norm=None,
                 args=None):

        self.actor_critic = actor_critic
        self.acktr = acktr

        self.value_loss_coef = value_loss_coef
        self.invaild_coef = invaild_coef
        self.max_grad_norm = max_grad_norm
        self.entropy_coef = entropy_coef
        self.args = args

        if acktr:
            # CORRECTED: Pass all KFAC hyperparameters to the optimizer
            self.optimizer = KFACOptimizer(
                actor_critic, 
                lr=lr, 
                kl_clip=kfac_clip,
                damping=kfac_damping,
                stat_decay=kfac_stat_decay
            )
        else:
            self.optimizer = optim.RMSprop(
                actor_critic.parameters(), lr, eps=eps, alpha=alpha)

    def update(self, rollouts):
        obs_shape = rollouts.obs.size()[2:]
        action_shape = rollouts.actions.size()[-1]
        num_steps, num_processes, _ = rollouts.rewards.size()

        obs_tensor = rollouts.obs[:-1].view(-1, 6, self.args.container_size[0], self.args.container_size[1])
        gt_masks = obs_tensor[:, 4:6, :, :]

        values, action_log_probs, dist_entropy, _, infeasibility_loss = self.actor_critic.evaluate_actions(
            rollouts.obs[:-1].view(-1, *obs_shape),
            rollouts.recurrent_hidden_states[0].view(
                -1, self.actor_critic.recurrent_hidden_state_size),
            rollouts.masks[:-1].view(-1, 1),
            rollouts.actions.view(-1, action_shape),
            gt_masks)

        values = values.view(num_steps, num_processes, 1)
        action_log_probs = action_log_probs.view(num_steps, num_processes, 1)

        advantages = rollouts.returns[:-1] - values
        value_loss = advantages.pow(2).mean()
        action_loss = -(advantages.detach() * action_log_probs).mean()

        if self.acktr and self.optimizer.steps % self.optimizer.Ts == 0:
            self.actor_critic.zero_grad()
            pg_fisher_loss = -action_log_probs.mean()
            value_noise = torch.randn(values.size(), device=values.device)
            sample_values = values + value_noise
            vf_fisher_loss = -(values - sample_values.detach()).pow(2).mean()
            fisher_loss = pg_fisher_loss + vf_fisher_loss
            
            # This block correctly tells KFAC to accumulate stats for this backward pass
            self.optimizer.acc_stats = True
            fisher_loss.backward(retain_graph=True)
            self.optimizer.acc_stats = False

        self.optimizer.zero_grad()
        
        loss = (value_loss * self.value_loss_coef 
                + action_loss 
                - dist_entropy * self.entropy_coef
                + infeasibility_loss * self.invaild_coef)
        
        loss.backward()

        if not self.acktr:
            nn.utils.clip_grad_norm_(
                self.actor_critic.parameters(), self.max_grad_norm)

        self.optimizer.step()

        return value_loss.item(), action_loss.item(), dist_entropy.item(), infeasibility_loss.item()

# Example of how you might instantiate it in your main script
if __name__ == '__main__':
    # This is a placeholder for your actual actor_critic model
    class DummyActorCritic(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(10, 5)
        def forward(self, x):
            return self.fc(x)
        def evaluate_actions(self, *args, **kwargs):
            # Dummy outputs for demonstration
            return torch.randn(1), torch.randn(1), torch.randn(1), None, torch.randn(1)

    # This is a placeholder for your actual args namespace
    class DummyArgs:
        def __init__(self):
            self.container_size = (10, 10)

    # --- How to create the ACKTR agent ---
    actor_critic_model = DummyActorCritic()
    
    # When using ACKTR, you now pass the KFAC-specific learning rates and params
    acktr_agent = ACKTR(
        actor_critic=actor_critic_model,
        value_loss_coef=0.5,
        entropy_coef=0.01,
        invaild_coef=1.0,
        acktr=True,
        lr=0.1, # KFAC learning rate
        kfac_clip=0.01, # KFAC kl_clip
        kfac_damping=0.001, # KFAC damping
        args=DummyArgs()
    )

    print("Successfully created ACKTR agent with KFAC optimizer.")
    print("Optimizer:", acktr_agent.optimizer)

