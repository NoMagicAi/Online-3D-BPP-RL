import time
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from typing import Tuple, List

# ==============================================================================
# Helper Modules & JIT-Compiled Functions
# ==============================================================================

class AddBias(nn.Module):
    """
    A module that adds a learnable bias to the input. This is used to separate
    the bias from the main weight matrix of a layer, which is a requirement for KFAC.
    """
    def __init__(self, bias: torch.Tensor):
        super(AddBias, self).__init__()
        self._bias = nn.Parameter(bias.unsqueeze(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            # For Linear layers
            bias = self._bias.t().view(1, -1)
        else:
            # For Conv2d layers
            bias = self._bias.t().view(1, -1, 1, 1)
        return x + bias

class SplitBias(nn.Module):
    """
    A wrapper module that splits a standard PyTorch module (like Linear or Conv2d)
    into two parts: the original module without bias, and a separate AddBias module.
    """
    def __init__(self, module: nn.Module):
        super(SplitBias, self).__init__()
        self.module = module
        self.add_bias = AddBias(module.bias.data)
        self.module.bias = None  # Remove bias from the original module

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x = self.module(input)
        return self.add_bias(x)

@torch.jit.script
def _extract_patches(x: torch.Tensor, kernel_size: List[int], stride: List[int], padding: List[int]) -> torch.Tensor:
    """
    JIT-compiled function to extract patches from a feature map, equivalent to the 'im2col' operation.
    This is a critical step for computing the covariance of activations in convolutional layers.
    """
    if padding[0] > 0 or padding[1] > 0:
        x = F.pad(x, (padding[1], padding[1], padding[0], padding[0]))
    x = x.unfold(2, kernel_size[0], stride[0])
    x = x.unfold(3, kernel_size[1], stride[1])
    x = x.transpose_(1, 2).transpose_(2, 3).contiguous()
    return x.view(x.size(0), x.size(1), x.size(2), -1)

@torch.jit.script
def compute_cov_a(a: torch.Tensor, classname: str, layer_info: Tuple[List[int], List[int], List[int], int], is_add_bias: bool) -> torch.Tensor:
    """
    JIT-compiled function to compute the covariance matrix of activations 'a'.
    """
    batch_size = a.size(0)

    if is_add_bias:
        # Simplified case for bias terms
        a = torch.ones(batch_size, 1, device=a.device)
        cov_a = a.t() @ (a / batch_size)
    elif classname == 'Conv2d':
        kernel_size, stride, padding, groups = layer_info
        a = _extract_patches(a, kernel_size, stride, padding)
        a = a.view(batch_size, -1, a.size(-1)).mean(1) # Average over spatial locations
        if groups > 1:
            # Handle grouped convolutions
            a = a.view(batch_size, groups, -1).transpose(0, 1)
            cov_a = torch.bmm(a.transpose(1, 2), a) / batch_size
        else:
            cov_a = a.t() @ a / batch_size
    else: # Linear
        cov_a = a.t() @ a / batch_size
    
    return cov_a

@torch.jit.script
def compute_cov_g(g: torch.Tensor, classname: str, layer_info: Tuple[List[int], List[int], List[int], int], fast_cnn: bool, is_add_bias: bool) -> torch.Tensor:
    """
    JIT-compiled function to compute the covariance matrix of pre-activation gradients 'g'.
    """
    batch_size = g.size(0)
    
    if is_add_bias:
        g = g.view(g.size(0), g.size(1), -1).sum(-1)
        g_ = g * batch_size
        cov_g = g_.t() @ g_ / g.size(0)
    elif classname == 'Conv2d':
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
            cov_g = torch.bmm(g_.transpose(1, 2), g_) / g.size(1)
        else:
            g_ = g * batch_size
            cov_g = g_.t() @ g_ / g.size(0)
    else: # Linear
        g_ = g * batch_size
        cov_g = g_.t() @ g_ / g.size(0)
        
    return cov_g

# ==============================================================================
# KFAC Optimizer (Optimized Version)
# ==============================================================================

class KFACOptimizer(optim.Optimizer):
    """
    K-FAC optimizer with optional low-rank approximation and Fisher update subsampling.
    """
    def __init__(self, model, lr=0.25, momentum=0.9, stat_decay=0.99, kl_clip=0.001, 
                 damping=1e-2, weight_decay=0, fast_cnn=False, Ts=1, Tf=10, 
                 kfac_approx_rank=64, fisher_frac=1.0):
        """
        Args:
            model (torch.nn.Module): The model to optimize.
            lr (float): Learning rate.
            momentum (float): Momentum for the SGD update.
            stat_decay (float): Decay factor for running averages of covariance matrices.
            kl_clip (float): Clipping parameter for the KL divergence.
            damping (float): Damping factor for the Fisher matrix inverse.
            weight_decay (float): L2 penalty.
            fast_cnn (bool): Use a faster but less accurate method for Conv2D layers.
            Ts (int): Frequency (in steps) to update Fisher statistics.
            Tf (int): Frequency (in steps) to update the eigendecomposition.
            kfac_approx_rank (int): Rank for the low-rank SVD approximation. If None, use full SVD.
            fisher_frac (float): Fraction of the batch to use for Fisher statistics. 1.0 uses the full batch.
        """
        
        def split_bias(module):
            """
            Recursively splits bias parameters from linear and conv layers.
            This is a standard K-FAC technique to handle biases separately.
            """
            for mname, child in module.named_children():
                if hasattr(child, 'bias') and child.bias is not None:
                    module._modules[mname] = SplitBias(child)
                else:
                    split_bias(child)
        
        # Prepare the model by splitting bias terms
        split_bias(model)
        
        # Initialize the optimizer
        super(KFACOptimizer, self).__init__(model.parameters(), dict())

        self.known_modules = {'Linear', 'Conv2d', 'AddBias'}
        self.modules = []
        self.model = model
        self._prepare_model()
        
        self.steps = 0
        
        # Dictionaries to store K-FAC statistics
        self.m_aa, self.m_gg = {}, {}
        self.Q_a, self.Q_g = {}, {}
        self.d_a, self.d_g = {}, {}
        
        # Hyperparameters
        self.stat_decay = stat_decay
        self.momentum = momentum
        self.lr = lr
        self.kl_clip = kl_clip
        self.damping = damping
        self.weight_decay = weight_decay
        self.fast_cnn = fast_cnn
        self.Ts = Ts
        self.Tf = Tf
        self.kfac_approx_rank = kfac_approx_rank
        self.fisher_frac = fisher_frac # New parameter for subsampling
        
        self.acc_stats = True
        # Use a standard SGD optimizer for the final parameter update
        self.optim = optim.SGD(model.parameters(), lr=self.lr * (1 - self.momentum), momentum=self.momentum)

    def _save_input(self, module, input):
        """Hook to save module inputs for covariance calculation."""
        if torch.is_grad_enabled() and self.steps % self.Ts == 0:
            a = input[0].detach().clone()

            # === NEW: Subsampling Logic ===
            # If fisher_frac is less than 1, use a random subset of the batch
            # to compute the activation covariance matrix (m_aa).
            if self.fisher_frac < 1.0:
                batch_size = a.size(0)
                sample_size = int(batch_size * self.fisher_frac)
                if sample_size < 1: sample_size = 1 # Ensure at least one sample
                indices = torch.randperm(batch_size, device=a.device)[:sample_size]
                a = a.index_select(0, indices)
            # ============================

            classname = module.__class__.__name__
            is_add_bias = classname == 'AddBias'
            
            layer_info = ([], [], [], 1) # Default for non-conv layers
            if classname == 'Conv2d':
                layer_info = (module.kernel_size, module.stride, module.padding, module.groups)
            
            aa = compute_cov_a(a, classname, layer_info, is_add_bias)
            
            if self.steps == 0:
                self.m_aa[module] = torch.zeros_like(aa)
            
            # Update running average of activation covariance
            self.m_aa[module].mul_(self.stat_decay).add_(aa, alpha=1 - self.stat_decay)

    def _save_grad_output(self, module, grad_input, grad_output):
        """Hook to save module gradient outputs for covariance calculation."""
        if self.acc_stats:
            g = grad_output[0].detach().clone()

            # === NEW: Subsampling Logic ===
            # If fisher_frac is less than 1, use a random subset of the batch
            # to compute the gradient covariance matrix (m_gg).
            if self.fisher_frac < 1.0:
                batch_size = g.size(0)
                sample_size = int(batch_size * self.fisher_frac)
                if sample_size < 1: sample_size = 1 # Ensure at least one sample
                indices = torch.randperm(batch_size, device=g.device)[:sample_size]
                g = g.index_select(0, indices)
            # ============================

            classname = module.__class__.__name__
            is_add_bias = classname == 'AddBias'
            
            layer_info = ([], [], [], 1) # Default for non-conv layers
            if classname == 'Conv2d':
                layer_info = (module.kernel_size, module.stride, module.padding, module.groups)

            gg = compute_cov_g(g, classname, layer_info, self.fast_cnn, is_add_bias)
            
            if self.steps == 0:
                self.m_gg[module] = torch.zeros_like(gg)

            # Update running average of gradient covariance
            self.m_gg[module].mul_(self.stat_decay).add_(gg, alpha=1 - self.stat_decay)

    def _prepare_model(self):
        """Register hooks for all known module types."""
        for module in self.model.modules():
            classname = module.__class__.__name__
            if classname in self.known_modules:
                self.modules.append(module)
                module.register_forward_pre_hook(self._save_input)
                module.register_full_backward_hook(self._save_grad_output)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step."""
        # Apply weight decay to raw gradients
        if self.weight_decay > 0:
            for group in self.param_groups:
                for p in group['params']:
                    if p.grad is not None:
                        p.grad.add_(p, alpha=self.weight_decay)
        
        preconditioned_grads = {}
        vg_sum = 0.0

        for m in self.modules:
            p = next(m.parameters())
            if p.grad is None:
                continue

            la = self.damping + self.weight_decay
            
            # Update eigendecomposition of covariance matrices periodically
            if self.steps % self.Tf == 0:
                if self.kfac_approx_rank is not None:
                    # Fast, approximate SVD for efficiency
                    U_g, S_g, _ = torch.svd_lowrank(self.m_gg[m], q=self.kfac_approx_rank)
                    self.d_g[m], self.Q_g[m] = S_g, U_g
                    U_a, S_a, _ = torch.svd_lowrank(self.m_aa[m], q=self.kfac_approx_rank)
                    self.d_a[m], self.Q_a[m] = S_a, U_a
                else:
                    # Exact, slower eigendecomposition
                    self.d_g[m], self.Q_g[m] = torch.linalg.eigh(self.m_gg[m])
                    self.d_a[m], self.Q_a[m] = torch.linalg.eigh(self.m_aa[m])

                # Filter out small eigenvalues to improve numerical stability
                self.d_a[m].mul_((self.d_a[m] > 1e-6).float())
                self.d_g[m].mul_((self.d_g[m] > 1e-6).float())
            
            # Precondition the gradient
            grad = p.grad.detach()
            is_grouped = self.Q_g[m].dim() == 3
            
            if is_grouped: # Grouped convolutions
                groups = m.groups
                p_grad_mat = grad.view(groups, self.Q_g[m].size(1), -1)
                # v = Q_g @ ( (Q_g.T @ G @ Q_a) / (d_g * d_a.T + la) ) @ Q_a.T
                v1 = torch.bmm(self.Q_g[m].transpose(1, 2), torch.bmm(p_grad_mat, self.Q_a[m]))
                denominator = self.d_g[m].unsqueeze(2) * self.d_a[m].unsqueeze(1) + la
                v2 = v1 / denominator
                v = torch.bmm(self.Q_g[m], torch.bmm(v2, self.Q_a[m].transpose(1, 2)))
            else: # Linear or standard Conv2D
                classname = m.__class__.__name__
                p_grad_mat = grad.view(grad.size(0), -1) if 'Conv' in classname else grad
                # Using torch.linalg.multi_dot can be faster by optimizing multiplication order
                v1 = torch.linalg.multi_dot([self.Q_g[m].t(), p_grad_mat, self.Q_a[m]])
                denominator = self.d_g[m].unsqueeze(1) * self.d_a[m].unsqueeze(0) + la
                v2 = v1 / denominator
                v = torch.linalg.multi_dot([self.Q_g[m], v2, self.Q_a[m].t()])

            preconditioned_grad = v.view(grad.size())
            preconditioned_grads[p] = preconditioned_grad
            vg_sum += (preconditioned_grad * grad * self.lr * self.lr).sum()
        
        # Compute KL clipping factor
        nu = min(1.0, math.sqrt(self.kl_clip / vg_sum) if vg_sum > 0 else 1.0)
        
        # Update gradients with the preconditioned and clipped values
        for p in self.model.parameters():
            if p.grad is not None and p in preconditioned_grads:
                # This is the key step: replace the original gradient with the
                # preconditioned, scaled, and clipped version.
                p.grad = preconditioned_grads[p] * nu

        # Perform the final parameter update using the underlying SGD optimizer
        self.optim.step()
        self.steps += 1



# ==============================================================================
# ACKTR Algorithm (Optimized with AMP Support)
# ==============================================================================

class ACKTR():
    def __init__(self,
                 actor_critic,
                 value_loss_coef,
                 entropy_coef,
                 invaild_coef,
                 acktr=False,
                 # KFAC-specific hyperparameters
                 lr=0.25, 
                 kfac_clip=0.001,
                 kfac_damping=1e-2,
                 kfac_stat_decay=0.99,
                 # Standard hyperparameters
                 eps=1e-5,
                 alpha=0.99,
                 max_grad_norm=0.5,
                 # AMP flag
                 use_amp=False,
                 args=None):

        self.actor_critic = actor_critic
        self.acktr = acktr
        self.value_loss_coef = value_loss_coef
        self.invaild_coef = invaild_coef
        self.max_grad_norm = max_grad_norm
        self.entropy_coef = entropy_coef
        self.args = args
        self.use_amp = use_amp and torch.cuda.is_available()

        if acktr:
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
        
        if self.use_amp:
            self.scaler = torch.cuda.amp.GradScaler()

    def update(self, rollouts):
        obs_shape = rollouts.obs.size()[2:]
        action_shape = rollouts.actions.size()[-1]
        num_steps, num_processes, _ = rollouts.rewards.size()

        # FIXED: Updated to modern torch.amp.autocast syntax
        with torch.amp.autocast(device_type='cuda', dtype=torch.float16, enabled=self.use_amp):
            obs_tensor = rollouts.obs[:-1].view(-1, 6, self.args.container_size[0], self.args.container_size[1])
            gt_masks = obs_tensor[:, 4:6, :, :]

            #start_time_2 = time.perf_counter()
            values, action_log_probs, dist_entropy, _, infeasibility_loss = self.actor_critic.evaluate_actions(
                rollouts.obs[:-1].view(-1, *obs_shape),
                rollouts.recurrent_hidden_states[0].view(-1, self.actor_critic.recurrent_hidden_state_size),
                rollouts.masks[:-1].view(-1, 1),
                rollouts.actions.view(-1, action_shape),
                gt_masks
            )
            #end_time_2 = time.perf_counter()
            #print(f"actor critic eval: {end_time_2 - start_time_2} seconds")

            values = values.view(num_steps, num_processes, 1)
            action_log_probs = action_log_probs.view(num_steps, num_processes, 1)

            advantages = rollouts.returns[:-1] - values
            value_loss = advantages.pow(2).mean()
            action_loss = -(advantages.detach() * action_log_probs).mean()
            
            loss = (value_loss * self.value_loss_coef 
                    + action_loss 
                    - dist_entropy * self.entropy_coef
                    + infeasibility_loss * self.invaild_coef)

        # Fisher loss calculation for KFAC
        if self.acktr and self.optimizer.steps % self.optimizer.Ts == 0:
            self.actor_critic.zero_grad()
            with torch.amp.autocast(device_type='cuda', dtype=torch.float16, enabled=self.use_amp):
                pg_fisher_loss = -action_log_probs.mean()
                value_noise = torch.randn_like(values)
                sample_values = values + value_noise
                vf_fisher_loss = -(values - sample_values.detach()).pow(2).mean()
                fisher_loss = pg_fisher_loss + vf_fisher_loss
            
            self.optimizer.acc_stats = True
            if self.use_amp:
                self.scaler.scale(fisher_loss).backward(retain_graph=True)
            else:
                #start_time = time.perf_counter()
                fisher_loss.backward(retain_graph=True)
                #end_time = time.perf_counter()
                #elapsed_time = end_time - start_time
                #print(f"fisher loss: {elapsed_time} seconds")
            self.optimizer.acc_stats = False

        self.optimizer.zero_grad()
        
        # Main backward pass
        if self.use_amp:
            self.scaler.scale(loss).backward()
            if not self.acktr:
                # Unscale gradients before clipping
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            #start_time_3 = time.perf_counter()
            loss.backward()
            #end_time_3 = time.perf_counter()
            #print(f"loss backward: {end_time_3 - start_time_3} seconds")
            if not self.acktr:
                nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
            #start_time_4 = time.perf_counter()
            self.optimizer.step()
            #end_time_4 = time.perf_counter()
            #print(f"optimizer step: {end_time_4 - start_time_4} seconds")

        return value_loss.item(), action_loss.item(), dist_entropy.item(), infeasibility_loss.item()

# ==============================================================================
# Example Instantiation
# ==============================================================================

if __name__ == '__main__':
    # This is a placeholder for your actual actor_critic model
    class DummyActorCritic(nn.Module):
        def __init__(self):
            super().__init__()
            # A more complex model to better showcase KFAC
            self.conv1 = nn.Conv2d(6, 16, kernel_size=3, stride=1, padding=1)
            self.fc = nn.Linear(16 * 10 * 10, 5)
            self.recurrent_hidden_state_size = 1 # Dummy value

        def forward(self, x):
            x = F.relu(self.conv1(x))
            x = x.view(x.size(0), -1)
            return self.fc(x)
        
        def evaluate_actions(self, obs, rnn_hhs, masks, actions, gt_masks):
            # Dummy outputs for demonstration
            num_samples = obs.size(0)
            device = obs.device
            return (torch.randn(num_samples, 1, device=device), 
                    torch.randn(num_samples, 1, device=device), 
                    torch.randn(1, device=device), 
                    None, 
                    torch.randn(1, device=device))

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
        lr=0.1,             # KFAC learning rate
        kfac_clip=0.01,     # KFAC kl_clip
        kfac_damping=0.001, # KFAC damping
        use_amp=False,      # Set to True if using a CUDA GPU
        args=DummyArgs()
    )

    print("Successfully created ACKTR agent with optimized KFAC optimizer.")
    print("Optimizer:", acktr_agent.optimizer)
    if acktr_agent.use_amp:
        print("Automatic Mixed Precision (AMP) is enabled.")

