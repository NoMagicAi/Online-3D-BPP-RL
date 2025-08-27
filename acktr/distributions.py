import torch
import torch.nn as nn
import torch.nn.functional as F

from acktr.utils import AddBias, init

"""
Modify standard PyTorch distributions so they are compatible with this code.
"""

#
# Standardize distribution interfaces
#

# Categorical
FixedCategorical = torch.distributions.Categorical

old_sample = FixedCategorical.sample
FixedCategorical.sample = lambda self: old_sample(self).unsqueeze(-1)

log_prob_cat = FixedCategorical.log_prob
FixedCategorical.log_probs = lambda self, actions: log_prob_cat(
    self, actions.squeeze(-1)).view(actions.size(0), -1).sum(-1).unsqueeze(-1)

FixedCategorical.mode = lambda self: self.probs.argmax(dim=-1, keepdim=True)

# Normal
FixedNormal = torch.distributions.Normal

log_prob_normal = FixedNormal.log_prob
FixedNormal.log_probs = lambda self, actions: log_prob_normal(self, actions).sum(-1, keepdim=True)

normal_entropy = FixedNormal.entropy
FixedNormal.entropy = lambda self: normal_entropy(self).sum(-1)

FixedNormal.mode = lambda self: self.mean

# Bernoulli
FixedBernoulli = torch.distributions.Bernoulli

log_prob_bernoulli = FixedBernoulli.log_prob
FixedBernoulli.log_probs = lambda self, actions: log_prob_bernoulli(
    self, actions).view(actions.size(0), -1).sum(-1).unsqueeze(-1)

bernoulli_entropy = FixedBernoulli.entropy
FixedBernoulli.entropy = lambda self: bernoulli_entropy(self).sum(-1)
FixedBernoulli.mode = lambda self: torch.gt(self.probs, 0.5).float()

# remove the mask
def mask_softmax(mat, mask, dim=-1):
    """
    Applies a softmax after masking invalid positions.
    Invalid positions are marked with 0 in the mask.
    """
    mask = mask.float()
    
    # Create an inverse mask where invalid positions are 1.0
    inverse_mask = 1.0 - mask
    
    # Subtract a very large number from the logits of invalid actions.
    # This effectively makes their probability zero after softmax.
    masked_mat = mat - (inverse_mask * 1e9)
    
    # Use PyTorch's built-in softmax, which is numerically stable.
    return F.softmax(masked_mat, dim=dim)

class Categorical(nn.Module):

    def __init__(self, num_inputs, num_outputs):
        super(Categorical, self).__init__()

        init_ = lambda m: init(m, nn.init.orthogonal_,
                               lambda x: nn.init.constant_(x, 0),
                               gain=0.01)

        self.linear = init_(nn.Linear(num_inputs, num_outputs))

    def forward(self, x, mask):
        x = self.linear(x)

        p_ones = torch.ones_like(x)
        ones = torch.ones_like(mask)
        inver_mask = ones - mask

        # Mask the original logits by subtracting a large number from invalid actions.
        # A value like 1e9 is standard for ensuring near-zero probability.
        masked_logits = x - inver_mask * 1e9

        # branch 2
        # Create the distribution directly from logits. This is more numerically stable
        # than creating it from probabilities (probs=...). PyTorch handles the softmax internally.
        fat_cat = FixedCategorical(logits=masked_logits)

        # branch 1
        # minimaze invaild actions
        ax = F.softmax(x, dim=-1)
        bx = ax
        bx = bx * inver_mask

        # branch 3
        # magnify vaild actions dist_entropy
        dx = mask_softmax(x, mask) + 1e-12
        dx = dx * torch.log(dx)
        dx = dx * mask
        dx = -dx
        # dx = F.softmax(x - inver_mask * 14, dim=-1) + 1e-12
        # dx = dx * torch.log(dx)
        # dx = -dx

        return fat_cat, bx, dx

    def get_policy_distribution(self, x):
        x = self.linear(x)
        return x


class DiagGaussian(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(DiagGaussian, self).__init__()

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0))
        self.fc_mean = init_(nn.Linear(num_inputs, num_outputs))
        self.logstd = AddBias(torch.zeros(num_outputs))

    def forward(self, x):
        action_mean = self.fc_mean(x)
        #  An ugly hack for my KFAC implementation.
        zeros = torch.zeros(action_mean.size())
        if x.is_cuda:
            zeros = zeros.cuda()

        action_logstd = self.logstd(zeros)
        return FixedNormal(action_mean, action_logstd.exp())


class Bernoulli(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(Bernoulli, self).__init__()

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0))

        self.linear = init_(nn.Linear(num_inputs, num_outputs))

    def forward(self, x):
        x = self.linear(x)
        return FixedBernoulli(logits=x)
