from torch import nn
from torch.nn import functional as F 
import torch
import math

class Policy(nn.Module):
  def __init__(self):
    super(Policy, self).__init__()
    self.net = nn.Sequential(
      nn.Linear(6, 128),
      nn.ReLU(),
      nn.Linear(128, 64),
      nn.ReLU(),
      nn.Linear(64, 3),
      nn.Softmax()
    )
    # action & reward buffer
    self.saved_actions = []
    self.rewards = []
  
  def forward(self, x):
    return self.net(x)

class Value(nn.Module):
  def __init__(self):
    super(Value, self).__init__()
    self.net  = nn.Sequential(
      nn.Linear(6, 128),
      nn.ReLU(),
      nn.Linear(128, 64),
      nn.ReLU(),
      nn.Linear(64, 1)
    )
  def forward(self, x):
    return self.net(x)

class Policy_Value(nn.Module):
    """
    implements both actor and critic in one model
    """
    def __init__(self):
        super(Policy_Value, self).__init__()
        self.linear = nn.Linear(6, 128)

        # actor's layer
        self.action_head = nn.Linear(128, 3)

        # critic's layer
        self.value_head = nn.Linear(128, 1)

        # action & reward buffer
        self.saved_actions = []
        self.rewards = []

    def forward(self, x):
        """
        forward of both actor and critic
        """
        x = F.relu(self.linear(x))

        # actor: choses action to take from state s_t 
        # by returning probability of each action
        action_prob = F.softmax(self.action_head(x), dim=-1)

        # critic: evaluates being in the state s_t
        state_values = self.value_head(x)

        # return values for both actor and critic as a tuple of 2 values:
        # 1. a list with the probability of each action over the action space
        # 2. the value from state s_t 
        return action_prob, state_values

class NoisyLinear(nn.Linear):
    def __init__(self, in_features, out_features, initial_std=0.017, bias=True):
        super(NoisyLinear, self).__init__(in_features, out_features, bias=True)
        w = torch.full((out_features, in_features), initial_std)
        self.sigma_weight = nn.Parameter(w, requires_grad=True)
        we = torch.zeros((out_features, in_features))
        self.register_buffer('weight_epsilon', we)
        if bias is not None:
            b = torch.full((out_features,), initial_std)
            self.sigma_bias = nn.Parameter(b, requires_grad=True)
            be = torch.zeros(out_features)
            self.register_buffer("bias_epsilon", be)
        self.reset_parameters()

    def reset_parameters(self):
        """Recommended Initialization by Ref."""
        std = math.sqrt(3 / self.in_features)
        self.weight.data.uniform_(-std, std)
        self.bias.data.uniform_(-std, std)

    def forward(self, x):
        self.weight_epsilon.normal_()
        bias = self.bias
        if bias is not None:
            self.bias_epsilon.normal_()
            bias = bias + self.sigma_bias * self.bias_epsilon.data
        weight = self.weight + self.sigma_weight * self.weight_epsilon.data
        return F.linear(x, weight, bias)


class DuelingDQN(nn.Module):

    def __init__(self, observation_size, hidden_size, action_size):
        super(DuelingDQN, self).__init__()
        mid_size = int(hidden_size / 2)
        self.common = nn.Sequential(
            NoisyLinear(observation_size, 250),
            nn.ReLU(),
            NoisyLinear(250, 200),
            nn.ReLU(),
            NoisyLinear(200, mid_size),
            nn.ReLU(),
        )
        self.advantage = nn.Sequential(
            nn.Linear(mid_size, action_size)
        )
        self.value = nn.Sequential(
            nn.Linear(mid_size, 1)
        )

    def forward(self, x):
        x = self.common(x)
        advantage, value = self.advantage(x), self.value(x)
        return value + (advantage - advantage.mean())



