from torch import nn
import torch.nn.functional as F
import pfrl
import torch
import numpy as np
import math
from pfrl.nn import BoundByTanh, ConcatObsAndAction
from pfrl.policies import DeterministicHead

"""
Noisy Layers
"""
class NoisyLinear(nn.Linear):
    """
    https://github.com/PacktPublishing/Deep-Reinforcement-Learning-Hands-On-Second-Edition/blob/master/Chapter10/lib/models.py
    """
    def __init__(self, in_features, out_features, sigma_init=0.017, bias=True):
        super(NoisyLinear, self).__init__(in_features, out_features, bias=bias)
        self.sigma_weight = nn.Parameter(torch.full((out_features, in_features), sigma_init))
        self.register_buffer("epsilon_weight", torch.zeros(out_features, in_features))
        if bias:
            self.sigma_bias = nn.Parameter(torch.full((out_features,), sigma_init))
            self.register_buffer("epsilon_bias", torch.zeros(out_features))
        self.reset_parameters()

    def reset_parameters(self):
        std = math.sqrt(3 / self.in_features)
        self.weight.data.uniform_(-std, std)
        self.bias.data.uniform_(-std, std)

    def forward(self, input):
        self.epsilon_weight.normal_()
        bias = self.bias
        if bias is not None:
            self.epsilon_bias.normal_()
            bias = bias + self.sigma_bias * self.epsilon_bias
        return F.linear(input, self.weight + self.sigma_weight * self.epsilon_weight, bias)


""" 
Various NN Architectures
"""

class DRQN(nn.Module):

    def __init__(self, obs_size, n_actions):
        super().__init__()
        """
        Based on Financial Trading as a Game:A Deep Reinforcement Learning Approach
        src: https://arxiv.org/pdf/1807.02787.pdf
        """
        self.l1 = nn.Sequential(
            nn.Linear(obs_size, 256),
            nn.ReLU()
        )
        self.l2 = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU()
        )
        self.l3 = nn.LSTM(256, 256, 1)
        self.relu = nn.ReLU()
        self.l4 = nn.Linear(256, n_actions)

    def forward(self, x):
        h = self.l1(x)
        h = self.l2(h)
        h = h.unsqueeze(0)
        out, (_,_) = self.l3(h)
        h = out.unsqueeze(0)
        h = self.relu(h)
        h = self.l4(h)
        return pfrl.action_value.DiscreteActionValue(h)
    
class DRQN_CustomNet(nn.Module):
    def __init__(self, obs_size, n_actions, hidden_size, n_layers):
        super().__init__()
        self.l1 = nn.LSTM(obs_size, hidden_size, n_layers, batch_first=True)
        self.l2 = pfrl.nn.FactorizedNoisyLinear(nn.Linear(hidden_size, n_actions))
        self.dropout = nn.Dropout()
        self.relu = nn.ReLU()
        
    def forward(self, x):
        out = self.l1(x)[0]
        h = self.dropout(out[:,-1,:])
        h = self.relu(h)
        h = self.l2(h)
        return pfrl.action_value.DiscreteActionValue(h)
    
class DRQN_CustomNet2(nn.Module):
    def __init__(self, obs_size, n_actions, hidden_size, n_layers):
        super().__init__()
        self.l1 = nn.LSTM(obs_size, hidden_size, n_layers, batch_first=True)
        self.l2 = pfrl.nn.FactorizedNoisyLinear(nn.Linear(hidden_size, int(hidden_size/2)))
        self.l3 = pfrl.nn.FactorizedNoisyLinear(nn.Linear(int(hidden_size/2), n_actions))
        self.dropout = nn.Dropout()
        self.relu = nn.ReLU()
        
    def forward(self, x):
        out = self.l1(x)[0]
        h = self.dropout(out[:,-1,:])
        h = self.relu(h)
        h = self.l2(h)
        h = self.l3(h)
        return pfrl.action_value.DiscreteActionValue(h) 
    
class DRQN_CustomNet3(nn.Module):
    def __init__(self, obs_size, n_actions, hidden_size, n_layers):
        super().__init__()
        self.l1 = nn.LSTM(obs_size, hidden_size, n_layers, batch_first=True)
        self.l2 = nn.Sequential(
            pfrl.nn.FactorizedNoisyLinear(nn.Linear(hidden_size, int(hidden_size/2))),
            nn.Dropout(),
            nn.ReLU()
        )
        self.l3 = pfrl.nn.FactorizedNoisyLinear(nn.Linear(int(hidden_size/2), n_actions))
        self.dropout = nn.Dropout()
        self.relu = nn.ReLU()
        
    def forward(self, x):
        out = self.l1(x)[0]
        h = self.dropout(out[:,-1,:])
        h = self.relu(h)
        h = self.l2(h)
        h = self.l3(h)
        return pfrl.action_value.DiscreteActionValue(h) 
    
class GDQN_CustomNet(nn.Module):
    def __init__(self, obs_size, n_actions, hidden_size, n_layers):
        super().__init__()
        self.l1 = nn.GRU(obs_size, hidden_size, n_layers, dropout=0.5, batch_first=True)
        self.l2 = pfrl.nn.FactorizedNoisyLinear(nn.Linear(hidden_size, n_actions))
        self.dropout = nn.Dropout()
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.l1(x)[0]
        out = self.dropout(out[:,-1,:])
        out = self.relu(out)
        out = self.l2(out)
        return pfrl.action_value.DiscreteActionValue(out)
    
    
class DuellingNet(nn.Module):
    def __init__(self, obs_size, n_actions, hidden_size):
        super(DuellingNet, self).__init__()

        self.fc_val = nn.Sequential(
            pfrl.nn.FactorizedNoisyLinear(nn.Linear(obs_size, hidden_size)),
            nn.ReLU(),
            pfrl.nn.FactorizedNoisyLinear(nn.Linear(hidden_size, hidden_size)),
            nn.ReLU(),
            pfrl.nn.FactorizedNoisyLinear(nn.Linear(hidden_size, 1))
        )

        self.fc_adv = nn.Sequential(
            pfrl.nn.FactorizedNoisyLinear(nn.Linear(obs_size, hidden_size)),
            nn.ReLU(),
            pfrl.nn.FactorizedNoisyLinear(nn.Linear(hidden_size, hidden_size)),
            nn.ReLU(),
            pfrl.nn.FactorizedNoisyLinear(nn.Linear(hidden_size, n_actions))
        )

    def forward(self, x):
        val = self.fc_val(x)
        adv = self.fc_adv(x)
        h = val + (adv - adv.mean(dim=1, keepdim=True))
        return pfrl.action_value.DiscreteActionValue(h)
    
class DuellingGRU(nn.Module):
    def __init__(self, obs_size, n_actions, hidden_size):
        super(DuellingGRU, self).__init__()

        self.fc_val = nn.Sequential(
            nn.GRU(obs_size, hidden_size, 2, dropout=0.5, batch_first=True),
            nn.ReLU(),
            pfrl.nn.FactorizedNoisyLinear(nn.Linear(hidden_size, hidden_size)),
            nn.ReLU(),
            pfrl.nn.FactorizedNoisyLinear(nn.Linear(hidden_size, 1))
        )

        self.fc_adv = nn.Sequential(
            nn.GRU(obs_size, hidden_size, 2, dropout=0.5, batch_first=True),
            nn.ReLU(),
            pfrl.nn.FactorizedNoisyLinear(nn.Linear(hidden_size, hidden_size)),
            nn.ReLU(),
            pfrl.nn.FactorizedNoisyLinear(nn.Linear(hidden_size, n_actions))
        )

    def forward(self, x):
        val = self.fc_val(x)
        adv = self.fc_adv(x)
        h = val + (adv - adv.mean(dim=1, keepdim=True))
        return pfrl.action_value.DiscreteActionValue(h)
    
class DQNConv1D(nn.Module):
    def __init__(self, shape, actions_n):
        super(DQNConv1D, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv1d(shape[0], 128, 5),
            nn.ReLU(),
            nn.Conv1d(128, 128, 5),
            nn.ReLU(),
        )

        out_size = self._get_conv_out(shape)

        self.fc_val = nn.Sequential(
            nn.Linear(out_size, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )

        self.fc_adv = nn.Sequential(
            nn.Linear(out_size, 512),
            nn.ReLU(),
            nn.Linear(512, actions_n)
        )

    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, x):
        conv_out = self.conv(x).view(x.size()[0], -1)
        val = self.fc_val(conv_out)
        adv = self.fc_adv(conv_out)
        out = val + (adv - adv.mean(dim=1, keepdim=True))
        return pfrl.action_value.DiscreteActionValue(out)
    
class DQNConv1DLarge(nn.Module):
    def __init__(self, shape, actions_n):
        super(DQNConv1DLarge, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv1d(shape[0], 32, 3),
            nn.MaxPool1d(3, 2),
            nn.ReLU(),
            nn.Conv1d(32, 32, 3),
            nn.MaxPool1d(3, 2),
            nn.ReLU(),
            nn.Conv1d(32, 32, 3),
            nn.MaxPool1d(3, 2),
            nn.ReLU(),
            nn.Conv1d(32, 32, 3),
            nn.MaxPool1d(3, 2),
            nn.ReLU(),
            nn.Conv1d(32, 32, 3),
            nn.ReLU(),
            nn.Conv1d(32, 32, 3),
            nn.ReLU(),
        )

        out_size = self._get_conv_out(shape)

        self.fc_val = nn.Sequential(
            nn.Linear(out_size, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )

        self.fc_adv = nn.Sequential(
            nn.Linear(out_size, 512),
            nn.ReLU(),
            nn.Linear(512, actions_n)
        )

    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, x):
        conv_out = self.conv(x).view(x.size()[0], -1)
        val = self.fc_val(conv_out)
        adv = self.fc_adv(conv_out)
        return val + (adv - adv.mean(dim=1, keepdim=True))
    
class LSTMCritic(nn.Module):

    def __init__(self, obs_size, n_actions, low, high):
        super().__init__()
        
        self.l1 = nn.LSTM(obs_size, 100, 2, batch_first=True)
        self.l2 = nn.Linear(100, n_actions)
        self.l3 = nn.Sequential(
            BoundByTanh(low=low, high=high),
            DeterministicHead(),
        )

    def forward(self, x):
        x = x.unsqueeze(1)
        out, (_,_) = self.l1(x)
        h = out[:,-1,:]
        h = self.l2(h)
        out = self.l3(h)
        return out
    
class Convert1Dto2D(nn.Module):
    def __init__(self, window_size, feature_len, asset_count):
        super().__init__()
        self.window_size = window_size
        self.feature_len = feature_len
        self.asset_count = asset_count
        
    def forward(self, x):
        padded_x = nn.functional.pad(x, [0,self.feature_len - self.asset_count%self.feature_len], 'constant', 0)
        return padded_x.reshape(-1, self.window_size + self.asset_count//self.feature_len + 1, self.feature_len)
    
class LSTMCritic2(nn.Module):

    def __init__(self, n_actions, window_size, feature_len, asset_count):
        super().__init__()
        
        self.l0 = Convert1Dto2D(window_size, feature_len, asset_count)
        self.l1 = nn.LSTM(feature_len, 512, 2, batch_first=True)
        self.l23 = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, n_actions)
        )
        self.l4a = nn.Sigmoid()
        self.l4b = nn.Softmax(dim=1)
        self.l5 = DeterministicHead()


    def forward(self, x):
        h = self.l0(x)
        out, (_,_) = self.l1(h)
        h = out[:,-1,:]
        h = self.l23(h)
        a, w = h[:,0], h[:,1:]
        a = self.l4a(a)
        w = self.l4b(w)
        a = a.reshape(-1,1)
        h = torch.cat((a,w), 1)
        return self.l5(h)
    
# class SimActor(nn.Module):

#     def __init__(self, obs_size, action_size):
#         super().__init__()
#         """
#         Based on Financial Trading as a Game:A Deep Reinforcement Learning Approach
#         src: https://arxiv.org/pdf/1807.02787.pdf
#         """
#         self.l0 = ConcatObsAndAction()
#         self.l1 = nn.Linear(obs_size+action_size, 256)
#         self.l2 = nn.Linear(256, 128)
#         self.l3 = nn.Linear(128, 64)
#         self.l4 = nn.Linear(64, 1)

#     def forward(self, x):
#         h = self.l0(x)
#         h = self.l1(h)
#         h = self.l2(h)
#         h = self.l3(h)
#         h = self.l4(h)
#         return h
    
    
class SimpleActor(nn.Module):

    def __init__(self, obs_size, action_size):
        super().__init__()
        
        self.l1 = ConcatObsAndAction()
        self.l2 = nn.Linear(obs_size + action_size, 512)
        self.l3 = nn.ReLU()
        self.l4 = nn.Linear(512, 256)
        self.l5 = nn.ReLU()
        self.l6 = nn.Linear(256, 1)

    def forward(self, x):
        h = self.l1(x)
        h = self.l2(h)
        h = self.l3(h)
        h = self.l4(h)
        h = self.l5(h)
        out = self.l6(h)
        return out
    
class SimpleCritic(nn.Module):

    def __init__(self, obs_size, action_size, lower_bound, upper_bound):
        super().__init__()
        self.l1 = nn.Linear(obs_size, 512)
        self.l2 = nn.ReLU()
        self.l3 = nn.Linear(512, 256)
        self.l4 = nn.ReLU()
        self.l5 = nn.Linear(256, action_size)
        self.l6 = BoundByTanh(low=lower_bound, high=upper_bound)
        self.l7 = DeterministicHead()

    def forward(self, x):
        h = self.l1(x)
        h = self.l2(h)
        h = self.l3(h)
        h = self.l4(h)
        h = self.l5(h)
        h = self.l6(h)
        out = self.l7(h)
        return out  

    
class FCNet(nn.Module):

    def __init__(self, obs_size, n_actions, hidden_size):
        super().__init__()
        self.l1 = nn.Linear(obs_size, hidden_size)
        self.l2 = nn.Linear(hidden_size, int(hidden_size/2))
        self.l3 = nn.Linear(int(hidden_size/2), n_actions)

    def forward(self, x):
        h = x
        h = nn.functional.relu(self.l1(h))
        h = nn.functional.relu(self.l2(h))
        h = self.l3(h)
        return pfrl.action_value.DiscreteActionValue(h)