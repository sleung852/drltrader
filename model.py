from torch import nn
import pfrl
import torch
import numpy as np
from pfrl.nn import BoundByTanh, ConcatObsAndAction
from pfrl.policies import DeterministicHead

"""
Noisy Layers
"""
class NoisyLinear(nn.Linear):
    def __init__(self,
                 in_features,
                 out_features,
                 sigma_init=0.015,
                 bias=True):
        w = torch.full((in_features, out_features), sigma_init)


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
        self.l1 = nn.Linear(obs_size, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.LSTM(256, 256, 1)
        self.l4 = nn.Linear(256, n_actions)

    def forward(self, x):
        h = self.l1(x)
        h = self.l2(h)
        h = h.unsqueeze(0)
        out, (_,_) = self.l3(h)
        h = out.unsqueeze(0)
        h = self.l4(h)
        return pfrl.action_value.DiscreteActionValue(h)
    
class DRQN_CustomNet(nn.Module):
    def __init__(self, obs_size, n_actions, hidden_size, n_layers):
        super().__init__()
        self.l1 = nn.LSTM(obs_size, hidden_size, n_layers)
        self.l2 = nn.Linear(hidden_size, n_actions)

    def forward(self, x):
        h = x
        _, (h,_) = self.l1(h)
        h = self.dropout(h[-1,:,:])
        h = self.l2(h)
        return pfrl.action_value.DiscreteActionValue(h)
    
class GDQN_CustomNet(nn.Module):
    def __init__(self, obs_size, hidden_size, n_layers, n_actions):
        super().__init__()
        self.gru = nn.GRU(obs_size, hidden_size, n_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, n_actions)
        self.dropout = nn.Dropout()

    def forward(self, x):
        out = self.gru(x)[0]
        out = self.dropout(out[:,-1,:])
        out = self.fc(out)
        return pfrl.action_value.DiscreteActionValue(out)
    
class DuellingNet(nn.Module):
    def __init__(self, obs_len, actions_n, hidden_dim):
        super(DuellingNet, self).__init__()

        self.fc_val = nn.Sequential(
            nn.Linear(obs_len, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

        self.fc_adv = nn.Sequential(
            nn.Linear(obs_len, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, actions_n)
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

    def __init__(self, obs_size, n_actions):
        super().__init__()

        self.l1 = nn.LSTM(obs_size, 100, 2)
        self.l2 = nn.Linear(50, n_actions)
        self.l3 = nn.Sequential(
            BoundByTanh(low=0.0, high=1.0),
            DeterministicHead(),
        )

    def forward(self, x):
        x = x.unsqueeze(0)
        out, (_,_) = self.l1(x)
        h = out.squeeze(0)
        h = self.l2(h)
        out = self.l3(h)
        return out
    
class SimActor(nn.Module):

    def __init__(self, obs_size, n_actions, kind='discrete'):
        super().__init__()
        """
        Based on Financial Trading as a Game:A Deep Reinforcement Learning Approach
        src: https://arxiv.org/pdf/1807.02787.pdf
        """
        self.l0 = ConcatObsAndAction()
        self.l1 = nn.Linear(obs_size, 256)
        self.l2 = nn.Linear(256, 128)
        self.l3 = nn.Linear(128, 64)
        self.l4 = nn.Linear(64, n_actions)

    def forward(self, x):
        h = self.l0(x)
        h = self.l1(h)
        h = self.l2(h)
        h = self.l3(h)
        h = self.l4(h)
        return h
    
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