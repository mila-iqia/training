from milabench.perf import *

import argparse
import gym
import numpy as np
from itertools import count

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical



parser = parser_base(description='PyTorch REINFORCE example')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor (default: 0.99)')
parser.add_argument('--render', action='store_true',
                    help='render the environment')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='interval between training status logs (default: 10)')

exp = Experiment(__file__)
args = exp.get_arguments(parser, show=True)
device = exp.get_device()

env = gym.make('CartPole-v0')
env.seed(args.seed)


class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()
        self.affine1 = nn.Linear(4, 128)
        self.affine2 = nn.Linear(128, 2)

        self.saved_log_probs = []
        self.rewards = []

    def forward(self, x):
        x = F.relu(self.affine1(x))
        action_scores = self.affine2(x)
        return F.softmax(action_scores, dim=1)


policy = Policy()
optimizer = optim.Adam(policy.parameters(), lr=1e-2)
eps = np.finfo(np.float32).eps.item()


def select_action(state):
    state = torch.from_numpy(state).float().unsqueeze(0)
    probs = policy(state)
    m = Categorical(probs)
    action = m.sample()
    policy.saved_log_probs.append(m.log_prob(action))
    return action.item()


def finish_episode():
    R = 0
    policy_loss = []
    returns = []

    for r in policy.rewards[::-1]:
        R = r + args.gamma * R
        returns.insert(0, R)

    returns = torch.tensor(returns)
    returns = (returns - returns.mean()) / (returns.std() + eps)

    for log_prob, R in zip(policy.saved_log_probs, returns):
        policy_loss.append(-log_prob * R)

    optimizer.zero_grad()
    policy_loss = torch.cat(policy_loss).sum()
    policy_loss.backward()
    optimizer.step()

    del policy.rewards[:]
    del policy.saved_log_probs[:]


def main():
    global policy, optimizer

    running_reward = 10
    chrono = exp.chrono()

    for i_episode in range(0, args.repeat):

        with chrono.time('train') as timer:
            state, ep_reward = env.reset(), 0

            for t in range(0, args.number):  # Don't infinite loop while learning

                action = select_action(state)

                state, reward, done, _ = env.step(action)
                exp.log_batch_loss(reward)
                policy.rewards.append(reward)
                ep_reward += reward

                # we actually do not care about solving the thing
                if done:
                    state, ep_reward = env.reset(), 0

            running_reward = 0.05 * ep_reward + (1 - 0.05) * running_reward
            finish_episode()

        exp.show_eta(i_episode, timer)

    exp.report()


if __name__ == '__main__':
    main()
