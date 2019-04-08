from perf import *

import copy
import glob
import os
import time
from collections import deque

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from a2c_ppo_acktr import algo
from a2c_ppo_acktr.envs import make_vec_envs
from a2c_ppo_acktr.model import Policy
from a2c_ppo_acktr.storage import RolloutStorage
from a2c_ppo_acktr.utils import get_vec_normalize, update_linear_schedule
from a2c_ppo_acktr.visualize import visdom_plot


def get_args():
    parser = parser_base(description='RL')
    parser.add_argument('--algo', default='a2c',
                        help='algorithm to use: a2c | ppo | acktr')
    parser.add_argument('--lr', type=float, default=7e-4,
                        help='learning rate (default: 7e-4)')
    parser.add_argument('--eps', type=float, default=1e-5,
                        help='RMSprop optimizer epsilon (default: 1e-5)')
    parser.add_argument('--alpha', type=float, default=0.99,
                        help='RMSprop optimizer apha (default: 0.99)')
    parser.add_argument('--gamma', type=float, default=0.99,
                        help='discount factor for rewards (default: 0.99)')
    parser.add_argument('--use-gae', action='store_true', default=False,
                        help='use generalized advantage estimation')
    parser.add_argument('--tau', type=float, default=0.95,
                        help='gae parameter (default: 0.95)')
    parser.add_argument('--entropy-coef', type=float, default=0.01,
                        help='entropy term coefficient (default: 0.01)')
    parser.add_argument('--value-loss-coef', type=float, default=0.5,
                        help='value loss coefficient (default: 0.5)')
    parser.add_argument('--max-grad-norm', type=float, default=0.5,
                        help='max norm of gradients (default: 0.5)')

    parser.add_argument('--cuda-deterministic', action='store_true', default=False,
                        help="sets flags for determinism when using CUDA (potentially slow!)")

    parser.add_argument('--num-processes', type=int, default=16,
                        help='how many training CPU processes to use (default: 16)')
    parser.add_argument('--num-steps', type=int, default=5,
                        help='number of forward steps in A2C (default: 5)')
    parser.add_argument('--ppo-epoch', type=int, default=4,
                        help='number of ppo epochs (default: 4)')
    parser.add_argument('--num-mini-batch', type=int, default=32,
                        help='number of batches for ppo (default: 32)')
    parser.add_argument('--clip-param', type=float, default=0.2,
                        help='ppo clip parameter (default: 0.2)')
    parser.add_argument('--log-interval', type=int, default=10,
                        help='log interval, one log per n updates (default: 10)')
    parser.add_argument('--save-interval', type=int, default=100,
                        help='save interval, one save per n updates (default: 100)')
    parser.add_argument('--eval-interval', type=int, default=None,
                        help='eval interval, one eval per n updates (default: None)')
    parser.add_argument('--vis-interval', type=int, default=100,
                        help='vis interval, one log per n updates (default: 100)')
    parser.add_argument('--num-env-steps', type=int, default=10e6,
                        help='number of environment steps to train (default: 10e6)')
    parser.add_argument('--env-name', default='PongNoFrameskip-v4',
                        help='environment to train on (default: PongNoFrameskip-v4)')
    parser.add_argument('--log-dir', default='/tmp/gym/',
                        help='directory to save agent logs (default: /tmp/gym)')
    parser.add_argument('--save-dir', default='./trained_models/',
                        help='directory to save agent logs (default: ./trained_models/)')

    parser.add_argument('--add-timestep', action='store_true', default=False,
                        help='add timestep to observations')
    parser.add_argument('--recurrent-policy', action='store_true', default=False,
                        help='use a recurrent policy')
    parser.add_argument('--use-linear-lr-decay', action='store_true', default=False,
                        help='use a linear schedule on the learning rate')
    parser.add_argument('--use-linear-clip-decay', action='store_true', default=False,
                        help='use a linear schedule on the ppo clipping parameter')
    parser.add_argument('--vis', action='store_true', default=False,
                        help='enable visdom visualization')
    parser.add_argument('--port', type=int, default=8097,
                        help='port to run the server on (default: 8097)')

    return parser


exp = Experiment(__file__)
args = exp.get_arguments(get_args(), show=True)
device = exp.get_device()

# we compute steps/sec
args.batch_size = args.num_processes

assert args.algo in ['a2c', 'ppo', 'acktr']

if args.recurrent_policy:
    assert args.algo in ['a2c', 'ppo'], \
        'Recurrent policy is not implemented for ACKTR'


num_updates = int(args.num_env_steps) // args.num_steps // args.num_processes

try:
    os.makedirs(args.log_dir)
except OSError:
    files = glob.glob(os.path.join(args.log_dir, '*.monitor.csv'))
    for f in files:
        os.remove(f)

eval_log_dir = args.log_dir + "_eval"

try:
    os.makedirs(eval_log_dir)
except OSError:
    files = glob.glob(os.path.join(eval_log_dir, '*.monitor.csv'))
    for f in files:
        os.remove(f)


def main():
    chrono = exp.chrono()

    envs = make_vec_envs(args.env_name, args.seed, args.num_processes,
                         args.gamma, args.log_dir, args.add_timestep, device, False)

    actor_critic = Policy(envs.observation_space.shape, envs.action_space,
                          base_kwargs={'recurrent': args.recurrent_policy})
    actor_critic.to(device)

    if args.algo == 'a2c':
        agent = algo.A2C_ACKTR(actor_critic, args.value_loss_coef,
                               args.entropy_coef, lr=args.lr,
                               eps=args.eps, alpha=args.alpha,
                               max_grad_norm=args.max_grad_norm)
    elif args.algo == 'ppo':
        agent = algo.PPO(actor_critic, args.clip_param, args.ppo_epoch, args.num_mini_batch,
                         args.value_loss_coef, args.entropy_coef, lr=args.lr,
                         eps=args.eps,
                         max_grad_norm=args.max_grad_norm)
    elif args.algo == 'acktr':
        agent = algo.A2C_ACKTR(actor_critic, args.value_loss_coef,
                               args.entropy_coef, acktr=True)

    rollouts = RolloutStorage(args.num_steps, args.num_processes,
                              envs.observation_space.shape, envs.action_space,
                              actor_critic.recurrent_hidden_state_size)

    obs = envs.reset()
    rollouts.obs[0].copy_(obs)
    rollouts.to(device)

    episode_rewards = deque(maxlen=10)

    start = time.time()
    for j in range(args.repeat):
        with chrono.time('train') as t:
            for n in range(args.number):

                if args.use_linear_lr_decay:
                    # decrease learning rate linearly
                    if args.algo == "acktr":
                        # use optimizer's learning rate since it's hard-coded in kfac.py
                        update_linear_schedule(agent.optimizer, j, num_updates, agent.optimizer.lr)
                    else:
                        update_linear_schedule(agent.optimizer, j, num_updates, args.lr)

                if args.algo == 'ppo' and args.use_linear_clip_decay:
                    agent.clip_param = args.clip_param * (1 - j / float(num_updates))

                for step in range(args.num_steps):
                    # Sample actions
                    with torch.no_grad():
                        value, action, action_log_prob, recurrent_hidden_states = actor_critic.act(
                            rollouts.obs[step],
                            rollouts.recurrent_hidden_states[step],
                            rollouts.masks[step])

                    # Obser reward and next obs
                    obs, reward, done, infos = envs.step(action)

                    for info in infos:
                        if 'episode' in info.keys():
                            episode_rewards.append(info['episode']['r'])

                    # If done then clean the history of observations.
                    masks = torch.FloatTensor([[0.0] if done_ else [1.0]
                                               for done_ in done])
                    rollouts.insert(obs, recurrent_hidden_states, action, action_log_prob, value, reward, masks)

                with torch.no_grad():
                    next_value = actor_critic.get_value(rollouts.obs[-1],
                                                        rollouts.recurrent_hidden_states[-1],
                                                        rollouts.masks[-1]).detach()
                # ---
                rollouts.compute_returns(next_value, args.use_gae, args.gamma, args.tau)

                value_loss, action_loss, dist_entropy = agent.update(rollouts)

                exp.log_batch_loss(action_loss)
                exp.log_metric('value_loss',  value_loss)

                rollouts.after_update()

                # save for every interval-th episode or for the last epoch
                if (j % args.save_interval == 0 or j == num_updates - 1) and args.save_dir != "":
                    save_path = os.path.join(args.save_dir, args.algo)
                    try:
                        os.makedirs(save_path)
                    except OSError:
                        pass

                    # A really ugly way to save a model to CPU
                    save_model = actor_critic
                    if args.cuda:
                        save_model = copy.deepcopy(actor_critic).cpu()

                    save_model = [save_model,
                                  getattr(get_vec_normalize(envs), 'ob_rms', None)]

                    torch.save(save_model, os.path.join(save_path, args.env_name + ".pt"))

                total_num_steps = (j + 1) * args.num_processes * args.num_steps

                if j % args.log_interval == 0 and len(episode_rewards) > 1:
                    end = time.time()
                    print(
                        "Updates {}, num timesteps {}, FPS {} \n Last {} training episodes: mean/median reward {:.1f}/{:.1f}, min/max reward {:.1f}/{:.1f}\n".
                        format(j, total_num_steps,
                               int(total_num_steps / (end - start)),
                               len(episode_rewards),
                               np.mean(episode_rewards),
                               np.median(episode_rewards),
                               np.min(episode_rewards),
                               np.max(episode_rewards), dist_entropy,
                               value_loss, action_loss))

                if (args.eval_interval is not None
                        and len(episode_rewards) > 1
                        and j % args.eval_interval == 0):
                    eval_envs = make_vec_envs(
                        args.env_name, args.seed + args.num_processes, args.num_processes,
                        args.gamma, eval_log_dir, args.add_timestep, device, True)

                    vec_norm = get_vec_normalize(eval_envs)
                    if vec_norm is not None:
                        vec_norm.eval()
                        vec_norm.ob_rms = get_vec_normalize(envs).ob_rms

                    eval_episode_rewards = []

                    obs = eval_envs.reset()
                    eval_recurrent_hidden_states = torch.zeros(args.num_processes,
                                                               actor_critic.recurrent_hidden_state_size, device=device)
                    eval_masks = torch.zeros(args.num_processes, 1, device=device)

                    while len(eval_episode_rewards) < 10:
                        with torch.no_grad():
                            _, action, _, eval_recurrent_hidden_states = actor_critic.act(
                                obs, eval_recurrent_hidden_states, eval_masks, deterministic=True)

                        # Obser reward and next obs
                        obs, reward, done, infos = eval_envs.step(action)

                        eval_masks = torch.tensor([[0.0] if done_ else [1.0]
                                                   for done_ in done],
                                                  dtype=torch.float32,
                                                  device=device)

                        for info in infos:
                            if 'episode' in info.keys():
                                eval_episode_rewards.append(info['episode']['r'])

                    eval_envs.close()

                    print(" Evaluation using {} episodes: mean reward {:.5f}\n".
                          format(len(eval_episode_rewards),
                                 np.mean(eval_episode_rewards)))

            # -- number
        # -- chrono
        exp.show_eta(j, t)
    # -- epoch
    exp.report()


if __name__ == "__main__":
    main()
