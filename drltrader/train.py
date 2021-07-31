from torch.utils.tensorboard import SummaryWriter
import numpy as np
import logging
from pathlib import Path
from .util import check_and_create_folder

import os
import sys

class DRLAlgoTraderTrainer:
    def __init__(self,
                 name,
                 agent,
                 max_steps=100000,
                 train_env=None,
                 eval_env=None,
                 test_env=None,
                 train_max_episode_len=200,
                 eval_max_episode_len=int(1e6),
                 eval_episode_interval=1000,
                 eval_n_episodes=1,
                 outdir='result'
                 ):
        self.agent = agent
        self.max_steps = max_steps
        self.train_env = train_env
        self.train_max_episode_len = train_max_episode_len
        self.eval_episode_interval = eval_episode_interval
        self.eval_env = eval_env
        self.eval_n_episodes = eval_n_episodes
        self.eval_max_episode_len = eval_max_episode_len
        self.test_env = test_env
        self.outdir = os.path.join(outdir, name)
        # check_and_create_folder(self.outdir)
        self.writer = SummaryWriter(comment=name)
        
        
    def save_agent(self, step, reward, final=False):
        if final:
            folder_name = f'final_{step}_{reward}'
        else:
            folder_name = f'step_{step}_{reward}'
        folder_path = os.path.join(self.outdir, folder_name)
        check_and_create_folder(folder_path)
        self.agent.save(folder_path)

    def train_agent(self):
        test_R = None
        try:
            step = 0
            episode_i = 0
            eval_i = 0
            best_eval_reward = float('-inf')
            eval_tracker = 0
            while step < self.max_steps:
                obs = self.train_env.reset()
                R = 0  # return (sum of rewards)
                t = 0  # time step
                # for each episode
                while True:
                    step += 1
                    action = self.agent.act(obs)
                    obs, reward, done, _ = self.train_env.step(action)
                    R += reward
                    t += 1
                    reset = t == self.train_max_episode_len
                    self.agent.observe(obs, reward, done, reset)
                    if done or reset:
                        break
                self.writer.add_scalar('reward/episode', R, episode_i)
                episode_i += 1
                eval_tracker += 1
                # report
                if episode_i % 10 == 0:
                    logging.info(f'episode: {episode_i} R: {R}')
                    self.writer.add_scalar('train_reward/episode_i', R, episode_i)       
                if episode_i % 50 == 0:
                    logging.info(f'statistics: {self.agent.get_statistics()}')
                    for stat_tuple in self.agent.get_statistics():
                        name, info = stat_tuple
                        self.writer.add_scalar(name, info, episode_i)
                if eval_tracker >= self.eval_episode_interval:
                    avg_eval_reward = self.eval_agent()
                    self.writer.add_scalar('avg_eval_reward/eval_i', avg_eval_reward, eval_i)
                    eval_i += 1
                    if avg_eval_reward > best_eval_reward:
                        best_eval_reward = avg_eval_reward
                        self.save_agent(step, avg_eval_reward)
                    eval_tracker = 0
            logging.info('Finished training.')
            logging.info('Testing agent')
            test_R = self.test_agent()
            self.save_agent(step, test_R, final=True)
        except KeyboardInterrupt:
            if test_R is not None:
                self.save_agent(step, test_R)
            sys.exit(0)

    def eval_agent(self):
        rewards = []
        with self.agent.eval_mode():
            for i in range(self.eval_n_episodes):
                obs = self.eval_env.reset()
                R = 0
                t = 0
                while True:
                    action = self.agent.act(obs)
                    obs, r, done, _ = self.eval_env.step(action)
                    R += r
                    t += 1
                    reset = t == self.eval_max_episode_len
                    self.agent.observe(obs, r, done, reset)
                    if done or reset:
                        break
                logging.info(f'evaluation episode: {i} R: {R}')
                rewards.append(R)
        rewards = np.array(rewards)
        return rewards.mean()
    
    def test_agent(self):
        with self.agent.eval_mode():
            obs = self.test_env.reset()
            R = 0
            t = 0
            while True:
                action = self.agent.act(obs)
                obs, r, done, _ = self.test_env.step(action)
                R += r
                t += 1
                reset = t == self.eval_max_episode_len
                self.writer.add_scalar('test_reward/step', r, t)
                self.writer.add_scalar('test_cum_reward/step', R, t)
                self.agent.observe(obs, r, done, reset)
                if done or reset:
                    break
        logging.info(f'Test result - Reward: {R}')
        return R
    
    def test_agent_detail(self):
        with self.agent.eval_mode():
            obs = self.test_env.reset()
            R = 0
            t = 0
            rewards = []
            actions = []
            ts = []
            infos = {}
            while True:
                action = self.agent.act(obs)
                obs, r, done, info = self.test_env.step(action)
                R += r
                t += 1
                rewards.append(r)
                actions.append(action)
                ts.append(t)
                for key in info.keys():
                    if key not in infos:
                        infos[key] = [info[key]]
                    else:
                        infos[key].append(info[key])
                reset = t == self.eval_max_episode_len
                self.writer.add_scalar('test_reward/step', r, t)
                self.writer.add_scalar('test_cum_reward/step', R, t)
                self.agent.observe(obs, r, done, reset)
                if t % (6.5*60*5) == 0:
                    logging.info(f'Step: {t} Cum. Reward: {R:.4f}')
                if done or reset:
                    break
        logging.info(f'Test result - Reward: {R}')
        return rewards, actions, ts, infos
    
    def benchmark_BnH(self):
        _ = self.test_env.reset()
        R = 0
        t = 0
        rewards = []
        actions = []
        ts = []
        time = []
        action = 1 # Buy
        # first step is buy
        _, r, done, info = self.test_env.step(action)
        R += r
        t += 1
        rewards.append(r)
        actions.append(action)
        ts.append(t)
        time.append(info['time'])
        action = 0 # Hold
        self.writer.add_scalar('benchmark_reward/step', r, t)
        self.writer.add_scalar('benchmark_cum_reward/step', R, t)
        while True:
            _, r, done, info = self.test_env.step(action)
            R += r
            t += 1
            rewards.append(r)
            actions.append(action)
            ts.append(t)
            time.append(info['time'])
            self.writer.add_scalar('benchmark_reward/step', r, t)
            self.writer.add_scalar('benchmark_cum_reward/step', R, t)
            if t % (6.5*60*5) == 0:
                logging.info(f'Step: {t} Cum. Reward: {R:.4f}')
            if done:
                break
        logging.info(f'Test result - Reward: {R}')
        return rewards, actions, ts, time
    
    def benchmark_BnH_multi(self, size, version):
        _ = self.test_env.reset()
        R = 0
        t = 0
        rewards = []
        actions = []
        ts = []
        if version == 1:
            action = np.array([0] + [1.0/size]*size)
        elif version == 2:
            action = np.array([1.0/size]*size)
        else:
            raise ValueError
        while True:
            _, r, done, _ = self.test_env.step(action)
            R += r
            t += 1
            rewards.append(r)
            actions.append(action)
            ts.append(t)
            self.writer.add_scalar('benchmark_reward/step', r, t)
            self.writer.add_scalar('benchmark_cum_reward/step', R, t)
            if t % (6.5*60*5) == 0:
                logging.info(f'Step: {t} Cum. Reward: {R:.4f}')
            if done:
                break
        logging.info(f'Test result - Reward: {R}')
        return rewards, actions, ts

        


