import numpy as np
import gymnasium as gym
import random
import imageio
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from huggingface_hub import HfApi, snapshot_download
from huggingface_hub.repocard import metadata_eval_result, metadata_save
from pathlib import Path
import datetime
import json
import pickle

map_name = '4x4'
is_slippery = False

desc=["SFFF", "FHFH", "FFFH", "HFFG"]
# “S” for Start tile
# “G” for Goal tile
# “F” for frozen tile
# “H” for a tile with a hole

env = gym.make("FrozenLake-v1", map_name=map_name, is_slippery=is_slippery, desc=desc, render_mode="rgb_array")

def greedy(qtab, state):
    return np.argmax(qtab[state])

def epsilon_greedy(qtab, state, epsi):
    rand = np.random.uniform(0,1)
    if rand > epsi:
        action = np.argmax(qtab[state])
    else:
        action = env.action_space.sample()

    return action

def initialize_qtab(n_states, n_actions):
    """
    0: LEFT
    1: DOWN
    2: RIGHT
    3: UP
    """
    return np.zeros((n_states, n_actions))

def train(env, qtab, min_epsi=0.05, max_epsi=1, decay_epsi=5e-5, n_epochs=1e5, max_steps=100, lr=0.5, gamma=0.95):
    for e in tqdm(range(int(n_epochs))):
        truncated = False
        terminated = False
        state, info = env.reset()

        epsi = min_epsi + (max_epsi - min_epsi) * np.exp(-decay_epsi*e)

        for step in range(max_steps):
            action = epsilon_greedy(qtab, state, epsi)
            new_state, reward, terminated, truncated, info = env.step(action)

            qtab[state][action] = qtab[state][action] + \
                lr*(reward + gamma * np.max(qtab[new_state]) - qtab[state][action])
            
            if terminated or truncated:
                break

            state = new_state

    return qtab
    
def eval(env, qtab, max_steps=100, n_eval_epochs=100, seed=None):
    episode_rewards = []
    for ep in tqdm(range(n_eval_epochs)):
        if seed:
            state, info = env.reset(seed=seed[ep])
        else:
            state, info = env.reset()

        truncated = False
        terminated = False
        total_reward = 0

        for step in range(max_steps):
            action = greedy(qtab, state)
            state, reward, terminated, truncated, info = env.step(action)
            total_reward += reward

            if terminated or truncated:
                break

        episode_rewards.append(total_reward)

    mean = np.mean(episode_rewards)
    std = np.std(episode_rewards)

    return mean, std

def record_video(env, Qtable, out_directory, fps=2):
    images = []
    terminated = False
    truncated = False
    state, info = env.reset(seed=random.randint(0, 500))
    img = env.render()
    images.append(img)
    while not (terminated or truncated):
        action = np.argmax(Qtable[state])
        state, reward, terminated, truncated, info = env.step(
            action
        )
        img = env.render()
        images.append(img)
    imageio.mimsave(out_directory, [np.array(img) for img in (images)], fps=fps, loop=0)

qtab = initialize_qtab(env.observation_space.n, env.action_space.n)

qtab = train(env, qtab)

plt.figure(figsize=(6,12))
qtab_ = pd.DataFrame(qtab, columns=['left', 'down', 'right', 'up'])
sns.heatmap(qtab_, xticklabels=qtab_.columns)
plt.tight_layout()
plt.savefig('./img/qtab_frozenlake.png')

mean, std = eval(env, qtab)
print(f'Model evaluation mean score: {mean:.3f} std: {std:.3f}')
# Model evaluation mean score: 1.000 std: 0.000

record_video(env, qtab, './img/frozenlake_vid.gif')