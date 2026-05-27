import os
from tqdm import tqdm
import pprint
import torch
import numpy as np
from numpy.typing import NDArray

import gymnasium as gym
from gymnasium import ObservationWrapper, spaces
import imageio
import wandb
from copy import deepcopy

import gym_multigrid
from gym_multigrid.envs.prey_pred import (
    GreedyPredatorPolicy,
    GreedyPredatorActionOption,
    Prey,
    Predator,
)
from gym_multigrid.core.agent import NavigationActions
from gym_multigrid.core.object import WorldObj

from agents import *
from toroid import ToroidPredPrey
from sklearn.linear_model import LinearRegression
from dataclasses import dataclass

@dataclass
class Team:
    learner: IndSFDQNAgent
    partner1: GreedyPredatorPolicy
    pref1: NDArray
    partner2: GreedyPredatorPolicy
    pref2: NDArray

def get_dr(env, agents, num_preds, ac_dim, a_p):
    rewards = np.zeros(ac_dim)
    for a in range(ac_dim):
        all_actions = [a] + a_p
        obs_next, rew, terminated, truncated, info_next = copy.deepcopy(env).step(all_actions)
        rewards[a] = rew
    return np.mean(rewards)

def get_prey_actions(env, agents, num_preds):
    prey_actions = [
        ag.act(env.unwrapped._get_obs(), {}) for ag in agents[num_preds :]
    ]
    return prey_actions

def collect_traj(team, env_str, episodes):
    env = ToroidPredPrey(gym.make(env_str))
    X = []
    y = []
    for ep in tqdm(range(episodes), desc="agent-rollouts"):
        obs, info = env.reset()

        while True:
            # Create inputs for the greedy policies
            prey_agents = env.unwrapped.prey_agents
            grid = env.unwrapped.grid
            agent_1_option: GreedyPredatorActionOption = {
                "current_pos": env.unwrapped.agents[1].pos,
                "preys": prey_agents,
                "grid": grid,
            }
            agent_2_option: GreedyPredatorActionOption = {
                "current_pos": env.unwrapped.agents[2].pos,
                "preys": prey_agents,
                "grid": grid,
            }

            # Get actions from the greedy policies
            partner_actions = [
                team.partner1.act(agent_1_option),
                team.partner2.act(agent_2_option),
            ]
            prey_actions = get_prey_actions(env, env.unwrapped.agents, env.unwrapped.num_preds)
            p_actions = partner_actions + prey_actions

            dr = get_dr(env, env.unwrapped.agents, env.unwrapped.num_preds, env.unwrapped.ac_dim, p_actions)

            # step env with selected action
            action = team.learner.get_action(obs[0].flatten())
            actions = [action] + p_actions
            obs_next, rew, terminated, truncated, info_next = env.step(actions)
            y.append(np.sum(rew) - dr)
            X.append(team.learner.phi(obs[0], obs_next[0]))

            if terminated or truncated:
                break
            obs = obs_next
            info = info_next
    return X, y

def compute_w_dr():
    num_runs = 10
    model_dir = ""
    path_suffix = "learner_for_team2"
    env = ToroidPredPrey(gym.make("multigrid-preypred-v0"))
    episodes = 10
    lr = 3e-5
    gamma = 0.95
    w_team = np.array([1.0, 1.0, 1.0, 1.0])
    #partner1 = [1, 0, 1, 0]
    #partner2 = [0, 0, 1, 1]
    partner1 = [0, 1, 0, 1]
    partner2 = [0, 0, 1, 1]
    #partner1 = [0, 1, 1, 0]
    #partner2 = [0, 0, 1, 1]

    for i in range(num_runs):
        print("beginning replicate " + str(i))
        model_dir = model_dir #+ f"run{i}/"
        a1 = SFPartnerAgent(
            dirname=model_dir,
            filename=path_suffix,
            w=w_team
        )
        greedy_policy_1 = GreedyPredatorPolicy(partner1, random_generator=env.np_random)
        greedy_policy_2 = GreedyPredatorPolicy(partner2, random_generator=env.np_random)
        team = Team(learner=a1, 
                        partner1=greedy_policy_1, 
                        pref1=partner1, 
                        partner2=greedy_policy_2, 
                        pref2=partner2)
        
        print("computing weights for " + path_suffix)
        X, y = collect_traj(team=team, env_str="multigrid-preypred-v0", episodes=episodes)
        w = LinearRegression().fit(X, y)
        np.save(model_dir + path_suffix + '.npy', np.array(w.coef_))

if __name__ == "__main__":
    compute_w_dr()
