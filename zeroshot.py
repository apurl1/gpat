import gymnasium as gym
from toroid import ToroidPredPrey
from gym_multigrid.envs.prey_pred import (
    GreedyPredatorPolicy,
    GreedyPredatorActionOption,
)
import torch
import numpy as np
from agents import *
#import tensorboard_reducer as tbr
from stats import *
from collections import Counter
import wandb
import os
import imageio

num_runs = 1
env = ToroidPredPrey(gym.make("multigrid-preypred-newteam-v0"))
lr = 3e-05
gamma = 0.95
w_team = np.array([1.0, 1.0, 1.0, 1.0])
cat = 'predprey'
cat_name = 'Predator Prey'
pi1_name = r"$\pi_1$"
pi1_pref = [[1, 0, 1, 0], [0, 0, 1, 1]]
pi2_name = r"$\pi_2$"
pi2_pref = [[0, 1, 0, 1], [0, 0, 1, 1]]
robust_name = "rob"
mesh_name = "mesh"
gpat_lin_name = "w-gpat"
pistar_name = r"$\pi^*$"
new_pref = [[0, 1, 1, 0], [0, 0, 1, 1]]
algs = [
        pi1_name, 
        pi2_name, 
        robust_name,
        mesh_name,
        gpat_lin_name,
        pistar_name
    ]
base_dir = "/Users/rupaln/Documents/uiuc/research/SFCollect/experiments/rlc/pred-prey/"

for i in range(num_runs):
    print("beginning replicate " + str(i))
    model_dir = base_dir + f"models/run{i}/"
    tb_dir = base_dir + f"dr-zeroshot/run{i}/"
    pi1 = SFPartnerAgent(
        dirname=model_dir,
        filename=f"learner_for_team1",
        w=w_team,
        #w=np.load(f"{model_dir}learner_for_team1.npy")
    )
    pi2 = SFPartnerAgent(
        dirname=model_dir,
        filename=f"learner_for_team2",
        w=w_team,
        #w=np.load(f"{model_dir}learner_for_team2.npy")
    )
    rob_agent = SFPartnerAgent(
        dirname=model_dir,
        filename=f"robust_learner",
        w=w_team
    )
    library = [
        f"learner_for_team1_psi.torch",
        f"learner_for_team2_psi.torch",
    ]
    pretrained = [torch.load(model_dir + l).eval() for l in library]
    mesh_agent = MeshAgent(
        pretrained=pretrained,
        state_dim=np.prod(env.observation_space.shape),
        action_dim=env.unwrapped.ac_dim,
        feat_dim=4,
        epsilon=0
    )
    w_gpat = SFGPIAgent(
        pretrained=pretrained,
        ws=[
            torch.from_numpy(np.load(f"{model_dir}learner_for_team1.npy")), 
            torch.from_numpy(np.load(f"{model_dir}learner_for_team2.npy"))
            ],
        learn_new=False,
        state_dim=np.prod(env.observation_space.shape),
        action_dim=env.unwrapped.ac_dim,
        feat_dim=4,
        epsilon=0,
        rate=0
    )
    pistar = SFPartnerAgent(
        dirname=model_dir,
        filename=f"learner_for_newteam",
        w=w_team,
        #w=np.load(f"{model_dir}learner_for_newteam.npy")
    )
    greedy_policy_1 = GreedyPredatorPolicy(new_pref[0], random_generator=env.np_random)
    greedy_policy_2 = GreedyPredatorPolicy(new_pref[1], random_generator=env.np_random)
    agent_tuples = [
                    # (pi1_name, pi1), 
                    # (pi2_name, pi2), 
                    # (robust_name, rob_agent),
                    (mesh_name, mesh_agent),
                    # (gpat_lin_name, w_gpat), 
                    # (pistar_name, pistar)
                    ]
    run = wandb.init(
        project="pred-prey1-mesh",
        config={
            "learning_rate": lr,
        }
    )
    
    for agent in agent_tuples:
        agent_name = agent[0]
        cur_agent =agent[1]
        print("running rollouts for " + agent_name)
        animation_path = model_dir + f"{agent_name}_with_new.gif"

        episodes = 1000
        rew_arr = []
        ret_arr = []
        frames = []
        for ep in range(episodes):
            ep_rew = 0.0
            ep_ret = 0.0
            obs, info = env.reset()
            timestep = 0
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
                action = cur_agent.get_action(obs[0].flatten(), action_epsilon=0.0)
                actions = [action] + [
                    greedy_policy_1.act(agent_1_option),
                    greedy_policy_2.act(agent_2_option),
                ]
                obs_next, reward, terminated, truncated, info_next = env.step(actions)
                ep_rew += reward
                ep_ret += gamma**timestep * reward
                if ep == episodes - 1:
                    frames.append(env.render())
                if terminated or truncated:
                    break
                obs = obs_next
                info = info_next
                timestep += 1
            run.log({f"{agent_name}/reward": ep_rew, f"{agent_name}/return": ep_ret, f"{agent_name}/episode": ep})
            cap_info = info["captured_preys"]
            for cp in cap_info:
                cp_key = cp["prey_id"]
                cp_pred = cp["captured_by"]
                run.log({f"{agent_name}/{cp_key}": cp_pred})
            rew_arr.append(ep_rew)
            ret_arr.append(ep_ret)
        print(agent_name, np.mean(np.array(rew_arr)), np.std(np.array(rew_arr)))
        print(agent_name, np.mean(np.array(ret_arr)), np.std(np.array(ret_arr)))
        os.makedirs(os.path.dirname(animation_path), exist_ok=True)
        imageio.mimsave(animation_path, frames, duration=2, loop=20)
        if agent_name == 'gpi' or agent_name == 'w-gpat' or agent_name == 'q-gpat':
            counter = Counter(cur_agent.pol_idx)
            output_file = model_dir + agent_name +f"_policy_usage.txt"
            with open(output_file, "w") as file:
                for item, count in counter.items():
                    file.write(f"{item}: {count}\n")
    run.finish()
