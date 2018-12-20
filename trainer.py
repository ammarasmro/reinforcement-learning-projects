import torch
import numpy as np

from dqn_agent import Agent
from collections import deque
from pathlib import Path

from unityagents import UnityEnvironment

env = UnityEnvironment(file_name='/data/Banana_Linux_NoVis/Banana.x86_64')

agent = Agent(seed=0)

def dqn(n_episodes=2000, max_t=1000, eps_start=1.0, eps_end=0.01, eps_decay=0.995):
    """Deep Q-Learning training function
    
    Params
    ======
        n_episodes (int): maximum number of training episodes
        max_t (int): maximum number of timesteps per episode
        eps_start (float): starting value of epsilon, for epsilon-greedy action selection
        eps_end (float): minimum value of epsilon
        eps_decay (float): multiplicative factor (per episode) for decreasing epsilon
    """  
    # get the default brain
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]
    
    scores = []                        # list containing scores from each episode
    statuses = []                      # list of logged statuses
    scores_deque = deque(maxlen=100)  # last 100 scores
    eps = eps_start                    # initialize epsilon
    for i_episode in range(1, n_episodes+1):
        env_info = env.reset(train_mode=True)[brain_name] # reset the environment
        score = 0
        for t in range(max_t):
            state = env_info.vector_observations[0]
            action = agent.act(state, eps)
            env_info = env.step(action)[brain_name]        # send the action to the environment
            next_state = env_info.vector_observations[0]   # get the next state
            reward = env_info.rewards[0]                   # get the reward
            done = env_info.local_done[0]                  # see if episode has finished
            agent.step(state, action, reward, next_state, done)
            state = next_state
            score += reward
            if done:
                break 
        scores_deque.append(score)       # save most recent score
        scores.append(score)              # save most recent score
        eps = max(eps_end, eps_decay*eps) # decrease epsilon
        status = (i_episode, np.mean(scores_deque), np.mean(score))
        statuses.append(status)
#         print(f'Episode {status[0]}\tAverage Score: {status[1]:.2f}\tScore: {status[2]:.2f}')
        if i_episode % 100 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_deque)))
        if np.mean(scores_deque) >= 13:
            print('\n==================================================================\n')
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode-100, np.mean(scores_deque)))
            torch.save(agent.qnetwork_local.state_dict(), 'checkpoint.pth')
            break
    
    with (Path('.') / ('results.csv')).open(mode='a+') as f:
        for score in scores:
            f.write(str(score))
            f.write('\n')

    with (Path('.') / ('statuses.csv')).open(mode='a+') as f:
        for status in statuses:
            f.write(f'{status[0]},{status[1]},{status[2]}')
            f.write('\n')
    return scores

def train():
    """Training function"""
    return dqn()

if __name__ == '__main__':
    scores = train()