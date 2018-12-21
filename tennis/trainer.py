import torch
import numpy as np
import matplotlib.pyplot as plt

from collections import deque
from pathlib import Path

from unityagents import UnityEnvironment

env = UnityEnvironment(file_name='/data/Tennis_Linux_NoVis/Tennis')

from ddpg_agent import Agent

# get the default brain
brain_name = env.brain_names[0]
brain = env.brains[brain_name]

agent = Agent(state_size=24, action_size=2, random_seed=0)

def ddpg(n_episodes=3000, max_t=700):
    scores_deque = deque(maxlen=100)
    scores = []
    statuses = []
    for i_episode in range(1, n_episodes+1):
        env_info = env.reset(train_mode=True)[brain_name]      # reset the environment
        agent.reset()
        state = env_info.vector_observations                  # get the current state (for each agent)

        score = np.zeros(2)
        for t in range(max_t):
            action = agent.act(state)
            env_info = env.step(action)[brain_name]
            next_state = env_info.vector_observations         # get next state (for each agent)
            reward = env_info.rewards                         # get reward (for each agent)
            done = env_info.local_done                        # see if episode finished
            agent.step(state, action, reward, next_state, done)
            state = next_state
            score += reward
            if np.any(done):
                break
        scores_deque.append(np.mean(score))
        scores.append(np.mean(score))
        status = (i_episode, np.mean(scores_deque), np.mean(score))
        statuses.append(status)
        print(f'Episode {status[0]}\tAverage Score: {status[1]:.2f}\tScore: {status[2]:.2f}')
        if i_episode >= 100 and np.mean(scores_deque) >= 0.5:
            torch.save(agent.actor_local.state_dict(), 'checkpoint_actor.pth')
            torch.save(agent.critic_local.state_dict(), 'checkpoint_critic.pth')
            print('Episode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_deque)))
            print(f'Solved in {i_episode+1} episodes!')
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
    return ddpg()

if __name__ == '__main__':
    scores = train()
