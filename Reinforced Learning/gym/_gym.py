import gym
import numpy as np
import random

env= gym.make("FrozenLake-v1")

q_table=np.zeros([env.observation_space.n, env.action_space.n])

gamma=0.95
epsilon=0.10
alpha=0.80


reward_list=[]

episode_number= 100000

for i in range(1,episode_number):

    state= env.reset()

    reward_count=0

    while True:

        if random.uniform(0,1) < epsilon:
            action=env.action_space.sample()

        else:
            action=np.argmax(q_table[state])

        next_state, reward, done, _ =env.step(action)

        # Q-learning
        old_value=q_table[state,action]
        next_max=np.max(q_table[next_state])
        next_value=(1-alpha)*old_value + alpha*(reward+ gamma*next_max)


        q_table[state,action]=next_value

        state=next_state

        reward_count+=reward

        if done:
            break

    if i%5000 ==0:
        print(f"Episdode : {i}")

    if i%1000==0:
        reward_list.append(reward_count)


















