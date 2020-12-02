import numpy as np
import gym
import tensorflow as tf
from redisai import Client
client = Client(host='192.168.1.16',port=6380,db=0)

def policy_pi(observation,n_actions=4):
    action=np.random.randint(2)
    #print('action=',action)
    return action


if __name__ == '__main__':

    env = gym.make('CartPole-v0')
    obs = env.reset()


    # action=pi(obs)

    totals = []
    done = False
    for episode in range(10):
        episode_rewards = 0
        obs = env.reset()
        env.render()

        for step in range(200):
            action = policy_pi(obs)
            obs, reward, done, info = env.step(action)

            client.tensorset('observation', obs)
            #print(client.tensorget('observation'))

            #client.tensorset('sars', [obs,action])
            client.lpush('mylist', obs)
            print(client.lpop('mylist'))

            env.render()
            episode_rewards += reward
            if done:
                break
        print(f'Episode reward {episode_rewards}')


#print(client.keys())