import gym
from redis_functions import RegulatorInterface
import numpy as np


class PolicyPi():
    def __init__(self,regulator_interface,random=True):
        self.reg_i=regulator_interface
        self.action_discrete=self.reg_i.action_discrete
        self.n_actions=self.reg_i.n_actions
        self.random=random

    def __call__(self, observation):
        if self.random:  #initiate random regulator
            if not self.action_discrete:  #for continous environments
                action=np.random.uniform(low=self.reg_i.min_action,high=self.reg_i.max_action,size=(self.n_actions))
            else: #for discreete environments
                action=np.random.randint(2)

        return action

if __name__=='__main__':

    reg_i=RegulatorInterface(host='192.168.1.16',port=6379)
    pi=PolicyPi(regulator_interface=reg_i,random=True)

    env=gym.make(reg_i.env_name)

    totals=[]
    done=False
    episode=0
    while 1:
        episode_rewards=0
        obs=env.reset()
        for step in range(1601):
            action=pi(obs)
            obs,reward,done, info=env.step(action)
            obs_=obs
            reg_i.storage_data(obs,action,reward,obs_,done)
            episode_rewards+=reward
            if done:
                episode += 1
                break
        print(f'Episode reward {episode_rewards}, episode nr {episode}, mem_cnt {reg_i.mem_cntr}')