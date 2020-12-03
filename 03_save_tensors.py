import numpy as np
import gym
import tensorflow as tf
from redisai import Client
import re
import time


class TrainerInterface(Client):
    def __init__(self,host,port,db=0):
        super().__init__()
        #redis data
        self.host=host
        self.port=port
        self.db = db
        self.client = Client(host=self.host, port=self.port, db=self.db) #decode_responses=True

        # Environment data
        self.env_name=str(self.client.get('env_name'),encoding='utf-8')
        self.input_dims=self.client.tensorget('input_dims')
        self.n_actions=int(self.client.get('n_actions'))
        self.action_continous=int(self.client.get('action_continous'))
        self.max_action=float(self.client.get('max_action')) #maximum value for action output
        self.min_action = float(self.client.get('min_action'))#minimum value for action output

        #Prcess control data
        self.stop_gathering=int(self.client.get('stop_gathering')) #information for regulators if 1 stop data collecting
        self.mem_cntr = self.client.get('mem_cntr')
        self.mem_size = float(self.client.get('mem_size'))

        self.show_info()

    def show_info(self):
        print('\n ---------------- INFO ---------------------------------')
        print(f'The environment: {self.env_name} has been created\n')
        print(' -------------- observations --------------------------')
        print(f'Input observation dimension: {self.input_dims}\n')
        print(' -------------- actions --------------------------')
        print(f'Number of actions: {self.n_actions}')
        print(f'Action continous: {self.action_continous}')
        print(f'Maximum action value: {self.max_action}')
        print(f'Minimum action value: {self.min_action}')
        try:
            print(f'Action meanings: {self.env.unwrapped.get_action_meanings()}')
        except:
            print(f'Action meanings not decribed in env')

    def return_batch(self,batch_size):
        pass
        #return observations,actions,rewards,observations_,dones

if __name__=='__main__':
    tr_i=TrainerInterface(host='192.168.1.16',port=6379)


    batch_size=5
    start = time.time()   #time count
    observations=np.zeros(shape=(batch_size,24))
    observations_ = np.zeros(shape=(batch_size, 24))
    actions=np.zeros(shape=(batch_size, 4))
    rewards=np.zeros(shape=(batch_size))
    dones = np.zeros(shape=(batch_size))

    trajectories=tr_i.client.srandmember('trajectories',batch_size)
    for i in range(batch_size):
        trajectory=trajectories[i]
        #print(trajectory)

        number=re.findall(r'\d+', str(trajectory,encoding='utf-8'))[0]
        #print('number=',number)
        sarsd=tr_i.client.smembers(trajectory)
        #print(sarsd)
        reward,done=tr_i.client.mget(f'reward{number}',f'done{number}')
        #reward=float(reward)
        done=int(done)

        obs=np.array([tr_i.client.tensorget(f'obs{number}')])
        obs_ = np.array([tr_i.client.tensorget(f'obs_{number}')])
        action = np.array([tr_i.client.tensorget(f'action{number}')])
        #print(f'obs_= {obs_}, reward={reward}  done={done} ')

        dones[i]=done
        rewards[i]=reward
        observations[i,]=obs
        observations_[i,] = obs_
        actions[i,]=action

    observations=tf.convert_to_tensor(observations,dtype=tf.float32)
    observations_ = tf.convert_to_tensor(observations_,dtype=tf.float32)
    actions = tf.convert_to_tensor(actions,dtype=tf.float32)
    rewards = tf.convert_to_tensor(rewards,dtype=tf.float32)
    dones = tf.convert_to_tensor(dones,dtype=tf.int32)

    end = time.time()

    print('dones=',dones)
    print('rewards=', rewards)
    print('observations=', observations)
    print('observations_=', observations_)
    print('actions=', actions)
    print('time delay',end-start)


    """
    batch1=tr_i.client.srandmember('trajectories',batch_size)
    print(batch1)

    batch2=tr_i.client.smembers('trajectory235216')
    print(batch2)

    batch3=tr_i.client.pipeline() mget(batch2)
    print(batch3)
    """
