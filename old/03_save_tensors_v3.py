import numpy as np
import gym
import tensorflow as tf
from redisai import Client
import re
import time


class TrainerInterface(Client):
    def __init__(self, host, port, db=0,batch_size=5):
        super().__init__()
        # redis data
        self.host = host
        self.port = port
        self.db = db
        self.client = Client(host=self.host, port=self.port, db=self.db)  # decode_responses=True

        # Environment data
        self.env_name = str(self.client.get('env_name'), encoding='utf-8')
        self.input_dims = self.client.tensorget('input_dims')  # need to improve

        #print('self.input_dims=', self.input_dims)

        self.n_actions = int(self.client.get('n_actions'))
        self.action_continous = int(self.client.get('action_continous'))
        self.max_action = float(self.client.get('max_action'))  # maximum value for action output
        self.min_action = float(self.client.get('min_action'))  # minimum value for action output

        # Prcess control data
        self.batch_size=batch_size
        self.stop_gathering = int(
            self.client.get('stop_gathering'))  # information for regulators if 1 stop data collecting
        self.mem_cntr = int(self.client.get('mem_cntr'))
        self.mem_size = int(self.client.get('mem_size'))


        #self.show_info()

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

    def get_batch(self):

        self.mem_cntr = int(self.client.get('mem_cntr'))
        idx_max=np.min([self.mem_cntr,self.mem_size])
        #print('idx_max=',idx_max)
        batch = np.random.choice(idx_max, self.batch_size, replace=False) #rand indexes to get from redis database
        #print('batch=',batch)


        #Get tensors with randomized indexes and stack them into batches
        self.observations=tf.stack([tf.Variable(self.client.tensorget(f'obs{batch[i]}'), dtype=tf.float32) for i in range(self.batch_size)],axis=0)
        self.observations_ = tf.stack([tf.Variable(self.client.tensorget(f'obs_{batch[i]}'), dtype=tf.float32) for i in range(self.batch_size)],axis=0)
        self.actions = tf.stack([tf.Variable(self.client.tensorget(f'action{batch[i]}'), dtype=tf.float32) for i in range(self.batch_size)],axis=0)



        #get rewards from database
        reward=self.client.mget([f'reward{batch[i]}' for i in range(self.batch_size)])
        self.rewards=np.array(reward,dtype=np.float)
        self.rewards=tf.convert_to_tensor(self.rewards,dtype=tf.float32)

        #get dones from database
        done=self.client.mget([f'done{batch[i]}' for i in range(self.batch_size)])
        self.dones=np.array(done,dtype=np.int)
        self.dones=tf.convert_to_tensor(self.dones,dtype=tf.int32)

        #print('self.observations=',self.observations)
        #print('self.observations_=', self.observations_)
        #print('self.actions=', self.actions)
        #print('rewards=',self.rewards)
        #print('dones=', self.dones)


        return self.observations, self.actions, self.rewards, self.observations_, self.dones


if __name__ == '__main__':


    batch_size = 5
    tr_i = TrainerInterface(host='192.168.1.16', port=6379, batch_size=5)

    delays=[]
    for i in range(100):
        print(f'Epoch:{i}')
        start = time.time()  # time count
        observations, actions, rewards, observations_, dones=tr_i.get_batch()
        delay = time.time()-start
        delays.append(delay)

        M=np.mean(delays)
        S=np.std(delays)
    #print('observations=', observations)
    #print('actions=', actions)
    #print('observations_=', observations_)
    #print('rewards=', rewards)
    #print('dones=', dones)
    print('delays=',delays)
    print('time delay mean', M)
    print('time delay std', S)
    print('\nMaximum time=',M+2*S)
    print('\nMaximum time for single trajectory:',(M+2*S)/batch_size)
