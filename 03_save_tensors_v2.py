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

    def get_batch(self):
        input_dims=24

        self.observations = np.zeros(shape=(self.batch_size, input_dims))
        self.observations_ = np.zeros(shape=(self.batch_size, input_dims))
        self.actions = np.zeros(shape=(self.batch_size, self.n_actions))
        self.rewards = np.zeros(shape=(self.batch_size))
        self.dones = np.zeros(shape=(self.batch_size))

        # get random trajectories from redis server
        self.trajectories = self.client.srandmember('trajectories', self.batch_size)

        # iterate for single trajectories in set
        for i in range(self.batch_size):
            trajectory = self.trajectories[i]
            # print(trajectory)

            number = re.findall(r'\d+', str(trajectory, encoding='utf-8'))[0]
            # print('number=',number)
            sarsd = self.client.smembers(trajectory)
            # print(sarsd)
            reward, done = self.client.mget(f'reward{number}', f'done{number}')
            reward = float(reward)
            done = int(done)

            obs = np.array([self.client.tensorget(f'obs{number}')])
            obs_ = np.array([self.client.tensorget(f'obs_{number}')])
            action = np.array([tr_i.client.tensorget(f'action{number}')])
            # print(f'obs_= {obs_}, reward={reward}  done={done} ')

            self.dones[i] = done
            self.rewards[i] = reward
            self.observations[i,] = obs
            self.observations_[i,] = obs_
            self.actions[i,] = action

        self.observations = tf.convert_to_tensor(self.observations, dtype=tf.float32)
        self.observations_ = tf.convert_to_tensor(self.observations_, dtype=tf.float32)
        self.actions = tf.convert_to_tensor(self.actions, dtype=tf.float32)
        self.rewards = tf.convert_to_tensor(self.rewards, dtype=tf.float32)
        self.dones = tf.convert_to_tensor(self.dones, dtype=tf.int32)

        return self.observations, self.actions, self.rewards, self.observations_, self.dones


if __name__ == '__main__':


    batch_size = 50
    tr_i = TrainerInterface(host='192.168.1.16', port=6379, batch_size=5)

    delays=[]
    for i in range(100):
        print(f'Epoch:{i}')
        start = time.time()  # time count
        observations, actions, rewards, observations_, dones=tr_i.get_batch()
        delay = time.time()-start
        delays.append(delay)


    #print('observations=', observations)
    #print('actions=', actions)
    #print('observations_=', observations_)
    #print('rewards=', rewards)
    #print('dones=', dones)
    print('delays=',delays)
    print('time delay mean', np.mean(delays))
    print('time delay std', np.std(delays))
    print('\nMaximum time=',np.mean(delays)+2*np.std(delays))
    print('\nMaximum time for single trajectory:',(np.mean(delays)+2*np.std(delays))/batch_size)
