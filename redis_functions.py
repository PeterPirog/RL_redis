import gym
from redisai import Client
import numpy as np
import tensorflow as tf
import sys
import time


class InfoInRedis:
    """
    This class read information about environment from redis database and show it
    """
    def __init__(self,host, port, db=0):
        '''
         Function gets from redis database information about parameters of gym environment
        :param host: ip of redis database host e.g. '192.168.1.16' or 'local'
        :param port: redis service port, check proper port forwarding if using docker images
        :param db: redis databse index, default 0, use other values if there are more databases in redis server
        :param get_info:
        '''
        self.host = host
        self.port = port
        self.db = db
        self.client = Client(host=self.host, port=self.port, db=self.db)

        # Environment data
        self.env_name = str(self.client.get('env_name'), encoding='utf-8')
        self.input_dims = self.client.tensorget('input_dims')  # need to improve

        self.n_actions = int(self.client.get('n_actions'))
        self.action_discrete = int(self.client.get('action_discrete'))
        self.max_action = float(self.client.get('max_action'))  # maximum value for action output
        self.min_action = float(self.client.get('min_action'))  # minimum value for action output

        #Process control data
        self.stop_collecting=int(self.client.get('stop_collecting')) #information for regulators if 1 stop regulator, 2 suspend regulator
        self.stop_training = int(self.client.get('stop_training'))  # information for trainers if 1 stop regulator, 2 suspend trainer
        self.mem_cntr = self.client.get('mem_cntr')
        self.mem_size = float(self.client.get('mem_size'))
        self.batch_size=int(self.client.get('batch_size'))

    def show_info(self):
        print('\n ---------------- INFO ---------------------------------')
        print(f'The environment: {self.env_name} has been created\n')
        print(' -------------- observations --------------------------')
        print(f'Input observation dimension: {self.input_dims}\n')
        print(' -------------- actions --------------------------')
        print(f'Number of actions: {self.n_actions}')
        print(f'Action discrete: {self.action_discrete}')
        print(f'Maximum action value: {self.max_action}')
        print(f'Minimum action value: {self.min_action}\n')

class RedisInitializer(InfoInRedis):
    '''
    RedisInitializeer  is connecting with redisai database in address = host in specified port and initiate gym environment
    '''
    def __init__(self,host,port,environment,db=0,mem_size=1000000,clean_all_keys=True,batch_size=10,stop_collecting=0,stop_training=1):

        #redis data
        self.host=host
        self.port=port
        self.db=db

        self.mem_size=mem_size
        self.client = Client(host=self.host, port=self.port, db=self.db)

        #Environment data
        self.env_name=environment
        self.env=gym.make(self.env_name)
        self.input_dims=np.asarray(self.env.reset().shape,dtype=np.int)

        self.n_actions=4#self.env.action_space.n
        self.action_discrete = 0 # 1 for discrete, 0 for continuous
        self.max_action=1 #maximum value for action output
        self.min_action = -1 #minimum value for action output

        #Prcess control data
        self.stop_collecting=stop_collecting #information for regulators if 1 stop data collecting
        self.stop_training= stop_training  # information for trainers, stopped if 1, running if 0, prevent start training before number of keys is smaller than batch size
        self.mem_cntr = 0
        self.batch_size=batch_size

        #Write data to redis base
        if clean_all_keys:
            self.client.flushall() #delete all keys    <<-------  delete all keys at the begining of the process

            #environmental data
        self.client.set('env_name',self.env_name)
        self.client.tensorset('input_dims', self.input_dims)
        self.client.set('n_actions', self.n_actions)
        self.client.set('max_action', self.max_action)
        self.client.set('min_action', self.min_action)
        self.client.set('action_discrete', self.action_discrete)
            #Process control data
        self.client.set('stop_collecting', self.stop_collecting)
        self.client.set('stop_training', self.stop_training)
        self.client.set('mem_size', self.mem_size)
        self.client.set('mem_cntr', self.mem_cntr)
        self.client.set('batch_size',self.batch_size)

class RegulatorInterface(InfoInRedis):
    def __init__(self,host,port,db=0):

        self.host=host
        self.port=port
        self.db=db

        super().__init__(host=self.host, port=self.port, db=self.db)
        #Show info about environment and process from redis database
        self.show_info()

    def storage_data(self,obs,action,reward,obs_,done):
        '''
        :param obs:  current state  - type -> numpy array
        :param action: action taken  - type -> numpy array
        :param reward:  reward from transition (obs,action) to obs_, type -> float value
        :param obs_: next state  - type -> numpy array
        :param done: value 1 for teminal state, value 0 for non-teminal state, type -> int value
        :return:
        '''
        #check if data should be collected
        self.stop_collecting=int(self.client.get('stop_collecting')) # 0 - collecting, 1 - stop, 2 - suspend

        if self.stop_collecting == 1: #regulator is stopped
            print(' Script has been stopped because "stop_collecting" flag in equal 1')
            sys.exit()

        elif self.stop_collecting == 2: #regulator is suspended
            print(' Script has been suspended because "stop_collecting" flag in equal 2, change flag value to 0 to resume')
            time.sleep(1)

        else: #regulator is running and collecting samples
            obs = np.array(obs, dtype=np.float)
            obs_ = np.array(obs_, dtype=np.float)
            action = np.array(action, dtype=np.float)
            done = int(done)

            self.mem_cntr = int(self.client.get('mem_cntr'))  # get free database index
            self.client.incr('mem_cntr')  # increment index to lock value in use
            index = int(
                self.mem_cntr % self.mem_size)  # if the counter is bigger than allocated buffer refill from oldest samples

            # store tensors into a redis database
            self.client.tensorset(f'obs{index}', obs)
            self.client.tensorset(f'obs_{index}', obs_)
            self.client.tensorset(f'action{index}', action)

            # store tfloat and int values into a redis database
            self.client.mset({f'reward{index}': reward,
                              f'done{index}': done})

class TrainerInterface(InfoInRedis):
    def __init__(self, host, port, db=0, batch_size=5):
        self.host=host
        self.port=port
        self.db=db

        #change batch size if changed in Trainer interface
        if batch_size is not self.batch_size:
            self.batch_size=batch_size
            self.client.set('batch_size',self.batch_size)

        super().__init__(host=self.host, port=self.port, db=self.db)
        #Show info about environment and process from redis database
        self.show_info()

    def __get_tensors(self,tensor_name,batch_indexes):
        #Function gets single tensors form redis database, stack them by rows into numpy array and converts to 2D tensorflow tensor
        tensors = np.stack([self.client.tensorget(f'{tensor_name}{batch_indexes[i]}') for i in range(len(batch_indexes))])
        tensors = tf.convert_to_tensor(tensors, dtype=tf.float32)
        return tensors

    def __get_multiple_keys(self,key_name,batch_indexes,tf_dtype=tf.float32):
        '''
        Function gets multiple keys from redis database, converts to numpy vector and next to tensorflow 1D tensor
        :param key_name:
        :param batch_indexes:
        :param tf_dtype:
        :return:
        '''
        values = self.client.mget([f'{key_name}{batch_indexes[i]}' for i in range(len(batch_indexes))])
        values = np.array(values, dtype=np.float)
        values = tf.convert_to_tensor(values, dtype=tf_dtype)
        return values

    def get_batch(self):
        #add waiting from batch
        self.mem_cntr = int(self.client.get('mem_cntr'))    #get current counter
        idx_max = int(np.min([self.mem_cntr, self.mem_size]))
        batch_indexes = np.random.choice(idx_max, self.batch_size, replace=False)  # draw indexes to get trajectories from redis database

        #get tensors from database
        self.observations=self.__get_tensors('obs',batch_indexes)
        self.observations_ = self.__get_tensors('obs_', batch_indexes)
        self.actions=self.__get_tensors('action', batch_indexes)

        #get multiple keys from database
        self.rewards=self.__get_multiple_keys('reward',batch_indexes)
        self.dones=self.__get_multiple_keys('done',batch_indexes,tf_dtype=tf.int32)

        return self.observations, self.actions, self.rewards, self.observations_, self.dones