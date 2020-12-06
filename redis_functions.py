import gym
from redisai import Client
import numpy as np
import tensorflow as tf

class Info():
    """
    This class read information about environment from redis database
    """
    def __init__(self,host, port, db=0):
        pass

    def get_env_info_from_redis(self,host, port, db=0,show_info=True):
        self.host = host
        self.port = port
        self.db = db
        self.client = Client(host=self.host, port=self.port, db=self.db)

        # Environment data
        self.env_name = str(self.client.get('env_name'), encoding='utf-8')
        self.input_dims = self.client.tensorget('input_dims')  # need to improve

        self.n_actions = int(self.client.get('n_actions'))
        self.action_continous = int(self.client.get('action_continous'))
        self.max_action = float(self.client.get('max_action'))  # maximum value for action output
        self.min_action = float(self.client.get('min_action'))  # minimum value for action output
        if show_info:
            print('\n ---------------- INFO ---------------------------------')
            print(f'The environment: {self.env_name} has been created\n')
            print(' -------------- observations --------------------------')
            print(f'Input observation dimension: {self.input_dims}\n')
            print(' -------------- actions --------------------------')
            print(f'Number of actions: {self.n_actions}')
            print(f'Action continous: {self.action_continous}')
            print(f'Maximum action value: {self.max_action}')
            print(f'Minimum action value: {self.min_action}\n')


class RedisInitializeer(Client):
    def __init__(self,host,port,environment,db=0,mem_size=1000000):
        super().__init__()
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
        self.action_continous = 1 # 1 for continous, 0 for discreete
        self.max_action=1 #maximum value for action output
        self.min_action = -1 #minimum value for action output


        #Prcess control data
        self.stop_gathering=0 #information for regulators if 1 stop data collecting
        self.mem_cntr = 0
        self.batch_size=3


        #Write data to redis base
        #self.client.flushall() #delete all keys    <<-------  delete all keys at the begining of the process
            #environmental data
        self.client.set('env_name',self.env_name)
        self.client.tensorset('input_dims', self.input_dims)
        self.client.set('n_actions', self.n_actions)
        self.client.set('max_action', self.max_action)
        self.client.set('min_action', self.min_action)
        self.client.set('action_continous', self.action_continous)
            #Process control data
        self.client.set('stop_gathering', self.stop_gathering)
        self.client.set('mem_size', self.mem_size)
        self.client.set('mem_cntr', self.mem_cntr)



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

class RegulatorInterface(Client):
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

    def storage_data(self,obs,action,reward,obs_,done):
        obs=np.array(obs,dtype=np.float)
        obs_ = np.array(obs_, dtype=np.float)
        action=np.array(action, dtype=np.float)
        done=int(done)

        self.mem_cntr = int(self.client.get('mem_cntr')) #get free database index
        self.client.incr('mem_cntr')  #increment index to lock value in use
        index=int(self.mem_cntr % self.mem_size)  #if the counter is bigger than allocated buffer refill from oldest samples

        self.client.tensorset(f'obs{index}', obs)
        self.client.tensorset(f'obs_{index}', obs_)
        self.client.tensorset(f'action{index}', action)
        #self.client.set(f'reward{index}', reward)
        #self.client.set(f'done{index}', done)


        self.client.mset({f'reward{index}': reward,
                          f'done{index}': done})
        #combine sarsd into one trajectory
        self.client.sadd(f'trajectory{index}',f'obs{index}',f'obs_{index}',f'action{index}',f'reward{index}',f'done{index}')
        #and add this n-th trajectory to all trajectories
        self.client.sadd('trajectories',f'trajectory{index}')


        #check if data should be collected
        self.stop_gathering=int(self.client.get('stop_gathering'))
        if self.stop_gathering:
            print('Stop_gathering flag has been set. Finishing episode')

        #self.client.sadd(f'trajectory{index}', f'obs{index}')

        #print('obs=',obs)
        #print('obs_=', obs_)
        #print('action=', action)
        #print('reward', self.client.get(f'reward{index}'))
        #print('done', done)
        #print('index=',index,'mem_cnt=',self.mem_cntr)
        #print(f'trajectory{index}',self.smembers(f'trajectory{index}'))
        pass

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

class TrainerInterface(Info):
    def __init__(self, host, port, db=0, batch_size=5):
        super().__init__(host, port)
        self.get_env_info_from_redis(host, port)

        # Prcess control data
        self.batch_size = batch_size
        self.stop_gathering = int(
            self.client.get('stop_gathering'))  # information for regulators if 1 stop data collecting
        self.mem_cntr = int(self.client.get('mem_cntr'))
        self.mem_size = int(self.client.get('mem_size'))

    def __get_tensors(self,tensor_name,batch_indexes):
        #Function gets single tensors form redis database, stack them by rows into numpy array and converts to 2D tensorflow tensor
        tensors = np.stack([self.client.tensorget(f'{tensor_name}{batch_indexes[i]}') for i in range(len(batch_indexes))])
        tensors = tf.convert_to_tensor(tensors, dtype=tf.float32)
        return tensors

    def __get_multiple_keys(self,key_name,batch_indexes,tf_dtype=tf.float32):
        # Function gets multiple keys from redis database, converts to numpy vector and next to tensorflow 1D tensor
        values = self.client.mget([f'{key_name}{batch_indexes[i]}' for i in range(len(batch_indexes))])
        values = np.array(values, dtype=np.float)
        values = tf.convert_to_tensor(values, dtype=tf_dtype)
        return values

    def get_batch(self):
        self.mem_cntr = int(self.client.get('mem_cntr'))    #get current counter
        idx_max = np.min([self.mem_cntr, self.mem_size])
        batch_indexes = np.random.choice(idx_max, self.batch_size, replace=False)  # draw indexes to get trajectories from redis database

        #get tensors from database
        self.observations=self.__get_tensors('obs',batch_indexes)
        self.observations_ = self.__get_tensors('obs_', batch_indexes)
        self.actions=self.__get_tensors('action', batch_indexes)

        #get multiple keys from database
        self.rewards=self.__get_multiple_keys('reward',batch_indexes)
        self.dones=self.__get_multiple_keys('done',batch_indexes,tf_dtype=tf.int32)

        return self.observations, self.actions, self.rewards, self.observations_, self.dones