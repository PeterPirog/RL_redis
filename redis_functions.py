import gym
from redisai import Client
import numpy as np

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
        self.input_dims=self.env.reset().shape
        self.n_actions=2 #str(self.env.action_space)
        self.max_action=1 #maximum value for action output
        self.min_action = -1 #minimum value for action output

        #Prcess control data
        self.stop_gathering=0 #information for regulators if 1 stop data collecting
        self.mem_cntr = 0
        self.batch_size=3


        #Write data to redis base
        self.client.flushall() #delete all keys
            #environmental data
        self.client.set('env_name',self.env_name)
        self.client.set('input_dims', *self.input_dims)
        self.client.set('n_actions', self.n_actions)
        self.client.set('max_action', self.max_action)
        self.client.set('min_action', self.min_action)
            #Process control data
        self.client.set('stop_gathering', self.stop_gathering)
        self.client.set('mem_size', self.mem_size)
        self.client.set('mem_cntr', self.mem_cntr)

        #make initial trajectories
        for i in range(self.mem_size):
            cnt = str(self.client.get('mem_cntr'))
            self.client.incr('mem_cntr')

            state = np.zeros((1,*self.input_dims))#np.random.random_sample((3, 2))
            state_ = np.zeros((1, *self.input_dims))
            action=np.zeros((1, self.n_actions))
            reward=0
            done=0

            self.client.tensorset(f'state{cnt}', state)
            self.client.tensorset(f'state_{cnt}', state_)
            self.client.tensorset(f'action{cnt}', action)
            self.client.set(f'reward{cnt}',reward)
            self.client.set(f'done{cnt}', done)


            #add sarsd to one trajectory
            self.client.sadd(f'trajectory{cnt}', f'state{cnt}')
            self.client.sadd(f'trajectory{cnt}', f'state_{cnt}')
            self.client.sadd(f'trajectory{cnt}', f'action{cnt}')
            self.client.sadd(f'trajectory{cnt}', f'reward{cnt}')
            self.client.sadd(f'trajectory{cnt}', f'done{cnt}')

            #add trajectory to the set of trajectories
            self.client.sadd(f'sarsd', f'trajectory{cnt}')



        print('keys=',self.client.keys())
        print(self.client.get('input_dims'))
        print(self.client.smembers('sarsd'))
        print(self.client.srandmember('sarsd', self.batch_size))
        #print(self.client.info())

class RegulatorInterface(Client):
    def __init__(self,host,port,db=0):
        super().__init__()
        #redis data
        self.host=host
        self.port=port
        self.db = db
        self.client = Client(host=self.host, port=self.port, db=self.db)

        # Environment data
        self.env_name=self.client.get('env_name')
        self.input_dims=self.client.get('input_dims')
        self.n_actions=self.client.get('n_actions')
        self.max_action=self.client.get('max_action') #maximum value for action output
        self.min_action = self.client.get('min_action') #minimum value for action output

        #Prcess control data
        self.stop_gathering=self.client.get('stop_gathering') #information for regulators if 1 stop data collecting
        self.mem_cntr = self.client.get('mem_cntr')

        print(self.client.info())
