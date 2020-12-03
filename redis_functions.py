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
        self.client.flushall() #delete all keys
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
        """
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

        """

        print('keys=',self.client.keys())
        #print(self.client.get('input_dims'))
        #print(self.client.smembers('sarsd'))
        #print(self.client.srandmember('sarsd', self.batch_size))
        #print(self.client.info())

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
        print('reward', self.client.get(f'reward{index}'))
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