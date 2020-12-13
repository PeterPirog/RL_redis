import numpy as np
import gym
import tensorflow as tf

import time
from redis_functions import TrainerInterface

if __name__ == '__main__':

    batch_size = 20
    tr_i = TrainerInterface(host='192.168.1.16', port=6379, db=0, batch_size=batch_size)
    """
    delays = []
    for i in range(2):
        print(f'Epoch:{i}')
        start = time.time()  # time count
        observations, actions, rewards, observations_, dones = tr_i.get_batch()
        delay = time.time() - start
        delays.append(delay)

        M = np.mean(delays)
        S = np.std(delays)
    
    print('observations=', observations)
    print('actions=', actions)
    print('observations_=', observations_)
    print('rewards=', rewards)
    print('dones=', dones)

    print('delays=', delays)
    print('time delay mean', M)
    print('time delay std', S)
    print('\nMaximum time=', M + 2 * S)
    print('\nMaximum time for single trajectory:', (M + 2 * S) / batch_size)
    """
    tr_i.save_trajectories_to_file()