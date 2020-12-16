import pickle
import numpy as np

if __name__ == '__main__':
    """
    with open('trajectories.pickle', 'rb') as f:
        observations, actions, rewards, observations_, dones = pickle.load(f)

        print(actions)
        
    """
    batches=2
    state_shape=5

    obs=np.random.randn(batches,5)
    action=np.random.randn(batches,3)
    print('obs=',obs)
    print('action=', action)