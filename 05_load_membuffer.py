import pickle

if __name__ == '__main__':

    with open('trajectories.pickle', 'rb') as f:
        observations, actions, rewards, observations_, dones = pickle.load(f)

        print(actions)