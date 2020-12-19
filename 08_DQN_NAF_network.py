# adding summary to subclassing models https://stackoverflow.com/questions/55235212/model-summary-cant-print-output-shape-while-using-subclass-model
import tensorflow as tf
from tensorflow import keras
import numpy as np


class NetworkUtils:
    def list_to_tuple(self, list_variable):
        '''

        :param list_variable: list ot tuple variable
        :return: variable ocnverted to tuple
        '''
        if type(list_variable) is list:
            return tuple(list_variable)
        elif type(list_variable) is float or type(list_variable) is int:
            return tuple([list_variable])
        else:
            return list_variable


class DqnNaf(tf.keras.Model, NetworkUtils):
    def __init__(self, state_shape, action_shape, fc1_dims=200, fc2_dims=100,lr=1e-4):
        super().__init__()
        """
        if type(action_shape) is list: #check if shape is list or tuple
            self.action_shape=tuple(action_shape)
        else:
            self.action_shape=action_shape
        """
        self.action_shape = self.list_to_tuple(action_shape)
        self.state_shape=self.list_to_tuple(state_shape)

        print('action_shape=', self.action_shape)
        print('states_shape=', self.state_shape)

        self.lr=lr
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims

        # define layers

        self.flat_states=keras.layers.Flatten() #layer to flatten states
        self.flat_actions=keras.layers.Flatten() #layer to flatten actions
        self.dense1=keras.layers.Dense(units=self.fc1_dims,kernel_initializer=keras.initializers.glorot_normal, activation=tf.nn.elu)
        self.dense2 = keras.layers.Dense(units=self.fc2_dims, kernel_initializer=keras.initializers.glorot_normal, activation=tf.nn.elu)
        self.V=keras.layers.Dense(units=1, kernel_initializer=keras.initializers.glorot_normal,activation=keras.activations.linear)

        self.compile(optimizer=tf.optimizers.Adam(learning_rate=self.lr),loss=tf.losses.mse)

    def build(self, input_shape):
        pass

    def call(self,inputs, training=False, mask=None):
        states,actions=inputs

        #flat inputs
        x = self.flat_states(states)
        if actions is not None: u=self.flat_actions(actions)
        else:
            u=None

        # define common dense layers
        x=self.dense1(x)
        dense_output=self.dense2(x)

        V=self.V(dense_output)


        return V

    #def summary(self, line_length=None, positions=None, print_fn=None):

        #self.summary()


if __name__ == '__main__':
    batch_size = 2

    #action_shape = [2, 4]
    action_shape = 3
    state_shape = [3, 4]

    #action_size = tuple(action_shape)  # conversion list to tuple bacause redis database storage lists not tuples

    #print('action_size=', *action_size)

    states = np.random.rand(batch_size, 3, 4)
    actions = np.random.rand(batch_size, 1, 4)

    print('states=', states)
    print('actions=', actions)
    naf_network = DqnNaf(state_shape=state_shape,action_shape=action_shape)

    inputs=[states,actions]
    out=naf_network(inputs)

    print('out=',out)
    naf_network.summary()