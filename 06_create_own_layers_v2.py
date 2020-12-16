#Implementation of layer of P network form publication Continuous Deep Q-Learning with Model-based Acceleration
#https://arxiv.org/abs/1603.00748

import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras.layers import Layer,Dense,Flatten
import numpy as np


class StateDependedSquareMatrix(Layer):
    """
    This is the last layer of P(state) network which returns matrix P(state)=L(state)*L(state).T where L(state) is low triangle matrix, P(state) shape is [n_actions x n_actions]
    """
    def __init__(self,n_actions):
        super().__init__()
        self.n_actions=n_actions
        self.n_dense_units=self.n_actions*(self.n_actions+1)/2 #convert data to shape which makes convertion to traingle matrix possible

        #define layers
        self.flatten_layer=Flatten(dtype=tf.float32)
        self.dense_layer=Dense(self.n_dense_units,activation=tf.keras.activations.elu,kernel_initializer=tf.keras.initializers.glorot_normal,dtype=tf.float32)

    def call(self, inputs):  # Defines the computation from inputs to outputs
        outputs=self.flatten_layer(inputs)
        outputs=self.dense_layer(outputs)
        low_triangular_array = tfp.math.fill_triangular(outputs, upper=False)
        outputs=tf.matmul(a=low_triangular_array, b=low_triangular_array, transpose_b=True)

        return outputs

if __name__ == '__main__':

    #construct state_dependent_square_layer
    P_layer = StateDependedSquareMatrix(n_actions=3)

    #input=np.random.randint(100,size=(2,3,4))
    input=np.random.rand(2,3,4)

    print('input=',input)

    output = P_layer(input)
    print('output square=',output)



