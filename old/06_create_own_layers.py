import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras.layers import Layer,Dense,Flatten
import numpy as np

"""
class SimpleDense(Layer):

  def __init__(self, units=32):
      super(SimpleDense, self).__init__()
      self.units = units

  def build(self, input_shape):  # Create the state of the layer (weights)
    w_init = tf.random_normal_initializer()
    self.w = tf.Variable(initial_value=w_init(shape=(input_shape[-1],self.units),dtype='float32'),
                         trainable=True)
    b_init = tf.zeros_initializer()
    self.b = tf.Variable(initial_value=b_init(shape=(self.units,), dtype='float32'),
                         trainable=True)
    print('input_shape=',input_shape[-1])
    print('self.w=',self.w)

  def call(self, inputs):  # Defines the computation from inputs to outputs
      return tf.matmul(inputs, self.w) + self.b
      """
################################################################
class FillTriangular(Layer):
    """
    FillTriangula layer converts vector to values of triangle matrix. The len of vector must be equal n*(n+1)/2 for n- integers
    """
    def __init__(self,upper=False):
        super().__init__()
        self.upper=upper

    def call(self, inputs):  # Defines the computation from inputs to outputs
        outputs=tf.reshape(inputs,[-1]) # [-1] for flatten
        outputs=tfp.math.fill_triangular(outputs,upper=self.upper)
        return outputs

class StateDependedSquareMatrix(Layer):
    """
    FillTriangula layer converts vector to values of triangle matrix. The len of vector must be equal n*(n+1)/2 for n- integers
    """
    def __init__(self,n_actions):
        super().__init__()
        self.n_actions=n_actions
        self.n_dense_units=self.n_actions*(self.n_actions+1)/2 #convert data to shape which makes convertion to traingle matrix possible

    def call(self, inputs):  # Defines the computation from inputs to outputs
        #dense_outputs=Dense(6).call(inputs) #,units=self.n_dense_units,activation=tf.keras.activations.elu,kernel_initializer=tf.keras.initializers.glorot_normal
        #print('dense_outputs=',dense_outputs)
        lower_matrix=FillTriangular().call(inputs)
        print('lower=',lower_matrix)
        square_matrix=tf.matmul(a=lower_matrix,b=lower_matrix,transpose_b=True)
        return square_matrix





# Instantiates the layer.
#linear_layer = FillTriangular(upper=True)
flatten_layer=Flatten()
dense_layer=Dense(6)
state_dependent_square_layer = StateDependedSquareMatrix(n_actions=3)


# This will also call `build(input_shape)` and create the weights.
#input=tf.ones((2,3))
input=np.random.randint(100,size=(1,2,4))
print('input=',input)

output=flatten_layer(input)
print('output flatten=',output)

output=dense_layer(output)
print('output dense=',output)


output = state_dependent_square_layer(output)
print('output square=',output)




#assert len(linear_layer.weights) == 2

# These weights are trainable, so they're listed in `trainable_weights`:
#assert len(linear_layer.trainable_weights) == 2
