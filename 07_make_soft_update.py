import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Sequential

shape=(4,)

model1 = Sequential()
model1.add(Dense(10, kernel_initializer=tf.initializers.glorot_normal, activation='relu', input_shape=(10,)))
model1.add(Dense(5, activation='relu',kernel_initializer=tf.initializers.glorot_normal))
model1.add(Dense(1, activation='sigmoid'))
model1.compile(optimizer='adam', loss='categorical_crossentropy')


model2 = Sequential()
model2.add(Dense(10, kernel_initializer=tf.initializers.glorot_normal, activation='relu', input_shape=(10,)))
model2.add(Dense(5, activation='relu',kernel_initializer=tf.initializers.glorot_normal))
model2.add(Dense(1, activation='sigmoid'))
model2.compile(optimizer='adam', loss='categorical_crossentropy')


network_trained=model1()#input(20,)
network_target=model2()
tau=0.01

w_trained=np.array(network_trained.get_weights())
w_target=np.array(network_target.get_weights())

#w_target=tau*w_trained+(1-tau)*w_target

mse = np.mean((w_target - w_trained)**2)
print('mse=',mse)
network_target.set_weights(w_target)


"""
a = np.array(model.get_weights())         # save weights in a np.array of np.arrays
model.set_weights(a + 1)                  # add 1 to all weights in the neural network
b = np.array(model.get_weights())         # save weights a second time in a np.array of np.arrays
print(a[2])                              # print changes in weights
"""