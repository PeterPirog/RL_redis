"""
https://realpython.com/python-redis/
https://gist.github.com/alexland/ce02d6ae5c8b63413843

https://github.com/RedisAI/redisai-examples
https://www.youtube.com/watch?v=_xubnCnHNgs

https://pypi.org/project/redisai/0.5.0/
"""

import redis
import redisai
import numpy as np
import tensorflow as tf

#r = redis.Redis(host='192.168.1.16', port=6380, db=0) #r = redis.Redis(host='localhost', port=6379, db=0)


#r.set('foo', 'bar')

#print(r.get('foo'))





#x=np.random.random(10)
#print(x)

#r.set('arr1', x)

#print(r.get('arr1'))


from redisai import Client
client = Client(host='192.168.1.16',port=6380,db=0)


arr = np.array([[2.1, 3.3]])

client.tensorset('x',arr)

arr_t=client.tensorget('x')
print(arr_t)

for i in range(10):
    arr=np.random.randn(4,2)
    client.tensorset(f'key{i}', arr)
print(client.keys())
print(client.tensorget('x2',arr))