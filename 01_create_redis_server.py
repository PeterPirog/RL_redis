
"""
Windows server install https://medium.com/@binary10111010/redis-cli-installation-on-windows-684fb6b6ac6b

sudo docker run --name redis-server -p 6379:6379 -d redis
----->>>>>   sudo docker run --name redis-AI -p 6379:6379 -it --rm redisai/redisai


docker run --name redis-AI -p 6379:6379 --gpus all -it --rm redisai/redisai:latest-gpu

https://redis.io/topics/rediscli
"""

from redis_functions import RedisInitializeer

#ri=RedisInitializeer(host='192.168.1.16',port=6379,environment='CartPole-v0',mem_size=10)

ri=RedisInitializeer(host='192.168.1.16',port=6379,environment='BipedalWalker-v3',mem_size=1000000)
#ri=RedisInitializeer(host='192.168.1.16',port=6379,environment='MsPacman-v0',mem_size=10) #3D state shape