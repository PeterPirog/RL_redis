#https://github.com/anyscale/academy/blob/master/ray-rllib/02-Introduction-to-RLlib.ipynb
from ray import tune
from ray.rllib.agents.ppo import PPOTrainer
tune.run(PPOTrainer,
    config={"env": "CartPole-v1"},
    stop={"training_iteration": 20},
    checkpoint_at_end=True,
    verbose=2            # 2 for INFO; change to 1 or 0 to reduce the output.
    )