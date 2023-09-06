from ray.rllib.algorithms.ppo import PPOConfig
from pettingzoo.mpe import simple_tag_v3
from ray.tune.registry import register_env

env = simple_tag_v3.env()

def env_creator(hele):
    return env

register_env('simple_tag_v3', env_creator)

# check this
# https://stackoverflow.com/questions/74637712/how-do-you-use-openai-gym-wrappers-with-a-custom-gym-environment-in-ray-tune
config = PPOConfig().environment('simple_tag_v3',render_env=True).rollouts(num_rollout_workers=2).framework('torch').training(model={'fcnet_hiddens':[64,64]}).evaluation(evaluation_num_workers=1)

algo = config.build()

for _ in range(5):
    print(algo.train())

algo.evaluate()