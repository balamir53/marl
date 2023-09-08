# from ray.rllib.algorithms.ppo import PPOConfig
from pettingzoo.sisl import waterworld_v4
# from pettingzoo.mpe import simple_tag_v3
from ray.tune.registry import register_env
from ray.rllib.env.wrappers.pettingzoo_env import PettingZooEnv

from ray import air, tune

# check this
# https://stackoverflow.com/questions/74637712/how-do-you-use-openai-gym-wrappers-with-a-custom-gym-environment-in-ray-tune
# config = PPOConfig().environment('simple_tag_v3',render_env=True).rollouts(num_rollout_workers=2).framework('torch').training(model={'fcnet_hiddens':[64,64]}).evaluation(evaluation_num_workers=1)

# algo = config.build()

# for _ in range(5):
#     print(algo.train())

# algo.evaluate()

if __name__ == "__main__":

    # register_env('simple_tag_v3', lambda _: PettingZooEnv(simple_tag_v3.env()))
    register_env('waterworld', lambda _: PettingZooEnv(waterworld_v4.env()))

    tune.Tuner(
        "PPO",
        run_config=air.RunConfig(
            stop={'episodes_total': 60000},
            checkpoint_config=air.CheckpointConfig(
                checkpoint_frequency=10,
            ),
        ),
        param_space={
            'env': 'waterworld',
            'num_gpus': 0,
            'num_workers': 1,
            'num_envs_per_worker': 8,
            'rollout_fragment_length': 32,
            'train_batch_size': 512,
            'gamma': 0.99,
            'n_step': 3,
            'lr': 0.0001,
            'policies': {'shared_policy'},
            'policy_mapping_fn': (
                lambda agent_id, episode, worker, **kwargs: 'shared_policy'
            ),
        },
    ).fit()