from pettingzoo.mpe import simple_tag_v2
from ray.tune.registry import register_env
from ray.rllib.env.wrappers.pettingzoo_env import PettingZooEnv

from ray import air, tune

if __name__ == "__main__":

    register_env('simple_tag_v2', lambda _: PettingZooEnv(simple_tag_v2.env()))

    tune.Tuner(
        "PPO",
        run_config=air.RunConfig(
            stop={'episodes_total': 60000},
            checkpoint_config=air.CheckpointConfig(
                checkpoint_frequency=10,
            ),
        ),
        param_space={
            'env': 'simple_tag_v2',
            'num_gpus': 0,
            'num_workers': 1,
            'num_envs_per_worker': 8,
            'rollout_fragment_length': 32,
            'train_batch_size': 512,
            'gamma': 0.99,
            # 'n_step': 3,
            'lr': 0.0001,
            # 'policies': {'shared_policy'},
            'multiagent':{
                'policies': {'adversary_0', 'adversary_1', 'adversary_2', 'agent_0'},
                'policy_mapping_fn': (
                lambda agent_id, episode, worker, **kwargs: agent_id
                ),
            }
            
        },
    ).fit()