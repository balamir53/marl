from pettingzoo.butterfly   import pistonball_v6

env = pistonball_v6.env(render_mode='human')

env.reset()
for agent in env.agent_iter():
    obs, rew, done, trunc, info = env.last()
    if done or trunc:
        action = None
    else:
        action = env.action_space(agent).sample()
    env.step(action)
env.close()