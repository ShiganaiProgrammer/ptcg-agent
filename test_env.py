"""Test the CabtEnv wrapper with random actions."""
from env_wrapper import CabtEnv
from main import DECK
import numpy as np

env = CabtEnv(DECK, DECK.copy())

obs_dict = env.reset()
print(f"Obs shape: {obs_dict['obs'].shape}")
print(f"Obs: {obs_dict['obs']}")
print(f"Action mask sum: {obs_dict['action_mask'].sum()}")
print(f"Select type: {obs_dict['select_type']}, max count: {obs_dict['max_count']}")
print()

total_reward = 0.0
step = 0
done = False
while not done and step < 200:
    # Random valid action
    valid_actions = np.where(obs_dict["action_mask"])[0]
    if len(valid_actions) == 0:
        action = np.array([0])
    else:
        num_to_select = min(obs_dict["max_count"], len(valid_actions))
        action = np.random.choice(valid_actions, size=num_to_select, replace=False)
    
    obs_dict, reward, done, info = env.step(action)
    total_reward += reward
    step += 1
    
    if done:
        print(f"Episode done at step {step}, reward={reward}, total={total_reward}")
    elif step % 20 == 0:
        print(f"  Step {step}, select_type={obs_dict['select_type']}, options={obs_dict['action_mask'].sum()}")

env.close()
print(f"Total steps: {step}, total_reward: {total_reward}")
