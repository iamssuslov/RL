import gym
import time

env = gym.make("CartPole-v1", render_mode="human")
state, _ = env.reset()
total_reward = 0

for _ in range(200):
    action = env.action_space.sample()
    state, reward, terminated, truncated, info = env.step(action)
    total_reward += reward
    time.sleep(0.05)
    if terminated or truncated:
        break

print(f"Total reward: {total_reward}")
env.close()