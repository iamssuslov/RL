import gym
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback, ProgressBarCallback
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Tuple


class RewardLoggerCallback(BaseCallback):
    """
    Кастомный callback для сбора данных о вознаграждениях
    """

    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []
        self.current_episode_reward = 0
        self.current_episode_length = 0

    def _on_step(self) -> bool:
        reward = self.locals['rewards'][0] if isinstance(self.locals['rewards'], np.ndarray) else self.locals['rewards']
        self.current_episode_reward += reward
        self.current_episode_length += 1

        if self.locals['dones'][0] if isinstance(self.locals['dones'], np.ndarray) else self.locals['dones']:
            self.episode_rewards.append(self.current_episode_reward)
            self.episode_lengths.append(self.current_episode_length)
            self.current_episode_reward = 0
            self.current_episode_length = 0

        return True


def train_dqn(env: gym.Env, total_timesteps: int = 50000) -> Tuple[DQN, List[float], List[int]]:
    """
    Обучает модель DQN и возвращает модель, вознаграждения и длины эпизодов
    """
    # Инициализация callback для сбора данных
    reward_logger = RewardLoggerCallback()

    # Создание модели DQN
    model = DQN(
        "MlpPolicy",
        env,
        verbose=1,
        learning_starts=1000,
        batch_size=64,
        buffer_size=100000,
        learning_rate=1e-3,
        gamma=0.99,
        target_update_interval=500,
        tensorboard_log="./dqn_cartpole_tensorboard/",
    )

    # Обучение модели
    model.learn(
        total_timesteps=total_timesteps,
        callback=[reward_logger, ProgressBarCallback()],
    )

    return model, reward_logger.episode_rewards, reward_logger.episode_lengths


def plot_training_results(rewards: List[float], window_size: int = 100) -> None:
    """
    Визуализация результатов обучения
    """
    # Скользящее среднее для сглаживания графика
    moving_avg = np.convolve(rewards, np.ones(window_size) / window_size, mode='valid')

    plt.figure(figsize=(12, 6))

    # График вознаграждений за эпизоды
    plt.subplot(1, 2, 1)
    plt.plot(rewards, label='Награда за эпизод')
    plt.plot(range(window_size - 1, len(rewards)), moving_avg, label=f'Скользящее среднее ({window_size} эп.)',
             color='red')
    plt.xlabel('Номер эпизода')
    plt.ylabel('Награда')
    plt.title('Награды за эпизоды')
    plt.legend()
    plt.grid()

    # График распределения наград
    plt.subplot(1, 2, 2)
    plt.hist(rewards, bins=20, edgecolor='black')
    plt.xlabel('Награда')
    plt.ylabel('Частота')
    plt.title('Распределение наград')
    plt.grid()

    plt.tight_layout()
    plt.show()


def test_model(model: DQN, env: gym.Env, episodes: int = 5) -> List[float]:
    """
    Тестирование модели и возврат наград за эпизоды
    """
    test_rewards = []

    for episode in range(episodes):
        state, _ = env.reset()
        done = False
        total_reward = 0

        while not done:
            action, _ = model.predict(state, deterministic=True)
            state, reward, done, truncated, _ = env.step(action)
            total_reward += reward
            env.render()

        test_rewards.append(total_reward)
        print(f"Эпизод {episode + 1}, Общее вознаграждение: {total_reward}")

    return test_rewards


def main():
    # Создание среды
    env = gym.make("CartPole-v1")

    # Обучение модели
    print("Начинаем обучение...")
    model, training_rewards, _ = train_dqn(env)

    # Визуализация результатов обучения
    plot_training_results(training_rewards)

    # Тестирование модели
    print("\nТестирование модели...")
    test_env = gym.make("CartPole-v1", render_mode="human")
    test_rewards = test_model(model, test_env)
    test_env.close()

    # Вывод результатов тестирования
    print("\nРезультаты тестирования:")
    print(f"Среднее вознаграждение: {np.mean(test_rewards):.2f}")
    print(f"Максимальное вознаграждение: {np.max(test_rewards)}")
    print(f"Минимальное вознаграждение: {np.min(test_rewards)}")


if __name__ == "__main__":
    main()