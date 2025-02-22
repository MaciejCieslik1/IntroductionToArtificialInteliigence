import numpy as np
import random
import time
import gymnasium as gym
import matplotlib.pyplot as plt

class QLearningSolver:
    """Class containing the Q-learning algorithm that might be used for different discrete environments."""

    def __init__(
        self,
        observation_space: int,
        action_space: int,
        learning_rate: float = 0.1,
        gamma: float = 0.9,
        epsilon: float = 0.1,
    ):
        self.observation_space = observation_space
        self.action_space = action_space
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.Q_table = np.zeros((observation_space, action_space))

    def __call__(self, state: int, action: int) -> float:
        """Return Q-value of given state and action."""
        return self.Q_table[state][action]

    def update(self, state: int, action: int, reward: float, next_state: int) -> None:
        """Update Q-value of given state and action."""
        self.Q_table[state, action] += self.learning_rate * (reward + self.gamma * np.max(self.Q_table[next_state]) - self(state, action))

    def get_best_action(self, state: int) -> int:
        """Return action that maximizes Q-value for a given state."""
        return np.argmax(self.Q_table[state])

    def __repr__(self):
        """Elegant representation of Q-learning solver."""
        return f"QLearningSolver(observation_space={self.observation_space}, action_space={self.action_space}, Q_table=\n{self.Q_table})"

    def __str__(self):
        return self.__repr__()

def plot_average_rewards(average_rewards: list[float], episodes: list[int], learning_rate: float, gamma: float, epsilon: float):
    """Generate a plot showing the average rewards during training."""
    plt.plot(episodes, average_rewards, label="Average award sum", color="blue")
    plt.xlabel("Episodes")
    plt.ylabel("Average award sum")
    plt.title(f"Agent's progress during training, LR: {learning_rate}, gamma: {gamma}, epsilon: {epsilon}")
    plt.grid()
    plt.legend()
    plt.savefig("average_rewards_plot.png")
    plt.show()

def plot_average_steps(steps: list[int], episodes: list[int], learning_rate: float, gamma: float, epsilon: float):
    """Generate a plot showing the average steps number during training."""
    plt.plot(episodes, steps, label="Average steps number", color="blue")
    plt.xlabel("Episodes")
    plt.ylabel("Average steps")
    plt.title(f"Agent's progress during training, LR: {learning_rate}, gamma: {gamma}, epsilon: {epsilon}")
    plt.grid()
    plt.legend()
    plt.savefig("average_steps_plot.png")
    plt.show()

def main():
    STATES = 500
    ACTIONS = 6
    LEARNING_RATE = 0.9
    GAMMA = 0.9
    EPSILON = 0.1
    EPOCHS = 5000
    MOVING_AVG_WINDOW = 100 

    env = gym.make("Taxi-v3", render_mode="ansi")
    q_learning_solver = QLearningSolver(STATES, ACTIONS, LEARNING_RATE, GAMMA, EPSILON)

    rewards_per_episode = []
    average_rewards = []
    episodes = []
    steps = []

    for episode in range(EPOCHS):
        state, _ = env.reset()
        total_reward = 0
        done = False
        steps_in_episode = 0

        while not done:
            if random.random() < q_learning_solver.epsilon:
                action = env.action_space.sample()
            else:
                action = q_learning_solver.get_best_action(state)
            next_state, reward, done, _, _ = env.step(action)
            q_learning_solver.update(state, action, reward, next_state)
            state = next_state
            total_reward += reward
            steps_in_episode += 1

        rewards_per_episode.append(total_reward)

        if episode >= MOVING_AVG_WINDOW:
            avg_reward = np.mean(rewards_per_episode[-MOVING_AVG_WINDOW:])
            average_rewards.append(avg_reward)
            episodes.append(episode)
            steps.append(steps_in_episode)
    plot_average_rewards(average_rewards, episodes, LEARNING_RATE, GAMMA, EPSILON)
    plot_average_steps(steps, episodes, LEARNING_RATE, GAMMA, EPSILON)

    state, _ = env.reset()
    done = False
    total_reward = 0
    while not done:
        action = q_learning_solver.get_best_action(state)
        state, reward, done, _, _ = env.step(action)
        total_reward += reward
        time.sleep(0.1)
        print(f"State: {state}, Final_state: {done}, Reward: {reward}")
    print(f"Total reward: {total_reward}")

main()
