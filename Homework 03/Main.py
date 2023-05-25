from gridworld import GridWorld
from sarsa import SARSA
from mccontrol import MCControl
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


def main():
    # env = GridWorld(winds=[(2, 2, 3, 0.3)], step_cost=-0.1)
    env = GridWorld(
        size=(5, 4),
        start=(0, 0),
        goal=(4, 0),
        walls=[()],
        winds=[(3, 0, 3, 0.99), (3, 1, 3, 0.99), (3, 2, 3, 0.99)],
        step_cost=-0.05,
    )

    # SARSA
    sarsa = SARSA(
        env,
        num_episodes=10,
        epsilon=0.01,
        alpha=0.1,
        gamma=0.99,
        plot="episode",
        load="half_trained-sarsa.pkl",
        save=None,
    )
    # sarsa.run()

    # MC Control
    mccontrol = MCControl(
        env,
        num_episodes=100,
        epsilon=0.01,
        gamma=0.8,
        plot="episode",
        load="half_trained-mccontrol.pkl",
        save=None,
    )
    # mccontrol.run()

    # Visualizing Variance-Bias Tradeoff
    q_sarsa = np.empty(shape=(10, 1000))
    q_mccontrol = np.empty(shape=(10, 1000))

    for i in tqdm(range(10)):
        sarsa = SARSA(
            env,
            num_episodes=1000,
            epsilon=0.01,
            alpha=0.1,
            gamma=0.99,
            plot=None,
            load="half_trained-sarsa.pkl",
            save=None,
        )
        q_sarsa[i] = sarsa.run()

        mccontrol = MCControl(
            env,
            num_episodes=1000,
            epsilon=0.01,
            gamma=0.8,
            plot=None,
            load="half_trained-mccontrol.pkl",
            save=None,
        )
        q_mccontrol[i] = mccontrol.run()

    mean_sarsa = np.mean(q_sarsa, axis=0)
    std_sarsa = np.std(q_sarsa, axis=0)

    mean_mccontrol = np.mean(q_mccontrol, axis=0)
    std_mccontrol = np.std(q_mccontrol, axis=0)

    # Plotting SARSA
    plt.plot(range(1000), mean_sarsa, label="SARSA")
    plt.fill_between(
        range(1000), mean_sarsa - std_sarsa, mean_sarsa + std_sarsa, alpha=0.2
    )

    # Plotting MC Control
    plt.plot(range(1000), mean_mccontrol, label="MC Control")
    plt.fill_between(
        range(1000),
        mean_mccontrol - std_mccontrol,
        mean_mccontrol + std_mccontrol,
        alpha=0.2,
    )

    plt.xlabel("Episodes")
    plt.ylabel("Q Values")
    plt.title("Mean and Standard Deviation of first Q Value over Episodes")
    plt.legend()
    plt.show()


if "__main__" == __name__:
    try:
        main()
    except KeyboardInterrupt:
        print("Keyboard Interrupted")
