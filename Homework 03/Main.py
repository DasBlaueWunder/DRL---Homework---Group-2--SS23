from gridworld import GridWorld
from sarsa import SARSA
from mccontrol import MCControl


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
        env, num_episodes=1000, epsilon=1, alpha=0.1, gamma=0.99, plot="wallclock"
    )
    # sarsa.run()

    # MC Control
    mccontrol = MCControl(
        env, num_episodes=5000, epsilon=0.1, gamma=0.8, plot="episode"
    )
    mccontrol.run()


if "__main__" == __name__:
    try:
        main()
    except KeyboardInterrupt:
        print("Keyboard Interrupted")
