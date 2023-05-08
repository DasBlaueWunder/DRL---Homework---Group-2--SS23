from gridworld import GridWorld


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
    print("\n" + "Initial state, representation of the environment")
    print("S: start, G: goal, W: wall, ^>v<: wind direction \n")
    env.render()

    # Example run (with visualization)
    # env.main_loop()
    env.first_visit_mc_control(num_iterations=100, epsilon=0.1, plot="iteration")


if "__main__" == __name__:
    try:
        main()
    except KeyboardInterrupt:
        print("Keyboard Interrupted")
