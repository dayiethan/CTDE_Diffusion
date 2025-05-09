# simulation.py
import time
import numpy as np
import matplotlib.pyplot as plt
from environment import HerdingEnvironment
from agents import ShepherdAgent, SheepAgent

def run_simulation(num_shepherds=2, num_sheep=10, total_time=20, dt=0.1, render=True):
    env = HerdingEnvironment(width=20, height=20, dt=dt)

    # Create shepherd agents with different (adjustable) policies if desired.
    for i in range(num_shepherds):
        # Here, for demonstration, they all go to a fixed target.
        # You can change the lambda to use more sophisticated policies.
        policy = lambda pos, sheep, env, target=np.array([18, 18]) : (target - pos) * 0.1
        shepherd = ShepherdAgent(position=[np.random.uniform(0, 2), np.random.uniform(0, 2)],
                                   policy_fn=policy)
        env.add_shepherd(shepherd)

    # Create sheep agents.
    for i in range(num_sheep):
        sheep = SheepAgent(position=[np.random.uniform(5, 15), np.random.uniform(5, 15)])
        env.add_sheep(sheep)

    env.reset_trajectories()

    num_steps = int(total_time / dt)

    plt.ion()
    fig, ax = plt.subplots(figsize=(8, 8))
    for step in range(num_steps):
        env.update()
        if render:
            env.render(ax)
        time.sleep(dt)

    # After simulation, you can save the trajectories
    env.save_trajectories("herding_trajectories.pkl")
    plt.ioff()
    plt.show()
    return env.trajectory_log

if __name__ == "__main__":
    run_simulation()
