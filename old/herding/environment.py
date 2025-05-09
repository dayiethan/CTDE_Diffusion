# environment.py
import numpy as np
import pickle
import matplotlib.pyplot as plt

class HerdingEnvironment:
    def __init__(self, width=20, height=20, dt=0.1):
        """
        width, height: size of the 2D world.
        dt: time step for the simulation.
        """
        self.width = width
        self.height = height
        self.dt = dt

        # Lists to hold agents (shepherds and sheep)
        self.shepherds = []
        self.sheep = []

        # A list to store trajectories for recording (each element is a snapshot of positions)
        self.trajectory_log = {"shepherds": [], "sheep": []}

    def add_shepherd(self, shepherd):
        self.shepherds.append(shepherd)

    def add_sheep(self, sheep_agent):
        self.sheep.append(sheep_agent)

    def reset_trajectories(self):
        self.trajectory_log = {"shepherds": [], "sheep": []}

    def record(self):
        # Record current positions of all agents as a snapshot
        shepherd_positions = np.array([s.position for s in self.shepherds])
        sheep_positions = np.array([s.position for s in self.sheep])
        self.trajectory_log["shepherds"].append(shepherd_positions)
        self.trajectory_log["sheep"].append(sheep_positions)

    def update(self):
        # First update the sheep based on their own policy.
        for s in self.sheep:
            s.update(self.dt, self, self.sheep, self.shepherds)

        # Then update shepherds; their policy could depend on the current sheep distribution.
        for d in self.shepherds:
            d.update(self.dt, self, self.sheep, self.shepherds)
            
        self.record()

    def render(self, ax=None):
        if ax is None:
            plt.figure(figsize=(8,8))
            ax = plt.gca()
        ax.cla()
        # Set limits (you may change these to your liking)
        ax.set_xlim(0, self.width)
        ax.set_ylim(0, self.height)
        # Plot sheep as red circles
        sheep_positions = np.array([s.position for s in self.sheep])
        if sheep_positions.size > 0:
            ax.scatter(sheep_positions[:, 0], sheep_positions[:, 1], c="red", label="Sheep")
        # Plot shepherds as blue squares
        shepherd_positions = np.array([d.position for d in self.shepherds])
        if shepherd_positions.size > 0:
            ax.scatter(shepherd_positions[:, 0], shepherd_positions[:, 1], c="blue", marker="s", label="Shepherd")
        ax.legend()
        plt.pause(0.001)

    def save_trajectories(self, filename="trajectories.pkl"):
        with open(filename, "wb") as f:
            pickle.dump(self.trajectory_log, f)

    def load_trajectories(self, filename="trajectories.pkl"):
        with open(filename, "rb") as f:
            self.trajectory_log = pickle.load(f)
