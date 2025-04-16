# agents.py
import numpy as np

class Agent:
    def __init__(self, position, velocity=None):
        self.position = np.array(position, dtype=np.float32)
        if velocity is None:
            velocity = np.zeros(2, dtype=np.float32)
        self.velocity = np.array(velocity, dtype=np.float32)

    def update(self, dt, env, sheep_list, shepherd_list):
        """Dummy base update method. Override in subclasses."""
        pass


class ShepherdAgent(Agent):
    def __init__(self, position, policy_fn=None, velocity=None):
        super().__init__(position, velocity)
        # policy_fn is a callable that returns a desired force or acceleration.
        # For example: lambda pos, sheep, env: some_vector
        if policy_fn is None:
            # Default: move to a fixed target (e.g., center of the field)
            self.policy_fn = lambda pos, sheep, env: (np.array([env.width, env.height]) / 2.0 - pos) * 0.1
        else:
            self.policy_fn = policy_fn

    def update(self, dt, env, sheep_list, shepherd_list):
        # Use the policy function to compute the force
        force = self.policy_fn(self.position, sheep_list, env)
        # Simple physics: assume unit mass, update velocity and position.
        self.velocity = force  # you could blend the current velocity with the new force.
        self.position += self.velocity * dt
        # Optionally, enforce domain boundaries
        self.position = np.clip(self.position, 0, env.width)


class SheepAgent(Agent):
    def __init__(self, position, velocity=None):
        super().__init__(position, velocity)
        # Parameters for the simple sheep behavior:
        self.collision_distance = 1.0
        self.repulsion_strength = 0.5
        self.attraction_strength = 0.2

    def update(self, dt, env, sheep_list, shepherd_list):
        # Calculate a repulsive force from nearby sheep.
        repulsion = np.zeros(2)
        for other in sheep_list:
            if other is not self:
                diff = self.position - other.position
                distance = np.linalg.norm(diff)
                if distance < self.collision_distance and distance > 1e-6:
                    repulsion += self.repulsion_strength * diff / (distance**2)
        # Calculate an attractive force toward the average position of the shepherds.
        if shepherd_list:
            dog_positions = np.array([d.position for d in shepherd_list])
            avg_dog = np.mean(dog_positions, axis=0)
            attraction = self.attraction_strength * (avg_dog - self.position)
        else:
            attraction = np.zeros(2)
        # Combine forces. (You can adjust the weights.)
        force = repulsion + attraction
        self.velocity = force
        self.position += self.velocity * dt
        # Enforce boundaries
        self.position = np.clip(self.position, 0, env.width)
