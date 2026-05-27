import gymnasium as gym
from gymnasium import ObservationWrapper, spaces
import numpy as np

class ToroidPredPrey(ObservationWrapper):
    """
    Transforms the CollectGame observation grid into agent-centric,
    toroidal observations as described in the appendix of 10.1073/pnas.1907370117
    """

    def __init__(self, env: gym.Env):
        """
        Initialize ToroidObservation Wrapper

        Parameters
        ----------
        env : gymnasium.Env
            gym environment for which the wrapper is being used
        """
        super().__init__(env)
        self.env = env
        self.depth = env.unwrapped.num_preys + 3
        self.observation_space = spaces.Box(
            shape=(env.unwrapped.width, env.unwrapped.height, self.depth), low=-np.inf, high=np.inf
        )

    def observation(self, obs):
        """
        Modifies default env observation into toroidal-wrapped
        observations for each agent's POV

        Parameters
        ----------
        obs
            default observation used by environment

        Returns
        -------
        obs : List[NDArray]
            list of length num_agents containing transformed observations
        """
        toroids = []
        left_top = [(2, 2), (9, 2), (2, 9), (9, 9)]
        for a in self.env.unwrapped.agents:
            if a.type == "predator":
                pos = a.pos
                tor = np.zeros(self.observation_space.shape, dtype="float32")
                for i in range(self.env.unwrapped.width):
                    for j in range(self.env.unwrapped.height):
                        new_coords = [i - pos[0], j - pos[1]]
                        obj = self.env.unwrapped.grid.get(i, j)
                        if new_coords[0] < 0:
                            new_coords[0] += self.env.unwrapped.width
                        if new_coords[1] < 0:
                            new_coords[1] += self.env.unwrapped.height
                        if obj is None:
                            continue
                        elif obj.type == "wall":
                            tor[new_coords[1], new_coords[0], self.depth - 1] = 1
                        elif obj.type == "prey_area":
                            tor[new_coords[1], new_coords[0], self.depth - 2] = 1
                        elif obj.type == "easy_prey" or obj.type == "hard_prey":
                            depth_idx = left_top.index(obj.prey_config["territory_left_top_corner"])
                            tor[new_coords[1], new_coords[0], depth_idx] = 1
                        elif obj.type == "predator" and not np.array_equal(obj.pos, pos):
                            tor[new_coords[1], new_coords[0], self.depth - 3] = 1
                toroids.append(tor)
        return toroids