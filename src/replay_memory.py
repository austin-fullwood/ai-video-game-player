from collections import deque, namedtuple
import random

"""
Defines a transition, which is use to store the agent's experience.
"""
Transition = namedtuple(
    "Transition",
    ("state", "action", "next_state", "reward")
)

"""
Saves the transitions that the agent experiences, also called experience replay.
This is later used to sample random batches of transitions, which are then used
to train the network.
"""
class ReplayMemory:
    """
    Initializes the queue.

    Args:
        capacity (int): The maximum number of transitions to store in the memory.
    """
    def __init__(self, capacity: int) -> None:
        self.memory = deque([], maxlen=capacity)

    """
    Saves a transition.

    Args:
        *args: The transition to save.
    """
    def push(self, *args: Transition) -> None:
        """Save a transition"""
        self.memory.append(Transition(*args))

    """
    Gets a random batch of transitions.

    Args:
        batch_size (int): The number of transitions to sample.
    """
    def sample(self, batch_size: int) -> list[Transition]:
        return random.sample(self.memory, batch_size)

    """
    Gets the number of transitions stored.

    Returns:
        int: The number of transitions stored.
    """
    def __len__(self) -> int:
        return len(self.memory)
