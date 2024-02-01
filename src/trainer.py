#! python

import math
import random
import matplotlib.pyplot as plt

import torch
import torch.optim as optim
import torch.nn as nn

from dqn import DQN
from game import Game
from replay_memory import ReplayMemory, Transition

import os

from datetime import datetime

"""
Trains the DQN model.

Uses two networks, the policy and target networks, to optimize the model. The
policy network is used to select actions and the target network is used to
compute the expected Q values.

This helps stabilize the training process by fixing the target values for a
while, and only updating them every few steps.
"""
class Trainer:
    # CONSTANTS
    BATCH_SIZE = 128    # number of transitions sampled from the replay buffer
    GAMMA = 0.99        # discount factor
    EPS_START = 0.9     # start value of epsilon
    EPS_END = 0.05      # final value of epsilon
    EPS_DECAY = 1000    # rate of exponential decay of epsilon (higher means a slower decay)
    TAU = 0.005         # update rate of the target network
    LR = 1e-4           # learning rate of the optimizer

    MODEL_FILE_PATH = os.path.join(os.getcwd(), "models", "dqn.pt")
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    """
    Initializes the trainer by creating the environment, the policy and target networks,
    """
    def __init__(self) -> None:
        self.env = Game()

        n_actions = len(self.env.action_space)
        state = self.env.reset()
        n_observations = len(state)
        self.policy_net = DQN(n_observations, n_actions).to(self.DEVICE)
        self.target_net = DQN(n_observations, n_actions).to(self.DEVICE)
        self.target_net.load_state_dict(self.policy_net.state_dict())

        self.optimizer = optim.AdamW(
            self.policy_net.parameters(),
            lr=self.LR,
            amsgrad=True
        )
        self.memory = ReplayMemory(10000)

        self.scores = []
        self.steps_done = 0

        if os.path.isfile(self.MODEL_FILE_PATH):
            print("loading model...")
            checkpoint = torch.load(self.MODEL_FILE_PATH)
            self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
            self.target_net.load_state_dict(checkpoint['target_net_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

            self.scores = checkpoint['scores']

    """
    Selects an action based on the current state.

    Args:
        state (list): The current state of the environment.
    Returns:
        torch.tensor: The action to take.
    """
    def _select_action(self, state: list) -> torch.tensor:
        sample = random.random()
        # Makes less random choices as the number of steps increases
        eps_threshold = self.EPS_END + (self.EPS_START - self.EPS_END) * math.exp(
            -1.0 * self.steps_done / self.EPS_DECAY
        )
        if sample > eps_threshold:
            # Select the action with the highest expected reward
            with torch.no_grad():
                return self.policy_net(state).max(1)[1].view(1, 1)
        else:
            # Select random action
            return torch.tensor(
                [[random.sample(self.env.action_space, 1)[0]]],
                device=self.DEVICE,
                dtype=torch.long
            )

    """
    Plots the scores of each epoch.

    Args:
        show_result (bool): Whether or not to show the result of the training.
        interactive (bool): Whether or not to show the plot interactively.
    """
    def plot(self, show_result: bool=False, interactive: bool=False) -> None:
        if interactive:
            plt.ion()
        else:
            plt.ioff()

        plt.figure(1)
        score_t = torch.tensor(self.scores, dtype=torch.float)
        if show_result:
            plt.title("Result")
        else:
            plt.clf()
            plt.title("Training...")
        plt.xlabel("Episode")
        plt.ylabel("Score")
        plt.plot(score_t.numpy())
        # Take 100 episode averages and plot them too
        if len(score_t) >= 100:
            means = score_t.unfold(0, 100, 1).mean(1).view(-1)
            means = torch.cat((torch.zeros(99), means))
            plt.plot(means.numpy())

        plt.pause(0.001)  # pause a bit so that plots are updated

    """
    Optimizes the model.
    """
    def _optimize_model(self) -> None:
        # Only optimize after a full batch can be grabbed from memory
        if len(self.memory) < self.BATCH_SIZE:
            return

        # Grab a batch of trasitions
        transitions = self.memory.sample(self.BATCH_SIZE)
        batch = Transition(*zip(*transitions))

        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        # Get the Q value for every Transition in the batch from the policy network.
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        # Create mask because if state was final, the expected Q value is 0.
        non_final_mask = torch.tensor(
            tuple(map(lambda s: s is not None, batch.next_state)),
            device=self.DEVICE,
            dtype=torch.bool,
        )
        # Get a tensor of zeros with the same length as the batch size
        next_state_values = torch.zeros(self.BATCH_SIZE, device=self.DEVICE)
        with torch.no_grad():
            # Get the expected Q values for the next states
            next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0]
        expected_state_action_values = (next_state_values * self.GAMMA) + reward_batch

        # Compute Huber loss of the actual and expected values
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()

        # In-place gradient clipping
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()

    """
    Train the model.

    Args:
        epochs (int): The number of epochs to train the model for.
    """
    def train(self, epochs: int=80) -> None:
        for _ in range(epochs):
            # Initialize the environment and get it's state
            state = self.env.reset()
            state = torch.tensor(state, dtype=torch.float32, device=self.DEVICE).unsqueeze(0)
            done = False
            while not done:
                self.steps_done += 1
                action = self._select_action(state)
                observation, reward, terminated = self.env.step(action.item())
                reward = torch.tensor([reward], device=self.DEVICE)
                done = terminated

                if terminated:
                    next_state = None
                else:
                    next_state = torch.tensor(
                        observation, dtype=torch.float32, device=self.DEVICE
                    ).unsqueeze(0)

                # Store the transition in memory
                self.memory.push(state, action, next_state, reward)

                # Move to the next state
                state = next_state

                # Perform one step of the optimization (on the policy network)
                self._optimize_model()

                # Soft update of the target network's weights
                target_net_state_dict = self.target_net.state_dict()
                policy_net_state_dict = self.policy_net.state_dict()

                for key in policy_net_state_dict:
                    target_net_state_dict[key] = policy_net_state_dict[
                        key
                    ] * self.TAU + target_net_state_dict[key] * (1 - self.TAU)
                self.target_net.load_state_dict(target_net_state_dict)

            self.scores.append(self.env.score)
            self.plot(interactive=True)

    """
    Saves the model.
    """
    def save(self) -> None:
        new_model_version_path = os.path.join(
            os.getcwd(),
            "models",
            "versions",
            f"dqn{datetime.now().strftime('%d-%m-%Y-%H%M%S')}.pt"
        )
        content = {
            'scores': self.scores,
            'policy_net_state_dict': self.policy_net.state_dict(),
            'target_net_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
        }
        torch.save(
            content,
            self.MODEL_FILE_PATH
        )
        torch.save(
            content,
            new_model_version_path
        )

if __name__ == "__main__":
    EPOCHS = 600 if torch.cuda.is_available() else 80

    train = Trainer()
    train.train(epochs=EPOCHS)
    train.save()
    train.plot(show_result=True, interactive=False)
