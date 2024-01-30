#! python3

import math
import pygame
import random

"""
Runs the simulation.
"""
class Game:
    # Constants
    PLAYER_SIZE: int = 40
    FOOD_SIZE: int = 20

    """
    Initializes the game.

    Args:
        is_human (bool): Whether or not the game is being played by a human.
    """
    def __init__(self, is_human: bool=True) -> None:
        self.is_human = is_human
        self.screen = pygame.display.set_mode((500, 500))
        self.player = pygame.Rect(
            self.screen.get_width() / 2,
            self.screen.get_height() / 2,
            Game.PLAYER_SIZE,
            Game.PLAYER_SIZE
        )

        food_x, food_y = self.__food_spawn()
        self.food = pygame.Rect(
            food_x,
            food_y,
            Game.FOOD_SIZE,
            Game.FOOD_SIZE
        )
        self.score = 0
        self.dt = 0.01
        self.clock = pygame.time.Clock()
        self.frames = 0

        self.action_space = [0, 1, 2, 3] # ["up", "down", "left", "right"]

    """
    Stops game on deletion.
    """
    def __del__(self) -> None:
        pygame.quit()

    """
    Runs one step of the game.

    Args:
        action (int): The action taken by the player (human or non-human).
    Returns:
        list: The state of the game.
        int: The reward for the action taken.
        bool: Whether or not the game is over.
    """
    def step(self, action: int=None) -> (list, int, bool):
        if self.is_human and action == None:
            action = self.__human_action()
        elif action not in self.action_space:
            print(f"Invalid action '{action}'")
            exit()

        reward = 0
        current_distance = math.dist(
            [self.player.x, self.player.y],
            [self.food.x, self.food.y]
        )

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                exit()

        # fill the screen with a color to wipe away anything from last frame
        self.screen.fill("purple")
        pygame.display.set_caption(f"Total Points: {self.score}")

        if self.__check_food_ate():
            self.food.x, self.food.y = self.__food_spawn()
            self.score += 1
            reward = 1000

        pygame.draw.rect(self.screen, "red", self.player)
        pygame.draw.rect(self.screen, "blue", self.food)

        if action == 0:
            self.player.y -= 300 * self.dt
        if action == 1:
            self.player.y += 300 * self.dt
        if action == 2:
            self.player.x -= 300 * self.dt
        if action == 3:
            self.player.x += 300 * self.dt
        if action == 4:
            self.reset()
            return

        self.player.y = max(self.player.y, 0)
        self.player.y = min(
            self.player.y,
            self.screen.get_height() - self.player.height
        )
        self.player.x = max(self.player.x, 0)
        self.player.x = min(
            self.player.x,
            self.screen.get_width() - self.player.width
        )

        # flip() the display to put your work on screen
        pygame.display.flip()

        # limits FPS to 60
        # dt is delta time in seconds since last frame, used for framerate-
        # independent physics.
        # self.dt = self.clock.tick(60) / 1000

        new_distance = math.dist(
            [self.player.x, self.player.y],
            [self.food.x, self.food.y]
        )
        reward += current_distance - new_distance

        self.frames += 1

        return self.__grab_screen(), reward, self.__is_done()

    """
    Restarts the game.

    Returns:
        list: The state of the game.
    """
    def reset(self) -> list:
        self.frames = 0
        self.score = 0
        self.dt = 0.01

        self.player.x = self.screen.get_width() / 2
        self.player.y = self.screen.get_height() / 2
        self.food.x, self.food.y = self.__food_spawn()

        return self.__grab_screen()

    """
    Checks if the game is over.

    Returns:
        bool: Whether or not the game is over.
    """
    def __is_done(self) -> bool:
        return self.frames >= 1000

    """
    Gets the action taken by the human player.

    Returns:
        int: The action taken by the human player.
    """
    def __human_action(self) -> int:
        keys = pygame.key.get_pressed()
        if keys[pygame.K_w]:
            return 0
        if keys[pygame.K_s]:
            return 1
        if keys[pygame.K_a]:
            return 2
        if keys[pygame.K_d]:
            return 3
        if keys[pygame.K_r]:
            return 4
        return None

    """
    Gets the state of the game.

    Returns:
        list: The state of the game.
    """
    def __grab_screen(self) -> list:
         # State of the game in pixels
        # return pygame.surfarray.array2d(self.screen).flatten()

        return [self.player.x, self.player.y, self.food.x, self.food.y]

    """
    Spawns food in a random location.

    Returns:
        int: The x of the food.
        int: The y of the food.
    """
    def __food_spawn(self) -> (int, int):
        y_exclude = [i for i in range(self.player.top, self.player.bottom)]
        x_exclude = [i for i in range(self.player.left, self.player.right)]
        y = random.choice(
            [i for i in range(0, self.screen.get_height() - Game.FOOD_SIZE) if i not in y_exclude]
        )
        x = random.choice(
            [i for i in range(0, self.screen.get_width() - Game.FOOD_SIZE) if i not in x_exclude]
        )

        return x, y

    """
    Checks if the food was eaten.

    Returns:
        bool: Whether or not the food was eaten.
    """
    def __check_food_ate(self) -> bool:
        return self.player.colliderect(self.food)

"""
Plays the game with a human player.
"""
if __name__ == "__main__":
    game = Game()
    while(True):
        game.step()
