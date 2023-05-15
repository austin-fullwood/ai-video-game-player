# Video Game Player

## Overview
The environment is a simple game that consist of a player block that tries to "eat" as many food blocks as possible in 10 seconds.

**Rules:**
* The blocks are spawned at random.
* The player starts in the middle of the map.
* Map is 500 x 500.
* Block is 40 x 40 square.
* Food is 20 x 20 square.

The score is posted at the top.

## Models

### Version 1
***Algorithm:*** A DQN model

***Stage:*** An array containing the blocks current coordinates and the foods current coordinates
```
[player.x, player.y, food.x, food.y]
```

***Action:*** An integer that maps to a direction of motion (the index of the below array)
```
["up", "down", "left", "right"]
```

***Reward:*** The change in distance between the player and food block (DENSE)

***Outcome***: The

## To Do
1. Use the screen's pixels as the stage
    * Transform data (grayscale, skip frames, )
2. Investigate why model performs worst on first iteration.
3. Test using sparse reward system (maybe after using dense).
4. Restructure model class (move saving, optimization, action selection to it)