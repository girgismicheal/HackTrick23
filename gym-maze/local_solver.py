import sys
import numpy as np
import math
import random
import json
import requests

import gym
import gym_maze
from gym_maze.envs.maze_manager import MazeManager
from riddle_solvers import *
import numpy as np
import warnings
warnings.filterwarnings("ignore")

def action_init():
    map_possible_movements = {(x,y) : ['N', 'W', 'E', 'S'] for x in range(10) for y in range(10)}

    for x in range(10):
        for y in range(10):
            actions = ['N', 'W', 'E', 'S']
            
            if x == 0:
                actions.remove('W')
            elif x==9:
                actions.remove('E')
            
            if y == 0:
                actions.remove('N')
            elif y==9:
                actions.remove('S')

            map_possible_movements[x,y] = actions
    return map_possible_movements

def eleminiate_action(x, y, action):
    print(map_possible_movements[x,y])
    print(map_possible_movements[x-1,y])
    print(action)
    if action == 'N':
        map_possible_movements[x,y].remove('N')
        map_possible_movements[x,y-1].remove('S')
    elif action == 'S':
        map_possible_movements[x,y].remove('S')
        map_possible_movements[x,y+1].remove('N')
    elif action == 'E':
        map_possible_movements[x,y].remove('E')
        map_possible_movements[x+1,y].remove('W')
    elif action == 'W':
        map_possible_movements[x,y].remove('W')
        map_possible_movements[x-1,y].remove('E')

def get_all_possible_goals(curr_pos, manhattan_distance, unit_vector_direction):
    if np.array_equal([1, 0], unit_vector_direction):
      return (curr_pos[0] + manhattan_distance, curr_pos[1])
    if np.array_equal([-1, 0], unit_vector_direction):
      return (curr_pos[0] - manhattan_distance, curr_pos[1])
    if np.array_equal([0, 1], unit_vector_direction):
      return (curr_pos[0], curr_pos[1] + manhattan_distance)
    if np.array_equal([0, -1], unit_vector_direction):
      return (curr_pos[0], curr_pos[1] - manhattan_distance)

    all_possible_goals = []
    max_x, max_y = 9, 9  # Define the boundaries of the maze
    for x in range(max_x + 1):
        for y in range(max_y + 1):
            if curr_pos[0] == x or curr_pos[1]==y:
              continue
             
            distance = abs(curr_pos[0] - x) + abs(curr_pos[1] - y)  # Calculate Manhattan distance
            if distance == manhattan_distance:
                all_possible_goals.append((x, y))  # Add the coordinate to the list of possible goals
    
    recommended_points = []
    for goal in all_possible_goals:
      if np.array_equal([1, 1], unit_vector_direction) and goal[0]>curr_pos[0] and goal[1]>curr_pos[1]:
        recommended_points.append(goal)
      elif np.array_equal([-1, 1], unit_vector_direction) and goal[0]<curr_pos[0] and goal[1]>curr_pos[1]:
        recommended_points.append(goal)
      elif np.array_equal([1, -1], unit_vector_direction) and goal[0]>curr_pos[0] and goal[1]<curr_pos[1]:
        recommended_points.append(goal)
      elif np.array_equal([-1, -1], unit_vector_direction) and goal[0]<curr_pos[0] and goal[1]<curr_pos[1]:
        recommended_points.append(goal)

    return recommended_points[0]

def manhattan_distance(x, y):
    return abs(x[0] - y[0]) + abs(x[1] - y[1])

def reconstruct_path(cameFrom, current):
    total_path = [current]
    while current in cameFrom.keys():
        current = cameFrom[current]
        total_path.append(current)
    return total_path
    

def get_neighbours(curr, map_possible_movements):
    neighbours = []
    moves = map_possible_movements[curr]
    for move in moves:
        if move == 'N':
            neighbours.append((curr[0], curr[1]-1))
        elif move == 'S':
            neighbours.append((curr[0], curr[1]+1))
        elif move == 'E':
            neighbours.append((curr[0]+1, curr[1]))
        elif move == 'W':
            neighbours.append((curr[0]-1, curr[1]))
    return neighbours

def a_star(start, goal, map_possible_movements):
    """
    This function implements the A* algorithm to find the shortest path between two points in a maze.
    param start: The starting point of the path
    param goal: The goal point of the path
    return: The shortest path points between the start and goal points
    """
    closedSet = set()
    openSet = set()
    cameFrom = {}

    openSet.add((start))

    # For each node, the cost of getting from the start node to that node.
    gScore = {}
    gScore[start] = 0

    # For each node, the total cost of getting from the start node to the goal
    # by passing by that node. That value is partly known, partly heuristic.
    fScore = {}
    fScore[start] = manhattan_distance(start, goal)

    while len(openSet) > 0:
        current = None
        currentFScore = sys.maxsize
        for pos in openSet:
            if fScore[pos] < currentFScore:
                currentFScore = fScore[pos]
                current = pos

        if current == goal:
            return cameFrom # reconstruct_path(cameFrom, current)

        openSet.remove(current)
        closedSet.add(current)

        for neighbor in get_neighbours(current, map_possible_movements):
            if neighbor in closedSet: # visited
                continue

            tentative_gScore = gScore[current] + 1

            if neighbor not in openSet:
                openSet.add(neighbor)
            elif tentative_gScore >= gScore[neighbor]:
                continue

            # This path is the best until now. Record it!
            cameFrom[neighbor] = current
            gScore[neighbor] = tentative_gScore
            fScore[neighbor] = gScore[neighbor] + manhattan_distance(neighbor, goal) # solved

    return None

map_possible_movements = action_init()

preiveous_position = None
preiveous_action = None
import time


def get_actions(dict_path):
    path_actions = []
    for key, value in dict_path.items():
        if key[0]-value[0] == 1:
            path_actions.append('W')
        elif key[0]-value[0] == -1:
            path_actions.append('E')
        elif key[1]-value[1] == 1:
            path_actions.append('N')
        elif key[1]-value[1] == -1:
            path_actions.append('S')
    return path_actions

# def get_actions(list_path):
#     path_actions = []
#     print(list_path)
#     for next_ in range(1, len(list_path)):
#         if list_path[next_-1][0]-list_path[next_][0] == 1:
#             path_actions.append('W')
#         elif list_path[next_-1][0]-list_path[next_][0] == -1:
#             path_actions.append('E')
#         elif list_path[next_-1][1]-list_path[next_][1] == 1:
#             path_actions.append('N')
#         elif list_path[next_-1][1]-list_path[next_][1] == -1:
#             path_actions.append('S')
#     return path_actions

def get_nearest_item(manhattan_distance, direction):
    if len(list(filter(lambda x: x > 0, manhattan_distance))) == 0:
        return (None, None)
    filtered_list = filter(lambda x: x[0] > 0, zip(manhattan_distance, direction))
    return sorted(filtered_list, key=lambda x: x[0])[0]

path_actions = []
def select_action(state):
    # This is a random agent 
    # This function should get actions from your trained agent when inferencing.
    global preiveous_position
    global preiveous_action
    global path_actions

    current_position = (state[0][0], state[0][1])
    
    # select the goal
    # selected_goal = 0
    # goal_manhattan_distance = state[1][selected_goal]
    # goal_direction = state[2][selected_goal]
    # goal = get_all_possible_goals(current_position, goal_manhattan_distance, goal_direction) # return the goal
    
    # path_actions = get_actions(current_position, goal, goal_manhattan_distance, map_possible_movements)

    # print(path_actions)

    # # we recalculate the path if the agent reached the goal
    # if len(path_actions) == 0:
    #     # TODO get the new goal
    #     goal_manhattan_distance, goal_direction=  get_nearest_item(state[1], state[2])
    #     if goal_manhattan_distance == None:
    #         goal = (9,9)
    #     else:
    #         goal = get_all_possible_goals(current_position, goal_manhattan_distance, goal_direction) # return the goal
        
    #     get_path_dict = a_star(current_position, goal, map_possible_movements)
    #     path_actions = get_actions(get_path_dict)
        # print(get_path_dict)
        # time.sleep(5)
    # we eleminate the actions that have walls in the way and the actions that are not possible
    # aslo we recalculate the path if the agent is stuck in wall and intarupted the path
    if (preiveous_position != None) and (preiveous_position == current_position):
        # time.sleep(5)
        eleminiate_action(preiveous_position[0], preiveous_position[1], preiveous_action)
        get_path_dict = a_star(current_position, goal, map_possible_movements)
        path_actions = get_actions(get_path_dict)
        # print(get_path_dict)
    
    
    # actions = path_actions.pop(0) # ["N", "S", "E", "W]
    # print(actions)
    actions = map_possible_movements[(state[0][0], state[0][1])]
    random_action = random.choice(actions)
    action_index = actions.index(random_action)
    
    preiveous_position = current_position
    preiveous_action= random_action

    # print(actions)
    # print(random_action)
    # print(state[1])
    # print(state[2])
    # print(preiveous_position)
    return random_action, action_index # action_index


def local_inference(riddle_solvers):

    obv = manager.reset(agent_id)

    for t in range(MAX_T):
        # Select an action
        state_0 = obv
        action, action_index = select_action(state_0) # Random action
        obv, reward, terminated, truncated, info = manager.step(agent_id, action)

        if not info['riddle_type'] == None:
            solution = riddle_solvers[info['riddle_type']](info['riddle_question'])
            obv, reward, terminated, truncated, info = manager.solve_riddle(info['riddle_type'], agent_id, solution)

        # THIS IS A SAMPLE TERMINATING CONDITION WHEN THE AGENT REACHES THE EXIT
        # IMPLEMENT YOUR OWN TERMINATING CONDITION
        if np.array_equal(obv[0], (9,9)):
            manager.set_done(agent_id)
            break # Stop Agent

        if RENDER_MAZE:
            manager.render(agent_id)

        states[t] = [obv[0].tolist(), action_index, str(manager.get_rescue_items_status(agent_id))]       
        


if __name__ == "__main__":

    sample_maze = np.load("hackathon_sample.npy")
    agent_id = "9" # add your agent id here
    
    manager = MazeManager()
    manager.init_maze(agent_id, maze_cells=sample_maze)
    env = manager.maze_map[agent_id]

    riddle_solvers = {'cipher': cipher_solver, 'captcha': captcha_solver, 'pcap': pcap_solver, 'server': server_solver}
    maze = {}
    states = {}

    
    maze['maze'] = env.maze_view.maze.maze_cells.tolist()
    maze['rescue_items'] = list(manager.rescue_items_dict.keys())

    MAX_T = 5000
    RENDER_MAZE = True
    

    local_inference(riddle_solvers)

    with open("./states.json", "w") as file:
        json.dump(states, file)

    

    with open("./maze.json", "w") as file:
        json.dump(maze, file)