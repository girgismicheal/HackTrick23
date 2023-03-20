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
import time
import traceback
import copy
import warnings
warnings.filterwarnings("ignore")


def calculate_final_score(agent_id, rescued_items, riddlesTimeDictionary):
    rescue_score = (1000*rescued_items)/(manager.maze_map[agent_id].steps)
    riddles_score = 0
    riddles_score_dict = dict()
    for riddle in manager.riddles_dict[agent_id].riddles.values():
        riddle_score = manager.riddle_scores[riddle.riddle_type]*riddle.solved()
        if riddle_score > 0:
            riddle_score = riddle_score / (riddlesTimeDictionary.get(riddle.riddle_type,1)*100)
        riddles_score += riddle_score
        riddles_score_dict[riddle.riddle_type] = riddle_score
        
    total_score = (rescue_score + riddles_score)
    # print(">>>>> rescue_score: ", rescue_score, "   riddles_score: ", riddles_score, riddles_score_dict)

    if(not tuple(manager.maze_map[agent_id].maze_view.robot)==(9,9) or not manager.maze_map[agent_id].terminated):
        total_score = 0.8 * total_score
        # print(">>>>> total_score: ", total_score)
    
    return total_score, riddles_score_dict


def select_action(state, explore_rate):
    # Select a random action
    if random.random() < explore_rate:
        action = env.action_space.sample()
    # Select the action with the highest q
    else:
        action = int(np.argmax(q_table[state]))
    return action


def get_explore_rate(t):
    return max(MIN_EXPLORE_RATE, min(0.8, 1.0 - math.log10((t+1)/DECAY_FACTOR)))


def get_learning_rate(t):
    return max(MIN_LEARNING_RATE, min(0.8, 1.0 - math.log10((t+1)/DECAY_FACTOR)))


def state_to_bucket(state):
    # Number of discrete states (bucket) per state dimension
    STATE_BOUNDS = [(0, 9), (0, 9)]
    bucket_indice = []
    for i in range(len(state[0])):
        if state[0][i] <= STATE_BOUNDS[i][0]:
            bucket_index = 0
        elif state[0][i] >= STATE_BOUNDS[i][1]:
            bucket_index = MAZE_SIZE - 1
        else:
            # Mapping the state bounds to the bucket array
            bound_width = STATE_BOUNDS[i][1] - STATE_BOUNDS[i][0]
            offset = (MAZE_SIZE-1)*STATE_BOUNDS[i][0]/bound_width
            scaling = (MAZE_SIZE-1)/bound_width
            bucket_index = int(round(scaling*state[0][i] - offset))
        bucket_indice.append(bucket_index)
    return tuple(bucket_indice)


def get_denisity_visiting(matrix, pos, area):
    col, row = pos
    st1 = 0 if col-area < 0 else col-area
    st2 = col+area+1 if col+area < matrix.shape[0] else matrix.shape[0]

    st3 = 0 if row-area < 0 else row-area
    st4 = row+area+1 if row+area < matrix.shape[1] else matrix.shape[1]
    mat = matrix[st1:st2, st3:st4]
    return np.sum(mat), mat.size


def check_valid_actions(state, prev_state, prev_action, visited_cells):
    col, row = state[0]
    actions = [0, 1, 2, 3]
    if row == 0:
        actions.remove(1)
    elif row == MAZE_SIZE-1:
        actions.remove(3)

    if col == 0:
        actions.remove(0)
    elif col == MAZE_SIZE-1:
        actions.remove(2)

    if prev_action != None:
        if row>0 and row<MAZE_SIZE-1 and np.array_equal(state[0], prev_state[0]):
            if prev_action in actions: actions.remove(prev_action)

        if col>0 and col<MAZE_SIZE-1 and np.array_equal(state[0], prev_state[0]):
            if prev_action in actions: actions.remove(prev_action)
    
    actions = list(set(actions) - visited_cells[tuple(state[0])][1])
    return actions


def local_inference(riddle_solvers):

    current_state = manager.reset(agent_id)
    prev_state = copy.deepcopy(current_state)
    riddles_solving_time = {}
    rescued_itesm = 0
    visited_items = 0
    is_riddle_visit = False
    default_explore_rate = 0#0.95
    explore_rate = default_explore_rate
    prev_action = None
    visited_cells = dict(((r,c), (0, set())) for r in range(MAZE_SIZE) for c in range(MAZE_SIZE))
    visited_space = np.zeros([MAZE_SIZE, MAZE_SIZE])
    visited_space[0][0] = 1
    denisity_threshold = 0.9
    max_cell_visit = 5

    for t in range(MAX_T):
        current_state = copy.deepcopy(current_state)

        valid_actions = check_valid_actions(current_state, prev_state, prev_action, visited_cells)
        # Select a random action
        if random.random() < explore_rate:
            action = random.choice(valid_actions)
        # Select the action with the highest q
        else:
            # Select an action
            state = state_to_bucket(current_state)
            action = list(filter(lambda x: x in valid_actions, np.argsort(q_table[state])[::-1]))[0]

        new_state, _, _, _, info = manager.step(agent_id, ACTIONS_DICT[action])
        visited_space[new_state[0][1]][new_state[0][0]] += 1

        denisity, cells = get_denisity_visiting(visited_space, new_state[0], 1)

        if (denisity/cells*max_cell_visit) > denisity_threshold:
            explore_rate -= EPSILON_INTERVAL / 100000
            explore_rate = max(explore_rate, MIN_EXPLORE_RATE)
        else:
            # increase explore rate
            explore_rate = default_explore_rate
           
                
        # in case the last action was invalid like a go throw a wall
        valid_actions = check_valid_actions(current_state, new_state, action, visited_cells)
        if not valid_actions or action not in valid_actions:
            visited_cells[tuple(current_state[0])][1].add(action)
        
        prev_action = action
        if not info['riddle_type'] == None:
            is_riddle_visit = True
            brevious_visited_itesm = visited_items
            visited_items += 1
            solution_time = time.time()
            solution = riddle_solvers[info['riddle_type']](info['riddle_question'])
            solution_time = time.time() - solution_time
            if info['riddle_type'] in riddles_solving_time:
                riddles_solving_time[info['riddle_type']] = riddles_solving_time[info['riddle_type']] + solution_time
            else:
                riddles_solving_time[info['riddle_type']] = solution_time

            new_state, _, _, _, info = manager.solve_riddle(info['riddle_type'], agent_id, solution)
            rescued_itesm = info['rescued_items']

        # THIS IS A SAMPLE TERMINATING CONDITION WHEN THE AGENT REACHES THE EXIT
        # IMPLEMENT YOUR OWN TERMINATING CONDITION
        if np.array_equal(new_state[0], (9,9)):
            final_score, _ = calculate_final_score(agent_id, rescued_itesm, riddles_solving_time)
            print(f"final Score : {final_score}")
            print(f"rescued items : {rescued_itesm}")
            print(f"visited items : {visited_items}")
            print(f"riddles solving time : {riddles_solving_time}")
            print("Steps : %d" % manager.maze_map[agent_id].steps)
            manager.set_done(agent_id)
            break # Stop Agent
        
        prev_state = current_state
        current_state = new_state
        manager.render(agent_id)

        if is_riddle_visit:
            # explore_rate = explore_rate * ((4-rescued_itesm)/4)
            # Decay probability of taking random action
            explore_rate -= EPSILON_INTERVAL / 1000
            explore_rate = max(explore_rate, MIN_EXPLORE_RATE)

        is_riddle_visit = False
        # print(visited_space)
        time.sleep(0.01)
        


def simulate(riddle_solvers):
    
    # Instantiating the learning related parameters
    learning_rate = get_learning_rate(0)
    explore_rate = get_explore_rate(0)
    discount_factor = 0.9
    num_streaks = 0

    for episode in range(NUM_EPISODES):
        
        # Reset the environment
        obv = manager.reset(agent_id)
        # the initial state
        total_reward = 0
        riddles_solving_time = {}
        rescued_itesm=0
        visited_items=0
        state_0 = state_to_bucket(obv)

        for t in range(MAX_T):

            action = select_action(state_0, explore_rate)
            explore_rate -= EPSILON_INTERVAL / 1000000
            explore_rate = max(explore_rate, MIN_EXPLORE_RATE)

            # execute the action
            obv, _, _, _, info = manager.step(agent_id, ACTIONS_DICT[action])
            reward = 0

            if not info['riddle_type'] == None:
                visited_items += 1
                solution_time = time.time()

                solution = riddle_solvers[info['riddle_type']](info['riddle_question'])

                solution_time = time.time() - solution_time
                if info['riddle_type'] in riddles_solving_time:
                    riddles_solving_time[info['riddle_type']] = riddles_solving_time[info['riddle_type']] + solution_time
                else:
                    riddles_solving_time[info['riddle_type']] = solution_time

                obv, _, _, _, info = manager.solve_riddle(info['riddle_type'], agent_id, solution)

                rescued_itesm = info['rescued_items']
                current_score, _ = calculate_final_score(agent_id, rescued_itesm, riddles_solving_time)
                reward += current_score+100

            if np.array_equal(obv[0], (9,9)):
                reward += 50
                done = True
            else:
                reward += -0.1/(MAZE_SIZE*MAZE_SIZE)
                done = False


            # Observe the result
            total_reward += reward

            # Update the Q based on the result
            state = state_to_bucket(obv)
            best_q = np.amax(q_table[state])
            q_table[state_0 + (action,)] += learning_rate * (reward + discount_factor * (best_q) - q_table[state_0 + (action,)])

            # Setting up for the next iteration
            state_0 = state

            # Print data
            if DEBUG_MODE == 2:
                print("\nEpisode = %d" % episode)
                print("t = %d" % t)
                print("Action: %d" % action)
                print("State: %s" % str(state))
                print("Reward: %f" % reward)
                print("Best Q: %f" % best_q)
                print("Explore rate: %f" % explore_rate)
                print("Learning rate: %f" % learning_rate)
                print("Streaks: %d" % num_streaks)
                print("")

            elif DEBUG_MODE == 1:
                if done or t >= MAX_T - 1:
                    print("\nDone ----- Episode = %d" % episode)
                    print("t = %d" % t)
                    print("Explore rate: %f" % explore_rate)
                    print("Learning rate: %f" % learning_rate)
                    print("Streaks: %d" % num_streaks)
                    score, score_riddels = calculate_final_score(agent_id, rescued_itesm, riddles_solving_time)
                    print("final Score : %f" % score)
                    print("Steps : %d" % manager.maze_map[agent_id].steps)
                    print('rescue items : %d' % rescued_itesm)
                    print("visited items : %d" % visited_items)
                    print('totla_time : ', riddles_solving_time)
                    print("Total reward: %f" % total_reward)
                    print("")

            if RENDER_MAZE:
                manager.render(agent_id)
            
            if done:
                print("Episode %d finished after %f time steps with total reward = %f (streak %d)."
                      % (episode, t, total_reward, num_streaks))

                if t <= SOLVED_T:
                    num_streaks += 1
                else:
                    num_streaks = 0
                break

            elif t >= MAX_T - 1:
                print("Episode %d timed out at %d with total reward = %f."
                      % (episode, t, total_reward))
            
        # It's considered done when it's solved over 120 times consecutively
        if num_streaks > STREAK_TO_END:
            break



if __name__ == "__main__":

    sample_maze = np.load("./bazoka_maze.npy") # np.load("hackathon_sample.npy")
    agent_id = "9" # add your agent id here
    
    manager = MazeManager()
    manager.init_maze(agent_id, maze_cells=sample_maze)
    env = manager.maze_map[agent_id]

    riddle_solvers = {'cipher': cipher_solver, 'captcha': captcha_solver, 'pcap': pcap_solver, 'server': server_solver}
    maze = {}
    states = {}

    maze['maze'] = env.maze_view.maze.maze_cells.tolist()
    maze['rescue_items'] = list(manager.rescue_items_dict.keys())

    # Number of discrete states (bucket) per state dimension
    MAZE_SIZE = 10

    # Number of discrete actions
    NUM_ACTIONS = env.action_space.n  # ["N", "S", "E", "W"]
    # Bounds for each discrete state
    STATE_BOUNDS = list(zip(env.observation_space.low, env.observation_space.high))
    print(STATE_BOUNDS)

    '''
    Learning related constants
    '''
    MIN_EXPLORE_RATE = 0#0.2
    EPSILON_MAX = 0#5.0  # Maximum epsilon greedy parameter
    EPSILON_INTERVAL = (
        EPSILON_MAX - MIN_EXPLORE_RATE
    ) 
    MIN_LEARNING_RATE = 0.1
    DECAY_FACTOR = np.prod([MAZE_SIZE, MAZE_SIZE], dtype=float) / 10.0

    '''
    Defining the simulation related constants
    '''
    NUM_EPISODES = 50000
    MAX_T = np.prod([MAZE_SIZE, MAZE_SIZE], dtype=int) * 50
    STREAK_TO_END = 100
    SOLVED_T = np.prod([MAZE_SIZE, MAZE_SIZE], dtype=int)*6
    DEBUG_MODE = 1
    Flage = True
    RENDER_MAZE = Flage
    ENABLE_RECORDING = Flage
    NUM_BUCKETS = (MAZE_SIZE, MAZE_SIZE)   # one bucket per grid


    # Actions dictionary
    ACTIONS_DICT = {
        0: 'W',
        1: 'N',
        2: 'E',
        3: 'S',
    }

    '''
    Creating a Q-Table for each state-action pair
    '''
    q_table = np.zeros(NUM_BUCKETS + (NUM_ACTIONS,), dtype=float)

    q_table = np.load(open('q_table.npy', 'rb'))
    local_inference(riddle_solvers)

    # simulate(riddle_solvers)
    # np.save('q_table.npy', q_table)


    # with open("./states.json", "w") as file:
    #     json.dump(states, file)

    
    # with open("./maze.json", "w") as file:
    #     json.dump(maze, file)
    


    # pos, manhanten distance, direction