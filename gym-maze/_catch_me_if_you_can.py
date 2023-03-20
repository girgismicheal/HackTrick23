#######################################
############ OUR TEST CODE ############
############  BAZOKAAAA    ############
#######################################
import sys
sys.path.append('./helpers')
import numpy as np
import random
from riddle_solvers import *
import copy
from gym_maze.envs.maze_manager import MazeManager
from gym_maze.envs.maze_generator import Maze, validate_maze
from riddle_solvers import cipher_solver, captcha_solver, pcap_solver, server_solver
import time
import warnings
warnings.filterwarnings("ignore")

from helpers.model import QTableSolver, A_Star, DQL


################### Helper functions ###################
def check_valid_actions(position, prev_position, prev_action, history):
    col, row = position
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
        if row>0 and row<MAZE_SIZE-1 and np.array_equal(position, prev_position):
            if prev_action in actions: actions.remove(prev_action)

        if col>0 and col<MAZE_SIZE-1 and np.array_equal(position, prev_position):
            if prev_action in actions: actions.remove(prev_action)

    actions = list(set(actions) - history.get_walls_of_position(position))
    return actions


def calculate_final_score(win_state, rescued_items, riddlesTimeDictionary, steps):
    if steps == 0: steps = 1
    rescue_score = ((rescued_items*250)*rescued_items)/(steps)
    
    time_taken, riddles_score = 0, 0
    for riddle, time in riddlesTimeDictionary.items():
        time_taken += time
        riddles_score += RIDDLE_SCORES[riddle]

    if time_taken != 0:
        riddles_score = riddles_score / (time_taken*100)
    else:
        riddles_score = 0

    total_score = (rescue_score + riddles_score)

    if not win_state:
        total_score = 0.8 * total_score

    return total_score



def select_action(state, prev_state, prev_action, history, epsilon):
    global PREDICTION_MODEL
    # Use epsilon-greedy for exploration
    valid_actions = check_valid_actions(state[0], prev_state[0], prev_action, history)
    if epsilon > np.random.rand(1)[0]:
        action = random.choice(valid_actions)
    else:
        # Select an action
        action = PREDICTION_MODEL.predict(state, valid_actions, history)
    return action



def load_maze(agent_id, maze_path:str=None):
    if maze_path:
        maze_obj = np.load(maze_path)
    else:
        while(True):
            maze = Maze(maze_size=(MAZE_SIZE, MAZE_SIZE), rescue_item_locations=[(MAZE_SIZE, MAZE_SIZE)])
            is_validated = validate_maze(maze.maze_cells)
            if is_validated:
                break

        maze_obj = maze.maze_cells
    
    manager = MazeManager()
    manager.init_maze(agent_id, maze_cells=maze_obj)
    env = manager.maze_map[agent_id]
    riddle_solvers = {'cipher': cipher_solver, 'captcha': captcha_solver, 'pcap': pcap_solver, 'server': server_solver}
    return manager, env, riddle_solvers, maze_obj


class History:
    def __init__(self):
        self.history = dict(((c,r), (0, set())) for r in range(MAZE_SIZE) for c in range(MAZE_SIZE)) #<<< need to test

    def visit(self, position):
        self.history[tuple(position)] = (self.history[tuple(position)][0]+1, self.history[tuple(position)][1])
    
    def add_wall(self, position, action):        
        self.history[tuple(position)][1].add(action)
        if ACTIONS[action] == 'N' and position[1] > 0:
            self.history[tuple([position[0], position[1]-1])][1].add(ACTIONS.index('S'))
        elif ACTIONS[action] == 'S' and position[1] < MAZE_SIZE-1:
            self.history[tuple([position[0], position[1]+1])][1].add(ACTIONS.index('N'))
        elif ACTIONS[action] == 'E' and position[0] < MAZE_SIZE-1:
            self.history[tuple([position[0]+1, position[1]])][1].add(ACTIONS.index('W'))
        elif ACTIONS[action] == 'W' and position[0] > 0:
            self.history[tuple([position[0]-1, position[1]])][1].add(ACTIONS.index('E')) 

    def get_density_matrix(self):
        return np.reshape([v[0] if v[0] > 0 else 1 for k,v in self.history.items()], (MAZE_SIZE, MAZE_SIZE))
    
    def get_walls_of_position(self, position):
        return self.history[tuple(position)][1]
    
    def get_visit_number(self, position):
        return self.history[tuple(position)][0]

#######################################################


def start_testing():

    # total rewards for each episode
    all_total_scores = []
    states_history = {}
    _wins = 0
    maze_obj = None
    for episode in range(NUM_EPISODES):

        # load maze
        _manager, _env, _riddle_solvers, maze_obj = load_maze(AGENT_ID, MAZE_PATH)

        # Reset the environment
        PREDICTION_MODEL.reset()
        _current_state = _manager.reset(AGENT_ID)

        _prev_state = copy.deepcopy(_current_state)
        _prev_action = None

        _riddles_solving_time = {}
        _rescued_itesm, _visited_items = 0, 0
        _explore_rate = EPSILON_DEFAULT
        
        _history = History()
        _history.visit(_current_state[0]) # visit initial state
        _win_state = False

        for step in range(MAX_STEPS):
            _current_state = copy.deepcopy(_current_state)

            # Select an action
            action = select_action(_current_state, _prev_state, _prev_action, _history, _explore_rate)

            # Take the action
            new_state, _, _, _, info = _manager.step(AGENT_ID, ACTIONS[action])
            
            # visit new cell
            _history.visit(new_state[0])

            # in case the last action was invalid like a go throw a wall
            valid_actions = check_valid_actions(_current_state[0], new_state[0], action, _history)
            # print('>>>>  ',valid_actions, action, new_state[0])
            if not valid_actions or action not in valid_actions:
                _history.add_wall(new_state[0], action)
            
            # Check if the agent is in a riddle
            if not info['riddle_type'] == None and info['riddle_type'] not in _riddles_solving_time:
                _visited_items += 1
                riddle_type = info['riddle_type']

                if SOLVE_RIDDLE:
                    solution_time = time.time()
                    # Solve the riddle
                    solution = _riddle_solvers[info['riddle_type']](info['riddle_question'])
                    solution_time = time.time() - solution_time
                    _rescued_itesm = info['rescued_items']
                else:
                    solution_time = 0.2
                    _rescued_itesm += 1
                    solution = ''
                
                new_state, _, _, _, info = _manager.solve_riddle(info['riddle_type'], AGENT_ID, solution)
                _riddles_solving_time[riddle_type] = _riddles_solving_time.get(riddle_type, 0) + solution_time
                
            score_to_terminate_now = calculate_final_score(win_state=False,
                                                           rescued_items=_rescued_itesm,
                                                           riddlesTimeDictionary=_riddles_solving_time,
                                                           steps=step)
            if score_to_terminate_now >= 50:
                _win_state = False
                break

            # Check termination condition
            if _visited_items == len(new_state[1]):
                if isinstance(PREDICTION_MODEL, A_Star):
                    

                    steps_to_end = len(PREDICTION_MODEL.get_heuristics_path( tuple(new_state[0]), (9,9) ))
                    score_to_terminate_at_end = calculate_final_score(win_state=True,
                                                                    rescued_items=_rescued_itesm,
                                                                    riddlesTimeDictionary=_riddles_solving_time,
                                                                    steps=step+steps_to_end)

                    # print(score_to_terminate_now, score_to_terminate_at_end, steps_to_end)                   
                    if score_to_terminate_now > score_to_terminate_at_end: # terminate now to get the max score
                        # print('terminate now')
                        _win_state = False
                        break

                    elif np.array_equal(new_state[0], (9,9)):  # go to the end to ge higher score
                        # print('terminate at end')
                        _win_state = True
                        break
                else:
                    if np.array_equal(new_state[0], (9,9)): # terminate at the end
                        print('terminate at end ---')
                        _win_state = True
                        break

            # Update parameters
            _prev_action = action
            _prev_state = _current_state
            _current_state = new_state

            # decrease exploration rate
            _explore_rate -= EPSILON_INTERVAL / EPSILON_DECAY_RATE
            _explore_rate = max(_explore_rate, EPSILON_MIN)

            if RENDER:
                _manager.render(AGENT_ID)
                time.sleep(SLEEP)

        episode_score = calculate_final_score(_win_state, _rescued_itesm, _riddles_solving_time, step)
        all_total_scores.append(episode_score)

        if _win_state: _wins += 1
        template = "%0d |  Win: %5d |  episode_score: %5.2f |  steps_num: %5d |  visited_item: %5d |  rescued_items: %5d |  epsilon: %5.2f"
        print(template % (episode, _win_state, episode_score, step, _visited_items, _rescued_itesm, _explore_rate))
        print("=============================================================")

        states_history[episode] = {"episod": episode,
                                   "score": episode_score,
                                   "win_state": _win_state,
                                   "steps_num": step,
                                   "visited_item": _visited_items,
                                   "rescued_items": _rescued_itesm,
                                   "epsilon": _explore_rate}
        
        # if _visited_items == 2:
        #     print(maze_obj)
        #     np.save("buggg.npy", maze_obj)
        #     break


    template = ">>>>> avg_score:: %5.2f | median_score:: %5.2f  | wins: %5d/%5d | avg_rescued_items: %5.2f | avg_steps: %5.2f | meadian_steps: %5.2f"
    avg_rescued_items = np.mean([v["rescued_items"] for k,v in states_history.items()])
    avg_steps = np.mean([v["steps_num"] for k,v in states_history.items()])
    meadian_steps = np.median([v["steps_num"] for k,v in states_history.items()])
    print(template % (np.mean(all_total_scores), np.median(all_total_scores), _wins, NUM_EPISODES, avg_rescued_items, avg_steps, meadian_steps))
    return all_total_scores, states_history



################# MAIN #################
if __name__ == "__main__":

    ################# GLOBALS #################
    # exploration rate
    EPSILON_DEFAULT = 0#0.9
    EPSILON_MIN = 0#0.2
    EPSILON_INTERVAL = EPSILON_DEFAULT - EPSILON_MIN
    EPSILON_DECAY_RATE = 10000

    MAZE_SIZE = 10

    AGENT_ID = "9" # add your agent id here

    RIDDLE_SCORES = {"cipher": 20, "server": 30, "pcap": 40, "captcha": 10}


    # ACTIONS = ['W', 'N', 'E', 'S']
    ACTIONS = ["N", "S", "E", "W"]
    ACTIONS_NUM = 4

    NUM_EPISODES = 100
    MAX_STEPS = 5000

    SOLVE_RIDDLE = False

    # debugging parameters
    SLEEP = 0
    RENDER = False

    # MAZE_PATH = './bazoka_maze.npy'
    MAZE_PATH = None

    PREDICTION_MODEL = A_Star(ACTIONS, MAZE_SIZE)
    # PREDICTION_MODEL = QTableSolver('q_table.npy')    # load model
    # PREDICTION_MODEL = QTableSolver('My_q_table1.npy')  # load model
    # PREDICTION_MODEL = DQL(ACTIONS, './models/saved_model_model/', './models/saved_model/')

    ################# MAIN LOOP #################
    start_time = time.time()
    states = start_testing()
    print("Total time: ", time.time() - start_time)


# 1- optemize riddels & test it  X <<< ++++++++   --Ahmed Khaled
# 2- optemize the A*               <<< +++++      --Gergis        ---> 2 ideas for optmization
# 3- Deep Q-learning               <<< +++        --Mostafa & Gaber & Umar
# 4- Video                       X <<< +          --Kareem
# 5- Presentation                  <<< +++        --Kareem        
# 6- our maze generating         X <<< +++++      --Ahmed Khaled
# 7- Submit maze & code
# 8- Trial submition on server
# 9- Prepare submission code 
##### Questions ####
# 1- Exit now or at (9,9), the stats says now not late :)



