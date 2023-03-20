import sys
import numpy as np
import math
import random
import json
import requests
import tensorflow as tf
from riddle_solvers import *
import copy
from helpers.model import QTableSolver, A_Star, DQL


### the api calls must be modified by you according to the server IP communicated with you
#### students track --> 16.170.85.45
#### working professionals track --> 13.49.133.141
server_ip = '13.49.133.141'


################# GLOBALS #################
# exploration rate
EPSILON_DEFAULT = 0#1
# EPSILON_MIN = 0#0.2
# EPSILON_INTERVAL = EPSILON_DEFAULT - EPSILON_MIN
# EPSILON_DECAY_RATE = 10000

MAZE_SIZE = 10
RIDDLE_SCORES = {"cipher": 20, "server": 30, "pcap": 40, "captcha": 10}
ACTIONS = ["N", "S", "E", "W"]
ACTIONS_NUM = 4
PREDICTION_MODEL = A_Star(ACTIONS, MAZE_SIZE)
# PREDICTION_MODEL = QTableSolver('q_table.npy')    # load model


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



def move(agent_id, action):
    response = requests.post(f'http://{server_ip}:5000/move', json={"agentId": agent_id, "action": action})
    return response

def solve(agent_id,  riddle_type, solution):
    response = requests.post(f'http://{server_ip}:5000/solve', json={"agentId": agent_id, "riddleType": riddle_type, "solution": solution}) 
    print(response.json()) 
    return response

def get_obv_from_response(response):
    directions = response.json()['directions']
    distances = response.json()['distances']
    position = response.json()['position']
    obv = [position, distances, directions] 
    return obv


def submission_inference(riddle_solvers):

    response = requests.post(f'http://{server_ip}:5000/init', json={"agentId": agent_id})
    current_state = get_obv_from_response(response)

    # Reset the environment
    PREDICTION_MODEL.reset()
    _current_state = current_state
    _prev_state = copy.deepcopy(_current_state)
    _prev_action = None

    _riddles_solving_time = {}
    _rescued_itesm, _visited_items = 0, 0
    _explore_rate = EPSILON_DEFAULT
    
    _history = History()
    _history.visit(_current_state[0]) # visit initial state
    _win_state = False
    steps = 0

    while(True):
        steps += 1
        _current_state = copy.deepcopy(_current_state)

        # Select an action
        action = select_action(_current_state, _prev_state, _prev_action, _history, _explore_rate)

        response = move(agent_id, ACTIONS[action])
        if not response.status_code == 200:
            print(response)
            break
        new_state = get_obv_from_response(response)
        print(response.json())

        # visit new cell
        _history.visit(new_state[0])
        # in case the last action was invalid like a go throw a wall
        valid_actions = check_valid_actions(_current_state[0], new_state[0], action, _history)
        if not valid_actions or action not in valid_actions:
            _history.add_wall(new_state[0], action)


        if not response.json()['riddleType'] == None:
            _visited_items += 1
            riddle_type = response.json()['riddleType']
            solution_time = time.time()
            solution = riddle_solvers[response.json()['riddleType']](response.json()['riddleQuestion'])
            solution_time = time.time() - solution_time
            response = solve(agent_id, response.json()['riddleType'], solution)
            _riddles_solving_time[riddle_type] = solution_time
            _rescued_itesm = int(response.json()['rescuedItems'])
            print(response, "  Rescue")


        score_to_terminate_now = calculate_final_score(win_state=False,
                                                       rescued_items=_rescued_itesm,
                                                       riddlesTimeDictionary=_riddles_solving_time,
                                                       steps=steps)
        if steps > 40 and score_to_terminate_now >= 50:
            _win_state = False
            print("nowwwwww   :",  score_to_terminate_now)
            break

        # Check termination condition
        if _visited_items == len(new_state[1]):
            if isinstance(PREDICTION_MODEL, A_Star):
                
                steps_to_end = len(PREDICTION_MODEL.get_heuristics_path( tuple(new_state[0]), (9,9) ))
                score_to_terminate_at_end = calculate_final_score(win_state=True,
                                                                  rescued_items=_rescued_itesm,
                                                                  riddlesTimeDictionary=_riddles_solving_time,
                                                                  steps=steps+steps_to_end)

                # print(score_to_terminate_now, score_to_terminate_at_end, steps_to_end)                   
                if score_to_terminate_now > score_to_terminate_at_end: # terminate now to get the max score
                    print(f'terminate now,  score_now :: {score_to_terminate_now}    score_late :: {score_to_terminate_at_end}, steps : {steps}, time_taken : {_riddles_solving_time}')
                    _win_state = False
                    break

                elif np.array_equal(new_state[0], (9,9)):  # go to the end to ge higher score
                    print(f'terminate at end,  score_now :: {score_to_terminate_now}    score_late :: {score_to_terminate_at_end}, steps: {steps}, time_taken : {_riddles_solving_time}')
                    _win_state = True
                    break
            else:
                if np.array_equal(new_state[0], (9,9)): # terminate at the end
                    print(f'terminate at end,  score_now :: {score_to_terminate_now}    score_late :: {score_to_terminate_at_end}, steps: {steps}, time_taken : {_riddles_solving_time}')
                    _win_state = True
                    break

        # Update parameters
        _prev_action = action
        _prev_state = _current_state
        _current_state = new_state
    
    # THIS IS A SAMPLE TERMINATING CONDITION WHEN THE AGENT REACHES THE EXIT
    # IMPLEMENT YOUR OWN TERMINATING CONDITION
    # if np.array_equal(response.json()['position'], (9,9)):
    response = requests.post(f'http://{server_ip}:5000/leave', json={"agentId": agent_id})
    print(response, "  Rescue")
    print(response.text, response.status_code)


if __name__ == "__main__":
    agent_id = "ZvB2tNpM8y"
    riddle_solvers = {'cipher': cipher_solver, 'captcha': captcha_solver, 'pcap': pcap_solver, 'server': server_solver}
    submission_inference(riddle_solvers)
    
