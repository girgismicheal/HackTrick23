import sys
import numpy as np
import pandas as pd
from rl.agents.sarsa import SARSAAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import EpisodeParameterMemory
import tensorflow as tf
import sys
import numpy as np
from tensorflow.keras.optimizers import Adam
import warnings
warnings.filterwarnings("ignore")



class Model:
    def load(self):
        # add you code here
        pass

    def predict(self, state:np.array, valid_actions:list, history):
        # add you code here
        pass

    def reset(self):
        # add you code here
        pass


class QTableSolver(Model):
    def __init__(self, path):
        self.model = None
        self.load(path)

    def load(self, path):
        self.model = np.load(open(path, 'rb'))

    def _state_to_vector(self, state, history):
        return state[0]

    def predict(self, state, valid_actions, history):
        state_vector = self._state_to_vector(state, history)
        action = list(filter(lambda x: x in valid_actions, np.argsort(self.model[tuple(state_vector)])[::-1]))[0]
        return  action
    
    def reset(self):
        pass





class A_Star(Model):
    def __init__(self, actions, maze_size):
        self.actions = actions
        self.maze_size = maze_size
        self.reset()

    def reset(self):
        self.preiveous_position = None
        self.preiveous_action = None
        self.path_actions = []
        self.goal = None
        self.map_possible_movements = self.__actions_init()
        self.goal_points = [(None,None) for i in range(4)]

        
    def __actions_init(self):
        map_possible_movements = {(x,y) : self.actions for x in range(self.maze_size) for y in range(self.maze_size)}
        def condetions(x, y):
            _actions = self.actions.copy()
            if x == 0:
                _actions.remove('W')
            elif x==9:
                _actions.remove('E')
            if y == 0:
                _actions.remove('N')
            elif y==9:
                _actions.remove('S')
            return _actions

        for x in range(10):
            for y in range(10):
                map_possible_movements[x,y] = condetions(x, y)
        return map_possible_movements
    

    def __get_all_possible_goals(self, curr_pos, manhattan_distance, unit_vector_direction):
        if np.array_equal([1, 0], unit_vector_direction):
            return (curr_pos[0] + manhattan_distance, curr_pos[1])
        if np.array_equal([-1, 0], unit_vector_direction):
            return (curr_pos[0] - manhattan_distance, curr_pos[1])
        if np.array_equal([0, 1], unit_vector_direction):
            return (curr_pos[0], curr_pos[1] + manhattan_distance)
        if np.array_equal([0, -1], unit_vector_direction):
            return (curr_pos[0], curr_pos[1] - manhattan_distance)

        all_possible_goals = []
        for x in range(self.maze_size):
            for y in range(self.maze_size):
                if curr_pos[0] == x or curr_pos[1]==y: continue
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



    def __manhattan_distance(self, x, y):
        return abs(x[0] - y[0]) + abs(x[1] - y[1])

    def __reconstruct_path(self, cameFrom, current):
        total_path = [current]
        while current in cameFrom.keys():
            current = cameFrom[current]
            total_path.append(current)
        return total_path

    def __get_neighbours(self, curr):
        neighbours = []
        moves = self.map_possible_movements[curr]
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


    def __update_goals_points(self, state):
        for i in range(len(state[2])):
            if self.goal_points[i] == (None,None):
                conition =  np.array_equal([-1, 0], state[2][i]) or \
                            np.array_equal([1, 0], state[2][i]) or \
                            np.array_equal([0, 1], state[2][i]) or \
                            np.array_equal([0, -1], state[2][i])
                if conition:
                    self.goal_points[i] = self.__get_all_possible_goals(state[0], state[1][i], state[2][i])


    def __get_goal_index(self, state, goal_manhattan_distance, goal_direction):
        index = -1 
        for i, dist in enumerate(state[1]):
            if dist == goal_manhattan_distance and state[2][i] == goal_direction:
                index = i
                break
        return index

    def __a_star(self, start, goal):
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
        fScore[start] = self.__manhattan_distance(start, goal)

        while len(openSet) > 0:
            current = None
            currentFScore = sys.maxsize
            for pos in openSet:
                if fScore[pos] < currentFScore:
                    currentFScore = fScore[pos]
                    current = pos

            if current == goal:
                return self.__reconstruct_path(cameFrom, current)

            openSet.remove(current)
            closedSet.add(current)

            for neighbor in self.__get_neighbours(current):
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
                fScore[neighbor] = gScore[neighbor] + self.__manhattan_distance(neighbor, goal) # solved
        return None
    

    def __get_actions(self, list_path):
        list_path = list_path[::-1]
        path_actions = []
        
        for next_ in range(1, len(list_path)):
            if list_path[next_-1][0]-list_path[next_][0] == 1:
                path_actions.append('W')
            elif list_path[next_-1][0]-list_path[next_][0] == -1:
                path_actions.append('E')
            elif list_path[next_-1][1]-list_path[next_][1] == 1:
                path_actions.append('N')
            elif list_path[next_-1][1]-list_path[next_][1] == -1:
                path_actions.append('S')
        return path_actions
    
    def __get_nearest_item(self, manhattan_distance, direction):
        if len(list(filter(lambda x: x > 0, manhattan_distance))) == 0:
            return (None, None)
        filtered_list = filter(lambda x: x[0] > 0, zip(manhattan_distance, direction))
        return_point = sorted(filtered_list, key=lambda x: x[0])[0]
        return return_point


    def __eleminiate_action(self, x, y, action):
        if action == 'N':
            self.map_possible_movements[x,y].remove('N')
            self.map_possible_movements[x,y-1].remove('S')
        elif action == 'S':
            self.map_possible_movements[x,y].remove('S')
            self.map_possible_movements[x,y+1].remove('N')
        elif action == 'E':
            self.map_possible_movements[x,y].remove('E')
            self.map_possible_movements[x+1,y].remove('W')
        elif action == 'W':
            self.map_possible_movements[x,y].remove('W')
            self.map_possible_movements[x-1,y].remove('E')



    def __select_action(self, state):
        # This is a random agent 
        # This function should get actions from your trained agent when inferencing.
        current_position = (state[0][0], state[0][1])

        # # we recalculate the path if the agent reached the goal
        if len(self.path_actions) == 0:
            # TODO get the new goal
            goal_manhattan_distance, goal_direction =  self.__get_nearest_item(state[1], state[2])
            if goal_manhattan_distance == None:
                self.goal = (9,9)
            else:
                index = self.__get_goal_index(state, goal_manhattan_distance, goal_direction)
                if index != -1 and self.goal_points[index] != (None,None):
                    self.goal = self.goal_points[index]
                else:
                    self.goal = self.__get_all_possible_goals(current_position, goal_manhattan_distance, goal_direction) # return the goal
                

            get_path_dict = self.__a_star(current_position, self.goal)
            self.path_actions = self.__get_actions(get_path_dict)

        # we eleminate the actions that have walls in the way and the actions that are not possible
        # aslo we recalculate the path if the agent is stuck in wall and intarupted the path
        if (self.preiveous_position != None) and (self.preiveous_position == current_position):
            self.__eleminiate_action(self.preiveous_position[0], self.preiveous_position[1], self.preiveous_action)
            get_path_dict = self.__a_star(current_position, self.goal)
            self.path_actions = self.__get_actions(get_path_dict)

        self.__update_goals_points(state=state) ## added update_goals_points
        # ["N", "S", "E", "W]
        action = self.path_actions.pop(0) # random.choice(actions)
        self.preiveous_position = current_position
        self.preiveous_action = action
        return action
    
    def get_heuristics_path(self, current_pos, goal):
        return self.__a_star(current_pos, goal)

    def predict(self, state, valid_actions, history):
        action = self.__select_action(state)
        return  self.actions.index(action)





class DQL(Model):
    def __init__(self, actions, model_path, agent_path) -> None:
        directions = [(0, 0), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (-1, 1)]
        self.one_hot_encoding = pd.get_dummies(pd.Series(directions))
        self.ACTIONS = actions
        self.model_path = model_path
        self.agent_path = agent_path
        self.WIND_VISITED_MAT_SIZE = 2
        self.DQN = self.load()
        
    def load(self):
        model = tf.keras.models.load_model(self.model_path)
        Adam._name = 'IBM :)'
        dqn = self.__build_agent(model)
        optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=2e-3)
        dqn.compile(optimizer)
        dqn.load_weights(self.agent_path)
        return dqn

    def __build_agent(self, model):
        policy = BoltzmannQPolicy()
        memory = EpisodeParameterMemory(limit=50000, window_length=1)
        qdn = SARSAAgent(model, len(self.ACTIONS), policy=policy, test_policy=None, gamma=0.99, nb_steps_warmup=10, train_interval=1)
        return qdn

    def __get_flatten_visited_cells(self, history, num=5):
        visiting_matrix = history.get_density_matrix()
        submatrices = [np.sum(visiting_matrix[i:i+num, j:j+num]) for i in range(0, 10, num) for j in range(0, 10, num)]
        submatrices = np.array(submatrices).reshape(int(10/num), int(10/num))
        norm_submatrices = submatrices/np.sum(submatrices)
        flatten_vec = 1-norm_submatrices.flatten()
        return flatten_vec/np.sum(flatten_vec)


    def __get_current_flatten_state(self, state_obv, history):
        state = []
        state.extend(state_obv[0])
        # add the new distance promising scores
        state_obv_dist = state_obv[1] + [-1 for _ in range(4 - len(state_obv[1]))] if len(state_obv[1]) < 4 else state_obv[1]
        distance = np.array(state_obv_dist)
        minus_one_flags = distance == -1
        sum_rest = sum(distance[~minus_one_flags])
        new_out = (sum_rest - distance)/sum_rest
        new_out[minus_one_flags] = 0
        state.extend(list(new_out))
        
        state_obv_direc = state_obv[2] + [[0, 0] for _ in range(4 - len(state_obv[2]))] if len(state_obv[2]) < 4 else state_obv[2]
        for direction in state_obv_direc:
            direction = self.one_hot_encoding[tuple(direction)].to_list()
            state.extend(direction)
        state.extend(self.__get_flatten_visited_cells(history, self.WIND_VISITED_MAT_SIZE))
        return np.array(state).squeeze()

    def predict(self, state:np.array, valid_actions:list, history):
        state_vec = self.__get_current_flatten_state(state, history)
        action_probs = self.DQN.model.predict(np.expand_dims([state_vec], axis=0))
        action = list(filter(lambda x: x in valid_actions, np.argsort(action_probs[0])[::-1]))[0]
        return action

    def reset(self):
        # add you code here
        pass









