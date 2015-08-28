import numpy as np
import random
import copy
import time
import lasagne
import lasagne.layers as ll
import theano
import theano.tensor as T

k_world_size = 20
k_memory_size = 1000000
k_replay_size = 500
k_rand_move_probability = 75
k_save_freq = 1000

Items = {
        'EMPTY': 0,
        'YOP': 10,
        'FOOD': 20,
        'WALL': 30
}

Directions = [
        np.matrix([0, 1]),
        np.matrix([0, -1]),
        np.matrix([1, 0]),
        np.matrix([-1, 0])
]

DirectionName = ['UP', 'DOWN', 'RIGHT', 'LEFT']

class State(object):
    food_pos = None
    yop_pos = None
    world = None

    def __init__(self):
        self.world = np.full((k_world_size, k_world_size), Items['EMPTY'])

        for i in range(k_world_size):
            self.world[(0, i)] = Items['WALL']
            self.world[(i, 0)] = Items['WALL']
            self.world[(k_world_size - 1, i)] = Items['WALL']
            self.world[(i, k_world_size - 1)] = Items['WALL']

        self.food_pos = np.matrix([1, 1]) # arbitraty
        self.move_food()
        self.yop_pos = np.matrix([random.randint(1, k_world_size - 2), random.randint(1,
            k_world_size - 2)])
        self.world[self.yop_pos[0, 0], self.yop_pos[0, 1]] = Items['YOP']

    def is_food_eaten(self):
        return (self.food_pos == self.yop_pos).all()

    def move_food(self):
        self.food_pos = np.matrix([random.randint(1, k_world_size - 2), random.randint(1,
            k_world_size - 2)])
        self.world[self.food_pos[0, 0], self.food_pos[0, 1]] = Items['FOOD']

    def move_if_possible(self, direction):
        new_pos = self.yop_pos + Directions[direction]
        if self.world[new_pos[0, 0], new_pos[0, 1]] != Items['WALL']:
            self.world[self.yop_pos[0, 0], self.yop_pos[0, 1]] = Items['EMPTY']
            self.yop_pos = new_pos
            self.world[self.yop_pos[0, 0], self.yop_pos[0, 1]] = Items['YOP']
            return True
        else:
            return False

    def distance_to_food(self):
        return np.absolute(self.food_pos - self.yop_pos).sum()

    def get_state_matrix(self):
        w_sz = float(k_world_size)
        return np.matrix([self.food_pos[0, 0] / w_sz, self.food_pos[0, 1] / w_sz,
                          self.yop_pos[0, 0] / w_sz, self.yop_pos[0, 1] / w_sz],
                         dtype=np.float32)

    def draw(self):
        print "\x1B[2J\x1B[0;0H"
        for j in range(k_world_size):
            for i in range(k_world_size):
                if self.world[i, j] == Items['EMPTY']:
                    print ' ',
                elif self.world[i, j] == Items['WALL']:
                    print '*',
                elif self.world[i, j] == Items['YOP']:
                    print 'X',
                elif self.world[i, j] == Items['FOOD']:
                    print 'O',
            print ''

class MemCell(object):
    r = None
    state = None
    action = None
    next_state = None

    def __init__(self, r, state, action, next_state):
        self.r = r
        self.state = copy.deepcopy(state)
        self.action = action
        self.next_state = copy.deepcopy(next_state)

class Brain(object):
    score = 0
    is_learning = True
    choose_move = None
    predict_rewards = None
    train_fn = None
    memory = []
    W = None

    def __init__(self):
        input_var = T.matrix('X')
        network = ll.InputLayer((None, 4), input_var)
        network = ll.DenseLayer(network, 10, name='hidden', nonlinearity=lasagne.nonlinearities.rectify)
        self.W = network.W
        network = ll.DenseLayer(network, 4, name='out', nonlinearity=lasagne.nonlinearities.linear)
        result = ll.get_output(network)
        self.predict_rewards = theano.function([input_var], result)
        self.choose_move = theano.function([input_var], T.argmax(result))

        corrected_rewards = T.matrix('Y')
        error = lasagne.objectives.squared_error(corrected_rewards, result).mean()
        params = lasagne.layers.get_all_params(network, trainable=True)
        updates = lasagne.updates.adagrad(error, params, 0.1)
        self.train_fn = theano.function([input_var, corrected_rewards], error, updates=updates)

    def compute_move(self, state):
        if random.random() < 1.0 / k_rand_move_probability:
            return random.choice(range(4))
        return self.choose_move(state.get_state_matrix())

    def reward(self, r, a, state, nxt_state):
        self.score += r
        print self.score, r

        if len(self.memory) < k_memory_size:
            self.memory.append(MemCell(r, state, a, nxt_state))
        else:
            self.memory[random.randint(0, k_memory_size - 1)] = MemCell(r, state, a, nxt_state)

        if len(self.memory) < k_replay_size:
            return

        replays = random.sample(self.memory, k_replay_size)

        replays_mat = np.empty((k_replay_size, 4), dtype=np.float32)
        replays_nxt_mat = np.empty((k_replay_size, 4), dtype=np.float32)
        for i in range(k_replay_size):
            replays_mat[i,:] = replays[i].state.get_state_matrix()
            replays_nxt_mat[i,:] = replays[i].next_state.get_state_matrix()

        predicted = self.predict_rewards(replays_mat)
        predicted_nxt = self.predict_rewards(replays_nxt_mat)

        for i in range(k_replay_size):
            predicted[i, replays[i].action] = replays[i].r + 0.75 * predicted_nxt[i,:].max()

        self.train_fn(replays_mat, predicted)
        self.train_fn(replays_mat, predicted)
        self.train_fn(replays_mat, predicted)
        self.train_fn(replays_mat, predicted)
        self.train_fn(replays_mat, predicted)


s = State()
b = Brain()
nb_iter = 0;
gah = None
while True:
    s.draw()
    direction = b.compute_move(s)
    nxt = copy.deepcopy(s)
    if nb_iter % 1000 == 0:
        print nb_iter, b.score
        gah = copy.deepcopy(b.W.get_value())
        print gah
        print ''
    if nxt.move_if_possible(direction):
        if nxt.is_food_eaten():
            nxt.move_food()
            b.reward(1, direction, s, nxt)
        else:
            b.reward(1 if nxt.distance_to_food() < s.distance_to_food() else -1, direction, s, nxt)
    else:
        b.reward(-1, direction, s, nxt)
    if nb_iter % 1000 == 0:
        print b.W.get_value() - gah
    #time.sleep(0.01)
    s = nxt
    nb_iter += 1
