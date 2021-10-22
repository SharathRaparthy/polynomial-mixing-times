import math
import random

import gym
import numpy as np
from gym.utils import seeding
from PIL import Image
from itertools import product


class NBottleneckClass(gym.Env):
    meta_data = {}

    def __init__(self, args=None):
        self.args = args
        self.x = self.initial_x = random.randint(0, args.length - 1)
        self.y = self.initial_y = random.randint(0, args.length - 1)
        self.gridtostate = {}
        self.statetogrid = {}
        count = 0
        for y in range(args.length):
            for x in range(args.length):
                self.gridtostate[(x, y)] = count
                self.statetogrid[count] = (x, y)
                count += 1

        self.state = self.gridtostate[(self.x, self.y)]
        self.all_combs = list(product(range(args.length), repeat=2))
        self.num_rooms = args.num_rooms
        self.current_room = 1
        self.current_idx = 1
        self.length = args.length
        self.goals = self.get_goals()
        self.goalx = self.goals[self.current_room - 1][0]
        self.goaly = self.goals[self.current_room - 1][1]

        self.state_num = count
        self.state_img = None
        self.destination_reached = False
        self.window = None

    def next_room(self):
        self.initial_x = self.goalx
        self.initial_y = self.goaly
        if self.args.task_evolution == 'cycles':
            if (self.current_room + 1) > self.num_rooms:
                self.current_room = 1
            else:
                self.current_room += 1
            self.goalx = self.goals[self.current_room - 1][0]
            self.goaly = self.goals[self.current_room - 1][1]
        elif self.args.task_evolution == 'random':
            room_idx = random.sample(range(1, self.num_rooms + 1), 1)[0]
            self.current_room = room_idx
            self.goalx = self.goals[room_idx - 1][0]
            self.goaly = self.goals[room_idx - 1][1]
        elif self.args.task_evolution == 'curricular':
            room_idx = int(self.find_number(self.current_idx))
            self.goalx = self.goals[room_idx - 1][0]
            self.goaly = self.goals[room_idx - 1][1]
            if room_idx == self.num_rooms:
                self.current_idx = 1
            else:
                self.current_room = room_idx
                self.current_idx += 1



    @staticmethod
    def find_number(n):
        # n = x(x + 1)/2 + 1
        x = int(math.floor((-1 + math.sqrt(1+ 8 * n - 8)) / 2))
        base = (x * (x + 1)) / 2 + 1
        # Value of n-th element
        return n - base + 1

    def get_goals(self):
        indices = np.random.randint(0, len(self.all_combs), size=self.num_rooms)
        all_combs = np.asarray(self.all_combs)
        np.random.shuffle(all_combs)
        return all_combs[indices]

    def to_img(self):
        state_img = np.zeros(shape=(self.length, self.length, 3), dtype=np.uint8)
        # state_img = np.pad(state_img, (1, ), 'constant', constant_values=255)
        # state_img[:, :, 1] = 0
        # state_img[:, :, 2] = 0
        if self.destination_reached:
            state_img[self.goalx, self.goaly, 1] = 255
        else:
            state_img[self.x, self.y, 0:2] = 255
            state_img[self.x, self.y, 2] = 0
            state_img[self.goalx, self.goaly, 0] = 255
            state_img[self.goalx, self.goaly, 1] = 0

        return state_img

    def render(self, mode="human", close=False):
        img = self.state_img
        if close:
            if self.window:
                self.window.close()
            return

        if mode == "human" and not self.window:
            from envs.rendering import Window

            self.window = Window("Taxi Domain")
            self.window.show(block=False)

        if mode == "human":
            img = Image.fromarray(img, "RGB")
            img = img.resize((50, 50))
            self.window.show_img(img)
            self.window.set_caption("Single Room")

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        if random.random() < 0.05:
            # failure of given action
            action = random.randint(0, 4)

        if action == 0:  # up
            if self.y > 0:
                self.y -= 1
                self.state = self.gridtostate[(self.x, self.y)]
            else:
                self.y += 1
                self.state = self.gridtostate[(self.x, self.y)]

        elif action == 1:  # down
            if self.y < self.length - 1:
                self.y += 1
                self.state = self.gridtostate[(self.x, self.y)]
            else:
                self.y -= 1
                self.state = self.gridtostate[(self.x, self.y)]

        elif action == 2:  # left
            if self.x > 0:
                self.x -= 1
                self.state = self.gridtostate[(self.x, self.y)]
            else:
                self.x += 1
                self.state = self.gridtostate[(self.x, self.y)]

        elif action == 3:  # right
            if self.x < self.length - 1:
                self.x += 1
                self.state = self.gridtostate[(self.x, self.y)]
            else:
                self.x -= 1
                self.state = self.gridtostate[(self.x, self.y)]

        elif action == 4:  # random stay put
            self.state = self.gridtostate[(self.x, self.y)]

        else:
            print("issue with provided action!")

        if self.x == self.goalx and self.y == self.goaly:
            reward = 1.0
            self.destination_reached = True
        else:
            reward = 0.0
            self.destination_reached = False
        if self.destination_reached:
            self.next_room()
        self.state_img = self.to_img()
        # print(f"Goals : {self.goalx} and {self.goaly} | Current pos : {[self.x, self.y]}")
        done = False  # continuing never ending environment
        if self.current_room == 0:
            current_pos = self.state
        else:
            current_pos = self.state + (self.current_room - 1) * self.state_num
        return current_pos, reward, done, 0

    def get_optimal_reward(self):
        manhattan_dist = abs(self.initial_x - self.goalx) + abs(self.initial_y - self.goaly)
        if manhattan_dist == 0:
            return 1
        else:
            return 1 / manhattan_dist

    def reset(self):
        self.x = random.randint(0, self.length - 1)
        self.y = random.randint(0, self.length - 1)
        self.state = self.gridtostate[(self.x, self.y)]
        current_pos = self.state
        self.state_img = self.to_img()
        return current_pos
