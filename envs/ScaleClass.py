import random

import gym
import numpy as np
from gym.utils import seeding
from PIL import Image


class ScaleClass(gym.Env):
    meta_data = {}

    def __init__(self, args=None):

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
        self.num_rooms = 1
        self.current_task = 0
        self.length = args.length
        self.goals = self.get_goals()
        self.goalx, self.goaly = self.goals[0], self.goals[1]

        self.state_num = count
        self.state_img = None
        self.destination_reached = False
        self.window = None
        self.max_steps = 10 * self.length

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

    def get_goals(self):
        return random.sample(range(self.length), 2)

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
        # self.state_img = self.to_img()

        done = self.destination_reached  # Episodic
        return self.state, reward, done, {}

    def get_optimal_reward(self):
        manhattan_dist = abs(self.initial_x - self.goalx) + abs(self.initial_y - self.goaly)
        return 1/manhattan_dist

    def reset(self):
        x = list(range(0, self.length))
        x.remove(self.goalx)
        self.x = self.initial_x = random.sample(x, 1)[0]
        y = list(range(0, self.length))
        y.remove(self.goaly)
        self.y = self.initial_y = random.sample(y, 1)[0]
        self.state = self.gridtostate[(self.x, self.y)]
        # self.state_img = self.to_img()
        return self.state
