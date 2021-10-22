#!/usr/bin/env python3
import gym
from PIL import Image

from envs import *
from envs.rendering import Window


def reset():
    obs = env.reset()
    img = Image.fromarray(obs, "RGB")
    img = img.resize((240, 180))
    window.show_img(img)


def step(action):
    obs, reward, done, mode_idx = env.step(action)

    # change_mode()

    if done:
        # print("Destination Reached")
        # print("Current task is {}".format(env.current_room))
        # change_mode()
        reset()
    else:
        img = Image.fromarray(obs, "RGB")
        img = img.resize((240, 180))
        window.show_img(img)


def key_handler(event):

    if event.key == "escape":
        window.close()
        return

    if event.key == "backspace":
        reset()
        return

    if event.key == "left":
        step(0)
        return
    if event.key == "right":
        step(1)
        return
    if event.key == "up":
        step(2)
        return
    if event.key == "down":
        step(3)
        return


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Single Room Environment")

    parser.add_argument(
        "--length",
        type=int,
        default=5,
        help="length of each wall of the length x length grid world",
    )

    parser.add_argument(
        "--seeds",
        type=int,
        default=25,
        help="number of seeds to consider for averaging",
    )
    parser.add_argument(
        "--num-rooms",
        type=int,
        default=4,
        help="length of each wall of the length x length grid world",
    )
    parser.add_argument('--task-evolution', type=str, default="cycles", nargs='?',
                        choices=['cycles', 'random', 'curricular'],
                        help='Periodic/Multi-task/Curricular task evolution')

    args = parser.parse_args()
    env = gym.make("NBottleneckClass-v0", args=args)
    print("Goals are {}".format(env.goals))
    window = Window("N Room Env")
    window.reg_key_handler(key_handler)
    print("Resetting")
    reset()

    # Blocking event loop
    window.show(block=True)
