import os
import random
import argparse
import sys

import gym
import envs
from algos.tabular_agents import *

parser = argparse.ArgumentParser(description='Single Room Environment')

parser.add_argument('--lr', type=float, default=0.3, help='learning rate')
parser.add_argument("--env-name", type=str, default="NCycleActive-v0", help="Batch size for DQN")
parser.add_argument('--gamma', type=float, default=1.0, help='gamma discount factor for q learning')
parser.add_argument('--epsilon', type=float, default=0.05, help='epsilon')
parser.add_argument('--agent', type=str, default="qlearning", help='which agent type?')
parser.add_argument('--lifetime', type=int, default=10000, help='how many steps will the agent live?')
parser.add_argument('--reporting', type=int, default=100, help='how many steps will we interact before reporting results?')
parser.add_argument('--length', type=int, default=5, help='length of each wall of the length x length grid world')
parser.add_argument('--num-states', type=int, default=10, help='length of each wall of the length x length grid world')
parser.add_argument('--seed', type=int, default=1, help='number of seeds to consider for averaging')
parser.add_argument('--steps', type=int, default=10, help='number of items in batch update')
parser.add_argument('--num-rooms', type=int, default=4, help='number of items in batch update')
parser.add_argument('--debug', dest='debug', action='store_true', help='print more when debug flag is on')
parser.add_argument('--unichain', dest='unichain', action='store_true',help='compute SSD w/ system of equations approach')
parser.add_argument('--lr_pi', type=float, dest='lr_pi', default=0.01, help='policy learning rate')
parser.add_argument('--priority', dest='priority', action='store_true', help='sample according to the estimated SSD rather than uniform sampling')
parser.add_argument('--baseline', dest='baseline', action='store_true', help='whether or not to use a baseline to reduce variance in gradient estimates')
parser.add_argument('--steps_pi', type=int, default=1, help='number of steps taken on pi per batch')
parser.add_argument('--task-evolution', type=str, default="cycles", nargs='?', choices=['cycles', 'random', 'curricular'], help='Periodic/Multi-task/Curricular task evolution')
parser.add_argument('--solver', type=str, default="cycles", nargs='?', choices=['numpy_solver', 'torch_solver'], help='SSD solver')
parser.add_argument('--x', type=int, default=1, help='Exponent for Scale class')


def train(args):
    seed = args.seed
    args.num_actions = 4
    args.num_rewards = 2
    if args.reporting == -1:
        args.reporting = args.num_states * args.num_actions
    if args.lifetime == -1:
        args.lifetime = 100 * args.num_states * args.num_actions
    args.numberoflogs = int(args.lifetime / args.reporting)
    # Set the seeds
    np.random.seed(seed)
    random.seed(seed)
    if args.env_name == "ScaleClass-v0":
        args.num_rooms = 1

    env = gym.make(args.env_name, args=args)
    args.num_states = env.state_num

    print("Usage:\n{0}\n".format(" ".join([x for x in sys.argv])))
    print("All settings used:")
    for k, v in sorted(vars(args).items()):
        print("{0}: {1}".format(k, v))

    if args.agent == "qlearning":
        model = QLearner(args)
    elif args.agent == "onpolicyrho":
        model = OnPolicyRho(args)
    elif args.agent == "offpolicyrho":
        model = OffPolicyRho(args)
    elif args.agent == "rhogradient":
        model = RhoGradient(args)
    elif args.agent == "dynalearning":
        model = DynaLearner(args)
    elif args.agent == "replaylearning":
        model = ReplayLearner(args)
    elif args.agent == "nstepTDlearning":
        model = NTDLearner(args)
    elif args.agent == "policygradient":
        model = PolicyGradientLearner(args)
    elif args.agent == "iterativegain":
        model = IterativeRhoLearner(args)
    else:
        print("agent type:", args.agent, "was not found!")

    state = env.reset()
    rewards = []
    regret = []
    opt_rew = 0
    running_rew = 0

    if args.env_name == "NBottleneckClass-v0":
        folder = os.getcwd() + f'/results/{args.env_name}/{args.agent}_{args.task_evolution}_{args.num_rooms}/'
    elif args.env_name == "ScaleClass-v0":
        if args.agent == "iterativegain":
            folder = os.getcwd() + f'/results/{args.env_name}/{args.agent}_{args.length}_{args.steps}/'
        else:
            folder = os.getcwd() + f'/results/{args.env_name}/{args.agent}_{args.length}/'
    else:
        folder = os.getcwd() + f'/results/{args.env_name}/{args.agent}_{args.x}/'

    if not os.path.exists(folder):
        os.makedirs(folder)

    time_taken = 0
    for i in range(args.lifetime):
        time_taken += 1
        action = model.getAction(state, epsilon=args.epsilon)
        next_state, reward, done, _ = env.step(action)
        model.learn(state, action, reward, next_state)
        state = next_state

        if args.env_name == "ScaleClass-v0":
            if done:
                state = env.reset()
        if args.env_name == "NCycleClass-v0":
            if time_taken > env.max_time:
                reward = reward
                env.next_room()
                time_taken = 0
        opt_rew += env.get_optimal_reward()
        running_rew += reward
        rewards.append(running_rew)
        regret.append(opt_rew - running_rew)
        if (i + 1) % 1000 == 0:
            print("Steps done are : {}".format(i + 1))
            np.savez(folder + "results_{}.npy".format(seed), rewards=np.asarray(rewards), regrets=np.asarray(regret))


if __name__ == "__main__":
    args = parser.parse_args()
    train(args)
    sys.exit(0)
