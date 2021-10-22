import numpy as np
import random
import torch
from torch.autograd import Variable
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class BaseClass:
    def __init__(self, args):
        num_states = args.num_rooms * args.num_states

        self.policy = np.random.randint(0, args.num_actions, size=num_states).astype(int)

        self.state_counts = np.zeros(shape=(num_states, args.num_actions, num_states)) + 0.0001

        self.reward_counts = np.zeros(shape=(num_states, args.num_actions, args.num_rewards)) + 0.0001

        self.state_probs = self.state_counts / self.state_counts.sum(axis=2, keepdims=True)

        self.reward_probs = self.reward_counts / self.reward_counts.sum(axis=2, keepdims=True)
        if args.solver == "torch_solver":
            self.ones = torch.ones(num_states, dtype=torch.float64).to(device)
            self.zeros = torch.zeros(num_states, dtype=torch.float64).to(device)
            self.eye = torch.eye(num_states, dtype=torch.float64).to(device)

        self.num_actions = args.num_actions
        self.args = args

    def getAction(self, state, epsilon=0.0):
        NotImplementedError

    def getPolicy(self):
        return self.policy

    def update(self, state, action, reward, next_state):
        self.state_counts[state][action][next_state] += 1.0
        self.reward_counts[state][action][int(reward)] += 1.0
        self.state_probs = self.state_counts / self.state_counts.sum(2, keepdims=True)

        self.reward_probs = self.reward_counts / self.reward_counts.sum(2, keepdims=True)

    def expectedreward(self, state_dist: np.ndarray, policy: np.ndarray, algo="onpolicy") -> float:
        if algo == "gain_gradient":
            mymatrix = []
            for mystate in range(self.state_probs.shape[0]):
                myvec = np.dot(policy[mystate], self.reward_probs[mystate])
                mymatrix.append(myvec)
            reward_state_probs = np.array(mymatrix)
        else:
            policy = np.expand_dims(policy, axis=1)
            policy = np.expand_dims(policy, axis=1)
            reward_state_probs = np.take_along_axis(self.reward_probs, policy, axis=1).sum(axis=1)
        reward_probs = np.dot(state_dist, reward_state_probs)
        reward_per_step = reward_probs[1]
        return reward_per_step

    def unichainsteadystate(self, policy, algo="onpolicy"):
        if algo == "gain_gradient":
            transition_matrix = []
            for mystate in range(self.state_probs.shape[0]):
                myvec = np.dot(policy[mystate], self.state_probs[mystate])
                transition_matrix.append(myvec)
            transition_matrix = np.asarray(transition_matrix)
        else:
            policy = np.expand_dims(policy, axis=1)
            policy = np.expand_dims(policy, axis=1)
            transition_matrix = np.take_along_axis(self.state_probs, policy, axis=1).sum(axis=1)
        if transition_matrix.shape == (1, 1):
            pi = np.array([1.0])
            return pi

        size = transition_matrix.shape[0]
        d_p = transition_matrix - np.eye(size, dtype=np.float64)
        # Replace the first equation by the normalizing condition.
        a = np.vstack([np.ones(size), d_p.T[1:, :]])
        rhs = np.zeros((size,), dtype=np.float64)
        rhs[0] = 1

        pi = np.linalg.solve(a, rhs)
        return pi

    def torch_steady_state(self, policy, algo="onpolicy") -> np.ndarray:
        # Use pytorch only
        if algo == "gain_gradient":
            transition_matrix = []
            for mystate in range(self.state_probs.shape[0]):
                myvec = np.dot(policy[mystate], self.state_probs[mystate])
                transition_matrix.append(myvec)
        else:
            policy = np.expand_dims(policy, axis=1)
            policy = np.expand_dims(policy, axis=1)
            transition_matrix = np.take_along_axis(self.state_probs, policy, axis=1).sum(axis=1)

        if transition_matrix.shape == (1, 1):
                pi = torch.tensor([1.0])
                return pi.numpy()
        transition_matrix = torch.FloatTensor(transition_matrix).to(device)
        d_p = transition_matrix - self.eye
        # Replace the first equation by the normalizing condition.
        a = torch.vstack([self.ones, d_p.T[1:, :]])
        rhs = self.zeros
        rhs[0] = 1

        pi = torch.linalg.solve(a, rhs)
        return pi.cpu().numpy()


class OnPolicyRho(BaseClass):

    def __init__(self, args):
        super().__init__(args)

    def getAction(self, state, epsilon=0.0):
        if random.random() < epsilon:
            optimal_action = random.randint(0, self.num_actions - 1)
        else:
            max_reward = -1e9
            for i in range(self.num_actions):
                temp_policy = self.policy
                temp_policy[state] = i
                if self.args.solver == "numpy_solver":
                    state_dist = self.unichainsteadystate(temp_policy)
                else:
                    state_dist = self.torch_steady_state(temp_policy)

                reward_per_step = self.expectedreward(state_dist, temp_policy)
                if reward_per_step > max_reward:
                    max_reward = reward_per_step
                    optimal_action = i

            self.policy[state] = optimal_action
        return optimal_action

    def learn(self, state, action, reward, next_state):
        self.update(state, action, reward, next_state)


class OffPolicyRho(BaseClass):

    def __init__(self, args):
        super().__init__(args)
        self.steps = args.steps
        self.priority = args.priority
        self.num_states = args.num_rooms * args.num_states

    def getAction(self, state, epsilon=0.0):
        if random.random() < epsilon:
            action = random.randint(0, self.num_actions - 1)
        else:
            action = self.policy[state]
        return action

    def learn(self, state, action, reward, next_state):
        self.update(state, action, reward, next_state)

        if self.priority:
            temp_policy = self.policy
            state_dist = self.unichainsteadystate(temp_policy)
            reward_per_step = self.expectedreward(state_dist, temp_policy)

        for j in range(self.steps):
            if self.priority:
                state = np.random.choice(self.num_states, 1, p=state_dist)[0]
            else:
                state = random.randint(0, self.num_states - 1)

            max_reward = -1e9
            for i in range(self.num_actions):
                temp_policy = self.policy
                temp_policy[state] = i
                if self.args.solver == "numpy_solver":
                    state_dist = self.unichainsteadystate(temp_policy)
                else:
                    state_dist = self.torch_steady_state(temp_policy)
                reward_per_step = self.expectedreward(state_dist, temp_policy)
                if reward_per_step > max_reward:
                    max_reward = reward_per_step
                    optimal_action = i

            self.policy[state] = optimal_action


class RhoGradient(BaseClass):

    def __init__(self, args):
        super().__init__(args)
        self.num_states = args.num_states * args.num_rooms
        self.pi = Variable(torch.randn(self.num_states, args.num_actions), requires_grad=True)
        self.num_actions = args.num_actions
        self.steps = args.steps
        self.steps_pi = args.steps_pi
        self.priority = args.priority
        self.lr = args.lr
        self.temp = 0.9
        self.baseline = args.baseline

    def getAction(self, state, epsilon=0.0):
        if random.random() < epsilon:
            action = random.randint(0, self.num_actions - 1)
        else:
            probs, logprobs = self.getProbs(state)
            action = int(np.random.choice(4, 1, p=probs)[0])
        return action

    def getProbs(self, state, numpy=True):
        my_input = self.pi[state] / self.temp
        if numpy:
            v = F.softmax(my_input, dim=0).data.numpy()
            logv = F.log_softmax(my_input, dim=0).data.numpy()
        else:
            v = F.softmax(my_input, dim=0)
            logv = F.log_softmax(my_input, dim=0)
        return v, logv

    def learn(self, state, action, reward, next_state):
        self.update(state, action, reward, next_state)
        temp_policy = F.softmax(self.pi / self.temp, dim=1).data.numpy()
        state_dist = self.unichainsteadystate(temp_policy, algo="gain_gradient")
        reward_per_step = self.expectedreward(state_dist, temp_policy, algo="gain_gradient")

        # batch = {}
        # for j in range(self.steps_pi):
        loss = 0.0
        for i in range(self.steps):
            # if j == 0:
            state = np.random.choice(self.num_states, 1, p=state_dist)[0]
            action = self.getAction(state)
            temp_policy = F.softmax(self.pi / self.temp, dim=1).data.numpy()
            temp_policy[state][:] = 0.0
            temp_policy[state][action] = 1.0
            if self.args.solver == "numpy_solver":
                new_state_dist = self.unichainsteadystate(temp_policy, algo="gain_gradient")
            else:
                new_state_dist = self.torch_steady_state(temp_policy, algo="gain_gradient")
            new_reward_per_step = self.expectedreward(new_state_dist, temp_policy, algo="gain_gradient")
            # batch[i] = (state,action,new_reward_per_step)

            # state,action,new_reward_per_step = batch[i]
            probs, logprobs = self.getProbs(state, numpy=False)
            logprob = logprobs[action]

            if self.baseline:
                loss -= logprob * (new_reward_per_step - reward_per_step)
            else:
                loss -= logprob * new_reward_per_step

        loss.backward()
        pi_grad = self.pi.grad
        self.old_pi = self.pi
        new_pi = self.pi - self.lr * pi_grad
        self.pi = Variable(new_pi.data, requires_grad=True)


class IterativeRhoLearner(BaseClass):
    def __init__(self, args):
        super().__init__(args)

        self.num_states = args.num_states * args.num_rooms  ## updated
        self.mu = np.ones(self.num_states) / self.num_states  ## updated
        self.steps = args.steps  ## updated

    def getAction(self, state, epsilon=0.0):
        if random.random() < epsilon:
            optimal_action = random.randint(0, self.num_actions - 1)
        else:
            max_reward = -1e9
            for i in range(self.num_actions):
                temp_policy = self.policy
                temp_policy[state] = i
                state_dist = self.iterativesteadystate(temp_policy)
                reward_per_step = self.expectedreward(state_dist, temp_policy)
                if reward_per_step > max_reward:
                    max_reward = reward_per_step
                    optimal_action = i
                    optimal_mu = state_dist  ## update

            self.policy[state] = optimal_action
            self.mu = optimal_mu  ## update
        return optimal_action

    def learn(self, state, action, reward, next_state):
        self.update(state, action, reward, next_state)

    def getPolicy(self):
        return self.policy

    def iterativesteadystate(self, policy):  ## updated full function
        policy = np.expand_dims(policy, axis=1)
        policy = np.expand_dims(policy, axis=1)
        transition_matrix = np.take_along_axis(self.state_probs, policy, axis=1).sum(axis=1)
        ###print("mu before=",self.mu)
        for k in range(self.steps):
            next_order = [i for i in range(self.num_states)]
            random.shuffle(next_order)
            for ind in range(self.num_states):
                next_state_ind = next_order[ind]
                total = 0.0
                for state_ind in range(self.num_states):
                    total += self.mu[state_ind] * transition_matrix[state_ind][next_state_ind]
                self.mu[next_state_ind] = total
            self.mu = self.mu / self.mu.sum()
        return self.mu


class DynaLearner(BaseClass):

    def __init__(self, args):
        super().__init__(args)
        self.num_states = args.num_states * args.num_rooms
        self.state_action_counts = np.zeros(shape=(self.num_states, args.num_actions))

        self.Q = np.zeros(shape=(self.num_states, args.num_actions))
        self.lr = args.lr
        self.steps = args.steps
        self.gamma = args.gamma
        self.states = []

    def getAction(self, state, epsilon=0.0):
        if random.random() < epsilon:
            action = random.randint(0, self.num_actions - 1)
        else:
            action = self.Q[state].argmax()
        return action

    def learn(self, state, action, reward, next_state):
        # update environment model
        self.update(state, action, reward, next_state)
        self.state_action_counts[state][action] += 1.0

        # add state to set of states considered
        # for i in range(self.args.num_rooms):
        if state not in self.states:
            self.states.append(state)

        # update Q function on current data
        self.Q[state][action] = ((1.0 - self.lr) * self.Q[state][action]) + (
                self.lr * (reward + self.gamma * self.Q[next_state].max()))

        # do the number of steps of model based updates on previous states

        num_states = len(self.states)
        if num_states > 0:
            sampled_indexes = np.random.randint(0, num_states, size=self.steps).astype(int)
            for index in sampled_indexes:
                state = self.states[index]

                # select random previous action
                actions = []
                for action in range(self.num_actions):
                    if self.state_action_counts[state][action] != 0.0:
                        actions.append(action)
                if not len(actions) > 0:
                    continue
                action_index = random.randint(0, len(actions) - 1)
                action = actions[action_index]
                reward_prob = self.reward_probs[state][action]
                state_prob = self.state_probs[state][action]
                reward = np.random.choice(2, 1, p=reward_prob)[0]
                next_state = np.random.choice(len(self.Q), 1, p=state_prob)[0]
                self.Q[state][action] = ((1.0 - self.lr) * self.Q[state][action]) + (
                        self.lr * (reward + self.gamma * self.Q[next_state].max()))

    def getPolicy(self):
        return self.Q.argmax(axis=1)


class QLearner:

    def __init__(self, args):
        num_states = args.num_rooms * args.num_states
        self.Q = np.zeros(shape=(num_states, args.num_actions))
        self.lr = args.lr
        self.gamma = args.gamma

    def getAction(self, state, epsilon=0.0):
        if random.random() < epsilon:
            action = random.randint(0, 3)
        else:
            action = self.Q[state].argmax()

        return action

    def learn(self, state, action, reward, next_state):

        self.Q[state][action] = ((1.0 - self.lr) * self.Q[state][action]) + (
                self.lr * (reward + self.gamma * self.Q[next_state].max()))


class ReplayLearner(BaseClass):

    def __init__(self, args):
        super().__init__(args)
        self.num_states = args.num_states * args.num_rooms
        self.state_action_counts = np.zeros(shape=(self.num_states, args.num_actions))
        self.Q = np.zeros(shape=(self.num_states, args.num_actions))
        self.lr = args.lr
        self.steps = args.steps
        self.gamma = args.gamma
        self.memory = []

    def getAction(self, state, epsilon=0.0):
        if random.random() < epsilon:
            action = random.randint(0, self.num_actions - 1)
        else:
            action = self.Q[state].argmax()
        return action

    def learn(self, state, action, reward, next_state):
        # update environment model
        # self.update(state, action, reward, next_state, task)
        self.state_action_counts[state][action] += 1.0

        # add state to set of states considered
        self.memory.append([state, action, reward, next_state])

        # update Q function on current data
        self.Q[state][action] = ((1.0 - self.lr) * self.Q[state][action]) + (
                self.lr * (reward + self.gamma * self.Q[next_state].max()))

        # do the number of steps of model based updates on previous states
        num_states = len(self.memory)
        if num_states > 0:
            sampled_indexes = np.random.randint(0, num_states, size=self.steps).astype(int)
            for k, index in enumerate(sampled_indexes):
                state, action, reward, next_state = self.memory[index]
                self.Q[state][action] = ((1.0 - self.lr) * self.Q[state][action]) + (
                        self.lr * (reward + self.gamma * self.Q[next_state].max()))

    def getPolicy(self):
        return self.Q.argmax(axis=1)


class NTDLearner(BaseClass):

    def __init__(self, args):
        super().__init__(args)
        self.num_states = args.num_states * args.num_rooms
        self.state_action_counts = np.zeros(shape=(self.num_states, args.num_actions))
        self.Q = np.zeros(shape=(self.num_states, args.num_actions))
        self.lr = args.lr
        self.steps = args.steps
        self.gamma = args.gamma
        self.states = []

    def getAction(self, state, epsilon=0.0):
        if random.random() < epsilon:
            action = random.randint(0, self.num_actions - 1)
        else:
            action = self.Q[state].argmax()
        return action

    def learn(self, state, action, reward, next_state):
        # update environment model
        self.update(state, action, reward, next_state)
        self.state_action_counts[state][action] += 1.0

        # update Q function on current data
        self.Q[state][action] = ((1.0 - self.lr) * self.Q[state][action]) + (
                self.lr * (reward + self.gamma * self.Q[next_state].max()))

        # do the number of steps of model based updates on previous states
        orig_state = state
        orig_action = action

        returns = reward
        for step in range(self.steps):
            state = next_state
            action = self.Q[state].argmax()
            reward_prob = self.reward_probs[state][action]
            state_prob = self.state_probs[state][action]
            reward = np.random.choice(2, 1, p=reward_prob)[0]
            next_state = np.random.choice(len(self.Q), 1, p=state_prob)[0]
            returns += (self.gamma ** (step + 1)) * reward

        returns += (self.gamma ** (self.steps + 1)) * self.Q[next_state].max()
        self.Q[orig_state][orig_action] = ((1.0 - self.lr) * self.Q[orig_state][orig_action]) \
                                          + (self.lr * returns)

    def getPolicy(self):
        return self.Q.argmax(axis=1)


class PolicyGradientLearner:

    def __init__(self, args):
        self.num_states = args.num_states * args.num_rooms
        self.Q = np.zeros(shape=(self.num_states, args.num_actions))
        self.pi = Variable(torch.randn(self.num_states, args.num_actions), requires_grad=True)
        self.lr = args.lr
        self.lr_pi = args.lr_pi
        self.steps_pi = args.steps_pi
        self.gamma = args.gamma
        self.temp = 1.0

    def getProbs(self, state, numpy=True):
        my_input = self.pi[state] / self.temp
        if numpy:
            v = F.softmax(my_input, dim=0).data.numpy()
            logv = F.log_softmax(my_input, dim=0).data.numpy()
        else:
            v = F.softmax(my_input, dim=0)
            logv = F.log_softmax(my_input, dim=0)
        return v, logv

    def getAction(self, state, epsilon=0.0):
        if random.random() < epsilon:
            action = random.randint(0, 3)
        else:
            probs, logprobs = self.getProbs(state)
            action = int(np.random.choice(4, 1, p=probs)[0])

        return action

    def learn(self, state, action, reward, next_state):
        # self.pi.grad.zero_()
        self.Q[state][action] = ((1.0 - self.lr) * self.Q[state][action]) + (
                self.lr * (reward + self.gamma * self.Q[next_state].max()))
        loss = 0.0
        for steps in range(self.steps_pi):
            probs, logprobs = self.getProbs(state, numpy=False)
            logprob = logprobs[action]
            loss -= logprob * self.Q[state][action]
        loss.backward()
        pi_grad = self.pi.grad
        self.old_pi = self.pi
        new_pi = self.pi - self.lr_pi * pi_grad
        self.pi = Variable(new_pi.data, requires_grad=True)

    def getPolicy(self):
        return self.pi.data.numpy().argmax(axis=1)
