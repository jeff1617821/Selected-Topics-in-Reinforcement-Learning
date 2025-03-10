import torch
import torch.nn as nn
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from base_agent import DQNBaseAgent
from models.atari_model import AtariNetDQN
import gym
import random
from gym.wrappers import AtariPreprocessing
from gym.wrappers.frame_stack import FrameStack

class AtariDQNAgent(DQNBaseAgent):
	def __init__(self, config):
		super(AtariDQNAgent, self).__init__(config)
		### TODO ###
		# initialize env
		# self.env = ???
		self.env = FrameStack(AtariPreprocessing(gym.make('ALE/MsPacman-v5'), 30, 1), 4)
		### TODO ###
		# initialize test_env
		# self.test_env = ???
		self.test_env = FrameStack(AtariPreprocessing(gym.make('ALE/MsPacman-v5', render_mode = "rgb_array"), 30, 1), 4)
		# initialize behavior network and target network
		self.behavior_net = AtariNetDQN(self.env.action_space.n)
		self.behavior_net.to(self.device)
		self.target_net = AtariNetDQN(self.env.action_space.n)
		self.target_net.to(self.device)
		self.target_net.load_state_dict(self.behavior_net.state_dict())
		# initialize optimizer
		self.lr = config["learning_rate"]
		self.optim = torch.optim.Adam(self.behavior_net.parameters(), lr=self.lr, eps=1.5e-4)
		
	def decide_agent_actions(self, observation, epsilon=0.0, action_space=None):
		### TODO ###
		# get action from behavior net, with epsilon-greedy selection
		
		# if random.random() < epsilon:
		# 	action = ???
		# else:
		# 	action = ???
		if random.random() < self.epsilon:
			return random.randrange(self.env.action_space.n)
		else:
			q = np.array([np.asarray(observation)])
			action = torch.argmax(self.behavior_net(torch.tensor(q, dtype = torch.float, device = self.device))).cpu()
			return action

	def update_behavior_network(self):
		# sample a minibatch of transitions
		state, action, reward, next_state, done = self.replay_buffer.sample(self.batch_size, self.device)
		### TODO ###
		# calculate the loss and update the behavior network
		# 1. get Q(s,a) from behavior net
		# 2. get max_a Q(s',a) from target net
		# 3. calculate Q_target = r + gamma * max_a Q(s',a)
		# 4. calculate loss between Q(s,a) and Q_target
		# 5. update behavior net

		# q_value = ???
		# with torch.no_grad():
			# q_next = ???
			# if episode terminates at next_state, then q_target = reward
			# q_target = ???
		q = self.behavior_net(state).gather(dim = 1, index = action.long())
		with torch.no_grad():
			q_next = self.target_net(next_state)
			q_target = reward + self.gamma * torch.max(q_next, dim = 1)[0].unsqueeze(1) * (1 - done)
		# criterion = ???
		criterion = nn.MSELoss()
		#loss = criterion(q_value, q_target)
		loss = criterion(q, q_target)
		self.writer.add_scalar('DQN/Loss', loss.item(), self.total_time_step)

		self.optim.zero_grad()
		loss.backward()
		self.optim.step()
	
	