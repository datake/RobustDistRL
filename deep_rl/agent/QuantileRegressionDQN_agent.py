#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

from ..network import *
from ..component import *
from ..utils import *
from .BaseAgent import *
from .DQN_agent import *


class QuantileRegressionDQNActor(DQNActor):
    def __init__(self, config):
        super().__init__(config)

    def compute_q(self, prediction):
        q_values = prediction['quantile'].mean(-1)
        return to_np(q_values)


class QuantileRegressionDQNAgent(DQNAgent): # based on the step of DQN agent
    def __init__(self, config):
        BaseAgent.__init__(self, config)
        self.config = config
        config.lock = mp.Lock()

        self.replay = config.replay_fn()
        self.actor = QuantileRegressionDQNActor(config)

        self.network = config.network_fn()
        self.network.share_memory()
        self.target_network = config.network_fn()
        self.target_network.load_state_dict(self.network.state_dict())
        self.optimizer = config.optimizer_fn(self.network.parameters())

        self.actor.set_network(self.network)
        self.total_steps = 0


        self.batch_indices = range_tensor(config.batch_size)

        # unique for QR-DQN
        self.quantile_weight = 1.0 / self.config.num_quantiles
        self.cumulative_density = tensor(
            (2 * np.arange(self.config.num_quantiles) + 1) / (2.0 * self.config.num_quantiles)).view(1, -1)


    def eval_step(self, state):
        self.config.state_normalizer.set_read_only()
        state = self.config.state_normalizer(state)
        q = self.network(state)['quantile'].mean(-1) 
        action = np.argmax(to_np(q).flatten()) 
        self.config.state_normalizer.unset_read_only()
        return [action]

    def eval_step_adv(self, state):
        self.config.state_normalizer.set_read_only()
        state = self.config.state_normalizer(state) 
        state0 = torch.from_numpy(state).to(self.config.DEVICE).detach()
        state_pgd = torch.from_numpy(state).to(self.config.DEVICE).detach()

        # random initializaiton
        random_noise = torch.FloatTensor(*state.shape).uniform_(-self.epsilon, self.epsilon).to(self.config.DEVICE)
        state_pgd = state_pgd.data + random_noise
        state_pgd.requires_grad = True

        for _ in range(self.PGD_iter):
            with torch.enable_grad():
                action_logit = self.network(state_pgd.float().to(self.config.DEVICE))['quantile'].mean(-1)
                y = action_logit.flatten().argmin().long().to(self.config.DEVICE).detach()
                y = torch.unsqueeze(y, 0) 
                loss0 = nn.CrossEntropyLoss()(action_logit, y)  
            loss0.backward()
            eta = self.step_size * state_pgd.grad.data.sign()
            state_pgd = state_pgd.data + eta
            eta = torch.clamp(state_pgd.data - state0.data, -self.epsilon, self.epsilon)
            state_pgd = state0.data + eta
            state_pgd.requires_grad = True

        q = self.network(state_pgd.float())['quantile'].mean(-1)  
        action = np.argmax(to_np(q).flatten())
        self.config.state_normalizer.unset_read_only()
        return [action]


    def compute_perturbation(self, states):
        if self.config.train_env == 1: 
            random_noise = torch.FloatTensor(*states.shape).normal_(0.0, self.config.random).to(self.config.DEVICE)
            states += random_noise
        else:
            state0 = states.to(self.config.DEVICE).detach()
            state_pgd = states.to(self.config.DEVICE).detach()
            # random initializaiton
            random_noise = torch.FloatTensor(*states.shape).uniform_(-self.epsilon, self.epsilon).to(self.config.DEVICE)
            state_pgd = state_pgd.data + random_noise
            state_pgd.requires_grad = True
            for _ in range(self.PGD_iter):
                with torch.enable_grad():
                    action_logit = self.network(state_pgd.float().to(self.config.DEVICE))['quantile'].mean(-1)
                    y = action_logit.argmin(1).long().to(self.config.DEVICE).detach()
                    loss0 = nn.CrossEntropyLoss()(action_logit, y) 
                loss0.backward()
                eta = self.step_size * state_pgd.grad.data.sign()
                state_pgd = state_pgd.data + eta
                eta = torch.clamp(state_pgd.data - state0.data, -self.epsilon, self.epsilon)
                state_pgd = state0.data + eta
                state_pgd.requires_grad = True
            states = state_pgd.detach()
        return states


    def compute_loss(self, transitions):
        states = self.config.state_normalizer(transitions.state)
        next_states = self.config.state_normalizer(transitions.next_state)

        # 1: random state, 2: adv state
        if self.config.train_env in [1, 2]:
            states = self.compute_perturbation(states)
        else:  # normal training
            pass

        quantiles_next = self.target_network(next_states)['quantile'].detach()
        a_next = torch.argmax(quantiles_next.sum(-1), dim=-1) 
        quantiles_next = quantiles_next[self.batch_indices, a_next, :] 

        rewards = tensor(transitions.reward).unsqueeze(-1)
        masks = tensor(transitions.mask).unsqueeze(-1)
        quantiles_next = rewards + self.config.discount ** self.config.n_step * masks * quantiles_next

        quantiles = self.network(states)['quantile']
        actions = tensor(transitions.action).long() 
        quantiles = quantiles[self.batch_indices, actions, :] 
        quantiles_next = quantiles_next.t().unsqueeze(-1)
 
        diff = quantiles_next - quantiles 
        loss = huber(diff) * (self.cumulative_density - (diff.detach() < 0).float()).abs() 
        return loss.sum(-1).mean(1) 

    def reduce_loss(self, loss):
        return loss.mean()
