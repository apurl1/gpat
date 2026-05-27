from models import PsiNet, NNet
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from numpy.typing import NDArray
import copy


class SFAgent:
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        feat_dim: int,
        w: NDArray,
        lr: float,
        gamma: float,
        epsilon: float,
        psi: PsiNet = None,
    ) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if psi:
            self.psi = copy.deepcopy(psi).train()
        else:
            self.psi = PsiNet(state_dim, action_dim, feat_dim)
        self.optim = optim.Adam(self.psi.parameters(), lr=lr)
        self.w = torch.from_numpy(w).float()
        self.gamma = gamma
        self.epsilon = epsilon
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.feat_dim = feat_dim
        self.batch_size = 10
        self.buffer = np.empty(self.batch_size, dtype=object)
        self.buffer_size = 0

    def phi(self, state, next_state):
        phi_vec = []
        for obj_type in range(self.feat_dim):
            phi_vec.append(np.sum(state[:, :, obj_type]) - np.sum(next_state[:, :, obj_type]))
        return np.array(phi_vec)

    def get_action(self, state, action_epsilon = None):
        # epsilon greedy
        if np.random.rand() < self.epsilon or (action_epsilon and np.random.rand() < action_epsilon):
            action = np.random.randint(self.action_dim)
        else:
            action = self.argmax_qvals(state)
        return action

    def get_qvals(self, state):
        with torch.no_grad():
            state = torch.from_numpy(state).to(self.device)
            self.psi.to(self.device)
            qvals = torch.matmul(
                self.psi(state).to(self.device), self.w.to(self.device)
            )
        return qvals

    def argmax_qvals(self, state):
        qvals = self.get_qvals(state)
        action = torch.argmax(qvals).item()
        return action

    def get_max_qval(self, state):
        qvals = self.get_qvals(state)
        max_val = torch.max(qvals).item()
        return max_val
    
    def reset_buffer(self):
        self.buffer = np.empty(self.batch_size, dtype=object)
        self.buffer_size = 0
    
    def compute_loss(self, cur_vals, target):
        # ref: https://github.com/deepmind/deepmind-research/blob/f5de0ede8430809180254ee957abf36ed62579ef/option_keyboard/keyboard_agent.py#L111
        loss = nn.MSELoss(reduction="sum")(cur_vals, target) / 2
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()
        return loss.item()

class RandomAgent():
    def __init__(self, env) -> None:
        self.env = env
        super().__init__()

    def get_action(self, state):
        return self.env.action_space.sample()

class GPEAgent(SFAgent):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        feat_dim: int,
        w: NDArray,
        lr: float,
        gamma: float,
        epsilon: float,
        psi: PsiNet,
    ) -> None:
        super().__init__(state_dim, action_dim, feat_dim, w, lr, gamma, epsilon)
        self.og_psi = copy.deepcopy(psi).eval()
        self.og_psi.to(self.device)

    def update(self, state, action, next_state, terminated):
        phi = self.phi(state, next_state)
        phi = torch.from_numpy(phi).to(self.device)
        state = torch.from_numpy(state).to(self.device)
        action = torch.from_numpy(np.array([action])).to(self.device).view(-1)
        next_state = torch.from_numpy(next_state).to(self.device)
        terminated = np.multiply(np.array([terminated]), 1)
        terminated = torch.from_numpy(terminated).to(self.device).view(-1)
        # add this transition to buffer
        self.buffer[self.buffer_size] = (state, action, phi, next_state, terminated)
        self.buffer_size += 1
        # once buffer is big enough do batch update
        if self.buffer_size == self.batch_size:
            # shuffle buffer
            indices = np.random.randint(
                low=0, high=self.buffer_size, size=(self.batch_size,)
            )
            states, actions, phis, next_states, terms = zip(*self.buffer[indices])
            # convert to torch tensors of appropriate shape
            states = torch.vstack(states).reshape((self.batch_size, self.state_dim)).to(self.device)
            actions = torch.vstack(actions).to(self.device)
            phis = torch.vstack(phis).reshape((self.batch_size, self.feat_dim)).to(self.device)
            next_states = torch.vstack(next_states).reshape((self.batch_size, self.state_dim)).to(self.device)
            terms = torch.vstack(terms).to(self.device)
            # compute current values
            cur_psi = (
                self.psi(states)
                .gather(1, actions.unsqueeze(-1).expand(-1, -1, self.feat_dim))
                .squeeze(1)
                .to(self.device)
            )
            # compute target values
            with torch.no_grad():
                next_psi = self.psi(next_states).to(self.device)
                next_qs = torch.matmul(
                    self.og_psi(next_states).to(self.device), self.w.to(self.device)
                )
                next_actions = torch.argmax(next_qs, dim=1).unsqueeze(-1)
                next_psi = (
                    next_psi.gather(
                        1, next_actions.unsqueeze(-1).expand(-1, -1, self.feat_dim)
                    )
                    .squeeze(1)
                    .to(self.device)
                )
                target = phis + self.gamma * (1 - terms) * next_psi
            self.reset_buffer()
            return self.compute_loss(cur_psi, target)
        return 0

class IndSFDQNAgent(SFAgent):

    def update(self, state, action, next_state, terminated):
        phi = self.phi(state, next_state)
        phi = torch.from_numpy(phi).to(self.device)
        state = torch.from_numpy(state).to(self.device)
        action = torch.from_numpy(np.array([action])).to(self.device).view(-1)
        next_state = torch.from_numpy(next_state).to(self.device)
        terminated = np.multiply(np.array([terminated]), 1)
        terminated = torch.from_numpy(terminated).to(self.device).view(-1)
        # add this transition to buffer
        self.buffer[self.buffer_size] = (state, action, phi, next_state, terminated)
        self.buffer_size += 1
        # once buffer is big enough do batch update
        if self.buffer_size == self.batch_size:
            # shuffle buffer
            indices = np.random.randint(
                low=0, high=self.buffer_size, size=(self.batch_size,)
            )
            states, actions, phis, next_states, terms = zip(*self.buffer[indices])
            # convert to torch tensors of appropriate shape
            states = torch.vstack(states).reshape((self.batch_size, self.state_dim)).to(self.device)
            actions = torch.vstack(actions).to(self.device)
            phis = torch.vstack(phis).reshape((self.batch_size, self.feat_dim)).to(self.device)
            next_states = torch.vstack(next_states).reshape((self.batch_size, self.state_dim)).to(self.device)
            terms = torch.vstack(terms).to(self.device)
            
            # compute current values
            cur_psi = (
                self.psi(states)
                .gather(1, actions.unsqueeze(-1).expand(-1, -1, self.feat_dim))
                .squeeze(1)
                .to(self.device)
            )
            # compute target values
            with torch.no_grad():
                next_psi = self.psi(next_states).to(self.device)
                next_qs = torch.matmul(next_psi, self.w.to(self.device))
                next_actions = torch.argmax(next_qs, dim=1).unsqueeze(-1)
                next_psi = (
                    next_psi.gather(
                        1, next_actions.unsqueeze(-1).expand(-1, -1, self.feat_dim)
                    )
                    .squeeze(1)
                    .to(self.device)
                )
                target = phis + self.gamma * (1 - terms) * next_psi
            self.reset_buffer()
            return self.compute_loss(cur_psi, target)
        return 0

class QdrAgent():
    def __init__(self, qdr) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.q_network = qdr
        self.action_dim = qdr.action_dim
        self.w = np.array([1.0, 1.0, 1.0])

    def get_action(self, state, action_epsilon=0.0):
        # epsilon greedy
        if np.random.rand() < action_epsilon:
            action = np.random.randint(self.action_dim)
        else:
            with torch.no_grad():
                state = torch.from_numpy(state).to(self.device)
                q_values = self.q_network(state).to(self.device)
                action = torch.argmax(q_values).item()
        return action

class DoubleSFAgent(SFAgent):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        feat_dim: int,
        w: NDArray,
        lr: float,
        gamma: float,
        epsilon: float,
        psi_a: PsiNet = None,
        psi_b: PsiNet = None,
    ) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if psi_a and psi_b:
            self.psi_a = copy.deepcopy(psi_a).train()
            self.psi_b = copy.deepcopy(psi_b).train()
        else:
            self.psi_a = PsiNet(state_dim, action_dim, feat_dim)
            self.psi_b = PsiNet(state_dim, action_dim, feat_dim)
        self.optim_a = optim.Adam(self.psi_a.parameters(), lr=lr)
        self.optim_b = optim.Adam(self.psi_b.parameters(), lr=lr)
        self.w = torch.from_numpy(w).float()
        self.gamma = gamma
        self.epsilon = epsilon
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.feat_dim = feat_dim
        self.batch_size = 10
        self.buffer_a = np.empty(self.batch_size, dtype=object)
        self.buffer_size_a = 0
        self.buffer_b = np.empty(self.batch_size, dtype=object)
        self.buffer_size_b = 0
    
    def update(self, state, action, next_state, terminated):
        phi = self.phi(state, next_state)
        phi = torch.from_numpy(phi).to(self.device)
        state = torch.from_numpy(state).to(self.device)
        action = torch.from_numpy(np.array([action])).to(self.device).view(-1)
        next_state = torch.from_numpy(next_state).to(self.device)
        terminated = np.multiply(np.array([terminated]), 1)
        terminated = torch.from_numpy(terminated).to(self.device).view(-1)
        transition = (state, action, phi, next_state, terminated)
        if np.random.random() < 0.5:
            return self.update_a(transition)
        else:
            return self.update_b(transition)
        return 0
    
    def update_a(self, transition):
        # add this transition to buffer
        self.buffer_a[self.buffer_size_a] = transition
        self.buffer_size_a += 1
        # once buffer is big enough do batch update
        if self.buffer_size_a == self.batch_size:
            # shuffle buffer
            indices = np.random.randint(
                low=0, high=self.buffer_size_a, size=(self.batch_size,)
            )
            states, actions, phis, next_states, terms = zip(*self.buffer_a[indices])
            # convert to torch tensors of appropriate shape
            states = torch.vstack(states).reshape((self.batch_size, self.state_dim)).to(self.device)
            actions = torch.vstack(actions).to(self.device)
            phis = torch.vstack(phis).reshape((self.batch_size, self.feat_dim)).to(self.device)
            next_states = torch.vstack(next_states).reshape((self.batch_size, self.state_dim)).to(self.device)
            terms = torch.vstack(terms).to(self.device)
            
            # compute current values
            cur_psi = (
                self.psi_a(states)
                .gather(1, actions.unsqueeze(-1).expand(-1, -1, self.feat_dim))
                .squeeze(1)
                .to(self.device)
            )
            # compute target values
            with torch.no_grad():
                next_psi = self.psi_a(next_states).to(self.device)
                next_qs = torch.matmul(next_psi, self.w.to(self.device))
                next_actions = torch.argmax(next_qs, dim=1).unsqueeze(-1)
                next_psi_b = self.psi_b(next_states).to(self.device)
                next_psi_b = (
                    next_psi_b.gather(
                        1, next_actions.unsqueeze(-1).expand(-1, -1, self.feat_dim)
                    )
                    .squeeze(1)
                    .to(self.device)
                )
                target = phis + self.gamma * (1 - terms) * next_psi_b
            self.buffer_a = np.empty(self.batch_size, dtype=object)
            self.buffer_size_a = 0
            loss = nn.MSELoss(reduction="sum")(cur_psi, target) / 2
            self.optim_a.zero_grad()
            loss.backward()
            self.optim_a.step()
            return loss.item()
        return 0
    
    def update_b(self, transition):
        # add this transition to buffer
        self.buffer_b[self.buffer_size_b] = transition
        self.buffer_size_b += 1
        # once buffer is big enough do batch update
        if self.buffer_size_b == self.batch_size:
            # shuffle buffer
            indices = np.random.randint(
                low=0, high=self.buffer_size_b, size=(self.batch_size,)
            )
            states, actions, phis, next_states, terms = zip(*self.buffer_b[indices])
            # convert to torch tensors of appropriate shape
            states = torch.vstack(states).reshape((self.batch_size, self.state_dim)).to(self.device)
            actions = torch.vstack(actions).to(self.device)
            phis = torch.vstack(phis).reshape((self.batch_size, self.feat_dim)).to(self.device)
            next_states = torch.vstack(next_states).reshape((self.batch_size, self.state_dim)).to(self.device)
            terms = torch.vstack(terms).to(self.device)
            
            # compute current values
            cur_psi = (
                self.psi_b(states)
                .gather(1, actions.unsqueeze(-1).expand(-1, -1, self.feat_dim))
                .squeeze(1)
                .to(self.device)
            )
            # compute target values
            with torch.no_grad():
                next_psi = self.psi_b(next_states).to(self.device)
                next_qs = torch.matmul(next_psi, self.w.to(self.device))
                next_actions = torch.argmax(next_qs, dim=1).unsqueeze(-1)
                next_psi_a = self.psi_a(next_states).to(self.device)
                next_psi_a = (
                    next_psi_a.gather(
                        1, next_actions.unsqueeze(-1).expand(-1, -1, self.feat_dim)
                    )
                    .squeeze(1)
                    .to(self.device)
                )
                target = phis + self.gamma * (1 - terms) * next_psi_a
            self.buffer_b = np.empty(self.batch_size, dtype=object)
            self.buffer_size_b = 0
            loss = nn.MSELoss(reduction="sum")(cur_psi, target) / 2
            self.optim_b.zero_grad()
            loss.backward()
            self.optim_b.step()
            return loss.item()
        return 0
    
    def get_qvals(self, state):
        with torch.no_grad():
            state = torch.from_numpy(state).to(self.device)
            self.psi_a.to(self.device)
            self.psi_b.to(self.device)
            qvals_a = torch.matmul(
                self.psi_a(state).to(self.device), self.w.to(self.device)
            )
            qvals_b = torch.matmul(
                self.psi_b(state).to(self.device), self.w.to(self.device)
            )
        return (qvals_a + qvals_b) / 2

class AvgSFAgent(SFAgent):
    def update(self, state, action, next_state):
        phi = self.phi(state, next_state)
        phi = torch.from_numpy(phi).to(self.device)
        state = torch.from_numpy(state).to(self.device)
        action = torch.from_numpy(np.array([action])).to(self.device).view(-1)
        next_state = torch.from_numpy(next_state).to(self.device)
        # add this transition to buffer
        self.buffer[self.buffer_size] = (state, action, phi, next_state)
        self.buffer_size += 1
        # once buffer is big enough do batch update
        if self.buffer_size == self.batch_size:
            # shuffle buffer
            indices = np.random.randint(
                low=0, high=self.buffer_size, size=(self.batch_size,)
            )
            states, actions, phis, next_states = zip(*self.buffer[indices])
            # convert to torch tensors of appropriate shape
            states = torch.vstack(states).reshape((self.batch_size, self.state_dim)).to(self.device)
            actions = torch.vstack(actions).to(self.device)
            phis = torch.vstack(phis).reshape((self.batch_size, self.feat_dim)).to(self.device)
            avg_phis = phis.mean(dim=0)
            next_states = torch.vstack(next_states).reshape((self.batch_size, self.state_dim)).to(self.device)
            
            # compute current values
            cur_psi = (
                self.psi(states)
                .gather(1, actions.unsqueeze(-1).expand(-1, -1, self.feat_dim))
                .squeeze(1)
                .to(self.device)
            )
            # compute target values
            with torch.no_grad():
                next_psi = self.psi(next_states).to(self.device)
                next_qs = torch.matmul(next_psi, self.w.to(self.device))
                next_actions = torch.argmax(next_qs, dim=1).unsqueeze(-1)
                next_psi = (
                    next_psi.gather(
                        1, next_actions.unsqueeze(-1).expand(-1, -1, self.feat_dim)
                    )
                    .squeeze(1)
                    .to(self.device)
                )
                target = (phis - avg_phis) + self.gamma * next_psi
            self.reset_buffer()
            return self.compute_loss(cur_psi, target)
        return 0

class SFPartnerAgent(SFAgent):
    def __init__(
        self, dirname: str, filename: str, w: NDArray
    ) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.psi = torch.load(dirname + filename + "_psi.torch", map_location=self.device)
        self.feat_dim = self.psi.feature_dim
        self.action_dim = self.psi.action_dim
        self.psi.eval()
        self.psi.to(self.device)
        self.w = np.array(w)
        self.w = torch.from_numpy(self.w).float()
        self.epsilon = 0.0

class DoubleSFPartnerAgent(SFAgent):
    def __init__(
        self, dirname: str, filename: str, w: NDArray
    ) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.psi_a = torch.load(dirname + filename + "_psi_a.torch", map_location=self.device)
        self.psi_b = torch.load(dirname + filename + "_psi_b.torch", map_location=self.device)
        self.feat_dim = self.psi_a.feature_dim
        self.action_dim = self.psi_a.action_dim
        self.psi_a.eval()
        self.psi_a.to(self.device)
        self.psi_b.eval()
        self.psi_b.to(self.device)
        self.w = np.array(w)
        self.w = torch.from_numpy(self.w).float()
        self.epsilon = 0.0
    
    def get_qvals(self, state):
        with torch.no_grad():
            state = torch.from_numpy(state).to(self.device)
            self.psi_a.to(self.device)
            self.psi_b.to(self.device)
            qvals_a = torch.matmul(
                self.psi_a(state).to(self.device), self.w.to(self.device)
            )
            qvals_b = torch.matmul(
                self.psi_b(state).to(self.device), self.w.to(self.device)
            )
        return (qvals_a + qvals_b) / 2

class GPIAgent():
    def __init__(
        self,
        pretrained,
        state_dim: int = None,
        action_dim: int = None,
        feat_dim: int = None,
    ) -> None:
        self.pretrained = [copy.deepcopy(model).eval() for model in pretrained]
        self.qdr_set = self.pretrained
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.w = np.array([1.0, 1.0, 1.0])
        self.pol_idx = []

    def get_action(self, state, action_epsilon=None):
        if np.random.rand() < action_epsilon:
            action = np.random.randint(self.action_dim)
        else:
            state = torch.from_numpy(state).to(self.device)
            qvals = []
            for q, qdr in enumerate(self.qdr_set):
                qdr.to(self.device)
                qvals.append(qdr(state).to(self.device))
            qvals = torch.stack(qvals)
            loc = torch.argmax(qvals).item()
            action = loc % self.action_dim
            idx = loc // self.action_dim
            self.pol_idx.append(idx)
        return action

    def get_max_qval(self, state):
        with torch.no_grad():
            state = torch.from_numpy(state).to(self.device)
            qvals = []
            for q, qdr in enumerate(self.qdr_set):
                qdr.to(self.device)
                vals = qdr(state).to(self.device)
                qvals.append(vals.squeeze(0).detach())
            qvals = torch.max(torch.stack(qvals), dim=1).values
        return qvals

    def get_qvals(self, state):
        with torch.no_grad():
            state = torch.from_numpy(state).to(self.device)
            qvals = []
            for q, qdr in enumerate(self.qdr_set):
                qdr.to(self.device)
                vals = qdr(state).to(self.device)
                qvals.append(vals.squeeze(0).detach())
            qvals = torch.max(torch.stack(qvals), dim=0).values
        return qvals.squeeze(0)
class SFGPIAgent(SFAgent):
    def __init__(
        self,
        pretrained,
        ws,
        learn_new: bool = None,
        state_dim: int = None,
        action_dim: int = None,
        feat_dim: int = None,
        epsilon=0.0,
        rate=0,
    ) -> None:
        self.pretrained = [copy.deepcopy(model).eval() for model in pretrained]
        self.psi_set = self.pretrained
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.feat_dim = feat_dim
        if learn_new is not None:
            self.learn_new = learn_new
            if learn_new:
                self.new_psi = PsiNet(state_dim, action_dim, feat_dim)
                self.optim = optim.Adam(self.new_psi.parameters(), lr=3e-5)
                self.gamma = 0.95
                self.device = torch.device(
                    "cuda" if torch.cuda.is_available() else "cpu"
                )
                self.batch_size = 10
                self.buffer = np.empty(self.batch_size, dtype=object)
                self.buffer_size = 0
                self.psi_set.append(self.new_psi)
        else:
            self.learn_new = False
        self.ws = ws
        self.w = np.array([1.0, 1.0, 1.0, 1.0])
        self.pol_idx = []
        self.epsilon = epsilon
        self.rate = rate
        self.max_epsilon = 1.0

    def get_action(self, state, action_epsilon=None):
        if np.random.rand() < action_epsilon:
            action = np.random.randint(self.action_dim)
        else:
            state = torch.from_numpy(state).to(self.device)
            qvals = []
            if self.learn_new and np.random.rand() > self.epsilon:
                # for early training phase of new psi
                psi_using = self.psi_set[:-1]
            else:
                psi_using = self.psi_set
            self.epsilon = min(self.max_epsilon, self.epsilon * self.rate)
            for p, psi in enumerate(psi_using):
                psi.to(self.device)
                vals = torch.matmul(psi(state).to(self.device), self.ws[p].to(self.device))
                qvals.append(vals)
            qvals = torch.stack(qvals)
            loc = torch.argmax(qvals).item()
            action = loc % self.action_dim
            idx = loc // self.action_dim
            self.pol_idx.append(idx)
        return action

    def get_max_qval(self, state):
        with torch.no_grad():
            state = torch.from_numpy(state).to(self.device)
            qvals = []
            for p, psi in enumerate(self.psi_set):
                psi.to(self.device)
                vals = torch.matmul(psi(state).to(self.device), self.ws[p].to(self.device))
                qvals.append(vals.squeeze(0).detach())
            qvals = torch.max(torch.stack(qvals), dim=1).values
            #max_vals = torch.max(qvals, dim=0).values
        return qvals

    def get_qvals(self, state):
        with torch.no_grad():
            state = torch.from_numpy(state).to(self.device)
            qvals = []
            for p, psi in enumerate(self.psi_set):
                psi.to(self.device)
                vals = torch.matmul(psi(state).to(self.device), self.ws[p].to(self.device))
                qvals.append(vals.squeeze(0).detach())
            qvals = torch.max(torch.stack(qvals), dim=0).values
        return qvals.squeeze(0)

    def update_pretrained(self, pretrained) -> None:
        self.pretrained = [copy.deepcopy(model).eval() for model in pretrained]
        self.psi_set = self.pretrained
        if self.learn_new:
            self.psi_set.append(self.new_psi)

    def update_psi(self, state, action, next_state, terminated):
        if not self.learn_new:
            return
        phi = self.phi(state, next_state)
        phi = torch.from_numpy(phi).to(self.device)
        state = torch.from_numpy(state).to(self.device)
        action = torch.from_numpy(np.array([action])).to(self.device).view(-1)
        next_state = torch.from_numpy(next_state).to(self.device)
        terminated = np.multiply(np.array([terminated]), 1)
        terminated = torch.from_numpy(terminated).to(self.device).view(-1)
        # add this transition to buffer
        self.buffer[self.buffer_size] = (state, action, phi, next_state, terminated)
        self.buffer_size += 1
        # once buffer is big enough do batch update
        if self.buffer_size == self.batch_size:
            # shuffle buffer
            indices = np.random.randint(
                low=0, high=self.buffer_size, size=(self.batch_size,)
            )
            states, actions, phis, next_states, terms = zip(*self.buffer[indices])
            # convert to torch tensors of appropriate shape
            states = torch.vstack(states).reshape((self.batch_size, self.state_dim)).to(self.device)
            actions = torch.vstack(actions).to(self.device)
            phis = torch.vstack(phis).reshape((self.batch_size, self.feat_dim)).to(self.device)
            next_states = torch.vstack(next_states).reshape((self.batch_size, self.state_dim)).to(self.device)
            terms = torch.vstack(terms).to(self.device)
            # compute current values
            cur_psi = (
                self.new_psi(states)
                .gather(1, actions.unsqueeze(-1).expand(-1, -1, self.feat_dim))
                .squeeze(1)
                .to(self.device)
            )
            # compute target values
            with torch.no_grad():
                next_psi = self.new_psi(next_states).to(self.device)
                next_qs = torch.matmul(next_psi, self.w.to(self.device))
                next_actions = torch.argmax(next_qs, dim=1).unsqueeze(-1)
                next_psi = (
                    next_psi.gather(
                        1, next_actions.unsqueeze(-1).expand(-1, -1, self.feat_dim)
                    )
                    .squeeze(1)
                    .to(self.device)
                )
                target = phis + self.gamma * (1 - terms) * next_psi
            self.reset_buffer()
            return self.compute_loss(cur_psi, target)
        return 0

class MeshAgent(SFAgent):
    def __init__(
        self,
        pretrained,
        state_dim: int = None,
        action_dim: int = None,
        feat_dim: int = None,
        epsilon=0.0,
    ) -> None:
        self.pretrained = [copy.deepcopy(model).eval() for model in pretrained]
        self.psi_set = self.pretrained
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.feat_dim = feat_dim
        self.w = torch.from_numpy(np.ones(self.feat_dim)).float()
        self.epsilon = epsilon

    def get_action(self, state, action_epsilon=0.0):
        if np.random.rand() < action_epsilon:
            action = np.random.randint(self.action_dim)
        else:
            state = torch.from_numpy(state).to(self.device)
            qvals = []
            psi_using = self.psi_set
            for p, psi in enumerate(psi_using):
                psi.to(self.device)
                vals = torch.matmul(psi(state).to(self.device), self.w.to(self.device))
                qvals.append(vals)
            qvals = torch.stack(qvals)
            avg_qvals = torch.mean(qvals, 1)
            action = torch.argmax(avg_qvals)
        return action

class DoubleSFGPIAgent(SFAgent):
    def __init__(
        self,
        pretrained,
        w,
        learn_new: bool = None,
        state_dim: int = None,
        action_dim: int = None,
        feat_dim: int = None,
        epsilon=0.0,
        rate=0,
    ) -> None:
        self.pretrained = [copy.deepcopy(model).eval() for model in pretrained]
        self.psi_set = self.pretrained
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.feat_dim = feat_dim
        self.w = torch.from_numpy(w).float()
        self.pol_idx = []
        self.epsilon = epsilon
        self.rate = rate
        self.max_epsilon = 1.0
        if learn_new is not None:
            self.learn_new = learn_new
            if learn_new:
                self.new_agent = DoubleSFAgent(state_dim=self.state_dim, action_dim=self.action_dim, feat_dim=self.feat_dim, w=self.w, lr=3e-5, gamma=0.95, epsilon=0.1)
                self.psi_set.append(self.new_agent.psi_a)
                self.psi_set.append(self.new_agent.psi_b)
        else:
            self.learn_new = False

    def get_action(self, state, action_epsilon=None):
        if np.random.rand() < action_epsilon:
            action = np.random.randint(self.action_dim)
        else:
            state = torch.from_numpy(state).to(self.device)
            qvals = []
            if self.learn_new and np.random.rand() > self.epsilon:
                # for early training phase of new psi
                psi_using = self.psi_set[:-2]
            else:
                psi_using = self.psi_set
            self.epsilon = min(self.max_epsilon, self.epsilon * self.rate)
            num_policies = len(psi_using) // 2
            #print('models being used: ', len(psi_using))
            #print('number of policies: ', num_policies)
            for i in range(num_policies):
                psi_a = psi_using[i].to(self.device)
                psi_b = psi_using[i+1].to(self.device)
                qvals_a = torch.matmul(
                    psi_a(state).to(self.device), self.w.to(self.device)
                )
                qvals_b = torch.matmul(
                    psi_b(state).to(self.device), self.w.to(self.device)
                )
                vals = (qvals_a + qvals_b) / 2
                qvals.append(vals.squeeze(0).detach())
            qvals = torch.stack(qvals)
            #print('qvals shape: ', qvals.shape)
            loc = torch.argmax(qvals).item()
            action = loc % self.action_dim
            idx = loc // self.action_dim
            self.pol_idx.append(idx)
        return action

    def get_max_qval(self, state):
        with torch.no_grad():
            state = torch.from_numpy(state).to(self.device)
            qvals = []
            num_policies = len(self.psi_set) // 2
            for i in range(num_policies):
                psi_a = self.psi_set[i].to(self.device)
                psi_b = self.psi_set[i+1].to(self.device)
                qvals_a = torch.matmul(
                    psi_a(state).to(self.device), self.w.to(self.device)
                )
                qvals_b = torch.matmul(
                    psi_b(state).to(self.device), self.w.to(self.device)
                )
                vals = (qvals_a + qvals_b) / 2
                qvals.append(vals.squeeze(0).detach())
            qvals = torch.max(torch.stack(qvals), dim=1).values
        return qvals

    def get_qvals(self, state):
        with torch.no_grad():
            state = torch.from_numpy(state).to(self.device)
            qvals = []
            num_policies = len(self.psi_set) // 2
            for i in range(num_policies):
                psi_a = self.psi_set[i].to(self.device)
                psi_b = self.psi_set[i+1].to(self.device)
                qvals_a = torch.matmul(
                    psi_a(state).to(self.device), self.w.to(self.device)
                )
                qvals_b = torch.matmul(
                    psi_b(state).to(self.device), self.w.to(self.device)
                )
                vals = (qvals_a + qvals_b) / 2
                qvals.append(vals.squeeze(0).detach())
            qvals = torch.max(torch.stack(qvals), dim=0).values
        return qvals.squeeze(0)

    def update_pretrained(self, pretrained) -> None:
        self.pretrained = [copy.deepcopy(model).eval() for model in pretrained]
        self.psi_set = self.pretrained
        if self.learn_new:
            self.psi_set.append(self.new_agent.psi_a)
            self.psi_set.append(self.new_agent.psi_b)

    def update_psi(self, state, action, next_state, terminated):
        if not self.learn_new:
            return
        self.new_agent.update(state, action, next_state, terminated)
class DuelSFAgent(SFAgent):
    def __init__(self, *args) -> None:
        super().__init__(*args)
        self.psi = PsiNet(self.state_dim, self.action_dim, self.feat_dim)
        self.psi_a = PsiNet(self.state_dim, self.action_dim, self.feat_dim)