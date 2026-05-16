"""
train_v2.py — Improved Double DQN for SUMO Traffic Signal Control

Improvements over baseline train.py:
1. Double DQN (separate target network) — fixes Q-value chasing
2. Soft target updates (Polyak averaging, τ=0.005)
3. Richer state (12 dims: counts + waiting times + current phase one-hot)
4. Delta-reward (change in waiting time — better credit assignment)
5. Huber loss instead of MSE — robust to outlier rewards
6. Gradient clipping — training stability
7. Proper (s, a, r, s') tuple alignment (carries over from train.py fix)
8. Warmup phase before any learning starts
"""

from __future__ import absolute_import
from __future__ import print_function

import os
import sys
import optparse
import copy
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
import matplotlib.pyplot as plt

# SUMO setup
if "SUMO_HOME" in os.environ:
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    sys.path.append(tools)
else:
    sys.exit("please declare environment variable 'SUMO_HOME'")

from sumolib import checkBinary
import traci

# ── Helper functions ─────────────────────────────────────────────────────────

def get_vehicle_numbers(lanes):
    out = dict()
    for l in lanes:
        out[l] = 0
        for vid in traci.lane.getLastStepVehicleIDs(l):
            if traci.vehicle.getLanePosition(vid) > 10:
                out[l] += 1
    return out


def get_waiting_time_per_lane(unique_lanes):
    """Return per-lane waiting time (not summed) for richer state."""
    return {l: traci.lane.getWaitingTime(l) for l in unique_lanes}


def total_waiting(unique_lanes):
    return sum(traci.lane.getWaitingTime(l) for l in unique_lanes)


def phaseDuration(junction, phase_time, phase_state):
    traci.trafficlight.setRedYellowGreenState(junction, phase_state)
    traci.trafficlight.setPhaseDuration(junction, phase_time)


# ── Network ──────────────────────────────────────────────────────────────────

class QNet(nn.Module):
    def __init__(self, input_dims, n_actions, fc1=128, fc2=128):
        super().__init__()
        self.fc1 = nn.Linear(input_dims, fc1)
        self.fc2 = nn.Linear(fc1, fc2)
        self.out = nn.Linear(fc2, n_actions)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.out(x)


# ── Double DQN Agent ─────────────────────────────────────────────────────────

class DoubleDQNAgent:
    def __init__(self, input_dims, n_actions,
                 lr=5e-4, gamma=0.99,
                 epsilon_start=1.0, epsilon_end=0.05, epsilon_decay=0.995,
                 batch_size=128, memory_size=100000,
                 tau=0.005, warmup_steps=1000):
        self.input_dims  = input_dims
        self.n_actions   = n_actions
        self.gamma       = gamma
        self.epsilon     = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size  = batch_size
        self.tau         = tau
        self.warmup_steps = warmup_steps

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Online + target networks
        self.q_online = QNet(input_dims, n_actions).to(self.device)
        self.q_target = copy.deepcopy(self.q_online).to(self.device)
        self.q_target.eval()
        self.optim    = optim.Adam(self.q_online.parameters(), lr=lr)

        # Replay buffer
        self.max_mem = memory_size
        self.state_mem      = np.zeros((self.max_mem, input_dims), dtype=np.float32)
        self.next_state_mem = np.zeros((self.max_mem, input_dims), dtype=np.float32)
        self.action_mem     = np.zeros(self.max_mem, dtype=np.int64)
        self.reward_mem     = np.zeros(self.max_mem, dtype=np.float32)
        self.done_mem       = np.zeros(self.max_mem, dtype=np.bool_)
        self.mem_cntr       = 0
        self.train_steps    = 0

    def store(self, s, a, r, s_, done):
        idx = self.mem_cntr % self.max_mem
        self.state_mem[idx]      = s
        self.action_mem[idx]     = a
        self.reward_mem[idx]     = r
        self.next_state_mem[idx] = s_
        self.done_mem[idx]       = done
        self.mem_cntr += 1

    def choose_action(self, state):
        if np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)
        with torch.no_grad():
            s = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            q = self.q_online(s)
            return int(torch.argmax(q, dim=1).item())

    def soft_update_target(self):
        """Polyak averaging: θ_target = τ·θ_online + (1-τ)·θ_target"""
        for p_t, p_o in zip(self.q_target.parameters(), self.q_online.parameters()):
            p_t.data.copy_(self.tau * p_o.data + (1.0 - self.tau) * p_t.data)

    def learn(self):
        if self.mem_cntr < self.warmup_steps:
            return None  # still warming up

        # Sample batch
        max_mem = min(self.mem_cntr, self.max_mem)
        idx = np.random.choice(max_mem, self.batch_size, replace=False)

        s  = torch.tensor(self.state_mem[idx],      device=self.device)
        a  = torch.tensor(self.action_mem[idx],     device=self.device)
        r  = torch.tensor(self.reward_mem[idx],     device=self.device)
        s_ = torch.tensor(self.next_state_mem[idx], device=self.device)
        d  = torch.tensor(self.done_mem[idx],       device=self.device)

        # Current Q(s, a)
        q_pred = self.q_online(s).gather(1, a.unsqueeze(1)).squeeze(1)

        # Double DQN target:
        #   1. Pick best action using ONLINE net
        #   2. Evaluate that action with TARGET net
        with torch.no_grad():
            best_actions = self.q_online(s_).argmax(dim=1, keepdim=True)
            q_next       = self.q_target(s_).gather(1, best_actions).squeeze(1)
            q_next[d]    = 0.0
            q_target     = r + self.gamma * q_next

        # Huber loss (smoother than MSE for large reward magnitudes)
        loss = F.smooth_l1_loss(q_pred, q_target)

        self.optim.zero_grad()
        loss.backward()
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.q_online.parameters(), 10.0)
        self.optim.step()

        # Soft target update every step
        self.soft_update_target()

        self.train_steps += 1
        return float(loss.item())

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_end)

    def save(self, name):
        torch.save({
            'q_online': self.q_online.state_dict(),
            'q_target': self.q_target.state_dict(),
            'input_dims': self.input_dims,
            'n_actions': self.n_actions,
        }, f'models/{name}.bin')

    def load(self, name):
        ckpt = torch.load(f'models/{name}.bin', map_location=self.device)
        self.q_online.load_state_dict(ckpt['q_online'])
        if 'q_target' in ckpt:
            self.q_target.load_state_dict(ckpt['q_target'])


# ── Phase string builder ─────────────────────────────────────────────────────

def build_phase_strings(state_length, n_actions):
    signals_per_group = state_length // n_actions
    phases = []
    for i in range(n_actions):
        yellow = 'r' * (i * signals_per_group) + 'y' * signals_per_group + \
                 'r' * ((n_actions - i - 1) * signals_per_group)
        green  = 'r' * (i * signals_per_group) + 'G' * signals_per_group + \
                 'r' * ((n_actions - i - 1) * signals_per_group)
        yellow = yellow.ljust(state_length, 'r')[:state_length]
        green  = green.ljust(state_length, 'r')[:state_length]
        phases.append([yellow, green])
    return phases


# ── State builder ────────────────────────────────────────────────────────────

def build_state(unique_lanes, current_action, n_actions):
    """
    Rich state: [vehicle counts (n_actions) + waiting times normalized (n_actions) + phase one-hot (n_actions)]
    """
    counts = []
    waits  = []
    for lane in unique_lanes:
        # count vehicles on this lane (only those past 10m)
        c = 0
        for vid in traci.lane.getLastStepVehicleIDs(lane):
            if traci.vehicle.getLanePosition(vid) > 10:
                c += 1
        counts.append(c)
        # waiting time on this lane (normalized to seconds, then scaled)
        waits.append(traci.lane.getWaitingTime(lane) / 100.0)  # rough normalization

    # one-hot of current action
    phase_oh = [0.0] * n_actions
    if 0 <= current_action < n_actions:
        phase_oh[current_action] = 1.0

    state = counts + waits + phase_oh
    # pad if unique_lanes < n_actions
    while len(state) < 3 * n_actions:
        state.append(0.0)
    return state[:3 * n_actions]


# ── Training loop ────────────────────────────────────────────────────────────

def run_training(model_name="ddqn_model", epochs=2000, steps=500):
    # Detect network
    traci.start([checkBinary("sumo"), "-c", "configuration.sumocfg",
                 "--no-warnings", "true"])
    all_junctions  = traci.trafficlight.getIDList()
    sample         = all_junctions[0]
    state_length   = len(traci.trafficlight.getRedYellowGreenState(sample))
    controlled     = traci.trafficlight.getControlledLanes(sample)
    unique_lanes   = list(dict.fromkeys(controlled))
    n_actions      = len(unique_lanes)
    traci.close()

    input_dims  = 3 * n_actions  # counts + waits + phase one-hot
    select_lane = build_phase_strings(state_length, n_actions)

    print(f"Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    print(f"State length: {state_length}, n_actions: {n_actions}, input_dims: {input_dims}")
    print(f"Unique lanes: {unique_lanes}")

    # Build agent
    agent = DoubleDQNAgent(
        input_dims=input_dims,
        n_actions=n_actions,
        lr=5e-4,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.05,
        epsilon_decay=0.995,
        batch_size=128,
        memory_size=100000,
        tau=0.005,
        warmup_steps=1000,
    )

    best_time       = np.inf
    total_time_list = []
    loss_list       = []
    min_duration    = 5

    for e in range(epochs):
        traci.start([checkBinary("sumo"), "-c", "configuration.sumocfg",
                     "--no-warnings", "true"])

        step                  = 0
        total_time            = 0.0
        traffic_lights_time   = {j: 0 for j in all_junctions}
        pending               = {jn: None for jn in range(len(all_junctions))}
        pending_wait_start    = {jn: 0.0  for jn in range(len(all_junctions))}
        current_phase_action  = {jn: 0    for jn in range(len(all_junctions))}
        epoch_losses          = []

        while step <= steps:
            try:
                traci.simulationStep()
            except Exception:
                break

            for jn, junction in enumerate(all_junctions):
                try:
                    waiting_now = total_waiting(unique_lanes)
                except Exception:
                    break

                total_time += waiting_now

                if traffic_lights_time[junction] == 0:
                    # Build rich state
                    next_state = build_state(unique_lanes,
                                             current_phase_action[jn], n_actions)

                    # Close out previous decision with proper (s, a, r, s')
                    if pending[jn] is not None:
                        prev_state, prev_action = pending[jn]
                        # Delta reward: how much waiting time changed during prev action
                        delta_wait = waiting_now - pending_wait_start[jn]
                        # Reward = negative delta (lower waiting growth = better)
                        # Normalize so reward magnitudes stay manageable
                        reward = -delta_wait / 1000.0
                        agent.store(prev_state, prev_action, reward, next_state, (step == steps))
                        loss = agent.learn()
                        if loss is not None:
                            epoch_losses.append(loss)

                    # Choose new action
                    action = agent.choose_action(next_state)
                    pending[jn]               = (next_state, action)
                    pending_wait_start[jn]    = waiting_now
                    current_phase_action[jn]  = action

                    phaseDuration(junction, 3, select_lane[action][0])  # yellow
                    phaseDuration(junction, min_duration + 10, select_lane[action][1])  # green
                    traffic_lights_time[junction] = min_duration + 10
                else:
                    traffic_lights_time[junction] -= 1

            step += 1

        agent.decay_epsilon()

        try:
            traci.close()
        except Exception:
            pass

        mean_loss = np.mean(epoch_losses) if epoch_losses else 0.0
        total_time_list.append(total_time)
        loss_list.append(mean_loss)

        if total_time < best_time and agent.mem_cntr > agent.warmup_steps:
            best_time = total_time
            agent.save(model_name)
            saved_msg = " ★ BEST SAVED"
        else:
            saved_msg = ""

        if e % 10 == 0 or e < 5:
            print(f"epoch {e:>4d}  ε={agent.epsilon:.3f}  "
                  f"wait={total_time:>12,.0f}  loss={mean_loss:.4f}{saved_msg}")
        elif saved_msg:
            print(f"epoch {e:>4d}  ε={agent.epsilon:.3f}  "
                  f"wait={total_time:>12,.0f}  loss={mean_loss:.4f}{saved_msg}")

        sys.stdout.flush()

    # ── Plot ────────────────────────────────────────────────────────────
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

    # Top: total waiting time
    ax1.plot(total_time_list, alpha=0.3, color='steelblue', label='Per epoch')
    window = max(10, epochs // 50)
    if len(total_time_list) >= window:
        ma = np.convolve(total_time_list, np.ones(window)/window, mode='valid')
        ax1.plot(range(window-1, len(total_time_list)), ma,
                 color='steelblue', linewidth=2, label=f'Moving avg ({window})')
    best_epoch_idx = int(np.argmin(total_time_list))
    ax1.scatter([best_epoch_idx], [total_time_list[best_epoch_idx]],
                color='gold', s=120, zorder=5, label=f'Best (ep {best_epoch_idx})')
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Total Waiting Time")
    ax1.set_title(f"Double DQN Training — {model_name}")
    ax1.legend()
    ax1.grid(alpha=0.3)

    # Bottom: loss
    if any(l > 0 for l in loss_list):
        ax2.plot(loss_list, alpha=0.5, color='crimson')
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Mean Huber Loss")
        ax2.set_title("Training Loss")
        ax2.grid(alpha=0.3)
        ax2.set_yscale('log')

    plt.tight_layout()
    plt.savefig(f'plots/ddqn_training_{model_name}.png', dpi=120)
    plt.show()

    # Also save raw data for later analysis
    np.save(f'plots/ddqn_waiting_{model_name}.npy', np.array(total_time_list))
    np.save(f'plots/ddqn_loss_{model_name}.npy',    np.array(loss_list))


def get_options():
    p = optparse.OptionParser()
    p.add_option("-m", dest="model_name", default="ddqn_model")
    p.add_option("-e", dest="epochs",     type='int', default=2000)
    p.add_option("-s", dest="steps",      type='int', default=500)
    opts, _ = p.parse_args()
    return opts


if __name__ == "__main__":
    opts = get_options()
    run_training(model_name=opts.model_name, epochs=opts.epochs, steps=opts.steps)