"""
train_v3.py — Adaptive Duration Double DQN for SUMO Traffic Signal Control

Key advancement over train_v2.py:
  Action space expanded from 4 → 12 actions
  Each action = (lane, duration) pair:
      action 0  = lane 0 + 5s green
      action 1  = lane 0 + 15s green
      action 2  = lane 0 + 25s green
      action 3  = lane 1 + 5s green
      ...
      action 11 = lane 3 + 25s green

This lets the DQN learn:
  - short green on empty lanes  (matches 10s fixed timer behavior)
  - long  green on busy lanes   (clears queues efficiently)
  - adapts dynamically per intersection state

All other Double DQN improvements (soft target updates, rich state,
Huber loss, gradient clipping, delta reward) carry over from train_v2.py.
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

# ── Duration options (in simulation steps ≈ seconds) ─────────────────────────
DURATIONS = [5, 15, 25]   # short / medium / long green
N_LANES   = 4             # 4-way intersection
N_ACTIONS = N_LANES * len(DURATIONS)   # 12 total actions


def decode_action(a):
    """Map flat action index → (lane_index, duration_seconds)."""
    lane     = a // len(DURATIONS)
    dur_idx  = a %  len(DURATIONS)
    return lane, DURATIONS[dur_idx]


def encode_action(lane, dur_idx):
    return lane * len(DURATIONS) + dur_idx


# ── Helper functions ─────────────────────────────────────────────────────────

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
        self.input_dims    = input_dims
        self.n_actions     = n_actions
        self.gamma         = gamma
        self.epsilon       = epsilon_start
        self.epsilon_end   = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size    = batch_size
        self.tau           = tau
        self.warmup_steps  = warmup_steps

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.q_online = QNet(input_dims, n_actions).to(self.device)
        self.q_target = copy.deepcopy(self.q_online).to(self.device)
        self.q_target.eval()
        self.optim    = optim.Adam(self.q_online.parameters(), lr=lr)

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
        for p_t, p_o in zip(self.q_target.parameters(), self.q_online.parameters()):
            p_t.data.copy_(self.tau * p_o.data + (1.0 - self.tau) * p_t.data)

    def learn(self):
        if self.mem_cntr < self.warmup_steps:
            return None

        max_mem = min(self.mem_cntr, self.max_mem)
        idx = np.random.choice(max_mem, self.batch_size, replace=False)

        s  = torch.tensor(self.state_mem[idx],      device=self.device)
        a  = torch.tensor(self.action_mem[idx],     device=self.device)
        r  = torch.tensor(self.reward_mem[idx],     device=self.device)
        s_ = torch.tensor(self.next_state_mem[idx], device=self.device)
        d  = torch.tensor(self.done_mem[idx],       device=self.device)

        q_pred = self.q_online(s).gather(1, a.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            best_actions = self.q_online(s_).argmax(dim=1, keepdim=True)
            q_next       = self.q_target(s_).gather(1, best_actions).squeeze(1)
            q_next[d]    = 0.0
            q_target     = r + self.gamma * q_next

        loss = F.smooth_l1_loss(q_pred, q_target)

        self.optim.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_online.parameters(), 10.0)
        self.optim.step()
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


# ── Phase string builder ─────────────────────────────────────────────────────

def build_phase_strings(state_length, n_lanes):
    """Build (yellow, green) phase string pairs for each LANE (not action)."""
    signals_per_group = state_length // n_lanes
    phases = []
    for i in range(n_lanes):
        yellow = 'r' * (i * signals_per_group) + 'y' * signals_per_group + \
                 'r' * ((n_lanes - i - 1) * signals_per_group)
        green  = 'r' * (i * signals_per_group) + 'G' * signals_per_group + \
                 'r' * ((n_lanes - i - 1) * signals_per_group)
        yellow = yellow.ljust(state_length, 'r')[:state_length]
        green  = green.ljust(state_length, 'r')[:state_length]
        phases.append([yellow, green])
    return phases


# ── State builder ────────────────────────────────────────────────────────────

def build_state(unique_lanes, current_lane, n_lanes):
    """
    Rich state: [vehicle counts + waiting times normalized + lane one-hot]  = 3*n_lanes dims
    """
    counts = []
    waits  = []
    for lane in unique_lanes:
        c = 0
        for vid in traci.lane.getLastStepVehicleIDs(lane):
            if traci.vehicle.getLanePosition(vid) > 10:
                c += 1
        counts.append(c)
        waits.append(traci.lane.getWaitingTime(lane) / 100.0)

    phase_oh = [0.0] * n_lanes
    if 0 <= current_lane < n_lanes:
        phase_oh[current_lane] = 1.0

    state = counts + waits + phase_oh
    while len(state) < 3 * n_lanes:
        state.append(0.0)
    return state[:3 * n_lanes]


# ── Training loop ────────────────────────────────────────────────────────────

def run_training(model_name="adaptive_dqn_best", epochs=3000, steps=1500):
    # Detect network
    traci.start([checkBinary("sumo"), "-c", "configuration.sumocfg",
                 "--no-warnings", "true"])
    all_junctions = traci.trafficlight.getIDList()
    sample        = all_junctions[0]
    state_length  = len(traci.trafficlight.getRedYellowGreenState(sample))
    controlled    = traci.trafficlight.getControlledLanes(sample)
    unique_lanes  = list(dict.fromkeys(controlled))
    n_lanes       = len(unique_lanes)
    traci.close()

    assert n_lanes == N_LANES, f"Expected {N_LANES} lanes, got {n_lanes}"

    input_dims  = 3 * n_lanes
    select_lane = build_phase_strings(state_length, n_lanes)

    print(f"Device       : {'cuda' if torch.cuda.is_available() else 'cpu'}")
    print(f"State length : {state_length}")
    print(f"n_lanes      : {n_lanes}")
    print(f"n_actions    : {N_ACTIONS}  (4 lanes × 3 durations = 12)")
    print(f"Durations    : {DURATIONS}")
    print(f"Input dims   : {input_dims}")
    print(f"Unique lanes : {unique_lanes}")
    print()

    agent = DoubleDQNAgent(
        input_dims=input_dims,
        n_actions=N_ACTIONS,
        lr=5e-4,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.05,
        epsilon_decay=0.997,   # slightly slower decay because action space is 3x larger
        batch_size=128,
        memory_size=100000,
        tau=0.005,
        warmup_steps=1500,     # more warmup for richer action space
    )

    best_time       = np.inf
    total_time_list = []
    loss_list       = []
    action_use_list = []   # track distribution of selected actions

    for e in range(epochs):
        traci.start([checkBinary("sumo"), "-c", "configuration.sumocfg",
                     "--no-warnings", "true"])

        step                  = 0
        total_time            = 0.0
        traffic_lights_time   = {j: 0 for j in all_junctions}
        pending               = {jn: None for jn in range(len(all_junctions))}
        pending_wait_start    = {jn: 0.0  for jn in range(len(all_junctions))}
        current_lane_picked   = {jn: 0    for jn in range(len(all_junctions))}
        epoch_losses          = []
        epoch_action_counts   = np.zeros(N_ACTIONS, dtype=np.int32)

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
                    next_state = build_state(unique_lanes,
                                             current_lane_picked[jn], n_lanes)

                    # Close previous decision
                    if pending[jn] is not None:
                        prev_state, prev_action = pending[jn]
                        delta_wait = waiting_now - pending_wait_start[jn]
                        reward = -delta_wait / 1000.0
                        agent.store(prev_state, prev_action, reward, next_state,
                                    (step == steps))
                        loss = agent.learn()
                        if loss is not None:
                            epoch_losses.append(loss)

                    # Choose new action (lane + duration)
                    action = agent.choose_action(next_state)
                    chosen_lane, chosen_duration = decode_action(action)

                    pending[jn]              = (next_state, action)
                    pending_wait_start[jn]   = waiting_now
                    current_lane_picked[jn]  = chosen_lane
                    epoch_action_counts[action] += 1

                    # Apply phase with the chosen duration
                    phaseDuration(junction, 3, select_lane[chosen_lane][0])   # yellow
                    phaseDuration(junction, chosen_duration, select_lane[chosen_lane][1])  # green for chosen duration
                    traffic_lights_time[junction] = chosen_duration
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
        action_use_list.append(epoch_action_counts)

        if total_time < best_time and agent.mem_cntr > agent.warmup_steps:
            best_time = total_time
            agent.save(model_name)
            saved_msg = " ★ BEST SAVED"
        else:
            saved_msg = ""

        if e % 10 == 0 or e < 5 or saved_msg:
            # show top-3 most-used actions this epoch
            top3 = np.argsort(epoch_action_counts)[-3:][::-1]
            top3_str = " ".join(
                f"a{a}(L{a//3},{DURATIONS[a%3]}s)={epoch_action_counts[a]}"
                for a in top3 if epoch_action_counts[a] > 0
            )
            print(f"epoch {e:>4d}  ε={agent.epsilon:.3f}  "
                  f"wait={total_time:>12,.0f}  loss={mean_loss:.4f}  "
                  f"top:[{top3_str}]{saved_msg}")
            sys.stdout.flush()

    # ── Plot ────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(3, 1, figsize=(13, 10))

    # 1. Waiting time
    ax = axes[0]
    ax.plot(total_time_list, alpha=0.3, color='steelblue', label='Per epoch')
    window = max(20, epochs // 60)
    if len(total_time_list) >= window:
        ma = np.convolve(total_time_list, np.ones(window)/window, mode='valid')
        ax.plot(range(window-1, len(total_time_list)), ma,
                color='steelblue', linewidth=2, label=f'Moving avg ({window})')
    best_idx = int(np.argmin(total_time_list))
    ax.scatter([best_idx], [total_time_list[best_idx]],
               color='gold', s=120, zorder=5, label=f'Best (ep {best_idx})')
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Total Waiting Time")
    ax.set_title(f"Adaptive Duration Double DQN — {model_name}")
    ax.legend()
    ax.grid(alpha=0.3)

    # 2. Loss
    ax = axes[1]
    if any(l > 0 for l in loss_list):
        ax.plot(loss_list, alpha=0.6, color='crimson')
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Mean Huber Loss")
        ax.set_title("Training Loss")
        ax.grid(alpha=0.3)
        ax.set_yscale('log')

    # 3. Action distribution heatmap (which actions are used over time)
    ax = axes[2]
    if action_use_list:
        action_matrix = np.array(action_use_list).T  # (n_actions, n_epochs)
        # Normalize per column so each epoch sums to 1
        col_sums = action_matrix.sum(axis=0, keepdims=True)
        col_sums[col_sums == 0] = 1
        action_matrix = action_matrix / col_sums
        im = ax.imshow(action_matrix, aspect='auto', cmap='viridis',
                       origin='lower', interpolation='nearest')
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Action (lane,duration)")
        ax.set_title("Action Usage Over Training (brighter = chosen more often)")
        # Label y-axis with (lane, duration) pairs
        ax.set_yticks(range(N_ACTIONS))
        ax.set_yticklabels([f"L{a//3} {DURATIONS[a%3]}s" for a in range(N_ACTIONS)],
                           fontsize=7)
        plt.colorbar(im, ax=ax, label='Usage fraction per epoch')

    plt.tight_layout()
    plt.savefig(f'plots/adaptive_dqn_training_{model_name}.png', dpi=120)
    plt.show()

    np.save(f'plots/adaptive_dqn_waiting_{model_name}.npy', np.array(total_time_list))
    np.save(f'plots/adaptive_dqn_loss_{model_name}.npy',    np.array(loss_list))
    np.save(f'plots/adaptive_dqn_actions_{model_name}.npy', np.array(action_use_list))


def get_options():
    p = optparse.OptionParser()
    p.add_option("-m", dest="model_name", default="adaptive_dqn_best")
    p.add_option("-e", dest="epochs",     type='int', default=3000)
    p.add_option("-s", dest="steps",      type='int', default=1500)
    opts, _ = p.parse_args()
    return opts


if __name__ == "__main__":
    opts = get_options()
    run_training(model_name=opts.model_name, epochs=opts.epochs, steps=opts.steps)