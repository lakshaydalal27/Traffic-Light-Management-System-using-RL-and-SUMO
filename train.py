from __future__ import absolute_import
from __future__ import print_function

import os
import sys
import optparse
import random
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
import matplotlib.pyplot as plt

# we need to import python modules from the $SUMO_HOME/tools directory
if "SUMO_HOME" in os.environ:
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    sys.path.append(tools)
else:
    sys.exit("please declare environment variable 'SUMO_HOME'")

from sumolib import checkBinary  # noqa
import traci  # noqa


def get_vehicle_numbers(lanes):
    vehicle_per_lane = dict()
    for l in lanes:
        vehicle_per_lane[l] = 0
        for k in traci.lane.getLastStepVehicleIDs(l):
            if traci.vehicle.getLanePosition(k) > 10:
                vehicle_per_lane[l] += 1
    return vehicle_per_lane


def get_waiting_time(lanes):
    waiting_time = 0
    for lane in lanes:
        waiting_time += traci.lane.getWaitingTime(lane)
    return waiting_time


def phaseDuration(junction, phase_time, phase_state):
    traci.trafficlight.setRedYellowGreenState(junction, phase_state)
    traci.trafficlight.setPhaseDuration(junction, phase_time)


class Model(nn.Module):
    def __init__(self, lr, input_dims, fc1_dims, fc2_dims, n_actions):
        super(Model, self).__init__()
        self.lr = lr
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions

        self.linear1 = nn.Linear(self.input_dims, self.fc1_dims)
        self.linear2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.linear3 = nn.Linear(self.fc2_dims, self.n_actions)

        self.optimizer = optim.Adam(self.parameters(), lr=self.lr)
        self.loss = nn.MSELoss()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        actions = self.linear3(x)
        return actions


class Agent:
    def __init__(
        self,
        gamma,
        epsilon,
        lr,
        input_dims,
        fc1_dims,
        fc2_dims,
        batch_size,
        n_actions,
        junctions,
        max_memory_size=50000,
        epsilon_dec=0.02,   # FIX: decay per EPOCH not per step
        epsilon_end=0.05,
    ):
        self.gamma = gamma
        self.epsilon = epsilon
        self.lr = lr
        self.batch_size = batch_size
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.action_space = [i for i in range(n_actions)]
        self.junctions = junctions
        self.max_mem = max_memory_size
        self.epsilon_dec = epsilon_dec
        self.epsilon_end = epsilon_end

        self.Q_eval = Model(
            self.lr, self.input_dims, self.fc1_dims, self.fc2_dims, self.n_actions
        )

        # FIX: single shared replay buffer (not per junction, simpler & more stable)
        self.state_memory     = np.zeros((self.max_mem, self.input_dims), dtype=np.float32)
        self.new_state_memory = np.zeros((self.max_mem, self.input_dims), dtype=np.float32)
        self.reward_memory    = np.zeros(self.max_mem, dtype=np.float32)
        self.action_memory    = np.zeros(self.max_mem, dtype=np.int32)
        self.terminal_memory  = np.zeros(self.max_mem, dtype=np.bool_)
        self.mem_cntr         = 0

    def store_transition(self, state, state_, action, reward, done, junction=0):
        index = self.mem_cntr % self.max_mem
        self.state_memory[index]     = state
        self.new_state_memory[index] = state_
        self.reward_memory[index]    = reward
        self.terminal_memory[index]  = done
        self.action_memory[index]    = action
        self.mem_cntr += 1

    def choose_action(self, observation):
        state = torch.tensor([observation], dtype=torch.float).to(self.Q_eval.device)
        if np.random.random() > self.epsilon:
            actions = self.Q_eval.forward(state)
            action = torch.argmax(actions).item()
        else:
            action = np.random.choice(self.action_space)
        return action

    def decay_epsilon(self):
        """Call once per epoch — epsilon decays gradually across epochs."""
        self.epsilon = max(self.epsilon - self.epsilon_dec, self.epsilon_end)

    def save(self, model_name):
        torch.save(self.Q_eval.state_dict(), f'models/{model_name}.bin')

    def learn(self, junction=0):
        """FIX: Sample a random mini-batch from replay buffer — not all transitions."""
        if self.mem_cntr < self.batch_size:
            return  # not enough data yet

        self.Q_eval.optimizer.zero_grad()

        max_mem = min(self.mem_cntr, self.max_mem)
        batch = np.random.choice(max_mem, self.batch_size, replace=False)

        state_batch     = torch.tensor(self.state_memory[batch]).to(self.Q_eval.device)
        new_state_batch = torch.tensor(self.new_state_memory[batch]).to(self.Q_eval.device)
        reward_batch    = torch.tensor(self.reward_memory[batch]).to(self.Q_eval.device)
        terminal_batch  = torch.tensor(self.terminal_memory[batch]).to(self.Q_eval.device)
        action_batch    = self.action_memory[batch]

        batch_idx = np.arange(self.batch_size, dtype=np.int32)
        q_eval    = self.Q_eval.forward(state_batch)[batch_idx, action_batch]
        q_next    = self.Q_eval.forward(new_state_batch)
        q_next[terminal_batch] = 0.0
        q_target  = reward_batch + self.gamma * torch.max(q_next, dim=1)[0]

        loss = self.Q_eval.loss(q_target, q_eval).to(self.Q_eval.device)
        loss.backward()
        self.Q_eval.optimizer.step()


def build_phase_strings(state_length, n_actions):
    """Dynamically build yellow/green phase strings based on actual signal state length."""
    signals_per_group = state_length // n_actions
    select_lane = []
    for i in range(n_actions):
        yellow = 'r' * (i * signals_per_group) + \
                 'y' * signals_per_group + \
                 'r' * ((n_actions - i - 1) * signals_per_group)
        green  = 'r' * (i * signals_per_group) + \
                 'G' * signals_per_group + \
                 'r' * ((n_actions - i - 1) * signals_per_group)
        yellow = yellow.ljust(state_length, 'r')[:state_length]
        green  = green.ljust(state_length, 'r')[:state_length]
        select_lane.append([yellow, green])
    return select_lane


def run(train=True, model_name="model", epochs=50, steps=500, ard=False):
    ard = False

    best_time      = np.inf
    total_time_list = []

    # ── Detect network properties ─────────────────────────────────────────────
    traci.start(
        [checkBinary("sumo"), "-c", "configuration.sumocfg",
         "--tripinfo-output", "maps/tripinfo.xml"]
    )
    all_junctions    = traci.trafficlight.getIDList()
    junction_numbers = list(range(len(all_junctions)))
    sample_junction  = all_junctions[0]
    state_length     = len(traci.trafficlight.getRedYellowGreenState(sample_junction))
    controlled_lanes = traci.trafficlight.getControlledLanes(sample_junction)
    unique_lanes     = list(dict.fromkeys(controlled_lanes))
    input_dims       = len(unique_lanes)
    traci.close()

    select_lane = build_phase_strings(state_length, input_dims)

    print(f"Device: cpu")
    print(f"State length: {state_length}, Input dims: {input_dims}")
    print(f"Junctions: {list(all_junctions)}")
    print(f"Phase strings:")
    for i, phase in enumerate(select_lane):
        print(f"  Action {i}: yellow={phase[0]}  green={phase[1]}")

    # ── Build agent ───────────────────────────────────────────────────────────
    brain = Agent(
        gamma=0.99,
        epsilon=1.0 if train else 0.0,
        lr=0.0001,           # FIX: much lower lr — stable learning
        input_dims=input_dims,
        fc1_dims=256,
        fc2_dims=256,
        batch_size=64,       # FIX: small fixed batch size
        n_actions=input_dims,
        junctions=junction_numbers,
        epsilon_dec=0.02,    # FIX: decays per epoch (1.0 → 0.05 over ~48 epochs)
        epsilon_end=0.05,
    )

    if not train:
        brain.Q_eval.load_state_dict(
            torch.load(f'models/{model_name}.bin', map_location=brain.Q_eval.device))

    min_duration = 5

    for e in range(epochs):
        traci.start(
            [checkBinary("sumo"), "-c", "configuration.sumocfg",
             "--tripinfo-output", "tripinfo.xml"]
        )

        print(f"epoch: {e}  epsilon: {brain.epsilon:.3f}")

        step       = 0
        total_time = 0

        traffic_lights_time = {j: 0 for j in all_junctions}
        # pending[jn] holds (state_taken_in, action_chosen) from the previous
        # decision cycle. When that cycle ends we know its reward (waiting
        # time accumulated while the action was active) and the new state,
        # so we can store a CORRECT (s, a, r, s') tuple.
        pending = {jn: None for jn in junction_numbers}
        pending_wait_accum = {jn: 0.0 for jn in junction_numbers}

        while step <= steps:
            traci.simulationStep()

            for junction_number, junction in enumerate(all_junctions):
                try:
                    controled_lanes = traci.trafficlight.getControlledLanes(junction)
                except Exception:
                    break

                waiting_time = get_waiting_time(controled_lanes)
                total_time += waiting_time
                # Accumulate waiting time experienced under the action that's
                # currently active (= the action chosen in the previous cycle).
                pending_wait_accum[junction_number] += waiting_time

                if traffic_lights_time[junction] == 0:
                    # ── current observation (this is s' for the previous action) ──
                    vehicles_per_lane = get_vehicle_numbers(controled_lanes)
                    seen = {}
                    for lane, count in vehicles_per_lane.items():
                        seen[lane] = seen.get(lane, 0) + count
                    next_state = list(seen.values())[:input_dims]
                    while len(next_state) < input_dims:
                        next_state.append(0)

                    # ── close out the previous decision: (s, a, r, s') ──
                    if pending[junction_number] is not None and train:
                        prev_state, prev_act = pending[junction_number]
                        # reward = negative waiting accumulated while prev_act
                        # was the active phase (true credit assignment)
                        reward = -1.0 * pending_wait_accum[junction_number]
                        brain.store_transition(
                            prev_state, next_state, prev_act,
                            reward, (step == steps)
                        )
                        brain.learn()

                    # ── pick a new action for current state ──
                    action = brain.choose_action(next_state)
                    pending[junction_number] = (next_state, action)
                    pending_wait_accum[junction_number] = 0.0

                    phaseDuration(junction, 6, select_lane[action][0])
                    phaseDuration(junction, min_duration + 10, select_lane[action][1])
                    traffic_lights_time[junction] = min_duration + 10
                else:
                    traffic_lights_time[junction] -= 1

            step += 1

        print(f"total_time {total_time}")
        total_time_list.append(total_time)

        if total_time < best_time:
            best_time = total_time
            if train:
                brain.save(model_name)
                print(f"  ✓ Best model saved (epoch {e}, waiting={total_time:.0f})")

        traci.close()
        sys.stdout.flush()

        # FIX: decay epsilon ONCE per epoch
        if train:
            brain.decay_epsilon()

        if not train:
            break

    if train:
        # Plot with moving average for cleaner visualization
        window = 5
        ma = np.convolve(total_time_list, np.ones(window)/window, mode='valid')

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(total_time_list, alpha=0.3, color='steelblue', label='Per epoch')
        ax.plot(range(window-1, len(total_time_list)),
                ma, color='steelblue', linewidth=2, label='Moving avg (5)')
        ax.set_xlabel("Epochs")
        ax.set_ylabel("Total Waiting Time")
        ax.set_title(f"DQN Training — {model_name}")
        ax.legend()
        plt.tight_layout()
        plt.savefig(f'plots/time_vs_epoch_{model_name}.png')
        plt.show()


def get_options():
    optParser = optparse.OptionParser()
    optParser.add_option("-m", dest='model_city1', type='string', default="model",
                         help="name of model")
    optParser.add_option("--train", action='store_true', default=False,
                         help="training or testing")
    optParser.add_option("-e", dest='epochs', type='int', default=50,
                         help="Number of epochs")
    optParser.add_option("-s", dest='steps', type='int', default=500,
                         help="Number of steps")
    optParser.add_option("--ard", action='store_true', default=False,
                         help="Connect Arduino (disabled)")
    options, args = optParser.parse_args()
    return options


if __name__ == "__main__":
    options  = get_options()
    model_name = options.model_city1
    train    = options.train
    epochs   = options.epochs
    steps    = options.steps
    ard      = options.ard
    run(train=train, model_name=model_name, epochs=epochs, steps=steps, ard=ard)