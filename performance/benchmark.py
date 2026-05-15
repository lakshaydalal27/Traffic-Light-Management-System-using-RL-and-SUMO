"""
Benchmark: DQN agent vs Fixed-Timer baseline.

Runs both controllers on the same SUMO scenario N times each,
records total waiting time, and outputs a clean comparison chart.

Usage: python3 performance/benchmark.py
Output: performance/dqn_vs_fixed_timer.png
        performance/results.txt
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import torch

# ── Make project root importable ──────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

if "SUMO_HOME" in os.environ:
    sys.path.append(os.path.join(os.environ["SUMO_HOME"], "tools"))
else:
    sys.exit("Please declare environment variable 'SUMO_HOME'")

import traci
from sumolib import checkBinary

from train import (
    Model, Agent,
    get_vehicle_numbers, get_waiting_time, phaseDuration,
    build_phase_strings,
)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SUMO_CFG     = os.path.join(PROJECT_ROOT, "configuration.sumocfg")
MODEL_PATH   = os.path.join(PROJECT_ROOT, "models", "best_model.bin")
OUTPUT_DIR   = os.path.dirname(os.path.abspath(__file__))

STEPS  = 500
RUNS   = 5  # number of independent runs per controller


def detect_network():
    """Probe SUMO once to find junction config."""
    traci.start([checkBinary("sumo"), "-c", SUMO_CFG, "--no-warnings", "true"])
    junctions    = traci.trafficlight.getIDList()
    sample       = junctions[0]
    state_length = len(traci.trafficlight.getRedYellowGreenState(sample))
    lanes        = list(dict.fromkeys(traci.trafficlight.getControlledLanes(sample)))
    input_dims   = len(lanes)
    traci.close()
    return junctions, state_length, input_dims


def run_fixed_timer(junctions, state_length, input_dims, steps=STEPS, green_secs=15):
    """Fixed timer: cycle through each lane giving each `green_secs` of green."""
    select_lane = build_phase_strings(state_length, input_dims)

    traci.start([checkBinary("sumo"), "-c", SUMO_CFG,
                 "--tripinfo-output", os.path.join(PROJECT_ROOT, "tripinfo.xml"),
                 "--no-warnings", "true"])

    total_time = 0.0
    step = 0
    current_lane = 0
    time_left    = green_secs

    while step <= steps:
        try:
            traci.simulationStep()
        except Exception:
            break

        for junction in junctions:
            try:
                controled_lanes = traci.trafficlight.getControlledLanes(junction)
                total_time += get_waiting_time(controled_lanes)
            except Exception:
                pass

        if time_left == 0:
            # switch to next lane
            for junction in junctions:
                try:
                    phaseDuration(junction, 3, select_lane[current_lane][0])  # yellow
                    phaseDuration(junction, green_secs, select_lane[current_lane][1])  # green
                except Exception:
                    pass
            current_lane = (current_lane + 1) % input_dims
            time_left = green_secs
        else:
            time_left -= 1

        step += 1

    try:
        traci.close()
    except Exception:
        pass
    return total_time


def run_dqn(junctions, state_length, input_dims, steps=STEPS):
    """Use the trained DQN agent (best_model.bin) in inference mode."""
    select_lane = build_phase_strings(state_length, input_dims)

    brain = Agent(
        gamma=0.99,
        epsilon=0.0,   # full exploitation — no random actions
        lr=0.0001,
        input_dims=input_dims,
        fc1_dims=256,
        fc2_dims=256,
        batch_size=64,
        n_actions=input_dims,
        junctions=list(range(len(junctions))),
    )
    brain.Q_eval.load_state_dict(
        torch.load(MODEL_PATH, map_location=brain.Q_eval.device))
    brain.Q_eval.eval()

    traci.start([checkBinary("sumo"), "-c", SUMO_CFG,
                 "--tripinfo-output", os.path.join(PROJECT_ROOT, "tripinfo.xml"),
                 "--no-warnings", "true"])

    total_time  = 0.0
    step        = 0
    min_duration = 5
    traffic_lights_time = {j: 0 for j in junctions}

    while step <= steps:
        try:
            traci.simulationStep()
        except Exception:
            break

        for jn, junction in enumerate(junctions):
            try:
                controled_lanes = traci.trafficlight.getControlledLanes(junction)
            except Exception:
                break

            waiting_time = get_waiting_time(controled_lanes)
            total_time  += waiting_time

            if traffic_lights_time[junction] == 0:
                vehicles_per_lane = get_vehicle_numbers(controled_lanes)
                seen = {}
                for lane, count in vehicles_per_lane.items():
                    seen[lane] = seen.get(lane, 0) + count
                state_ = list(seen.values())[:input_dims]
                while len(state_) < input_dims:
                    state_.append(0)

                action = brain.choose_action(state_)
                phaseDuration(junction, 6, select_lane[action][0])
                phaseDuration(junction, min_duration + 10, select_lane[action][1])
                traffic_lights_time[junction] = min_duration + 10
            else:
                traffic_lights_time[junction] -= 1

        step += 1

    try:
        traci.close()
    except Exception:
        pass
    return total_time


def main():
    print("Detecting SUMO network...")
    junctions, state_length, input_dims = detect_network()
    print(f"  Junctions: {junctions}")
    print(f"  State length: {state_length}, Input dims: {input_dims}\n")

    # ── Run Fixed Timer N times ──────────────────────────────────────────────
    print(f"Running Fixed-Timer baseline × {RUNS} ...")
    fixed_results = []
    for i in range(RUNS):
        wt = run_fixed_timer(junctions, state_length, input_dims)
        fixed_results.append(wt)
        print(f"  Run {i+1}: total waiting time = {wt:,.0f}")

    # ── Run DQN N times ──────────────────────────────────────────────────────
    print(f"\nRunning DQN (best_model.bin) × {RUNS} ...")
    dqn_results = []
    for i in range(RUNS):
        wt = run_dqn(junctions, state_length, input_dims)
        dqn_results.append(wt)
        print(f"  Run {i+1}: total waiting time = {wt:,.0f}")

    # ── Stats ────────────────────────────────────────────────────────────────
    fixed_mean = np.mean(fixed_results)
    fixed_std  = np.std(fixed_results)
    dqn_mean   = np.mean(dqn_results)
    dqn_std    = np.std(dqn_results)
    improvement = (1 - dqn_mean / fixed_mean) * 100

    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    print(f"Fixed Timer  : {fixed_mean:>14,.0f}  ± {fixed_std:>10,.0f}")
    print(f"DQN (trained): {dqn_mean:>14,.0f}  ± {dqn_std:>10,.0f}")
    print(f"Improvement  : {improvement:>+13.1f}%")
    print("="*60)

    # ── Save text results ────────────────────────────────────────────────────
    with open(os.path.join(OUTPUT_DIR, "results.txt"), "w") as f:
        f.write("DQN vs Fixed-Timer Benchmark Results\n")
        f.write("="*60 + "\n\n")
        f.write(f"Runs per controller : {RUNS}\n")
        f.write(f"Steps per run       : {STEPS}\n\n")
        f.write("Fixed Timer baseline:\n")
        for i, wt in enumerate(fixed_results):
            f.write(f"  Run {i+1}: {wt:,.0f}\n")
        f.write(f"  Mean: {fixed_mean:,.0f}  Std: {fixed_std:,.0f}\n\n")
        f.write("DQN agent (best_model.bin):\n")
        for i, wt in enumerate(dqn_results):
            f.write(f"  Run {i+1}: {wt:,.0f}\n")
        f.write(f"  Mean: {dqn_mean:,.0f}  Std: {dqn_std:,.0f}\n\n")
        f.write(f"Improvement: {improvement:+.1f}%\n")
    print(f"\nText results saved: {os.path.join(OUTPUT_DIR, 'results.txt')}")

    # ── Plot ────────────────────────────────────────────────────────────────
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
    fig.patch.set_facecolor('#0a0e1a')

    for ax in (ax1, ax2):
        ax.set_facecolor('#111827')
        for s in ax.spines.values():
            s.set_color('#1e3a5f')
        ax.tick_params(colors='#64748b')
        ax.xaxis.label.set_color('#94a3b8')
        ax.yaxis.label.set_color('#94a3b8')
        ax.title.set_color('#e2e8f0')

    # Left: per-run bars
    x = np.arange(RUNS)
    w = 0.35
    ax1.bar(x - w/2, [v/1e6 for v in fixed_results], w,
            color='#ff3b5c', label='Fixed Timer', edgecolor='#1e3a5f')
    ax1.bar(x + w/2, [v/1e6 for v in dqn_results], w,
            color='#00ff88', label='DQN (trained)', edgecolor='#1e3a5f')
    ax1.set_xticks(x)
    ax1.set_xticklabels([f'Run {i+1}' for i in range(RUNS)])
    ax1.set_ylabel('Total Waiting Time (×10⁶ s)')
    ax1.set_title('Per-Run Comparison')
    ax1.legend(facecolor='#1a2235', edgecolor='#1e3a5f', labelcolor='#e2e8f0')
    ax1.grid(True, color='#1e3a5f', axis='y', alpha=0.4, linewidth=0.5)

    # Right: mean ± std
    means = [fixed_mean/1e6, dqn_mean/1e6]
    stds  = [fixed_std/1e6,  dqn_std/1e6]
    bars = ax2.bar(['Fixed Timer', 'DQN (trained)'], means, yerr=stds,
                   color=['#ff3b5c', '#00ff88'], capsize=10,
                   edgecolor='#1e3a5f', linewidth=1.5,
                   error_kw={'ecolor': '#94a3b8'})
    for bar, val in zip(bars, means):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(stds)*0.3,
                 f'{val:.2f}M', ha='center', color='#e2e8f0',
                 fontsize=11, fontweight='bold')
    ax2.set_ylabel('Total Waiting Time (×10⁶ s)')
    ax2.set_title(f'Mean ± Std over {RUNS} runs')
    ax2.text(0.98, 0.95,
             f'Improvement:\n{improvement:+.1f}%',
             transform=ax2.transAxes, ha='right', va='top',
             color='#00ff88' if improvement > 0 else '#ff3b5c',
             fontsize=11, fontweight='bold',
             bbox=dict(boxstyle='round',
                       facecolor='#0d2e1a' if improvement > 0 else '#2e0d0d',
                       edgecolor='#00ff88' if improvement > 0 else '#ff3b5c',
                       alpha=0.85))
    ax2.grid(True, color='#1e3a5f', axis='y', alpha=0.4, linewidth=0.5)

    plt.suptitle('DQN vs Fixed-Timer Traffic Control — Performance Benchmark',
                 fontsize=13, color='#00d4ff', y=0.98, fontweight='bold')
    plt.tight_layout()

    chart_path = os.path.join(OUTPUT_DIR, "dqn_vs_fixed_timer.png")
    plt.savefig(chart_path, dpi=150, bbox_inches='tight',
                facecolor='#0a0e1a', edgecolor='none')
    print(f"Chart saved   : {chart_path}\n")
    plt.show()


if __name__ == "__main__":
    main()