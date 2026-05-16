"""
Benchmark v2: Double DQN agent vs Fixed-Timer baseline.

Runs both controllers on the same SUMO scenario N times each,
records total waiting time, and outputs a clean comparison chart.

Usage: python3 performance/benchmark_v2.py
Output: performance/dqn_v2_vs_fixed_timer.png
        performance/results_v2.txt
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

# Import Double DQN bits from train_v2
from train_v2 import (
    DoubleDQNAgent, build_phase_strings, build_state,
    total_waiting, phaseDuration,
)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SUMO_CFG     = os.path.join(PROJECT_ROOT, "configuration.sumocfg")
MODEL_PATH   = os.path.join(PROJECT_ROOT, "models", "double_dqn_best.bin")
OUTPUT_DIR   = os.path.dirname(os.path.abspath(__file__))

STEPS = 1500   # match training (we used -s 1500)
RUNS  = 5


def detect_network():
    """Probe SUMO once to find junction config."""
    traci.start([checkBinary("sumo"), "-c", SUMO_CFG, "--no-warnings", "true"])
    junctions    = traci.trafficlight.getIDList()
    sample       = junctions[0]
    state_length = len(traci.trafficlight.getRedYellowGreenState(sample))
    lanes        = list(dict.fromkeys(traci.trafficlight.getControlledLanes(sample)))
    n_actions    = len(lanes)
    traci.close()
    return junctions, state_length, lanes, n_actions


def run_fixed_timer(junctions, state_length, n_actions, steps=STEPS, green_secs=15):
    """Round-robin fixed timer baseline."""
    select_lane = build_phase_strings(state_length, n_actions)
    unique_lanes = []  # populated after start

    traci.start([checkBinary("sumo"), "-c", SUMO_CFG,
                 "--no-warnings", "true"])

    # Get unique lanes once SUMO is running
    sample = junctions[0]
    unique_lanes = list(dict.fromkeys(traci.trafficlight.getControlledLanes(sample)))

    total_time   = 0.0
    step         = 0
    current_lane = 0
    time_left    = green_secs

    while step <= steps:
        try:
            traci.simulationStep()
        except Exception:
            break

        try:
            total_time += total_waiting(unique_lanes)
        except Exception:
            pass

        if time_left == 0:
            for junction in junctions:
                try:
                    phaseDuration(junction, 3, select_lane[current_lane][0])
                    phaseDuration(junction, green_secs, select_lane[current_lane][1])
                except Exception:
                    pass
            current_lane = (current_lane + 1) % n_actions
            time_left    = green_secs
        else:
            time_left -= 1

        step += 1

    try:
        traci.close()
    except Exception:
        pass
    return total_time


def run_dqn_v2(junctions, state_length, lanes, n_actions, steps=STEPS):
    """Use the trained Double DQN agent in greedy inference mode."""
    select_lane = build_phase_strings(state_length, n_actions)
    input_dims  = 3 * n_actions  # rich state

    agent = DoubleDQNAgent(
        input_dims=input_dims,
        n_actions=n_actions,
        epsilon_start=0.0,   # pure exploitation
        epsilon_end=0.0,
        warmup_steps=0,
    )
    # Load trained weights
    ckpt = torch.load(MODEL_PATH, map_location=agent.device)
    agent.q_online.load_state_dict(ckpt['q_online'])
    if 'q_target' in ckpt:
        agent.q_target.load_state_dict(ckpt['q_target'])
    agent.q_online.eval()
    agent.epsilon = 0.0  # no exploration during benchmark

    traci.start([checkBinary("sumo"), "-c", SUMO_CFG,
                 "--no-warnings", "true"])

    # Get unique lanes after SUMO start
    sample = junctions[0]
    unique_lanes = list(dict.fromkeys(traci.trafficlight.getControlledLanes(sample)))

    total_time           = 0.0
    step                 = 0
    min_duration         = 5
    traffic_lights_time  = {j: 0 for j in junctions}
    current_phase_action = {j: 0 for j in junctions}

    while step <= steps:
        try:
            traci.simulationStep()
        except Exception:
            break

        for junction in junctions:
            try:
                total_time += total_waiting(unique_lanes)
            except Exception:
                break

            if traffic_lights_time[junction] == 0:
                state = build_state(unique_lanes,
                                    current_phase_action[junction], n_actions)
                action = agent.choose_action(state)
                current_phase_action[junction] = action

                phaseDuration(junction, 3, select_lane[action][0])
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
    junctions, state_length, lanes, n_actions = detect_network()
    print(f"  Junctions   : {junctions}")
    print(f"  State length: {state_length}")
    print(f"  Lanes       : {lanes}")
    print(f"  n_actions   : {n_actions}\n")

    # ── Run Fixed Timer N times ──────────────────────────────────────────
    print(f"Running Fixed-Timer baseline × {RUNS} ...")
    fixed_results = []
    for i in range(RUNS):
        wt = run_fixed_timer(junctions, state_length, n_actions)
        fixed_results.append(wt)
        print(f"  Run {i+1}: total waiting time = {wt:,.0f}")

    # ── Run Double DQN N times ───────────────────────────────────────────
    print(f"\nRunning Double DQN (double_dqn_best.bin) × {RUNS} ...")
    dqn_results = []
    for i in range(RUNS):
        wt = run_dqn_v2(junctions, state_length, lanes, n_actions)
        dqn_results.append(wt)
        print(f"  Run {i+1}: total waiting time = {wt:,.0f}")

    # ── Stats ────────────────────────────────────────────────────────────
    fixed_mean = np.mean(fixed_results)
    fixed_std  = np.std(fixed_results)
    dqn_mean   = np.mean(dqn_results)
    dqn_std    = np.std(dqn_results)
    improvement = (1 - dqn_mean / fixed_mean) * 100

    print("\n" + "="*60)
    print("RESULTS — Double DQN vs Fixed Timer")
    print("="*60)
    print(f"Fixed Timer    : {fixed_mean:>14,.0f}  ± {fixed_std:>10,.0f}")
    print(f"Double DQN     : {dqn_mean:>14,.0f}  ± {dqn_std:>10,.0f}")
    print(f"Improvement    : {improvement:>+13.1f}%")
    print("="*60)

    # ── Save text results ────────────────────────────────────────────────
    with open(os.path.join(OUTPUT_DIR, "results_v2.txt"), "w") as f:
        f.write("Double DQN vs Fixed-Timer Benchmark Results\n")
        f.write("="*60 + "\n\n")
        f.write(f"Runs per controller : {RUNS}\n")
        f.write(f"Steps per run       : {STEPS}\n")
        f.write(f"Model file          : double_dqn_best.bin\n\n")
        f.write("Fixed Timer baseline:\n")
        for i, wt in enumerate(fixed_results):
            f.write(f"  Run {i+1}: {wt:,.0f}\n")
        f.write(f"  Mean: {fixed_mean:,.0f}  Std: {fixed_std:,.0f}\n\n")
        f.write("Double DQN (trained):\n")
        for i, wt in enumerate(dqn_results):
            f.write(f"  Run {i+1}: {wt:,.0f}\n")
        f.write(f"  Mean: {dqn_mean:,.0f}  Std: {dqn_std:,.0f}\n\n")
        f.write(f"Improvement: {improvement:+.1f}%\n")
    print(f"\nText results saved: {os.path.join(OUTPUT_DIR, 'results_v2.txt')}")

    # ── Plot (light cream theme to match dashboard) ──────────────────────
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
    fig.patch.set_facecolor('#f5f1e8')

    for ax in (ax1, ax2):
        ax.set_facecolor('#faf7f0')
        for s in ax.spines.values():
            s.set_color('#d4c9b5')
        ax.tick_params(colors='#6b5f4a')
        ax.xaxis.label.set_color('#2c2416')
        ax.yaxis.label.set_color('#2c2416')
        ax.title.set_color('#2c2416')

    # Left: per-run bars
    x = np.arange(RUNS)
    w = 0.35
    ax1.bar(x - w/2, [v/1e6 for v in fixed_results], w,
            color='#dc2626', label='Fixed Timer', edgecolor='#7a1818')
    ax1.bar(x + w/2, [v/1e6 for v in dqn_results], w,
            color='#16a34a', label='Double DQN (trained)', edgecolor='#0d5024')
    ax1.set_xticks(x)
    ax1.set_xticklabels([f'Run {i+1}' for i in range(RUNS)])
    ax1.set_ylabel('Total Waiting Time (×10⁶ s)')
    ax1.set_title('Per-Run Comparison', fontweight='bold')
    ax1.legend(facecolor='#ffffff', edgecolor='#d4c9b5')
    ax1.grid(True, color='#e8e0d0', axis='y', alpha=0.7, linewidth=0.6)

    # Right: mean ± std
    means = [fixed_mean/1e6, dqn_mean/1e6]
    stds  = [fixed_std/1e6,  dqn_std/1e6]
    bars = ax2.bar(['Fixed Timer', 'Double DQN'], means, yerr=stds,
                   color=['#dc2626', '#16a34a'], capsize=10,
                   edgecolor=['#7a1818', '#0d5024'], linewidth=1.5,
                   error_kw={'ecolor': '#6b5f4a'})
    for bar, val in zip(bars, means):
        ax2.text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() + max(stds + [0.05])*0.5,
                 f'{val:.2f}M', ha='center', color='#2c2416',
                 fontsize=12, fontweight='bold')

    ax2.set_ylabel('Total Waiting Time (×10⁶ s)')
    ax2.set_title(f'Mean ± Std over {RUNS} runs', fontweight='bold')
    box_color = '#16a34a' if improvement > 0 else '#dc2626'
    ax2.text(0.97, 0.95,
             f'Improvement:\n{improvement:+.1f}%',
             transform=ax2.transAxes, ha='right', va='top',
             color=box_color, fontsize=12, fontweight='bold',
             bbox=dict(boxstyle='round',
                       facecolor='#ffffff',
                       edgecolor=box_color, linewidth=2,
                       alpha=0.95))
    ax2.grid(True, color='#e8e0d0', axis='y', alpha=0.7, linewidth=0.6)

    plt.suptitle('Double DQN vs Fixed-Timer Traffic Control — Performance Benchmark',
                 fontsize=14, color='#2c2416', y=0.99, fontweight='bold')
    plt.tight_layout()

    chart_path = os.path.join(OUTPUT_DIR, "dqn_v2_vs_fixed_timer.png")
    plt.savefig(chart_path, dpi=150, bbox_inches='tight',
                facecolor='#f5f1e8', edgecolor='none')
    print(f"Chart saved   : {chart_path}\n")
    plt.show()


if __name__ == "__main__":
    main()