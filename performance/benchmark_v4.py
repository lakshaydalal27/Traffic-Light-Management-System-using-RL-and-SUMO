"""
Benchmark v4: 5-way comparison.

Compares:
  • Fixed Timer 10s
  • Fixed Timer 20s
  • Fixed Timer 30s
  • Double DQN (v2, fixed 15s green)
  • Adaptive DQN (v3, agent picks duration)

This is the final comparison chart for the project.

Usage: python3 performance/benchmark_v4.py
Output: performance/benchmark_v4_chart.png
        performance/results_v4.txt
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

if "SUMO_HOME" in os.environ:
    sys.path.append(os.path.join(os.environ["SUMO_HOME"], "tools"))
else:
    sys.exit("Please declare environment variable 'SUMO_HOME'")

import traci
from sumolib import checkBinary

# Import v2 (Double DQN with fixed duration)
from train_v2 import (
    DoubleDQNAgent as DoubleDQNAgentV2,
    build_phase_strings as build_phase_strings_v2,
    build_state as build_state_v2,
    total_waiting,
    phaseDuration,
)

# Import v3 (Adaptive duration DQN)
from train_v3 import (
    DoubleDQNAgent as DoubleDQNAgentV3,
    build_phase_strings as build_phase_strings_v3,
    build_state as build_state_v3,
    decode_action,
    N_ACTIONS as V3_N_ACTIONS,
    DURATIONS as V3_DURATIONS,
)

PROJECT_ROOT  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SUMO_CFG      = os.path.join(PROJECT_ROOT, "configuration.sumocfg")
MODEL_V2_PATH = os.path.join(PROJECT_ROOT, "models", "double_dqn_best.bin")
MODEL_V3_PATH = os.path.join(PROJECT_ROOT, "models", "adaptive_dqn_best.bin")
OUTPUT_DIR    = os.path.dirname(os.path.abspath(__file__))

STEPS = 1500
RUNS  = 3

FIXED_SETTINGS = [
    ("Fixed 10s", 10),
    ("Fixed 20s", 20),
    ("Fixed 30s", 30),
]


def detect_network():
    traci.start([checkBinary("sumo"), "-c", SUMO_CFG, "--no-warnings", "true"])
    junctions    = traci.trafficlight.getIDList()
    sample       = junctions[0]
    state_length = len(traci.trafficlight.getRedYellowGreenState(sample))
    lanes        = list(dict.fromkeys(traci.trafficlight.getControlledLanes(sample)))
    traci.close()
    return junctions, state_length, lanes


def run_fixed_timer(junctions, state_length, n_lanes, green_secs, steps=STEPS):
    select_lane = build_phase_strings_v2(state_length, n_lanes)

    traci.start([checkBinary("sumo"), "-c", SUMO_CFG, "--no-warnings", "true"])
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
            current_lane = (current_lane + 1) % n_lanes
            time_left    = green_secs
        else:
            time_left -= 1
        step += 1

    try: traci.close()
    except Exception: pass
    return total_time


def run_dqn_v2(junctions, state_length, n_lanes, steps=STEPS):
    select_lane = build_phase_strings_v2(state_length, n_lanes)
    input_dims  = 3 * n_lanes

    agent = DoubleDQNAgentV2(
        input_dims=input_dims, n_actions=n_lanes,
        epsilon_start=0.0, epsilon_end=0.0, warmup_steps=0,
    )
    ckpt = torch.load(MODEL_V2_PATH, map_location=agent.device)
    agent.q_online.load_state_dict(ckpt['q_online'])
    agent.q_online.eval()
    agent.epsilon = 0.0

    traci.start([checkBinary("sumo"), "-c", SUMO_CFG, "--no-warnings", "true"])
    sample = junctions[0]
    unique_lanes = list(dict.fromkeys(traci.trafficlight.getControlledLanes(sample)))

    total_time          = 0.0
    step                = 0
    min_duration        = 5
    traffic_lights_time = {j: 0 for j in junctions}
    current_action      = {j: 0 for j in junctions}

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
                state  = build_state_v2(unique_lanes, current_action[junction], n_lanes)
                action = agent.choose_action(state)
                current_action[junction] = action
                phaseDuration(junction, 3, select_lane[action][0])
                phaseDuration(junction, min_duration + 10, select_lane[action][1])
                traffic_lights_time[junction] = min_duration + 10
            else:
                traffic_lights_time[junction] -= 1
        step += 1

    try: traci.close()
    except Exception: pass
    return total_time


def run_dqn_v3(junctions, state_length, n_lanes, steps=STEPS):
    select_lane = build_phase_strings_v3(state_length, n_lanes)
    input_dims  = 3 * n_lanes

    agent = DoubleDQNAgentV3(
        input_dims=input_dims, n_actions=V3_N_ACTIONS,
        epsilon_start=0.0, epsilon_end=0.0, warmup_steps=0,
    )
    ckpt = torch.load(MODEL_V3_PATH, map_location=agent.device)
    agent.q_online.load_state_dict(ckpt['q_online'])
    agent.q_online.eval()
    agent.epsilon = 0.0

    traci.start([checkBinary("sumo"), "-c", SUMO_CFG, "--no-warnings", "true"])
    sample = junctions[0]
    unique_lanes = list(dict.fromkeys(traci.trafficlight.getControlledLanes(sample)))

    total_time          = 0.0
    step                = 0
    traffic_lights_time = {j: 0 for j in junctions}
    current_lane_picked = {j: 0 for j in junctions}

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
                state  = build_state_v3(unique_lanes, current_lane_picked[junction], n_lanes)
                action = agent.choose_action(state)
                chosen_lane, chosen_duration = decode_action(action)
                current_lane_picked[junction] = chosen_lane

                phaseDuration(junction, 3, select_lane[chosen_lane][0])
                phaseDuration(junction, chosen_duration, select_lane[chosen_lane][1])
                traffic_lights_time[junction] = chosen_duration
            else:
                traffic_lights_time[junction] -= 1
        step += 1

    try: traci.close()
    except Exception: pass
    return total_time


def main():
    print("Detecting SUMO network...")
    junctions, state_length, lanes = detect_network()
    n_lanes = len(lanes)
    print(f"  Junctions: {junctions}, n_lanes: {n_lanes}\n")

    all_results = {}

    # Fixed timers
    for name, secs in FIXED_SETTINGS:
        print(f"Running {name} × {RUNS} ...")
        rs = []
        for i in range(RUNS):
            wt = run_fixed_timer(junctions, state_length, n_lanes, secs)
            rs.append(wt)
            print(f"  Run {i+1}: {wt:,.0f}")
        all_results[name] = rs
        print()

    # Double DQN v2
    if os.path.exists(MODEL_V2_PATH):
        print(f"Running Double DQN v2 (fixed 15s) × {RUNS} ...")
        rs = []
        for i in range(RUNS):
            wt = run_dqn_v2(junctions, state_length, n_lanes)
            rs.append(wt)
            print(f"  Run {i+1}: {wt:,.0f}")
        all_results["Double DQN v2"] = rs
        print()

    # Adaptive DQN v3
    if os.path.exists(MODEL_V3_PATH):
        print(f"Running Adaptive DQN v3 (lane+duration) × {RUNS} ...")
        rs = []
        for i in range(RUNS):
            wt = run_dqn_v3(junctions, state_length, n_lanes)
            rs.append(wt)
            print(f"  Run {i+1}: {wt:,.0f}")
        all_results["Adaptive DQN v3"] = rs

    # Print summary
    print("\n" + "="*80)
    print(f"{'Controller':<22} {'Mean (×10⁶)':>14}  {'Std':>10}  {'Rank':>6}")
    print("="*80)
    sorted_items = sorted(all_results.items(), key=lambda kv: np.mean(kv[1]))
    for rank, (name, results) in enumerate(sorted_items, 1):
        m = np.mean(results); s = np.std(results)
        marker = " ★" if "DQN" in name else "  "
        print(f"{name:<22} {m/1e6:>14.3f}  {s/1e6:>10.4f}  {rank:>4}{marker}")
    print("="*80)
    best_name, best_results = sorted_items[0]
    print(f"\nBest controller: {best_name} ({np.mean(best_results)/1e6:.3f}M)")

    # Save results.txt
    with open(os.path.join(OUTPUT_DIR, "results_v4.txt"), "w") as f:
        f.write("Final 5-way Benchmark Comparison\n")
        f.write("="*80 + "\n")
        f.write(f"Steps per run: {STEPS}\nRuns per controller: {RUNS}\n\n")
        for name, results in all_results.items():
            m = np.mean(results); s = np.std(results)
            f.write(f"{name}:\n")
            for i, r in enumerate(results):
                f.write(f"  Run {i+1}: {r:,.0f}\n")
            f.write(f"  Mean: {m:,.0f}  Std: {s:,.0f}\n\n")
        f.write("\nRanking (best → worst):\n")
        for rank, (name, results) in enumerate(sorted_items, 1):
            m = np.mean(results)
            f.write(f"  {rank}. {name}: {m:,.0f}\n")
    print(f"\nResults saved: {os.path.join(OUTPUT_DIR, 'results_v4.txt')}")

    # ── Plot ─────────────────────────────────────────────────────────────
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6.5))
    fig.patch.set_facecolor('#f5f1e8')

    for ax in (ax1, ax2):
        ax.set_facecolor('#faf7f0')
        for s in ax.spines.values():
            s.set_color('#d4c9b5')
        ax.tick_params(colors='#6b5f4a')
        ax.xaxis.label.set_color('#2c2416')
        ax.yaxis.label.set_color('#2c2416')
        ax.title.set_color('#2c2416')

    # Color mapping
    color_map = {
        'Fixed 10s':        '#dc2626',
        'Fixed 20s':        '#ea580c',
        'Fixed 30s':        '#ca8a04',
        'Double DQN v2':    '#2563eb',
        'Adaptive DQN v3':  '#16a34a',
    }

    controllers = list(all_results.keys())
    n_ctrls     = len(controllers)
    x = np.arange(RUNS)
    width = 0.8 / n_ctrls

    for i, name in enumerate(controllers):
        results = all_results[name]
        color   = color_map.get(name, '#888')
        offset  = (i - (n_ctrls - 1) / 2) * width
        ax1.bar(x + offset, [v/1e6 for v in results], width,
                color=color, label=name, edgecolor='#2c2416', linewidth=0.5)

    ax1.set_xticks(x)
    ax1.set_xticklabels([f'Run {i+1}' for i in range(RUNS)])
    ax1.set_ylabel('Total Waiting Time (×10⁶ s)')
    ax1.set_title('Per-Run Comparison', fontweight='bold')
    ax1.legend(facecolor='#ffffff', edgecolor='#d4c9b5', loc='upper right', fontsize=9)
    ax1.grid(True, color='#e8e0d0', axis='y', alpha=0.7, linewidth=0.6)

    # Mean ± std with ranking
    names = [n for n, _ in sorted_items]   # sorted by performance
    means = [np.mean(all_results[n])/1e6 for n in names]
    stds  = [np.std(all_results[n])/1e6  for n in names]
    bar_colors = [color_map.get(n, '#888') for n in names]

    bars = ax2.bar(names, means, yerr=stds, color=bar_colors,
                   capsize=8, edgecolor='#2c2416', linewidth=1,
                   error_kw={'ecolor': '#6b5f4a'})

    for i, (bar, val) in enumerate(zip(bars, means)):
        # Rank emoji-ish (no actual emoji, just text)
        rank_label = ['1st', '2nd', '3rd', '4th', '5th'][i]
        ax2.text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() + max(stds + [0.01])*0.7,
                 f'{val:.3f}M\n({rank_label})', ha='center', color='#2c2416',
                 fontsize=10, fontweight='bold')

    # Highlight winner
    winner = names[0]
    winner_idx = list(all_results.keys()).index(winner)
    ax2.text(0.5, -0.15,
             f"Winner: {winner} → {np.mean(all_results[winner])/1e6:.3f}M waiting time",
             transform=ax2.transAxes, ha='center', va='top',
             color='#16a34a', fontsize=12, fontweight='bold')

    ax2.set_ylabel('Total Waiting Time (×10⁶ s)')
    ax2.set_title(f'Final Ranking — Mean ± Std over {RUNS} runs', fontweight='bold')
    ax2.grid(True, color='#e8e0d0', axis='y', alpha=0.7, linewidth=0.6)
    ax2.tick_params(axis='x', rotation=15)

    plt.suptitle('Final Benchmark: Adaptive DQN vs Double DQN vs Fixed Timers',
                 fontsize=15, color='#2c2416', y=1.00, fontweight='bold')
    plt.tight_layout()

    chart_path = os.path.join(OUTPUT_DIR, "benchmark_v4_chart.png")
    plt.savefig(chart_path, dpi=150, bbox_inches='tight',
                facecolor='#f5f1e8', edgecolor='none')
    print(f"Chart saved : {chart_path}\n")
    plt.show()


if __name__ == "__main__":
    main()