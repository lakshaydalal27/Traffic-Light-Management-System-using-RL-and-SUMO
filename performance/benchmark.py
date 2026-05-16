"""
Benchmark v3: Double DQN vs MULTIPLE Fixed-Timer baselines.

Real-world traffic engineers use varied green-time durations (10s, 20s, 30s)
depending on intersection load. We benchmark our trained DQN against several
realistic settings to show it learns a competitive adaptive policy.

Usage: python3 performance/benchmark_v3.py
Output: performance/dqn_v3_vs_fixed_timers.png
        performance/results_v3.txt
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

from train import (
    DoubleDQNAgent, build_phase_strings, build_state,
    total_waiting, phaseDuration,
)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SUMO_CFG     = os.path.join(PROJECT_ROOT, "configuration.sumocfg")
MODEL_PATH   = os.path.join(PROJECT_ROOT, "models", "double_dqn_best.bin")
OUTPUT_DIR   = os.path.dirname(os.path.abspath(__file__))

STEPS = 1500
RUNS  = 3   # 3 runs per controller (we have many controllers to test)

# Multiple realistic fixed-timer settings (real-world traffic engineering values)
FIXED_TIMER_SETTINGS = [
    ("Fixed 10s", 10),   # short cycle (heavy traffic city)
    ("Fixed 20s", 20),   # standard cycle
    ("Fixed 30s", 30),   # long cycle (suburban)
]


def detect_network():
    traci.start([checkBinary("sumo"), "-c", SUMO_CFG, "--no-warnings", "true"])
    junctions    = traci.trafficlight.getIDList()
    sample       = junctions[0]
    state_length = len(traci.trafficlight.getRedYellowGreenState(sample))
    lanes        = list(dict.fromkeys(traci.trafficlight.getControlledLanes(sample)))
    n_actions    = len(lanes)
    traci.close()
    return junctions, state_length, lanes, n_actions


def run_fixed_timer(junctions, state_length, n_actions, green_secs, steps=STEPS):
    select_lane = build_phase_strings(state_length, n_actions)

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
            current_lane = (current_lane + 1) % n_actions
            time_left    = green_secs
        else:
            time_left -= 1
        step += 1

    try: traci.close()
    except Exception: pass
    return total_time


def run_dqn_v2(junctions, state_length, lanes, n_actions, steps=STEPS):
    select_lane = build_phase_strings(state_length, n_actions)
    input_dims  = 3 * n_actions

    agent = DoubleDQNAgent(
        input_dims=input_dims, n_actions=n_actions,
        epsilon_start=0.0, epsilon_end=0.0, warmup_steps=0,
    )
    ckpt = torch.load(MODEL_PATH, map_location=agent.device)
    agent.q_online.load_state_dict(ckpt['q_online'])
    if 'q_target' in ckpt:
        agent.q_target.load_state_dict(ckpt['q_target'])
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
                state = build_state(unique_lanes, current_action[junction], n_actions)
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


def main():
    print("Detecting SUMO network...")
    junctions, state_length, lanes, n_actions = detect_network()
    print(f"  Junctions: {junctions}, n_actions: {n_actions}\n")

    all_results = {}  # {controller_name: [run_results]}

    # ── Fixed timer baselines ────────────────────────────────────────────
    for name, secs in FIXED_TIMER_SETTINGS:
        print(f"Running {name} (green={secs}s) × {RUNS} ...")
        results = []
        for i in range(RUNS):
            wt = run_fixed_timer(junctions, state_length, n_actions, secs)
            results.append(wt)
            print(f"  Run {i+1}: {wt:,.0f}")
        all_results[name] = results
        print()

    # ── Double DQN ────────────────────────────────────────────────────────
    print(f"Running Double DQN (trained) × {RUNS} ...")
    dqn_results = []
    for i in range(RUNS):
        wt = run_dqn_v2(junctions, state_length, lanes, n_actions)
        dqn_results.append(wt)
        print(f"  Run {i+1}: {wt:,.0f}")
    all_results["Double DQN"] = dqn_results

    # ── Print results table ──────────────────────────────────────────────
    print("\n" + "="*70)
    print(f"{'Controller':<20} {'Mean (×10⁶)':>14}  {'Std':>10}  {'vs DQN':>12}")
    print("="*70)
    dqn_mean = np.mean(dqn_results)
    for name, results in all_results.items():
        m  = np.mean(results)
        s  = np.std(results)
        if name == "Double DQN":
            cmp = "(baseline)"
        else:
            diff = (1 - dqn_mean / m) * 100
            cmp = f"{diff:+.1f}%" if diff > 0 else f"{diff:.1f}%"
        print(f"{name:<20} {m/1e6:>14.3f}  {s/1e6:>10.4f}  {cmp:>12}")
    print("="*70)

    # ── Save results.txt ─────────────────────────────────────────────────
    with open(os.path.join(OUTPUT_DIR, "results_v3.txt"), "w") as f:
        f.write("Double DQN vs Multiple Fixed-Timer Baselines\n")
        f.write("="*70 + "\n")
        f.write(f"Steps per run: {STEPS}\nRuns per controller: {RUNS}\n\n")
        for name, results in all_results.items():
            m = np.mean(results)
            s = np.std(results)
            f.write(f"{name}:\n")
            for i, r in enumerate(results):
                f.write(f"  Run {i+1}: {r:,.0f}\n")
            f.write(f"  Mean: {m:,.0f}  Std: {s:,.0f}\n\n")
        f.write("Improvement of Double DQN vs each baseline:\n")
        for name, results in all_results.items():
            if name == "Double DQN": continue
            m = np.mean(results)
            diff = (1 - dqn_mean / m) * 100
            f.write(f"  vs {name}: {diff:+.1f}%\n")
    print(f"\nResults saved: {os.path.join(OUTPUT_DIR, 'results_v3.txt')}")

    # ── Plot ──────────────────────────────────────────────────────────────
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    fig.patch.set_facecolor('#f5f1e8')

    for ax in (ax1, ax2):
        ax.set_facecolor('#faf7f0')
        for s in ax.spines.values():
            s.set_color('#d4c9b5')
        ax.tick_params(colors='#6b5f4a')
        ax.xaxis.label.set_color('#2c2416')
        ax.yaxis.label.set_color('#2c2416')
        ax.title.set_color('#2c2416')

    # Left: per-run bars (grouped)
    controllers = list(all_results.keys())
    n_ctrls = len(controllers)
    x = np.arange(RUNS)
    width = 0.8 / n_ctrls
    colors_fixed = ['#dc2626', '#ea580c', '#ca8a04']  # red/orange/yellow for fixed
    colors_dqn   = '#16a34a'

    for i, name in enumerate(controllers):
        results = all_results[name]
        color = colors_dqn if name == "Double DQN" else colors_fixed[i]
        offset = (i - (n_ctrls - 1) / 2) * width
        ax1.bar(x + offset, [v/1e6 for v in results], width,
                color=color, label=name, edgecolor='#2c2416', linewidth=0.5)

    ax1.set_xticks(x)
    ax1.set_xticklabels([f'Run {i+1}' for i in range(RUNS)])
    ax1.set_ylabel('Total Waiting Time (×10⁶ s)')
    ax1.set_title('Per-Run Comparison', fontweight='bold')
    ax1.legend(facecolor='#ffffff', edgecolor='#d4c9b5', loc='upper right')
    ax1.grid(True, color='#e8e0d0', axis='y', alpha=0.7, linewidth=0.6)

    # Right: mean ± std bar chart
    names = list(all_results.keys())
    means = [np.mean(all_results[n])/1e6 for n in names]
    stds  = [np.std(all_results[n])/1e6  for n in names]
    bar_colors = [colors_fixed[i] if n != "Double DQN" else colors_dqn
                  for i, n in enumerate(names)]
    bars = ax2.bar(names, means, yerr=stds, color=bar_colors,
                   capsize=10, edgecolor='#2c2416', linewidth=1,
                   error_kw={'ecolor': '#6b5f4a'})
    for bar, val in zip(bars, means):
        ax2.text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() + max(stds + [0.005])*0.5,
                 f'{val:.3f}M', ha='center', color='#2c2416',
                 fontsize=11, fontweight='bold')

    # Add improvement annotations vs each fixed timer
    annotations = []
    for n in names:
        if n == "Double DQN": continue
        m = np.mean(all_results[n])
        diff = (1 - dqn_mean / m) * 100
        annotations.append(f"vs {n}: {diff:+.1f}%")
    box_text = "DQN performance:\n" + "\n".join(annotations)
    # Use green if DQN wins overall, red if it loses overall
    avg_baseline = np.mean([np.mean(all_results[n])
                            for n in names if n != "Double DQN"])
    box_color = '#16a34a' if dqn_mean < avg_baseline else '#dc2626'
    ax2.text(0.98, 0.97, box_text,
             transform=ax2.transAxes, ha='right', va='top',
             color=box_color, fontsize=10, fontweight='bold',
             bbox=dict(boxstyle='round,pad=0.5',
                       facecolor='#ffffff',
                       edgecolor=box_color, linewidth=1.5, alpha=0.95))
    ax2.set_ylabel('Total Waiting Time (×10⁶ s)')
    ax2.set_title(f'Mean ± Std over {RUNS} runs', fontweight='bold')
    ax2.grid(True, color='#e8e0d0', axis='y', alpha=0.7, linewidth=0.6)

    plt.suptitle('Double DQN vs Multiple Fixed-Timer Settings — Real-World Comparison',
                 fontsize=14, color='#2c2416', y=0.99, fontweight='bold')
    plt.tight_layout()

    chart_path = os.path.join(OUTPUT_DIR, "dqn_v3_vs_fixed_timers.png")
    plt.savefig(chart_path, dpi=150, bbox_inches='tight',
                facecolor='#f5f1e8', edgecolor='none')
    print(f"Chart saved: {chart_path}\n")
    plt.show()


if __name__ == "__main__":
    main()