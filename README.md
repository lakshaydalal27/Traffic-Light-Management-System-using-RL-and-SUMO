# 🚦 Intelligent Traffic Management System — Double DQN

> **Branch: `double-dqn-improved`** — Advanced reinforcement learning implementation with Double DQN, soft target updates, richer state representation, and improved reward shaping.

A complete end-to-end Reinforcement Learning system that uses **Double DQN** to control traffic signals at a 4-way intersection, with a real-time web dashboard for visualization.

---

## 🎯 What This Branch Achieves

Unlike the basic DQN on `main`, this branch implements **state-of-the-art techniques** and produces a trained agent that **beats real-world fixed-timer baselines**:

| Comparison | Result |
|------------|--------|
| DQN vs Fixed Timer (20s green/lane) | **+12% better** ✓ |
| DQN vs Fixed Timer (30s green/lane) | **+45% better** ✓ |
| DQN vs Fixed Timer (10s green/lane) | −104% (10s happens to be near-optimal for this scenario) |

See `performance/REPORT.md` for full analysis.

---

## 🧠 Algorithm: Double DQN with Improvements

| Technique | Why |
|-----------|-----|
| Double DQN (online + target networks) | Eliminates Q-value overestimation |
| Soft target updates (Polyak τ=0.005) | Smoother, more stable learning |
| Rich state (12 dims) | Counts + waiting times + current phase |
| Delta-based reward | Better credit assignment |
| Huber loss + gradient clipping | Robust training |
| Warmup phase (1000 steps) | Stable initial learning |
| Larger replay buffer (100k) | Better off-policy diversity |

---

## 🏗️ Architecture

```
┌──────────────────┐    HTTP    ┌──────────────────┐   TraCI    ┌────────────┐
│  Web Dashboard   │ ─────────► │ FastAPI Backend  │ ────────► │    SUMO    │
│  (HTML/CSS/JS)   │ ◄───────── │  (RL Inference)  │ ◄──────── │ Simulator  │
└──────────────────┘    JSON    └──────────────────┘   State   └────────────┘
                                          │
                                          ▼
                                ┌──────────────────┐
                                │  Double DQN      │
                                │  (PyTorch)       │
                                │  12 → 128 → 128  │
                                │  → 4 actions     │
                                └──────────────────┘
```

---

## 📁 Project Structure

```
.
├── train.py                       # Baseline DQN (legacy)
├── train_v2.py                    # Double DQN (THIS BRANCH)
├── configuration.sumocfg          # SUMO config
├── dashboard/
│   ├── main.py                    # FastAPI server + RL integration
│   ├── templates/index.html       # Light-theme dashboard UI
│   └── static/                    # CSS / JS
├── maps/
│   ├── network.net.xml            # 4-way intersection
│   └── routes.rou.xml             # 150 vehicles, mixed routes
├── models/
│   ├── best_model.bin             # baseline DQN (main branch)
│   └── double_dqn_best.bin        # ★ TRAINED Double DQN (this branch)
├── plots/
│   ├── time_vs_epoch_*.png
│   └── ddqn_training_*.png        # Double DQN training curve + loss
├── performance/
│   ├── benchmark.py               # baseline benchmark
│   ├── benchmark_v2.py            # Double DQN vs Fixed 15s
│   ├── benchmark_v3.py            # ★ Double DQN vs Fixed 10s/20s/30s
│   ├── REPORT.md                  # Full performance analysis
│   ├── dqn_v3_vs_fixed_timers.png # Final comparison chart
│   └── results_v3.txt             # Raw benchmark numbers
└── training_log_v2.txt            # Training log
```

---

## 🚀 How to Run

### 1. Install dependencies
```bash
pip install fastapi uvicorn traci sumolib torch numpy matplotlib
export SUMO_HOME="/path/to/sumo/share/sumo"
```

### 2. Train Double DQN from scratch (~25 minutes)
```bash
python3 train_v2.py -e 2500 -s 1500 -m double_dqn_best 2>&1 | tee training_log_v2.txt
```

### 3. Run benchmark against fixed timers
```bash
python3 performance/benchmark_v3.py
```

### 4. Run the dashboard
```bash
python3 dashboard/main.py
```
Open browser at **http://localhost:8000** and click **START**.

---

## 📊 Dashboard Features

- 🎨 Light cream theme with white panels
- 🎯 Real intersection visualization in center with road, crosswalks, traffic lights
- 🚗 Vehicle blocks rendered on each approach lane
- 🚦 Traffic light bulbs (red/yellow/green) light up correctly
- ✨ Active lane glows green when DQN selects it
- 🤖 Q-values per lane displayed (best one highlighted)
- 📈 Live epoch / epsilon / waiting time

---

## 🔬 Technical Details

### State Space (12 dimensions)
```
[count_lane1, count_lane2, count_lane3, count_lane4,        ← vehicle counts
 wait_lane1,  wait_lane2,  wait_lane3,  wait_lane4,          ← waiting times (normalized)
 phase_oh_1,  phase_oh_2,  phase_oh_3,  phase_oh_4]          ← current phase one-hot
```

### Action Space
4 discrete actions: choose which lane gets green light.

### Reward
```
reward = -(waiting_time_now - waiting_time_when_action_chosen) / 1000
```
Negative delta of waiting time — rewards actions that reduce queue growth.

### Training
```
2500 epochs × 1500 steps = 3,750,000 environment interactions
```

---

## 🎓 Academic Foundation

References used:
- Mnih et al. (2015) — *Human-level control through deep reinforcement learning*
- van Hasselt et al. (2015) — *Deep Reinforcement Learning with Double Q-learning*
- Lillicrap et al. (2015) — *Continuous Control with Deep RL* (soft target updates)
- Sutton & Barto (2018) — *Reinforcement Learning: An Introduction*

All techniques are standard, all implementations from scratch in PyTorch.

---

## 🛠️ Tech Stack

- **PyTorch** — Double DQN neural network
- **SUMO** — traffic simulator (DLR, Germany)
- **TraCI** — Python ↔ SUMO API
- **FastAPI** + **Uvicorn** — backend
- **Vanilla HTML/CSS/JS** — dashboard

---

*Branch: double-dqn-improved · 6th Semester Project — Intelligent Traffic Management System*