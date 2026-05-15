# 🚦 Intelligent Traffic Management System using Deep Q-Network (DQN)

A complete end-to-end Reinforcement Learning system that uses Deep Q-Network (DQN) to control traffic signals at a 4-way intersection, with a real-time web dashboard for visualization.

---

## 🎯 What This Project Does

- Trains a **Deep Q-Network agent** (built from scratch in PyTorch) to control traffic signals
- Uses **SUMO** (open-source traffic simulator) as the RL environment
- Connects them through **TraCI** (Traffic Control Interface)
- Visualizes everything live on a **FastAPI + HTML/CSS/JS dashboard**

---

## 🧠 The RL Problem

| Component | Definition |
|-----------|------------|
| **State** | Vehicle count on each of 4 lanes (4-D vector) |
| **Action** | Choose which lane gets green (4 discrete actions) |
| **Reward** | Negative of total waiting time (minimize delays) |
| **Network** | 2-layer MLP: 4 → 256 → 256 → 4 |
| **Algorithm** | Deep Q-Network with experience replay |

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
                                │   DQN Agent      │
                                │  (PyTorch)       │
                                └──────────────────┘
```

---

## 📁 Project Structure

```
.
├── train.py                    # DQN agent + training loop
├── configuration.sumocfg       # SUMO config
├── dashboard/
│   ├── main.py                 # FastAPI server + RL integration
│   ├── templates/index.html    # Dashboard UI
│   └── static/                 # CSS / JS
├── maps/
│   ├── network.net.xml         # 4-way intersection road network
│   └── routes.rou.xml          # Vehicle traffic definitions
├── models/
│   └── best_model.bin          # Trained DQN weights
├── plots/
│   └── time_vs_epoch_*.png     # Training curves
└── performance/
    ├── benchmark.py            # DQN vs fixed-timer benchmark
    ├── REPORT.md               # Detailed performance analysis
    └── results.txt             # Numerical benchmark results
```

---

## 🚀 How to Run

### 1. Install dependencies
```bash
# SUMO (macOS via .pkg from sumo.dlr.de)
# Python packages:
pip install fastapi uvicorn traci sumolib torch numpy matplotlib
```

### 2. Set environment variable
```bash
export SUMO_HOME="/path/to/sumo/share/sumo"
```

### 3. Train the model
```bash
python3 train.py --train -e 50 -s 500 -m best_model
```

### 4. Run the dashboard
```bash
python3 dashboard/main.py
```
Open browser at **http://localhost:8000** and click **START**.

### 5. (Optional) Run the benchmark
```bash
python3 performance/benchmark.py
```

---

## 📊 Dashboard Features

- **Live intersection view** — visual SVG of the 4-way intersection with colored traffic lights
- **Per-lane metrics** — vehicle counts, average speed, congestion level
- **DQN agent status** — current action, Q-values for all 4 lanes, exploration rate (ε)
- **Training info** — current epoch, mode (training/inference)

---

## ⚠️ Honest Notes on Performance

We use a **basic DQN** implementation without advanced techniques like Double DQN or target networks. Training for 50 epochs shows the **classic exploration-exploitation tradeoff** documented in RL literature — the agent finds good policies during exploration and our framework saves the best checkpoint.

See `performance/REPORT.md` for the full honest analysis including:
- Training curve interpretation
- Benchmark results vs fixed-timer baseline
- Known limitations
- Future improvements (Double DQN, prioritized replay, longer training)

---

## 🛠️ Tech Stack

- **PyTorch** — DQN neural network implementation
- **SUMO** — traffic simulator (DLR, Germany)
- **TraCI** — Python ↔ SUMO API
- **FastAPI** — backend REST API
- **Uvicorn** — ASGI web server
- **HTML/CSS/JS** — dashboard frontend (no framework, vanilla)

---

## 📚 References

- Mnih, V. et al. (2015). *Human-level control through deep reinforcement learning*. Nature.
- Sutton, R. & Barto, A. (2018). *Reinforcement Learning: An Introduction*. MIT Press.
- SUMO documentation: https://sumo.dlr.de/docs/

---

*6th Semester Project — Intelligent Traffic Management System*