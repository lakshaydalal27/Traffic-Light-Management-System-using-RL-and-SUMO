# 🚦 Intelligent Traffic Management System using Deep Q-Network (DQN)

A real-time traffic signal control system that uses Deep Reinforcement Learning (DQN) to dynamically optimize traffic light phases at a 4-way intersection, replacing traditional fixed-timer systems with an intelligent adaptive agent.

---

## 🧠 What Makes This Different

Traditional traffic systems use **fixed timers** — every direction gets green for the same duration regardless of how many vehicles are waiting. This wastes time and creates unnecessary congestion.

This system uses a **Deep Q-Network (DQN)** agent that:
- Observes the number of vehicles queued on each of the 4 lanes in real time
- Decides **which lane gets the green light** based on learned Q-values
- Minimizes total vehicle waiting time across the intersection
- Improves its decisions through trial-and-error across training episodes

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────┐
│                   Web Dashboard                      │
│         (FastAPI + HTML/CSS/JS frontend)             │
│   Live intersection view · Q-values · RL decisions   │
└─────────────────────┬───────────────────────────────┘
                      │ REST API (/api/data, /api/start)
┌─────────────────────▼───────────────────────────────┐
│              FastAPI Backend (main.py)               │
│      Runs RL agent · Feeds data to dashboard         │
└─────────────────────┬───────────────────────────────┘
                      │ TraCI (Traffic Control Interface)
┌─────────────────────▼───────────────────────────────┐
│           SUMO Traffic Simulator (headless)          │
│     Simulates vehicles · Applies signal phases       │
└─────────────────────────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────┐
│              DQN Agent (train.py)                    │
│   State: vehicle counts per lane (4 values)          │
│   Actions: 4 (which lane gets green)                 │
│   Reward: negative of total waiting time             │
│   Network: 2-layer MLP (256 → 256 → 4)              │
└─────────────────────────────────────────────────────┘
```

---

## 📁 Project Structure

```
├── train.py                  # DQN agent + SUMO training loop
├── configuration.sumocfg     # SUMO simulation config
├── dashboard/
│   ├── main.py               # FastAPI server + RL integration
│   ├── templates/
│   │   └── index.html        # Dashboard UI
│   └── static/
│       ├── css/style.css
│       └── js/script.js
├── maps/
│   ├── network.net.xml       # 4-way intersection road network
│   └── routes.rou.xml        # Vehicle route definitions
├── models/
│   └── best_model.bin        # Trained DQN model weights
├── plots/
│   └── time_vs_epoch_best_model.png   # Training curve
└── performance/
    ├── comparison.md         # DQN vs Fixed Timer comparison
    └── generate_comparison.py # Performance visualization script
```

---

## 🚀 How to Run

### Prerequisites
- Python 3.12+
- SUMO 1.26+ (install from https://sumo.dlr.de)
- Set `SUMO_HOME` environment variable

### Install Dependencies
```bash
pip install fastapi uvicorn traci sumolib torch numpy matplotlib
```

### Train the Model
```bash
python3 train.py --train -e 50 -s 500 -m best_model
```

### Run the Dashboard
```bash
python3 dashboard/main.py
```

Open browser at **http://localhost:8000**

Click **START** to begin the RL simulation.

---

## 🤖 DQN Agent Details

| Parameter | Value |
|-----------|-------|
| State space | 4 (vehicle count per lane) |
| Action space | 4 (which lane gets green) |
| Hidden layers | 2 × 256 neurons |
| Activation | ReLU |
| Optimizer | Adam (lr=0.001) |
| Discount factor γ | 0.99 |
| Epsilon (start) | 1.0 (full exploration) |
| Epsilon (end) | 0.05 |
| Epsilon decay | 5e-4 per step |
| Reward function | −1 × total waiting time |

---

## 📊 Performance

See `performance/comparison.md` for detailed comparison between DQN and fixed-timer baseline.

Key result: **DQN reduces average vehicle waiting time by ~35-40%** compared to a fixed 30-second timer system in early training epochs.

---

## 🛠️ Tech Stack

- **SUMO** — open-source traffic simulator (DLR, Germany)
- **TraCI** — Python API for real-time SUMO control
- **PyTorch** — DQN neural network
- **FastAPI** — backend REST API
- **Uvicorn** — ASGI server
- **HTML/CSS/JS** — dashboard frontend (no framework)

---

## 👨‍💻 How It Works — Step by Step

1. SUMO spawns vehicles on 4 approach lanes toward a single intersection
2. Every signal cycle, TraCI reads vehicle counts on each lane → **state**
3. DQN agent picks the lane with highest Q-value → **action**
4. Selected lane gets green for 15 seconds, others stay red
5. Waiting time is measured → **reward = −waiting_time**
6. Agent learns via Bellman equation update (Q-learning)
7. Over episodes, agent learns to prioritize congested lanes

---

*Built as part of 6th semester project — Intelligent Traffic Management System*