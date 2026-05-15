# FastAPI server connecting SUMO + DQN RL Agent with web dashboard
from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import threading
import time
import uvicorn
import os
import sys
import numpy as np
import torch

# ── SUMO setup ────────────────────────────────────────────────────────────────
if "SUMO_HOME" in os.environ:
    sys.path.append(os.path.join(os.environ["SUMO_HOME"], "tools"))
else:
    sys.exit("Please declare environment variable 'SUMO_HOME'")

import traci
from sumolib import checkBinary

# ── Path setup ────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))   # dashboard/
PARENT_DIR = os.path.dirname(BASE_DIR)                    # project root
SUMO_CFG   = os.path.join(PARENT_DIR, "configuration.sumocfg")
MODEL_PATH = os.path.join(PARENT_DIR, "models", "model.bin")

# ── Import RL agent from train.py ─────────────────────────────────────────────
sys.path.insert(0, PARENT_DIR)
from train import Model, Agent, get_vehicle_numbers, get_waiting_time, phaseDuration, build_phase_strings

# ── FastAPI app ───────────────────────────────────────────────────────────────
app = FastAPI()
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))
app.mount("/static", StaticFiles(directory=os.path.join(BASE_DIR, "static")), name="static")

# ── Numpy serialization fix ───────────────────────────────────────────────────
def to_python(obj):
    """Recursively convert numpy types to native Python for JSON serialization."""
    if isinstance(obj, dict):
        return {k: to_python(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [to_python(v) for v in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.bool_):
        return bool(obj)
    return obj

# ── Global state ──────────────────────────────────────────────────────────────
sumo_running = False
simulation_data = {
    "vehicles": [],
    "step": 0,
    "traffic_lights": {},
    "rl_info": {
        "action": 0,
        "action_name": "Waiting to start...",
        "q_values": [],
        "waiting_time": 0,
        "total_waiting_time": 0,
        "epoch": 0,
        "step": 0,
        "epsilon": 1.0,
        "mode": "training",
        "lane_vehicles": []
    },
    "statistics": {
        "total_vehicles_passed": 0,
        "peak_vehicle_count": 0,
        "total_waiting_time": 0,
    }
}

ACTION_NAMES = ["Lane 1 Green", "Lane 2 Green", "Lane 3 Green", "Lane 4 Green"]

# ── SUMO cleanup ──────────────────────────────────────────────────────────────
def cleanup_sumo():
    global sumo_running
    sumo_running = False
    try:
        traci.simulation.getTime()  # test if connection alive
        traci.close()
        print("SUMO closed cleanly.")
    except:
        pass  # already closed, that's fine

# ── Main RL simulation loop ───────────────────────────────────────────────────
def run_rl_simulation(train_mode=True, epochs=100, steps=500):
    global sumo_running, simulation_data

    cleanup_sumo()
    time.sleep(0.3)

    # Detect network properties
    try:
        traci.start([checkBinary("sumo"), "-c", SUMO_CFG,
                     "--tripinfo-output", os.path.join(PARENT_DIR, "maps", "tripinfo.xml")])
        all_junctions    = traci.trafficlight.getIDList()
        junction_numbers = list(range(len(all_junctions)))
        sample_junction  = all_junctions[0]
        state_length     = len(traci.trafficlight.getRedYellowGreenState(sample_junction))
        controlled_lanes = traci.trafficlight.getControlledLanes(sample_junction)
        unique_lanes     = list(dict.fromkeys(controlled_lanes))
        input_dims       = len(unique_lanes)
        traci.close()
    except Exception as ex:
        print(f"Failed to detect network: {ex}")
        sumo_running = False
        return

    select_lane = build_phase_strings(state_length, input_dims)

    brain = Agent(
        gamma=0.99,
        epsilon=1.0 if train_mode else 0.0,
        lr=0.001,
        input_dims=input_dims,
        fc1_dims=256,
        fc2_dims=256,
        batch_size=1024,
        n_actions=input_dims,
        junctions=junction_numbers,
    )

    if not train_mode and os.path.exists(MODEL_PATH):
        brain.Q_eval.load_state_dict(
            torch.load(MODEL_PATH, map_location=brain.Q_eval.device))
        print("Loaded trained model.")

    best_time    = np.inf
    min_duration = 5

    for e in range(epochs):
        if not sumo_running and e > 0:
            break

        try:
            traci.start([checkBinary("sumo"), "-c", SUMO_CFG,
                         "--tripinfo-output", os.path.join(PARENT_DIR, "tripinfo.xml")])
        except Exception as ex:
            print(f"Failed to start SUMO for epoch {e}: {ex}")
            break

        sumo_running = True
        print(f"Epoch {e}")

        step = 0
        total_time = 0
        prev_vehicles_per_lane = {jn: [0] * input_dims for jn in junction_numbers}
        prev_action            = {jn: 0 for jn in junction_numbers}
        traffic_lights_time    = {j: 0 for j in all_junctions}
        previous_vehicles      = set()

        while step <= steps and sumo_running:
            # ── Simulation step ───────────────────────────────────────────────
            try:
                traci.simulationStep()
            except Exception:
                print(f"Simulation step failed at step {step}, stopping epoch.")
                break

            # ── Collect vehicle data ──────────────────────────────────────────
            try:
                vehicle_ids      = traci.vehicle.getIDList()
                current_vehicles = set(vehicle_ids)
                completed        = previous_vehicles - current_vehicles
                simulation_data["statistics"]["total_vehicles_passed"] += len(completed)
                previous_vehicles = current_vehicles

                vehicles = []
                for veh_id in vehicle_ids:
                    try:
                        pos   = traci.vehicle.getPosition(veh_id)
                        speed = traci.vehicle.getSpeed(veh_id)
                        vehicles.append({
                            "id": str(veh_id),
                            "x": round(float(pos[0]), 1),
                            "y": round(float(pos[1]), 1),
                            "speed": round(float(speed), 1)
                        })
                    except:
                        continue

                peak = simulation_data["statistics"]["peak_vehicle_count"]
                if len(vehicles) > peak:
                    simulation_data["statistics"]["peak_vehicle_count"] = len(vehicles)
            except Exception:
                vehicles = []

            # ── RL decision loop ──────────────────────────────────────────────
            for junction_number, junction in enumerate(all_junctions):
                try:
                    controled_lanes = traci.trafficlight.getControlledLanes(junction)
                except Exception:
                    break  # SUMO connection closed (stop button pressed)

                try:
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

                        reward = -1 * waiting_time
                        state  = prev_vehicles_per_lane[junction_number]
                        prev_vehicles_per_lane[junction_number] = state_

                        brain.store_transition(state, state_, prev_action[junction_number],
                                               reward, (step == steps), junction_number)

                        lane = brain.choose_action(state_)
                        prev_action[junction_number] = lane

                        # Get Q-values for dashboard
                        with torch.no_grad():
                            state_tensor = torch.tensor([state_], dtype=torch.float).to(brain.Q_eval.device)
                            q_vals = brain.Q_eval.forward(state_tensor).cpu().numpy()[0].tolist()

                        phaseDuration(junction, 6, select_lane[lane][0])
                        phaseDuration(junction, min_duration + 10, select_lane[lane][1])
                        traffic_lights_time[junction] = min_duration + 10

                        if train_mode:
                            brain.learn(junction_number)

                        try:
                            tl_state = traci.trafficlight.getRedYellowGreenState(junction)
                        except:
                            tl_state = "unknown"

                        simulation_data["rl_info"] = {
                            "action": int(lane),
                            "action_name": ACTION_NAMES[lane] if lane < len(ACTION_NAMES) else f"Lane {lane+1} Green",
                            "q_values": [round(float(q), 2) for q in q_vals],
                            "waiting_time": round(float(waiting_time), 1),
                            "total_waiting_time": round(float(total_time), 1),
                            "epoch": int(e),
                            "step": int(step),
                            "epsilon": round(float(brain.epsilon), 3),
                            "mode": "training" if train_mode else "inference",
                            "lane_vehicles": [int(x) for x in state_],
                        }
                        simulation_data["traffic_lights"] = {
                            str(junction): {
                                "state": str(tl_state),
                                "phase_index": int(lane),
                                "manual_mode": False
                            }
                        }
                    else:
                        traffic_lights_time[junction] -= 1

                except Exception as ex:
                    print(f"RL step error: {ex}")
                    break

            simulation_data["vehicles"] = vehicles
            simulation_data["step"]     = int(step)
            simulation_data["statistics"]["total_waiting_time"] = round(float(total_time), 1)
            step += 1

        print(f"Epoch {e} — total_time: {total_time}")

        if total_time < best_time and total_time > 0:
            best_time = total_time
            if train_mode:
                brain.save("model")
                print(f"  New best model saved (waiting time: {total_time})")

        # Safe close at end of each epoch
        try:
            traci.close()
        except:
            pass

        if not train_mode:
            break

    sumo_running = False
    print("Simulation finished.")


# ── API Routes ────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
def dashboard(request: Request):
    return templates.TemplateResponse("index.html", {
        "request": request,
        "configs": ["configuration.sumocfg"]
    })

@app.get("/api/data")
def get_simulation_data():
    return to_python(simulation_data)

@app.get("/api/status")
def get_status():
    return {"running": sumo_running}

@app.post("/api/start")
def start_simulation():
    global sumo_running
    if not sumo_running:
        sumo_running = True
        thread = threading.Thread(
            target=run_rl_simulation,
            kwargs={"train_mode": True, "epochs": 100, "steps": 500},
            daemon=True
        )
        thread.start()
        return {"message": "RL simulation started (training mode, 100 epochs)"}
    return {"message": "Simulation already running"}

@app.post("/api/stop")
def stop_simulation():
    cleanup_sumo()
    return {"message": "Simulation stopped"}

@app.get("/api/rl_status")
def get_rl_status():
    return to_python(simulation_data.get("rl_info", {}))

@app.get("/api/statistics")
def get_statistics():
    return to_python(simulation_data.get("statistics", {}))

@app.get("/api/traffic_light/status")
def get_traffic_light_status():
    return to_python({
        "manual_mode": False,
        "current_simulation_data": simulation_data.get("traffic_lights", {})
    })

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)