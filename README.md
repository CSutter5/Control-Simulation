# Control-Simulation

A lightweight, extensible simulation for control of rockets. This repository contains a simple physics-based simulator, example flight data, and an example control scenario (`example/CRL2026/CRL2026_Canards.py`). It is intended as a development playground for controller design, aerodynamics experiments, and rapid prototyping of guidance logic.

**Contents:**
- `environments/`: simulation environment (`CanardSim`, `ReactionWheelSim`) that steps physics and applies control commands.
- `properties/`: dataclasses for `Canard`, `ReactionWheel`, `Rocket`, and `Data` (air data and recorded signals).
- `example/CRL2026/`: example scenario, airfoil CSVs, and example flight CSV used by the example script.

Getting started
--------------

1. Install dependencies (recommended in a virtualenv):

```bash
pip install -r requirements.txt
python -m pip install --user -e . --break-system-packages
```

2. Run the example from the project root:

```bash
cd example/CRL2026/
python CRL2026_Canards.py
```

Notes: the example uses local CSV files in `example/CRL2026/airfoil` and a flight data CSV. You can run the script from the `example/CRL2026` folder directly if you prefer.

How the project is organized
---------------------------

- `CanardSim` (in `environments/CanardSim.py`) implements the simulation loop and calls into `properties` for aerodynamic and flight data.
- `properties/Canard.py` reads airfoil CL data and interpolates CL for angle/speed pairs.
- `properties/Data.py` provides simple helpers to query airspeed, density, and recorded roll rate by time.

Extending and customizing
-------------------------

- Add new examples: copy `example/CRL2026/` and tweak `targetFunction`, PID gains, or the CSV inputs.
- Add more accurate aerodynamics: replace the simple CSV-based CL lookup with a higher-fidelity model or use external aero libraries. Consider wrapping heavy computations in a dedicated module for unit testing.
- Add sensors and latency: introduce sensor models and actuator dynamics in `Canard` or `CanardSim` to test robustness of control laws.
- Logging and replay: extend `CanardSim` to save deterministic simulation checkpoints and a JSON/YAML run description to reproduce experiments.
- Parameter sweep and batch runs: add a small CLI or a `scripts/` runner that launches multiple simulations with different controller parameters and aggregates results into plots or CSV.
- Adding more control techniques like tvc, airbreaks or 6dof aero control
- Add system dampening


Plans for Rewrite
-----------------
Instead of having seperate simulation env's, there should only be 1 env and you can add different control mechanisms to a rocket.
That way you can have either use multiply control mechanisms or swap them out easily

The `rocket` will contain the simulation engine and there will be a generic class `control` that canards, reaction wheels, etc. will inherit from. \
To run a step you use kwargs to set the control mechanisms ex. `rocket.step(canardAngle=0,reactionWheelSpeed=200)`

Inside of the `rocket.step` method the relavent control mechanisms will be updated and queries for all the toqures applied to the main body. tl;dr/ex the rocket will update the canard angle and in return it will a tuple of torques

