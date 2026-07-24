# Control-Simulation

A lightweight, extensible simulation for rocket control experiments. The project now uses a single rocket model with pluggable control modules, making it easier to swap between control mechanisms or combine multiple ones in one simulation.

Overview
--------

The current design is centered around a unified `Rocket` class that owns the simulation state and advances the dynamics each step. Control objects live in the `Controls` package and implement a common interface that returns a torque tuple for the rocket to apply.

At the moment, the implementation focuses on roll control using canards, but the architecture is intended to support reaction wheels and other control surfaces in the future.

Project layout
--------------

- `Rocket/Rocket.py`: the main simulation engine and state container for the rocket.
- `Controls/Controls.py`: abstract base class for all control implementations.
- `Controls/Canards.py`: canard model with aerodynamic lift estimation from airfoil data and rate limiting.
- `Controls/ReactionWheel.py`: placeholder for a future reaction wheel implementation.
- `example/CRL2026/`: example scenario, airfoil data, flight data, and the canard-based controller demo.

Getting started
---------------

1. Install dependencies (recommended in a virtual environment):

```bash
pip install -r requirements.txt
python -m pip install --user -e . --break-system-packages
```

2. Run the example from the project root:

```bash
cd example/CRL2026/
python CRL2026_Canards.py
```

The example uses local CSV files from `example/CRL2026/airfoil` and the example rocket flight data in that folder.

How the simulation works
------------------------

- The `Rocket` object stores state such as position, velocity, orientation, and angular rates.
- Each simulation step calls every attached control object via `rocket.sim(...)`.
- Each control returns a torque tuple in the form `(yawTorque_Nm, pitchTorque_Nm, rollTorque_Nm)`.
- The rocket applies the aggregate torques and updates its state for the next step.

Current implementation details
-----------------------------

- Canards are modeled from a CSV-based lift table and interpolate lift coefficient based on angle of attack and velocity.
- The canard controller accepts a desired angle through the `canardAngle_deg` keyword argument.
- The current physics implementation is focused on roll dynamics; pitch and yaw are not yet fully implemented.
- In the current rocket step logic, any non-zero yaw or pitch torque will raise an error to indicate that those axes are still pending implementation.

Extending the project
---------------------

- Add a new control module by inheriting from `Controls` and implementing `sim(...)`.
- Combine multiple control mechanisms by passing several objects into the rocket's `controls` list.
- Improve the aerodynamic model by replacing the CSV lookup with a higher-fidelity physics model.
- Add sensors, latency, actuator dynamics, or additional state estimation.
- Implement pitch and yaw dynamics so the rocket can simulate non-zero yaw and pitch torques instead of raising an error.
- Add batch runs, parameter sweeps, or plotting utilities for controller tuning.

