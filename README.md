# Control-Simulation

A lightweight, extensible simulation for rocket control experiments. A single
`Rocket` model owns the flight state and physics; pluggable `Controls`
objects (canards, reaction wheels, etc.) generate torques against that
model, making it easy to swap between control mechanisms or combine several
in one simulation.

## Overview

The design centers on a unified `Rocket` class that owns the simulation
state (position, velocity, orientation, angular rates) and advances it one
timestep at a time. `Controls` objects implement a common interface —
`sim(rocket, **kwargs) -> (yawTorque_Nm, pitchTorque_Nm, rollTorque_Nm)` —
and the rocket polls every attached control each step, sums their torques,
and integrates its state forward.

At the moment, the reference implementation is a canard-based roll
controller (`Controls/Canards.py`), driven by aerodynamic lift estimated
from an airfoil CSV lookup table with actuator rate limiting. The
architecture is intended to support reaction wheels and other control
surfaces alongside or instead of canards.

## Current Status

- **Roll dynamics** are fully modeled: torque about the roll axis is
  integrated into angular velocity and orientation via quaternion
  kinematics.
- **Yaw and pitch** are computed using the same (decoupled) equations as
  roll, but without gyroscopic cross-coupling (`omega x I*omega`), so they
  aren't yet physically complete for non-roll maneuvers. There is currently
  no guard against non-zero yaw/pitch torque — it will run, just without
  the missing coupling term.
- **Position tracking** (X/Y/Z) assumes vertical flight with no lateral
  forces — the only environment input driving the physics is vertical
  velocity (`zVel_mps`) and air density from the sim data CSV. Lateral
  position targets/errors exist in the API but aren't exercised by the
  current physics model.

See `TODO.md` for a running list of known issues and their status.

## Project Layout

- `Rocket/Rocket.py` — the simulation engine and state container. Owns
  position, velocity, orientation, and angular rates; advances them via
  `sim()`; loads environment/flight data from a CSV.
- `Controls/Controls.py` — abstract base class all control implementations
  subclass. Defines the `sim(rocket, **kwargs)` contract every control must
  implement.
- `Controls/Canards.py` — canard model: aerodynamic lift from an airfoil CSV
  lookup, rate-limited and clamped actuator angle, roll-only torque output.
- `Controls/ReactionWheel.py` — placeholder for a future reaction wheel
  implementation.
- `example/CRL2026/` — reference example scenario:
  - `CRL2026_Canards.py` — builds a `Rocket` + `Canards`, drives a scripted
    roll maneuver with a PID controller, and plots the result. Start here
    when building a new scenario or control mechanism.
  - `Canard_Rocket.csv` — time-indexed flight/environment data (`zVel_mps`,
    `airDensity`) used to drive the example.
  - `airfoil/` — airfoil lift-coefficient (C_L) tables used by `Canards`.

## Installation

Recommended in a virtual environment:

```bash
pip install -r requirements.txt
python -m pip install --user -e . --break-system-packages
```

## Running the Example

```bash
cd example/CRL2026/
python CRL2026_Canards.py
```

This runs the scripted roll maneuver described in `CRL2026_Canards.py`
(hold 0°, ramp to 90°, hold, ramp back down) and plots rocket angle vs.
target vs. error, canard deflection, and airspeed/air density over time.

## How the Simulation Works

- A `Rocket` stores state such as position, velocity, orientation, and
  angular rates, and loads environment data (velocity, air density) from a
  CSV keyed by simulation time.
- A `targetFunc(time_s) -> (posX, posY, posZ, yaw, pitch, roll)` describes
  the desired trajectory; the rocket evaluates it every step and computes
  the error against current state.
- Each simulation step, `rocket.sim(**kwargs)` polls every attached control
  object via `control.sim(rocket=self, **kwargs)`. Each control returns a
  torque tuple `(yawTorque_Nm, pitchTorque_Nm, rollTorque_Nm)`.
- The rocket sums torques across all attached controls, integrates angular
  velocity and orientation (via quaternion kinematics), and updates
  position/attitude error terms against the target state.
- `**kwargs` passed into `rocket.sim(...)` are forwarded unchanged to every
  control's `sim()` — e.g. a `Canards` control requires a `canardAngle_deg`
  keyword each step (see the PID loop in `CRL2026_Canards.py` for a working
  example).

## Extending the Project

- **Add a new control mechanism** by subclassing `Controls` and overriding
  `sim(self, rocket, **kwargs)` to return a torque tuple. See
  `Controls/Controls.py` for the full subclassing contract, and
  `Controls/Canards.py` as a worked example.
- **Combine multiple controls** by passing several objects into the
  rocket's `controls` list — their torques are summed automatically.
- **Build a new example scenario** by copying the pattern in
  `CRL2026_Canards.py`: build your control(s), build a `Rocket` with a
  `targetFunc`, call `rocket.reset()`, then loop `while rocket.running:`
  calling `rocket.sim(**your_inputs)` each step.
- **Improve the aerodynamic model** by replacing the CSV lookup in
  `Canards` with a higher-fidelity physics model.
- **Add sensors, latency, or actuator dynamics** for more realistic control
  loops.
- **Implement pitch and yaw dynamics**, including gyroscopic coupling, to
  move beyond the current roll-only physics.
- **Add batch runs, parameter sweeps, or plotting utilities** for
  controller tuning.
