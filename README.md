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

Two control mechanisms are currently implemented:
- A canard-based roll controller (`Controls/Canards.py`), driven by
  aerodynamic lift estimated from an airfoil CSV lookup table with
  actuator rate limiting.
- A reaction-wheel roll controller (`Controls/ReactionWheel.py`), driven by
  conservation of angular momentum as an internal flywheel is spun up and
  down toward a commanded speed, with motor acceleration limiting.

The architecture is intended to support additional control surfaces
alongside or instead of these.

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
- **Reaction wheel PID gains** in the example (`ReactionWheelRocket.py`)
  are negative, unlike the canard example's positive gains — this is
  expected, since a reaction wheel's reaction torque opposes its own
  angular acceleration (see `Controls/ReactionWheel.py`). If you rework the
  wheel's inertia, direction convention, or timestep, re-tune these gains
  rather than assuming they still produce a stable response.

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
- `Controls/ReactionWheel.py` — reaction wheel model: an internal flywheel
  spun toward a commanded speed with motor-acceleration limiting; roll
  torque is the equal-and-opposite reaction to the wheel's own angular
  acceleration (conservation of angular momentum), roll-only torque output.
- `example/CanardControl/` — reference example scenario:
  - `CanardRocket.py` — builds a `Rocket` + `Canards`, drives a scripted
    roll maneuver with a PID controller, and plots the result. Start here
    when building a new scenario or control mechanism.
  - `.csv` — time-indexed flight/environment data (`zVel_mps`,
    `airDensity`) used to drive the example.
  - `airfoil/` — airfoil lift-coefficient (C_L) tables used by `Canards`.
- `example/ReactionWheelControl/` — reference example scenario for the
  reaction wheel:
  - `ReactionWheelRocket.py` — builds a `Rocket` + `ReactionWheel`, drives
    the same scripted roll maneuver with a PID controller, and plots
    rocket angle vs. target vs. error alongside wheel speed vs. rocket
    angular velocity.
  - `.csv` — time-indexed flight/environment data used to drive the
    example (no airfoil table is needed, since the reaction wheel has no
    aerodynamic dependency).

## Installation

Recommended in a virtual environment:

```bash
pip install -r requirements.txt
python -m pip install --user -e . --break-system-packages
```

## Running the Examples

```bash
cd example/CanardControl/
python CanardRocket.py
```

```bash
cd example/ReactionWheelControl/
python ReactionWheelRocket.py
```

Both run the same scripted roll maneuver (hold 0°, ramp to 90°, hold, ramp
back down) using their respective control mechanism, and plot rocket angle
vs. target vs. error alongside a second view of the controlling actuator's
own state (canard deflection and airspeed/air density for `CanardRocket.py`;
wheel speed vs. rocket angular velocity for `ReactionWheelRocket.py`).

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
  keyword, and a `ReactionWheel` control requires a `wheelSpeed_deg`
  keyword, each step (see the PID loops in `CanardRocket.py` and
  `ReactionWheelRocket.py` for working examples).

## Extending the Project

- **Add a new control mechanism** by subclassing `Controls` and overriding
  `sim(self, rocket, **kwargs)` to return a torque tuple. See
  `Controls/Controls.py` for the full subclassing contract, and
  `Controls/Canards.py` or `Controls/ReactionWheel.py` as worked examples.
- **Combine multiple controls** by passing several objects into the
  rocket's `controls` list — their torques are summed automatically.
- **Build a new example scenario** by copying the pattern in
  `CanardRocket.py` or `ReactionWheelRocket.py`: build your control(s),
  build a `Rocket` with a `targetFunc`, call `rocket.reset()`, then loop
  `while rocket.running:` calling `rocket.sim(**your_inputs)` each step.
- **Improve the aerodynamic model** by replacing the CSV lookup in
  `Canards` with a higher-fidelity physics model.
- **Add sensors, latency, or actuator dynamics** for more realistic control
  loops.
- **Implement pitch and yaw dynamics**, including gyroscopic coupling, to
  move beyond the current roll-only physics.
- **Add batch runs, parameter sweeps, or plotting utilities** for
  controller tuning.