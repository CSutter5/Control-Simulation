# Control-Simulation

A lightweight, extensible simulation for rocket control experiments. A single
`Rocket` model owns the flight state and physics; pluggable `Controls`
objects (canards, reaction wheels, etc.) generate forces against that
model, making it easy to swap between control mechanisms or combine several
in one simulation.

## Overview

The design centers on a unified `Rocket` class that owns the simulation
state (position, velocity, orientation, angular rates) and advances it one
timestep at a time. `Controls` objects implement a common interface —
`sim(rocket, **kwargs) -> list[Force]` — and the rocket polls every
attached control each step, converts each returned `Force` into torque
(via `r x F` about the center of gravity) and net translational force,
sums both across all controls and all forces, and integrates its state
forward.

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
- **Yaw and pitch** use the full Euler rigid-body rotation equation,
  including gyroscopic cross-coupling (`omega x I*omega`) between all
  three axes — this is not yet exercised by any current control (both
  `Canards` and `ReactionWheel` only ever produce roll torque), but the
  underlying physics is in place for a control that does (e.g. an
  upcoming Thrust Vector Control implementation).
- **Forces are converted to torque and net translational force every
  step** via the `Force`-based control contract (see Overview above) —
  including any lateral (X/Y) force a control produces. What's NOT yet
  implemented is integrating that net force into velocity and position:
  translational state is currently "on rails," driven only by
  externally-supplied vertical velocity (`zVel_mps`) and air density from
  the sim data CSV, rather than by the computed forces. Lateral position
  targets/errors exist in the API but aren't exercised by the current
  physics model as a result. See `TODO.md`.
- **Reaction wheel PID gains** in the example (`ReactionWheelRocket.py`)
  are negative, unlike the canard example's positive gains — this is
  expected, since a reaction wheel's reaction torque opposes its own
  angular acceleration (see `Controls/ReactionWheel.py`). If you rework the
  wheel's inertia, direction convention, or timestep, re-tune these gains
  rather than assuming they still produce a stable response.

## Project Layout

- `Rocket/Rocket.py` — the simulation engine and state container. Owns
  position, velocity, orientation, and angular rates; advances them via
  `sim()`; loads environment/flight data from a CSV.
- `Controls/Controls.py` — abstract base class all control implementations
  subclass. Defines the `sim(rocket, **kwargs) -> list[Force]` contract
  every control must implement.
- `Controls/Canards.py` — canard model: aerodynamic lift from an airfoil CSV
  lookup, rate-limited and clamped actuator angle, returns roll-only net
  torque as a symmetric set of per-fin forces.
- `Controls/ReactionWheel.py` — reaction wheel model: an internal flywheel
  spun toward a commanded speed with motor-acceleration limiting; roll
  torque is the equal-and-opposite reaction to the wheel's own angular
  acceleration (conservation of angular momentum), represented as a
  two-force internal couple (see `Controls/Force.py`).
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
  `list[Force]` — one `Force` (a vector plus its application point in the
  body frame) per distinct force it produces this step.
- The rocket converts each returned `Force` into torque about the center
  of gravity (`r x F`) and its own translational contribution, sums both
  across every force from every control, integrates angular velocity and
  orientation (via the full Euler rigid-body equation and quaternion
  kinematics), converts the net translational force to acceleration, and
  updates position/attitude error terms against the target state. (Net
  force -> acceleration is computed every step but not yet integrated into
  velocity/position — see Current Status above.)
- `**kwargs` passed into `rocket.sim(...)` are forwarded unchanged to every
  control's `sim()` — e.g. a `Canards` control requires a `canardAngle_deg`
  keyword, and a `ReactionWheel` control requires a `wheelSpeed_deg`
  keyword, each step (see the PID loops in `CanardRocket.py` and
  `ReactionWheelRocket.py` for working examples).

## Extending the Project

- **Add a new control mechanism** by subclassing `Controls` and overriding
  `sim(self, rocket, **kwargs)` to return a `list[Force]`. See
  `Controls/Controls.py` for the full subclassing contract, and
  `Controls/Canards.py` or `Controls/ReactionWheel.py` as worked examples.
- **Combine multiple controls** by passing several objects into the
  rocket's `controls` list — each control's forces are converted to torque
  and summed automatically.
- **Build a new example scenario** by copying the pattern in
  `CanardRocket.py` or `ReactionWheelRocket.py`: build your control(s),
  build a `Rocket` with a `targetFunc`, call `rocket.reset()`, then loop
  `while rocket.running:` calling `rocket.sim(**your_inputs)` each step.
- **Improve the aerodynamic model** by replacing the CSV lookup in
  `Canards` with a higher-fidelity physics model.
- **Add sensors, latency, or actuator dynamics** for more realistic control
  loops.
- **Add a control that produces non-zero yaw/pitch torque** (e.g. Thrust
  Vector Control, currently the next planned milestone — see `TODO.md`) to
  exercise the coupled yaw/pitch/roll dynamics, which are implemented but
  currently untested against a real non-roll maneuver since no existing
  control produces one.
- **Add batch runs, parameter sweeps, or plotting utilities** for
  controller tuning.