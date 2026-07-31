"""
Reaction Wheel Roll-Control Demo
================================

This is the reference example for the reaction-wheel control mechanism: it
builds a single `Rocket` with one `ReactionWheel` control attached, drives
the wheel with a simple PID controller trying to hit a scripted roll-angle
target, and plots the result. It follows the same pattern as
`CanardRocket.py` — see that file for a more heavily-commented walkthrough
of the overall Controls/Rocket API.

Run from the `example/ReactionWheelControl/` directory (paths below are
relative to it):

    cd example/ReactionWheelControl/
    python ReactionWheelRocket.py

What this script does, in order:
    1. Defines `target(time_s)` — the desired trajectory as a function of
       time. Here it's a scripted roll maneuver: hold 0 deg, ramp up to
       `turnAngle_deg`, hold, then ramp back down. Position and yaw/pitch
       targets are always 0 since this project currently only implements
       roll dynamics (see project README).
    2. Constructs a `ReactionWheel` control object with a wheel inertia,
       max acceleration, and starting speed.
    3. Constructs a `Rocket` wired up with that reaction-wheel control, the
       target function, a fixed timestep, and a CSV of environment/flight
       data.
    4. Runs a simple PID loop: each step, compute roll error against the
       target, feed a PID output into `rocket.sim(wheelSpeed_deg=...)` as
       the commanded wheel speed, and log the resulting state.
    5. Plots rocket angle vs. target vs. error, and wheel speed vs. rocket
       angular velocity over time.

Note on PID gains: the sign of the roll torque produced by a reaction
wheel is the opposite of the wheel's own angular acceleration (Newton's
third law / conservation of angular momentum — see `ReactionWheel.sim`),
which is why `Kp`/`Kd` below are negative rather than positive as in
`CanardRocket.py`. If you change the wheel's inertia, direction
convention, or timestep, re-check and re-tune these gains rather than
assuming they'll still produce a stable response.
"""

import math

import matplotlib.pyplot as plt
import pandas as pd

import Controls
import Rocket

# --- Scripted roll-maneuver timing ---
# The target function below holds 0 deg, ramps linearly up to
# `turnAngle_deg` between turnStart_s and turnStart_s + turnLerp_s, holds
# turnAngle_deg until turnEnd_s, then ramps back down to 0 over the next
# turnLerp_s seconds.
turnStart_s = 7
turnEnd_s   = 9
turnLerp_s  = 0.5

startingTargetAngle_deg = 0
turnAngle_deg = 90

def target(time_s: float) -> tuple[float, float, float, float, float, float]:
    """ Target Function

    Describes the desired trajectory as a function of simulation time.
    Passed into `Rocket(targetFunc=...)`; the rocket calls this every
    `sim()` step and stores the result as the current target state (used
    to compute `rocket.rollError_rad`, etc.).

    This particular target only varies roll: it's flat at
    `startingTargetAngle_deg` (0), linearly ramps to `turnAngle_deg` (90)
    between `turnStart_s` and `turnStart_s + turnLerp_s`, holds there until
    `turnEnd_s`, then linearly ramps back down to `startingTargetAngle_deg`
    over the next `turnLerp_s` seconds. Position and yaw/pitch targets are
    always 0, since only roll dynamics are implemented in this project
    currently.

    Args:
        time_s (float): Simulation Time Step

    Returns:
        tuple[float, float, float, float, float, float]: The target location and orientation [posX_m, posY_m, posZ_m, yaw_deg, pitch_deg, roll_deg]
    """

    def lerp(a: float, b: float, t: float) -> float:
        """Linear interpolation from `a` to `b` at fraction `t` (0..1)."""
        return (1 - t) * a + t * b

    rollAngle = 0

    # If we are at the start of the turn
    if time_s > turnStart_s and time_s < turnStart_s + turnLerp_s:
        rollAngle = lerp(startingTargetAngle_deg, turnAngle_deg, (time_s - turnStart_s) / turnLerp_s)

    # If we are at the end of the turn
    if time_s > turnEnd_s and time_s < turnEnd_s + turnLerp_s:
        rollAngle = lerp(turnAngle_deg, startingTargetAngle_deg, (time_s - turnEnd_s) / turnLerp_s)

    # If we are holding the turn
    if time_s >= turnStart_s + turnLerp_s and time_s <= turnEnd_s:
        rollAngle = turnAngle_deg

    
    return (0, 0, 0, 0, 0, rollAngle)

# Fixed simulation timestep in seconds. Passed to Rocket as `simTimeStep`
# and used directly in the PID loop below for the integral/derivative
# terms — if you change this, both the physics step size and the PID
# gains' effective behavior change together.
TIMESTEP = 0.01


# --- Control object ---
# One ReactionWheel instance modeling a single flywheel on the roll axis.
# I_kgm2 is the wheel's own moment of inertia (not the rocket's);
# maxAcceleration_dps2/startingSpeed_dps describe motor limits and initial
# spin-up state. See Controls/ReactionWheel.py for how these are used.
wheel = Controls.ReactionWheel(
    I_kgm2=0.026,
    maxAcceleration_dps2=100,
    startingSpeed_dps=400
)

# --- Rocket ---
# Wires the reaction-wheel control above into a Rocket along with the
# scripted target function and a CSV of environment/flight data.
rocket = Rocket.Rocket(
    simDataPath="FlightProfile.csv", # Requiremets are dependent on the control methon used
    Ix_kgm2=0.01, # Mass Moment of Inerta (MMOI) around the x-axis
    Iy_kgm2=0.01, # Mass Moment of Inerta (MMOI) around the y-axis
    Iz_kgm2=0.01, # Mass Moment of Inerta (MMOI) around the z-axis
    r_m=0.05,
    length_m=0,   # not used by the current (roll-only) physics model
    mass_kg=0,    # not used by the current (roll-only) physics model
    targetFunc=target,
    simTimeStep=TIMESTEP,
    controls=[
        wheel
    ]
)

# --- Plotting ---
def plot(df: pd.DataFrame):
    """
    Plot the logged simulation run.

    Expects `df` to have been built the same way as in the `__main__`
    block below, i.e. with columns 'time', 'targetAngle', 'rocketAngle',
    'rocketAngularVelocity', 'wheelSpeed', and 'error' (angles in degrees,
    angular velocities in degrees/sec). Produces two side-by-side subplots:
        1. Rocket roll angle vs. target roll angle vs. error, in degrees.
        2. Reaction wheel speed (left axis) vs. rocket roll angular
           velocity (right axis), both in degrees/sec.

    Args:
        df (pd.DataFrame): The logged run, one row per simulation step.

    Returns:
        None. Calls `plt.show()` to display the figure.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), sharey=False)

    # --- Plot 1: Rocket angle + Target angle ---
    ax1.plot(df['time'], df['rocketAngle'], color='black', label='Rocket Angle')
    ax1.plot(df['time'], df['targetAngle'], color='blue',  label='Target Angle')
    ax1.plot(df['time'], df['error'],       color='red',   label='Error')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Angle (deg)')
    ax1.legend(loc='lower left')
    ax1.grid(True)

    # --- Plot 2: Wheel & Rocket Angular Velocity ---
    ax2.plot(df['time'], df['wheelSpeed'],  color='red',   label='Wheel Speed')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Angular Velocity (Degree/Second)')
    ax2.legend(loc='upper right')
    ax2.grid(True)

    ax2_2 = ax2.twinx()
    ax2_2.plot(df['time'], df['rocketAngularVelocity'], color='green', label='Rocket Angular Velocity')
    ax2_2.set_ylabel('Angular Velocity (Degree/Second)')
    ax2_2.legend(loc='lower left')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Reset the rocket to t=0 before starting the run. Since `rocket` is a
    # module-level object constructed once above, calling reset() here
    # (rather than constructing a fresh Rocket) is what makes this script
    # safely re-runnable / importable without side effects from
    # construction alone.
    rocket.reset()

    df = pd.DataFrame(columns=[
        'time', 'targetAngle', 'rocketAngle', 'rocketAngularVelocity',
        'wheelSpeed', 'error'
    ])

    # Log the initial (t=0) state before the loop starts, so the plots
    # include the starting point.
    df.loc[0] = {
        'time':                  rocket.simTime,
        'targetAngle':           rocket.targetRoll_deg,
        'rocketAngle':           rocket.roll_deg,
        'rocketAngularVelocity': rocket.rollVel_dps,
        'wheelSpeed':           wheel.speed_dps,
        'error':                 rocket.rollError_deg
    }

    # --- PID gains for the roll controller ---
    # Negative here (unlike Canards' positive gains) because a reaction
    # wheel's reaction torque opposes its own angular acceleration — see
    # the module docstring above and Controls/ReactionWheel.py. Tuned by
    # hand for this specific wheel/rocket inertia and timestep; re-tune if
    # you change TIMESTEP, I_kgm2, or the rocket's moments of inertia.
    Kp = -200.0
    Ki = 0.0   # currently unused (0 gain) — integral term has no effect
    Kd = -100.0

    p = 0.0
    i = 0.0
    d = 0.0

    error = 0.0
    lastError = 0.0

    # Main simulation loop: runs until `rocket.simTime` reaches
    # `rocket.simStop` (set inside Rocket, default 15s), at which point
    # `rocket.running` becomes False.
    while rocket.running:
        # Roll error in degrees, target minus actual (see Rocket._eulerFromQuat
        # / Rocket.sim for how rollError_rad is derived each step).
        error = math.degrees(rocket.rollError_rad)

        # Standard discrete PID: proportional on the current error,
        # integral as a running sum scaled by TIMESTEP, derivative as a
        # finite difference against the previous step's error scaled by
        # TIMESTEP.
        p  = error
        i += error * TIMESTEP
        d  = (error - lastError) / TIMESTEP

        pid = \
            Kp * p + \
            Ki * i + \
            Kd * d
        
        lastError = error

        # Advance the simulation by one TIMESTEP. The PID output is passed
        # straight through as the commanded wheel speed in degrees/sec —
        # `rocket.sim(**kwargs)` forwards `wheelSpeed_deg` to every
        # attached control's `sim()`, which for `ReactionWheel` is a
        # required keyword argument (see Controls/ReactionWheel.py).
        rocket.sim(
            wheelSpeed_deg=pid
        )

        df.loc[len(df)] = {
            'time':                  rocket.simTime,
            'targetAngle':           rocket.targetRoll_deg,
            'rocketAngle':           rocket.roll_deg,
            'rocketAngularVelocity': rocket.rollVel_dps,
            'wheelSpeed':           wheel.speed_dps,
            'error':                 rocket.rollError_deg
        }

    plot(df)