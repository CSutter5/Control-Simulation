"""
CRL2026 Canard Roll-Control Demo
================================

This is the reference example for using this project: it builds a single
`Rocket` with one `Canards` control attached, drives the canards with a
simple PID controller trying to hit a scripted roll-angle target, and plots
the result. If you're adding a new scenario or a new control mechanism,
this file is the intended starting point to copy and modify.

Run from the `example/CRL2026/` directory (paths below are relative to it):

    cd example/CRL2026/
    python CRL2026_Canards.py

What this script does, in order:
    1. Defines `target(time_s)` — the desired trajectory as a function of
       time. Here it's a scripted roll maneuver: hold 0 deg, ramp up to
       `turnAngle_deg`, hold, then ramp back down. Position and yaw/pitch
       targets are always 0 since this project currently only implements
       roll dynamics (see project README).
    2. Constructs a `Canards` control object with physical fin geometry and
       actuator limits.
    3. Constructs a `Rocket` wired up with that canard control, the target
       function, a fixed timestep, and a CSV of environment/flight data.
    4. Runs a simple PID loop: each step, compute roll error against the
       target, feed a PID output into `rocket.sim(canardAngle_deg=...)` as
       the commanded canard angle, and log the resulting state.
    5. Plots rocket angle vs. target vs. error, canard deflection, and
       airspeed/air density over time.

If you're building a new example, the pattern to follow is:
    - Build your Controls object(s) first.
    - Build a Rocket, passing your control(s) in the `controls` list and a
      `targetFunc` describing what you want the rocket to do.
    - Call `rocket.reset()` once before the loop.
    - Loop `while rocket.running:`, computing whatever control input your
      control(s) need this step, then call `rocket.sim(**those_inputs)`.
    - Read back state via `rocket.<property>` after each `sim()` call for
      logging/plotting.
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
# One Canards instance modeling a pair of fins. Geometry values (root_m,
# tip_m, span_m, sweep_m, offset_m) describe the physical fin shape and
# placement; maxAngle_deg/rateLimit_dps describe actuator limits (how far
# and how fast the canards can physically deflect). See Controls/Canards.py
# for how these are used.
canards = Controls.Canards(
    airfoilDataPath="airfoil/0012_airfoil_data.csv", # Table of Cl for different velocities and aoa's
    root_m=0.0635,
    tip_m=0.01905,
    span_m=0.0254,
    sweep_m=0.04445,
    numCanards=2,
    offset_m=0.05,
    maxAngle_deg=10,
    rateLimit_dps=120,
    updateFreq_hz=1/50 # 50 hz
)

# --- Rocket ---
# Wires the canard control above into a Rocket along with the scripted
# target function and a CSV of environment/flight data.
rocket = Rocket.Rocket(
    simDataPath="Canard_Rocket.csv", # Requiremets are dependent on the control methon used
    # simDataPath="Test_Flight_Data.csv",
    Ix_kgm2=0.01, # Mass Moment of Inerta (MMOI) around the x-axis
    Iy_kgm2=0.01, # Mass Moment of Inerta (MMOI) around the y-axis
    Iz_kgm2=0.01, # Mass Moment of Inerta (MMOI) around the z-axis
    r_m=0.05,
    length_m=0,   # not used by the current (roll-only) physics model
    mass_kg=0,    # not used by the current (roll-only) physics model
    targetFunc=target,
    simTimeStep=TIMESTEP,
    controls=[
        canards
    ]
)

# --- Plotting ---
def plot(df: pd.DataFrame):
    """
    Plot the logged simulation run.

    Expects `df` to have been built the same way as in the `__main__` block
    below, i.e. with columns 'time', 'targetAngle', 'rocketAngle', 'error',
    'canardAngle', 'airSpeed', and 'airDensity' (all in the units named,
    degrees for angles). Produces three stacked subplots:
        1. Rocket roll angle vs. target roll angle vs. error, in degrees.
        2. Canard deflection angle, in degrees.
        3. Airspeed (m/s) with air density (kg/m^3) on a secondary y-axis.

    Note: `df` also carries a 'rocketAngularVelocity' column (logged in the
    main loop below) that isn't plotted here — it's collected but currently
    unused, in case you want to add a fourth subplot for it.

    Args:
        df (pd.DataFrame): The logged run, one row per simulation step.

    Returns:
        None. Calls `plt.show()` to display the figure.
    """
    fig, axs = plt.subplots(3, 1, figsize=(14, 5), sharey=False)

    # --- Plot 1: Rocket angle + Target angle ---
    axs[0].plot(df['time'], df['rocketAngle'], color='black', label='Rocket Angle')
    axs[0].plot(df['time'], df['targetAngle'], color='blue',  label='Target Angle')
    axs[0].plot(df['time'], df['error'],       color='red',   label='Error')
    axs[0].set_xlabel('Time (s)')
    axs[0].set_ylabel('Angle (deg)')
    axs[0].legend(loc='lower left')
    axs[0].grid(True)

    # --- Plot 2: Canard Deflection Angle ---
    axs[1].plot(df['time'], df['canardAngle'], color='red', label='Canard Deflection Angle')
    axs[1].set_xlabel('Time (s)')
    axs[1].set_ylabel('Canard Angle (deg)')
    axs[1].legend(loc='lower left')
    axs[1].grid(True)

    # --- Plot 3: Airspeed ---
    axs[2].plot(df['time'], df['airSpeed'], color='red', label='Airspeed')
    axs[2].set_xlabel('Time (s)')
    axs[2].set_ylabel('Airspeed (m/s)')
    axs[2].legend(loc='upper right')
    axs[2].grid(True)

    # Add Air Pressure to the airspeed plot
    ax3_2 = axs[2].twinx()
    ax3_2.plot(df['time'], df['airDensity'], color='blue', label='Air Density')
    ax3_2.set_ylabel('Air Density (kg/m^3)')
    ax3_2.legend(loc='lower left')

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
        'canardAngle', 'error', 'airSpeed', 'airDensity'
    ])

    # Log the initial (t=0) state before the loop starts, so the plots
    # include the starting point.
    df.loc[0] = {
        'time':                  rocket.simTime,
        'targetAngle':           rocket.targetRoll_deg,
        'rocketAngle':           rocket.roll_deg,
        'rocketAngularVelocity': rocket.rollVel_dps,
        'canardAngle':           canards.angle_deg,
        'error':                 rocket.rollError_deg,
        'airSpeed':              rocket.zVel_mps,
        'airDensity':            rocket.airDensity
    }

    # --- PID gains for the roll controller ---
    # Tuned by hand for this specific rocket/canard geometry and timestep;
    # re-tune if you change TIMESTEP, the canard geometry, or the rocket's
    # moments of inertia.
    Kp = 5.7
    Ki = 0.0   # currently unused (0 gain) — integral term has no effect
    Kd = 2.8

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
        # straight through as the commanded canard deflection angle in
        # degrees — `rocket.sim(**kwargs)` forwards `canardAngle_deg` to
        # every attached control's `sim()`, which for `Canards` is a
        # required keyword argument (see Controls/Canards.py).
        rocket.sim(
            canardAngle_deg=pid
        )

        df.loc[len(df)] = {
            'time':                  rocket.simTime,
            'targetAngle':           rocket.targetRoll_deg,
            'rocketAngle':           rocket.roll_deg,
            'rocketAngularVelocity': rocket.rollVel_dps,
            'canardAngle':           canards.angle_deg,
            'error':                 rocket.rollError_deg,
            'airSpeed':              rocket.zVel_mps,
            'airDensity':            rocket.airDensity
        }

    plot(df)