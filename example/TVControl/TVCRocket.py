"""
TVC Pitch/Yaw Sine-Wave Demo
================================

Reference example for the TVC control mechanism: builds a single `Rocket`
with one `TVC` control attached, drives the gimbal with two independent
PID controllers tracking a +/-5 degree sine wave target on both pitch and
yaw simultaneously, and plots the result. Unlike `CanardRocket.py` and
`ReactionWheelRocket.py` (which only ever exercise roll), this is the
first example that produces real pitch/yaw torque -- it's the first real
test of the gyroscopic cross-coupling terms in `Rocket._applyTorques`.

Run from the `example/TVCControl/` directory (paths below are relative to
it):

    cd example/TVCControl/
    python TVCRocket.py

What this script does, in order:
    1. Defines `target(time_s)` -- a +/-`AMPLITUDE_DEG` sine wave on BOTH
       pitch and yaw (same frequency, same phase -- see the module-level
       constants below to change either independently). Position and roll
       targets are always 0.
    2. Constructs a `TVC` control object using the F15 motor's thrust
       curve and estimated 3" cardboard/PLA airframe physical properties
       (see project notes -- these are engineering estimates, not
       measured values).
    3. Constructs a `Rocket` wired up with that TVC control, the target
       function, a fixed timestep, and a minimal environment CSV (TVC has
       no aerodynamic dependency, so unlike `CanardRocket.py` this CSV
       carries no `zVel_mps`/`airDensity` data).
    4. Runs two independent PID loops (one for pitch, one for yaw) each
       step, feeding `gimbalPitch_deg`/`gimbalYaw_deg` into
       `rocket.sim(...)`.
    5. Plots rocket pitch/yaw vs target vs error, and gimbal angle/thrust
       vs time.

IMPORTANT -- PID gains below are UNTUNED starting guesses, not verified
against synthetic flight data yet (see project TODO/testing step). Sign
is negative, same reasoning as `ReactionWheelRocket.py`'s Kp/Kd: the
nozzle sits aft of the CG, so a positive gimbal deflection produces a
NEGATIVE pitch/yaw torque (see `Controls/TVC.py` class docstring, sign
note under the formula section). Do not treat the magnitudes as tuned.

IMPORTANT -- motor burnout: the F15 burns out at ~3.45s. `rocket.simStop`
is set to 4.0s (past burnout) deliberately, so the plots show what
happens to tracking error once TVC loses all control authority (thrust
-> 0), rather than stopping the sim right at the interesting part.
"""

import math

import matplotlib.pyplot as plt

import Controls
import Rocket

# --- Scripted sine-wave maneuver ---
# Both pitch and yaw target the same +/-AMPLITUDE_DEG sine wave at
# FREQUENCY_HZ, in phase with each other (traces a diagonal line in
# pitch-yaw space, not a cone/circle -- offset YAW_PHASE_RAD from 0 if a
# circular maneuver is wanted instead).
AMPLITUDE_DEG  = 5.0
FREQUENCY_HZ   = 1.0
YAW_PHASE_RAD  = 0.0

def target(time_s: float) -> tuple[float, float, float, float, float, float]:
    """ Target Function

    Describes the desired trajectory as a function of simulation time.
    Passed into `Rocket(targetFunc=...)`; the rocket calls this every
    `sim()` step and stores the result as the current target state (used
    to compute `rocket.pitchError_rad`/`rocket.yawError_rad`).

    Unlike the roll-only ramp-hold-ramp target used in
    `CanardRocket.py`/`ReactionWheelRocket.py`, this target continuously
    varies pitch and yaw as a +/-AMPLITUDE_DEG sine wave at FREQUENCY_HZ
    -- a basic tracking-authority test for TVC's two gimbal axes. Roll
    and position targets are always 0, since TVC in this example is a
    single centerline-mounted engine with no roll authority (see
    `Controls/TVC.py` class docstring on multi-engine roll control).

    Args:
        time_s (float): Simulation time step.

    Returns:
        tuple[float, float, float, float, float, float]: The target location and orientation [posX_m, posY_m, posZ_m, yaw_deg, pitch_deg, roll_deg]
    """
    pitchAngle_deg = AMPLITUDE_DEG * math.sin(2 * math.pi * FREQUENCY_HZ * time_s)
    yawAngle_deg   = AMPLITUDE_DEG * math.sin(2 * math.pi * FREQUENCY_HZ * time_s + YAW_PHASE_RAD)

    return (0, 0, 0, yawAngle_deg, pitchAngle_deg, 0)

# Fixed simulation timestep in seconds.
TIMESTEP = 0.01


# --- Control object ---
# Estimated physical properties for a 3" cardboard/PLA TVC test rocket
# flying an Estes F15 (29mm, 25.26N peak thrust, 49.61 N-s total impulse,
# 3.45s burn) -- see project notes for the reasoning behind each value;
# these are engineering estimates, not measured hardware values.
tvc = Controls.TVC(
    thrustCurvePath="F15_Thrustcurve.csv",
    forceLocationZ_m=-0.58,   # nozzle, near the aft tip
    maxAngle_deg=5,           # mechanical gimbal range
    rateLimit_dps=300         # servo + linkage rate limit
    # forceLocationX_m/Y_m default to 0.0 -- single centerline engine,
    # no roll authority (see Controls/TVC.py class docstring)
    # yawMaxAngle_deg/yawRateLimit_dps default to the pitch values above
)

# --- Rocket ---
rocket = Rocket.Rocket(
    simDataPath="FlightProfile.csv", # minimal -- TVC has no aero dependency
    Ix_kgm2=0.018,   # pitch MMOI estimate (rod approximation)
    Iy_kgm2=0.018,   # yaw MMOI estimate (rod approximation)
    Iz_kgm2=0.0008,  # roll MMOI estimate -- not exercised by this single-engine example
    CG_m=-0.35,      # from nose; motor mass pulls CG aft for this short/light airframe
    r_m=0.0381,      # 3" diameter / 2
    length_m=0.6,
    mass_kg=0.6,     # includes ~0.104 kg F15 motor; propellant burn (~0.06 kg) not modeled as mass loss
    targetFunc=target,
    simTimeStep=TIMESTEP,
    controls=[
        tvc
    ]
)

if __name__ == "__main__":
    rocket.reset()
    rocket.simStop = 4.0  # past F15 burnout (3.45s) -- see module docstring

    # --- PID gains: pitch axis ---
    # UNTUNED starting guesses -- see module docstring's IMPORTANT note.
    KpPitch = -0.3
    KiPitch = 0.0
    KdPitch = -0.05

    pPitch = 0.0
    iPitch = 0.0
    dPitch = 0.0

    errorPitch = 0.0
    lastErrorPitch = 0.0

    # --- PID gains: yaw axis ---
    # Same sign/magnitude reasoning as pitch (symmetric gimbal geometry).
    KpYaw = -0.3
    KiYaw = 0.0
    KdYaw = -0.05

    pYaw = 0.0
    iYaw = 0.0
    dYaw = 0.0

    errorYaw = 0.0
    lastErrorYaw = 0.0

    while rocket.running:
        # --- Pitch PID ---
        errorPitch = math.degrees(rocket.pitchError_rad)

        pPitch  = errorPitch
        iPitch += errorPitch * TIMESTEP
        dPitch  = (errorPitch - lastErrorPitch) / TIMESTEP

        pidPitch = KpPitch * pPitch + KiPitch * iPitch + KdPitch * dPitch
        lastErrorPitch = errorPitch

        # --- Yaw PID ---
        errorYaw = math.degrees(rocket.yawError_rad)

        pYaw  = errorYaw
        iYaw += errorYaw * TIMESTEP
        dYaw  = (errorYaw - lastErrorYaw) / TIMESTEP

        pidYaw = KpYaw * pYaw + KiYaw * iYaw + KdYaw * dYaw
        lastErrorYaw = errorYaw

        rocket.sim(
            gimbalPitch_deg=pidPitch,
            gimbalYaw_deg=pidYaw
        )

    fig, axis = plt.subplots(3, 2, figsize=(12, 10), sharey=False)

    # Pitch/yaw tracking, reusing the same df columns Rocket already logs
    # (pitchError_rad/yawError_rad aren't currently exposed via a
    # plotRoll-style helper -- see note below)
    rocket.df['pitch_deg']       = rocket.df.apply(lambda row: math.degrees(row.pitch_rad), axis=1)
    rocket.df['targetPitch_deg'] = rocket.df.apply(lambda row: math.degrees(row.targetPitch_rad), axis=1)
    rocket.df['pitchError_deg']  = rocket.df.apply(lambda row: math.degrees(row.pitchError_rad), axis=1)

    rocket.df['yaw_deg']         = rocket.df.apply(lambda row: math.degrees(row.yaw_rad), axis=1)
    rocket.df['targetYaw_deg']   = rocket.df.apply(lambda row: math.degrees(row.targetYaw_rad), axis=1)
    rocket.df['yawError_deg']    = rocket.df.apply(lambda row: math.degrees(row.yawError_rad), axis=1)

    axis[0][0].plot(rocket.df.index, rocket.df['pitch_deg'],       color='black', label='Rocket Pitch')
    axis[0][0].plot(rocket.df.index, rocket.df['targetPitch_deg'], color='blue',  label='Target Pitch')
    axis[0][0].plot(rocket.df.index, rocket.df['pitchError_deg'],  color='red',   label='Pitch Error')
    axis[0][0].set_title('Pitch Tracking')
    axis[0][0].legend(loc='lower left')
    axis[0][0].grid(True)

    axis[0][1].plot(rocket.df.index, rocket.df['yaw_deg'],       color='black', label='Rocket Yaw')
    axis[0][1].plot(rocket.df.index, rocket.df['targetYaw_deg'], color='blue',  label='Target Yaw')
    axis[0][1].plot(rocket.df.index, rocket.df['yawError_deg'],  color='red',   label='Yaw Error')
    axis[0][1].set_title('Yaw Tracking')
    axis[0][1].legend(loc='lower left')
    axis[0][1].grid(True)

    tvc.plot(axis[1][0], axis[2][0])
    axis[1][1].axis('off')
    axis[2][1].axis('off')

    plt.tight_layout()
    plt.show()