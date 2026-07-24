import math

import matplotlib.pyplot as plt
import pandas as pd

import Controls
import Rocket

turnStart_s = 7
turnEnd_s   = 9
turnLerp_s  = 0.5

startingTargetAngle_deg = 0
turnAngle_deg = 90

def target(time_s: float) -> tuple[float, float, float, float, float, float]:
    """ Target Function

    Args:
        time_s (float): Simulation Time Step

    Returns:
        tuple[float, float, float, float, float, float]: The target location and orientation [posX_m, posY_m, posZ_m, yaw_deg, pitch_deg, roll_deg]
    """

    def lerp(a: float, b: float, t: float) -> float:
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

TIMESTEP = 0.01


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

rocket = Rocket.Rocket(
    simDataPath="Canard Rocket.csv", # Requiremets are dependent on the control methon used
    # simDataPath="Test Flight Data.csv",
    Ix_kgm2=0.01, # Mass Moment of Inerta (MMOI) around the x-axis
    Iy_kgm2=0.01, # Mass Moment of Inerta (MMOI) around the y-axis
    Iz_kgm2=0.01, # Mass Moment of Inerta (MMOI) around the z-axis
    r_m=0.05,
    length_m=0,
    mass_kg=0,
    targetFunc=target,
    simTimeStep=TIMESTEP,
    controls=[
        canards
    ]
)

# Plots
def plot(df: pd.DataFrame):
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
    rocket.reset()

    df = pd.DataFrame(columns=[
        'time', 'targetAngle', 'rocketAngle', 'rocketAngularVelocity',
        'canardAngle', 'error', 'airSpeed', 'airDensity'
    ])

    df.loc[0] = {
        'time':                  rocket.simTime,
        'targetAngle':           rocket.targetRoll_deg,
        'rocketAngle':           rocket.roll_deg,
        'rocketAngularVelocity': rocket.rollVel_dps/6,
        'canardAngle':           canards.angle_rad,
        'error':                 rocket.rollError_deg,
        'airSpeed':              rocket.zVel_mps,
        'airDensity':            rocket.airDensity
    }

    Kp = 5.7
    Ki = 0.0
    Kd = 2.8

    p = 0.0
    i = 0.0
    d = 0.0

    error = 0.0
    lastError = 0.0

    while rocket.running:
        error = math.degrees(rocket.rollError_rad)

        p  = error
        i += error * TIMESTEP
        d  = (error - lastError) / TIMESTEP

        pid = \
            Kp * p + \
            Ki * i + \
            Kd * d
        
        lastError = error

        rocket.sim(
            canardAngle_deg=pid
        )

        df.loc[len(df)] = {
            'time':                  rocket.simTime,
            'targetAngle':           rocket.targetRoll_deg,
            'rocketAngle':           rocket.roll_deg,
            'rocketAngularVelocity': rocket.rollVel_dps/6,
            'canardAngle':           canards.angle_deg,
            'error':                 rocket.rollError_deg,
            'airSpeed':              rocket.zVel_mps,
            'airDensity':            rocket.airDensity
        }

    plot(df)