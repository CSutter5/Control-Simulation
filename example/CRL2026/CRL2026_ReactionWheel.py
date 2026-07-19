import matplotlib.pyplot as plt
import pandas as pd

import math

from environments.ReactionWheelSim import ReactionWheelSim

from properties.ReactionWheel import ReactionWheel
from properties.Data import Data
from properties.Rocket import Rocket

# Define all rocket properties

MOTOR_BURNOUT = 1.7 # seconds

reactionWheel = ReactionWheel(
    mmoi_kgm2=0.026,
    maxSpeedChange_rps2=math.radians(100),
)
reactionWheel.speed_rps = math.radians(400 * 6) # rad/s (deg/sec * c = RPM)

rocket = Rocket(
    rollMMOI_kgm2=0.01004
)

# Define all flight properties

data = Data(
    dataPath="Canard Rocket.csv",
    # dataPath="Test Flight Data.csv",
    startingTemperature_c=6,
    startingPressure_pa=102998,
    humidity=0.9
)

STARTING_TARGET_ANGLE = 0
TURN_START  = 7   # seconds
TURN_END    = 9   # seconds
TURN_ANGLE  = math.radians(90)  # radians
TURN_LERP   = 0.5  # seconds

def targetFunction(time) -> float:
    def lerp(a: float, b: float, t: float) -> float:
        return (1 - t) * a + t * b
    
    # If we are at the start of the turn
    if time > TURN_START and time < TURN_START + TURN_LERP:
        return lerp(STARTING_TARGET_ANGLE, TURN_ANGLE, (time - TURN_START) / TURN_LERP)

    # If we are at the end of the turn
    if time > TURN_END and time < TURN_END + TURN_LERP:
        return lerp(TURN_ANGLE, STARTING_TARGET_ANGLE, (time - TURN_END) / TURN_LERP)

    # If we are holding the turn
    if time >= TURN_START + TURN_LERP and time <= TURN_END:
        return TURN_ANGLE

    return STARTING_TARGET_ANGLE

# Define all simulation properties

TIMESTEP = 0.01

sim = ReactionWheelSim(
    data=data,
    rocket=rocket,
    reactionWheel=reactionWheel,
    targetFunction=targetFunction,
    simulationEnd_s=15,
    timestep_s=TIMESTEP
)

# Plots
def plot(df: pd.DataFrame):
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
    ax2.plot(df['time'], df['wheelAngularVelocity'],  color='red',   label='Wheel Angular Velocity')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Angular Velocity (RPM)')
    ax2.legend(loc='upper right')
    ax2.grid(True)


    ax2_2 = ax2.twinx()
    ax2_2.plot(df['time'], df['rocketAngularVelocity'], color='green', label='Rocket Angular Velocity')
    ax2_2.set_ylabel('Angular Velocity (RPM)')
    ax2_2.legend(loc='lower left')

    # --- Mark angle changes with vertical lines == 1 ---
    eventTimes = df.loc[df['eventMarker'] == 1, 'time']
    for t in eventTimes:
        ax1.axvline(x=t, color='purple', linestyle='--', alpha=0.7)
        ax2.axvline(x=t, color='purple', linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    sim.reset()

    df = pd.DataFrame(columns=[
        'time', 'targetAngle', 'rocketAngle', 'rocketAngularVelocity',
        'wheelAngularVelocity', 'eventMarker', 'error', 'airSpeed', 'airDensity'
    ])

    df.loc[0] = {
        'time':                  0.0,
        'targetAngle':           math.degrees(targetFunction(0.0)),
        'rocketAngle':           math.degrees(rocket.rollAngle_rad),
        'rocketAngularVelocity': math.degrees(rocket.verticalVelocity_mps)/6,
        'wheelAngularVelocity':  math.degrees(reactionWheel.speed_rps),
        'eventMarker':           0,
        'error':                 math.degrees(targetFunction(0.0) - rocket.rollAngle_rad),
    }

    motorBurnedOut = 0
    
    # PID Values
    Kp = -5.0
    Ki = -0.0
    Kd = -50.0
    KiLimit = 0.5

    PID             = 0.0
    error           = 0.0
    lastError       = 0.0
    integralError   = 0.0
    derivativeError = 0.0

    while sim.running:
        integralError  += error * TIMESTEP                  # Integral
        derivativeError = (error - lastError) / TIMESTEP    # Derivative

        integralError = max(min(integralError, KiLimit), -KiLimit) # Anti-windup for integral term

        PID = \
            Kp * error + \
            Ki * integralError + \
            Kd * derivativeError   # The PID values is the torque applied to the rocket from the canard

        lastError = error
        error = sim.step(reactionWheel.speed_rps + PID * TIMESTEP)

        if (not sim.time_s < MOTOR_BURNOUT):
            motorBurnedOut += 1

        eventMarker = 0
        if sim.time_s == TURN_START or sim.time_s == TURN_END:
            eventMarker = 1
        if motorBurnedOut == 1:
            eventMarker = 1

        df.loc[len(df)] = {
            'time':                  sim.time_s,
            'targetAngle':           math.degrees(sim.targetRollAngle_rad),
            'rocketAngle':           math.degrees(rocket.rollAngle_rad),
            'rocketAngularVelocity': math.degrees(rocket.verticalVelocity_mps),
            'wheelAngularVelocity':  math.degrees(reactionWheel.speed_rps),
            'eventMarker':           eventMarker,
            'error':                 math.degrees(sim.targetRollAngle_rad - rocket.rollAngle_rad),
            'airSpeed':              data.getAirSpeed(sim.time_s),
            'airDensity':            data.getAirDensity(sim.time_s)
        }
    
    plot(df)


