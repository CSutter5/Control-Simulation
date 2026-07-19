import matplotlib.pyplot as plt
import pandas as pd

import math

from environments.CanardSim import CanardSim

from properties.Canard import Canard
from properties.Data import Data
from properties.Rocket import Rocket

# Define all rocket properties

MOTOR_BURNOUT = 1.7 # seconds

canards = Canard(
    airfoilDataPath="airfoil/0012_airfoil_data.csv",
    rootCoord_m=0.0635,
    tipCoord_m=0.01905,
    span_m=0.0254,
    sweep_m=0.04445,
    canardDistance_m=0.05,
    canardAngleRateLimit_rps=math.radians(120.0),
    minCanardAngle_rad=math.radians(-10.0),
    maxCanardAngle_rad=math.radians(10.0)
)

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

sim = CanardSim(
    data=data,
    rocket=rocket,
    canards=canards,
    numCanards=2,
    targetFunction=targetFunction,
    simulationEnd_s=15,
    timestep_s=TIMESTEP
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

    # --- Mark angle changes with vertical lines == 1 ---
    eventTimes = df.loc[df['eventMarker'] == 1, 'time']
    for t in eventTimes:
        axs[0].axvline(x=t, color='purple', linestyle='--', alpha=0.7)
        axs[1].axvline(x=t, color='purple', linestyle='--', alpha=0.7)
        axs[2].axvline(x=t, color='purple', linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    sim.reset()

    df = pd.DataFrame(columns=[
        'time', 'targetAngle', 'rocketAngle', 'rocketAngularVelocity',
        'canardAngle', 'eventMarker', 'error', 'airSpeed', 'airDensity'
    ])

    df.loc[0] = {
        'time':                  0.0,
        'targetAngle':           math.degrees(targetFunction(0.0)),
        'rocketAngle':           math.degrees(rocket.rollAngle_rad),
        'rocketAngularVelocity': math.degrees(rocket.verticalVelocity_mps)/6,
        'canardAngle':           math.degrees(canards.canardAngle_rad),
        'eventMarker':           0,
        'error':                 math.degrees(targetFunction(0.0) - rocket.rollAngle_rad),
        'airSpeed':              data.getAirSpeed(0.0),
        'airDensity':            data.getAirDensity(0.0)
    }

    motorBurnedOut = 0
    
    # PID Values
    Kp = 0.1
    Ki = 0.0
    Kd = 0.05
    KiLimit = 0.5

    canardPID       = 0.0
    error           = 0.0
    lastError       = 0.0
    integralError   = 0.0
    derivativeError = 0.0

    while sim.running:
        integralError  += error * TIMESTEP                  # Integral
        derivativeError = (error - lastError) / TIMESTEP    # Derivative

        integralError = max(min(integralError, KiLimit), -KiLimit) # Anti-windup for integral term

        canardPID = \
            Kp * error + \
            Ki * integralError + \
            Kd * derivativeError   # The PID values is the torque applied to the rocket from the canard

        lastError = error
        error = sim.step(canardPID)

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
            'canardAngle':           math.degrees(canards.canardAngle_rad),
            'eventMarker':           eventMarker,
            'error':                 math.degrees(sim.targetRollAngle_rad - rocket.rollAngle_rad),
            'airSpeed':              data.getAirSpeed(sim.time_s),
            'airDensity':            data.getAirDensity(sim.time_s)
        }
    
    plot(df)


