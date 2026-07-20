from typing import Callable

import random

from properties.Canard import Canard
from properties.Data import Data
from properties.Rocket import Rocket

class CanardSim:
    """
    A Simulation Environment for simulation Canard Roll Control on a Rocket

    Attributes:
        data (Data): Flight Data
        rocket (Rocket): Physical Rocket Properties
        canards (Canard): Physical Canard Properties
        numCanards (int): Number of canards on the rocket
        targetFunction (Callable[[float], float]): The function that takes sim time as an argument and returns the target angle
        targetRollAngle_rad (float): The current roll angle target in radians
        time_s (float): The current simulation time in seconds
        timestep_s (float): The simulation time step size in seconds
        running (boolean): Is the simulation still running
        simulationEnd_s (float): The time that the simulation will end at
        error_rad (float): Error between current angle and target angle in rad
        angularNoise (float): STD Dev of noise added to the rocket angle added at each timestep
    """

    def __init__(
            self, data: Data, 
            rocket: Rocket, canards: Canard, numCanards: int, 
            targetFunction: Callable[[float], float],
            simulationEnd_s: float, timestep_s: float = 0.01,
            angleNoise: float = 0.05,
        ):

        """
        Initialize the simulation environment
        
        Parameters:
            data (Data): Flight Data
            rocket (Rocket): Physical Rocket Properties
            canards (Canard): Physical Canard Properties
            numCanards (int): Number of canards on the rocket
            targetFunction (Callable[[float], float]): The function that takes sim time as an argument and returns the target angle
            simulationEnd_s (float): The time that the simulation will end at
            timestep_s (float): The simulation time step size in seconds
            angularNoise (float): STD Dev of noise added to the rocket angle added at each timestep
        """

        self.data = data

        self.rocket = rocket
        self.canards = canards
        self.numCanards = numCanards

        self.targetFunction = targetFunction

        self.time_s = 0.0
        self.simulationEnd_s = simulationEnd_s
        self.timestep_s = timestep_s

        self.angleNoise = angleNoise

        self.running = True

        self.error_rad = 0.0

        self.canards.dt = self.timestep_s

    def reset(self):
        self.time_s = 0.0

        self.rocket.rollAngle_rad = 0.0
        self.rocket.rollVelocity_rps = 0.0
        self.rocket.verticalVelocity_mps = self.data.getAirSpeed(self.time_s)

        self.canards.canardAngle_rad = 0.0

        self.targetRollAngle_rad = self.targetFunction(self.time_s)
        self.error_rad = 0.0

        self.running = True

    def step(self, commandAngle_rad: float) -> float:
        self.time_s += self.timestep_s

        self.targetRollAngle_rad = self.targetFunction(self.time_s)
        self.canards.canardAngle_rad = commandAngle_rad

        # Add random noise to the rocket angle to simulate real world inaccuracies
        self.rocket.rollAngle_rad += random.gauss(0, self.angleNoise) * self.timestep_s

        # Add any recorded roll data
        self.rocket.rollAngle_rad += self.data.getRecordedRollRate(self.time_s) * self.timestep_s

        self.rocket.verticalVelocity_mps = self.data.getAirSpeed(self.time_s)

        # Calculate torque generate by the canards
        generatedTorque  = self.__calculateFinLift() * self.canards.canardDistance_m
        generatedTorque *= -1 if self.canards.canardAngle_rad < 0 else 1
        generatedTorque *= self.numCanards

        # Apply torque to the rocket
        self.rocket.rollVelocity_rps += (generatedTorque / self.rocket.rollMMOI_kgm2) * self.timestep_s
        self.rocket.rollAngle_rad    += (self.rocket.rollVelocity_rps * self.timestep_s)

        if (self.time_s >= self.simulationEnd_s):
            self.running = False

        self.error_rad = self.targetRollAngle_rad - self.rocket.rollAngle_rad
        return self.error_rad

    def __calculateFinLift(self) -> float:
        # Get data for current time
        airDensity_kgm3 = self.data.getAirDensity(self.time_s)

        CL = self.canards.getCL(self.rocket.verticalVelocity_mps)

        if CL == "Out of bounds":
            # print(f"Warning: Angle {math.degrees(angle):.2f} degrees and velocity {airspeed:.2f} m/s are out of bounds for the CL interpolator.")
            # If the angle and velocity are outside the range of the CSV, we can assume the coefficient of lift is 0
            return 0
        
        generatedLift = CL * 0.5 * self.rocket.verticalVelocity_mps**2 * airDensity_kgm3 * self.canards.surfaceArea_m2

        return generatedLift
