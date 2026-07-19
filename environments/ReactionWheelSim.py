from typing import Callable

import random

from properties.Data import Data
from properties.ReactionWheel import ReactionWheel
from properties.Rocket import Rocket

class ReactionWheelSim:
    """
    A Simulation Environment for simulating Reaction Wheel Roll Control on a Rocket

    Attributes:
        data (Data): Flight Data
        rocket (Rocket): Physical Rocket Properties
        reactionWheel (ReactionWheel): Physical Reaction Wheel Properties
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
            rocket: Rocket, reactionWheel: ReactionWheel,
            targetFunction: Callable[[float], float],
            simulationEnd_s: float, timestep_s: float = 0.01,
            angleNoise: float = 0.05,
        ):

        """
        Initialize the simulation environment
        
        Parameters:
            data (Data): Flight Data
            rocket (Rocket): Physical Rocket Properties
            reactionWheel (ReactionWheel): Physical Reaction Wheel Properties
            targetFunction (Callable[[float], float]): The function that takes sim time as an argument and returns the target angle
            simulationEnd_s (float): The time that the simulation will end at
            timestep_s (float): The simulation time step size in seconds
            angularNoise (float): STD Dev of noise added to the rocket angle added at each timestep
        """

        self.data = data

        self.rocket = rocket
        self.reactionWheel = reactionWheel

        self.targetFunction = targetFunction

        self.time_s = 0.0
        self.simulationEnd_s = simulationEnd_s
        self.timestep_s = timestep_s

        self.angleNoise = angleNoise

        self.running = True

        self.error_rad = 0.0

        self.reactionWheel.dt = self.timestep_s

    def reset(self):
        self.time_s = 0.0

        self.rocket.rollAngle_rad = 0.0
        self.rocket.rollVelocity_rps = 0.0

        self.targetRollAngle_rad = self.targetFunction(self.time_s)
        self.error_rad = 0.0

        self.running = True

    def step(self, wheelSpeed_rps: float) -> float:
        self.time_s += self.timestep_s

        self.targetRollAngle_rad = self.targetFunction(self.time_s)

        lastReactionWheelSpeed_rps = self.reactionWheel.speed_rps
        self.reactionWheel.speed_rps = wheelSpeed_rps

        # Add random noise to the rocket angle to simulate real world inaccuracies
        self.rocket.rollAngle_rad += random.gauss(0, self.angleNoise) * self.timestep_s

        # Add any recorded roll data
        self.rocket.rollAngle_rad += self.data.getRecordedRollRate(self.time_s) * self.timestep_s

        # Calculate torque generate by the reaction wheel
        dWReactionWheel_rps = self.reactionWheel.speed_rps - lastReactionWheelSpeed_rps
        
        # Apply torque to the rocket
        self.rocket.rollVelocity_rps += -(self.reactionWheel.mmoi_kgm2 / self.rocket.rollMMOI_kgm2) * dWReactionWheel_rps
        self.rocket.rollAngle_rad    += (self.rocket.rollVelocity_rps * self.timestep_s)

        if (self.time_s >= self.simulationEnd_s):
            self.running = False

        self.error_rad = self.targetRollAngle_rad - self.rocket.rollAngle_rad
        return self.error_rad