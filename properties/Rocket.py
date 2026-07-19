from dataclasses import dataclass, field

import math

@dataclass
class Rocket:
    """
    Physical Rocket Properties

    Attributes:
        rollMMOI_kgm2 (float): The rockets mass moment of inertia (MMOI) in kg m^2
        rollAngle_rad (float): The rockets current angle in radians
        rollVelocity_rps (float): The rockets current anglular velocity in radians / sec
        verticalVelocity_mps (float): The rockets current vertical velocity in meters / sec
    """
    rollMMOI_kgm2:          float

    _rollAngle_rad:          float = 0.0
    rollVelocity_rps:       float = 0.0
    verticalVelocity_mps:   float = 0.0


    @property
    def rollAngle_rad(self) -> float:
        return self._rollAngle_rad

    @rollAngle_rad.setter
    def rollAngle_rad(self, angle: float):
        self._rollAngle_rad = (angle + math.pi) % (2 * math.pi) - math.pi