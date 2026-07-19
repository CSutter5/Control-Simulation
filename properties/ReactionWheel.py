from dataclasses import dataclass, field

@dataclass
class ReactionWheel:
    """
    Physical ReactionWheel Properties
    
    Attributes:
        mmoi_kgm2 (float): The mass moment of inertia (MMOI) of the reaction wheel in kg meters^2
        speed_rps (float): The speed of the reaction wheel in radian per second
        maxSpeedChange_rps2 (float): The maximum rate of change of the reaction wheel in radians per second^2
    """
    mmoi_kgm2:           float
    maxSpeedChange_rps2: float
    _speed_rps:          float = field(init=False, default=0.0)

    dt: float = field(init=False, default=0.01)

    @property
    def speed_rps(self) -> float:
        return self._speed_rps

    @speed_rps.setter
    def speed_rps(self, targetWheelSpeed):
        if (self._speed_rps == 0): self._speed_rps = targetWheelSpeed
        maxChange = self.maxSpeedChange_rps2 * self.dt

        delta = targetWheelSpeed - self._speed_rps
        delta = max(-maxChange, min(maxChange, delta))

        self._speed_rps = self._speed_rps + delta