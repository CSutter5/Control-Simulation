from dataclasses import dataclass, field

from scipy.interpolate import LinearNDInterpolator
import pandas as pd
import numpy as np

import math

@dataclass
class Canard:
    """
    Physical Canard Properties
    
    Attributes:
        airfoilDataPath (str): The path to the airfoil data
        rootCoord_m (float): The canards root coord length in meters
        tipCoord_m (float): The canards tip coord length in meters
        span_m (float): The canards span length in meters
        sweep_m (float): The canards sweep length in meters
        surfaceArea_m2 (float): The canards surface in meters^2
        canardAngle_rad (float): The canard current angle in radians
        canardDistance_m (float): The canards distance from the center axis of the rocket
        minCanardAngle_rad (float): Min canard angle in radians
        maxCanardAngle_rad (float): Max canard angle in radians
        canardAngleRateLimit_rps (float): Maximum rate that the canard is able to move at
    """
    airfoilDataPath:    str
    canardData:         pd.DataFrame = field(init=False)
    liftInterpFunc:     any = field(init=False)
    
    rootCoord_m:        float
    tipCoord_m:         float
    span_m:             float
    sweep_m:            float
    canardDistance_m:   float
    
    canardAngleRateLimit_rps: float
    canardAngleRate_rps:      float = field(init=False)

    surfaceArea_m2:     float = field(init=False)

    _canardAngle_rad:   float = math.radians(0.0)
    minCanardAngle_rad: float = math.radians(-10.0)
    maxCanardAngle_rad: float = math.radians(10.0)

    dt: float = field(init=False, default=0.01)

    def __post_init__(self):
        self.surfaceArea_m2 = (self.rootCoord_m + self.tipCoord_m) / 2 * self.span_m

        self.canardData = pd.read_csv(self.airfoilDataPath)

        self._points = self.canardData[['alpha', 'Velocity']].values
        self._values = self.canardData['CL'].values

        # Create the interpolator
        # This handles non-exact matches by lerping between the nearest points
        self.liftInterpFunc = LinearNDInterpolator(self._points, self._values, fill_value=0.0)

    @property
    def canardAngle_rad(self) -> float:
        return self._canardAngle_rad

    @canardAngle_rad.setter
    def canardAngle_rad(self, angle):
        maxChange = self.canardAngleRateLimit_rps * self.dt

        self.canardAngleRate_rps = angle - self._canardAngle_rad
        self.canardAngleRate_rps = max(-maxChange, min(maxChange, self.canardAngleRate_rps))

        self._canardAngle_rad = max(min(self._canardAngle_rad + self.canardAngleRate_rps, self.maxCanardAngle_rad), self.minCanardAngle_rad)

    def getCL(self, velocity: float) -> float:
        res = self.liftInterpFunc(abs(math.degrees(self._canardAngle_rad)), velocity)
        return float(res) if not np.isnan(res) else math.nan
