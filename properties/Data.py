from dataclasses import dataclass, field
import math

import pandas as pd

@dataclass
class Data:
    """
    Flight Data

    Attributes:
        dataPath (str): The path to the flight data
        startingTemperature_c (float): The temperature at ground level in degrees C
        startingPressure_pa (float): The pressure at ground level in Pa
        humidity (float): The relative humidity at ground representated as a decimal 
    
    """
    dataPath:    str
    data: pd.DataFrame = field(init=False)

    startingTemperature_c:  float
    startingPressure_pa:    float
    humidity:               float = 0.9

    def __post_init__(self):
        self.data = pd.read_csv(self.dataPath)

    def getAirDensity(self, time: float) -> float:
        idx = (self.data['time'] - time).abs().idxmin()
        if "density" in self.data.columns:
            return self.data.loc[idx, 'density']

        elif "altitude" in self.data.columns:
            p1 = 6.1078 * 10 ** (7.5 * self.startingTemperature_c / (self.startingTemperature_c + 237.3))
            pv = self.humidity * p1

            pressure = self.startingPressure_pa * (1 - 0.0065 * self.data.loc[idx, 'altitude'] / (self.startingTemperature_c + 273.15)) ** 5.2561

            pd = pressure - pv

            return pd / (287.05 * (self.startingTemperature_c + 273.15)) + pv / (461.495 * (self.startingTemperature_c + 273.15))
        
    def getAirSpeed(self, time: float) -> float:
        idx = (self.data['time'] - time).abs().idxmin()
        return self.data.loc[idx, 'speed']
    
    def getRecordedRollRate(self, time: float) -> float:
        if "roll-rate" in self.data.columns:
            idx = (self.data['time'] - time).abs().idxmin()
            return math.radians(self.data.loc[idx, 'roll-rate'])
        
        return 0
