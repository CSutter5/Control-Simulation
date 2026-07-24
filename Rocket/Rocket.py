from typing import Callable
import math

import pandas as pd

from Controls import Controls

class Rocket:
    simDataPath:    str
    simData: pd.DataFrame

    Ix_kgm2:    float
    Iy_kgm2:    float
    Iz_kgm2:    float
    r_m:        float
    length_m:   float
    mass_kg:    float
    targetFunc: Callable[[float], tuple[float, float, float, float, float, float]]
    simTimeStep: float
    controls: list[Controls]

    airDensity: float

    xPos_m    = 0.0
    xVel_mps  = 0.0
    xZcc_mps2 = 0.0

    yPos_m    = 0.0
    yVel_mps  = 0.0
    yZcc_mps2 = 0.0

    zPos_m    = 0.0
    zVel_mps  = 0.0
    zZcc_mps2 = 0.0

    yaw_rad     = 0.0
    yawVel_rps  = 0.0

    pitch_rad     = 0.0
    pitchVel_rps  = 0.0

    roll_rad     = 0.0
    rollVel_rps  = 0.0

    targetXPos_m    = 0.0
    targetYPos_m    = 0.0
    targetZPos_m    = 0.0
    targetYaw_rad   = 0.0
    targetPitch_rad = 0.0
    targetRoll_rad  = 0.0

    xPosError_m    = 0.0
    yPosError_m    = 0.0
    zPosError_m    = 0.0
    yawError_rad   = 0.0
    pitchError_rad = 0.0
    rollError_rad  = 0.0

    simStop = 15
    simTime = 0.0
    running = True

    def __init__(self, simDataPath: str, 
        Ix_kgm2: float, Iy_kgm2: float, Iz_kgm2: 
        float, r_m: float, length_m: float, mass_kg:float, 
        targetFunc: Callable[[float], tuple[float, float, float, float, float, float]], 
        simTimeStep: float, controls: list[Controls]
    ):
        self.simDataPath = simDataPath
        self.simData = pd.read_csv(self.simDataPath)

        if Ix_kgm2 == 0: raise ValueError("Ix_kgm2 must not be 0!")
        if Iy_kgm2 == 0: raise ValueError("Iy_kgm2 must not be 0!")
        if Iz_kgm2 == 0: raise ValueError("Iz_kgm2 must not be 0!")

        self.Ix_kgm2  = Ix_kgm2
        self.Iy_kgm2  = Iy_kgm2
        self.Iz_kgm2  = Iz_kgm2
        self.r_m      = r_m
        self.length_m = length_m
        self.mass_kg    = mass_kg

        self.targetFunc = targetFunc
        self.simTimeStep = simTimeStep

        self.controls = controls

        self.airDensity = 0.0

        self.simTime = 0.0
        self.running = True

    def reset(self):
        self.simTime = 0.0

        self._updateAllSimData()

        return

    def _updateAllSimData(self):
        self.xVel_mps = self._getSimData('xVel_mps', self.simTime)
        self.yVel_mps = self._getSimData('yVel_mps', self.simTime)
        self.zVel_mps = self._getSimData('zVel_mps', self.simTime)

        self.airDensity = self._getSimData('airDensity', self.simTime)

    def _getSimData(self, columnName: str, time: float) -> float:
        if not columnName in self.simData:
            return None

        idx = (self.simData['time'] - time).abs().idxmin()
        return self.simData.loc[idx, columnName]

    def sim(self, **kwargs):
        self.simTime += self.simTimeStep
        self.running = self.simTime < self.simStop

        self._updateAllSimData()

        self.targetPosX_m, self.targetPosY_m, self.targetPosZ_m, self.targetYaw_deg, self.targetPitch_deg, self.targetRoll_deg = self.targetFunc(self.simTime)

        yawTorque_Nm    = 0.0
        pitchTorque_Nm  = 0.0
        rollTorque_Nm   = 0.0

        for control in self.controls:
            torques = control.sim(rocket=self, **kwargs)
            yawTorque_Nm    += torques[0]
            pitchTorque_Nm  += torques[1]
            rollTorque_Nm   += torques[2]

        if yawTorque_Nm != 0 or pitchTorque_Nm != 0: raise NotImplementedError(f"Pitch and Yaw Simulation is not implemented")

        self.rollVel_rps += rollTorque_Nm / self.Iz_kgm2 * self.simTimeStep
        self.roll_rad    += self.rollVel_rps * self.simTimeStep

        self.posXError_m = self.targetXPos_m - self.xPos_m
        self.posYError_m = self.targetYPos_m - self.yPos_m
        self.posZError_m = self.targetZPos_m - self.zPos_m

        self.yawError_rad   = self.targetYaw_rad - self.yaw_rad
        self.pitchError_rad = self.targetPitch_rad - self.pitch_rad
        self.rollError_rad  = self.targetRoll_rad - self.roll_rad

    @property
    def yaw_deg(self) -> float: return math.degrees(self.yaw_rad)

    @property
    def yawVel_dps(self) -> float: return math.degrees(self.yawVel_rps)

    @property
    def targetYaw_deg(self) -> float: return math.degrees(self.targetYaw_rad)

    @targetYaw_deg.setter
    def targetYaw_deg(self, targetYaw_deg): self.targetRoll_rad = math.radians(targetYaw_deg)

    @property
    def yawError_deg(self) -> float: return math.degrees(self.yawError_rad)


    @property
    def pitch_deg(self) -> float: return math.degrees(self.pitch_rad)

    @property
    def pitchVel_dps(self) -> float: return math.degrees(self.pitchVel_rps)

    @property
    def targetPitch_deg(self) -> float: return math.degrees(self.targetPitch_rad)

    @targetPitch_deg.setter
    def targetPitch_deg(self, targetPitch_deg): self.targetPitch_rad = math.radians(targetPitch_deg)

    @property
    def pitchError_deg(self) -> float: return math.degrees(self.pitchError_rad)


    @property
    def roll_deg(self) -> float: return math.degrees(self.roll_rad)

    @property
    def rollVel_dps(self) -> float: return math.degrees(self.rollVel_rps)

    @property
    def targetRoll_deg(self) -> float: return math.degrees(self.targetRoll_rad)

    @targetRoll_deg.setter
    def targetRoll_deg(self, targetRoll_deg): self.targetRoll_rad = math.radians(targetRoll_deg)

    @property
    def rollError_deg(self) -> float: return math.degrees(self.rollError_rad)