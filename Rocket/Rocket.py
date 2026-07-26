from typing import Callable
import math

from scipy.spatial.transform import Rotation
import numpy as np
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

    yaw_rad    = 0.0
    yawVel_rps = 0.0

    pitch_rad    = 0.0
    pitchVel_rps = 0.0

    roll_rad    = 0.0
    rollVel_rps = 0.0

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

        self.q = np.array([1.0, 0.0, 0.0, 0.0])

    def reset(self):
        self.simTime = 0.0

        self._updateAllSimData()

        self.q = np.array([1.0, 0.0, 0.0, 0.0])

        self.yawVel_rps = 0.0
        self.pitchVel_rps = 0.0
        self.rollVel_rps = 0.0

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

        self._applyTorques(yawTorque_Nm, pitchTorque_Nm, rollTorque_Nm)

        self.yaw_rad, self.pitch_rad, self.roll_rad = self._eulerFromQuat()

        self.posXError_m = self.targetXPos_m - self.xPos_m
        self.posYError_m = self.targetYPos_m - self.yPos_m
        self.posZError_m = self.targetZPos_m - self.zPos_m

        self.yawError_rad   = self.targetYaw_rad - self.yaw_rad
        self.pitchError_rad = self.targetPitch_rad - self.pitch_rad
        self.rollError_rad  = self.targetRoll_rad - self.roll_rad

    def _applyTorques(self, yawTorque_Nm: float, pitchTorque_Nm: float, rollTorque_Nm: float) -> None:
        """
        Apply torques given in the rocket's body reference frame to update angular velocity
        and integrate the absolute orientation quaternion.

        Angular acceleration is computed per-axis via a decoupled form of Euler's rotation
        equation (no gyroscopic cross-coupling — see TODO). The resulting body-frame angular
        velocity is then used to propagate the orientation quaternion self.q via the standard
        quaternion kinematic equation q_dot = 0.5 * q (x) [0, omega], integrated with a
        forward-Euler step and renormalized to counteract numerical drift.

        Axis convention: body x = yaw, body y = pitch, body z = roll.

        TODO: Add gyroscopic coupling (omega x I*omega) to the angular acceleration calculation
        once inertia asymmetry or high spin rates make the decoupled approximation inaccurate.

        Args:
            yawTorque_Nm (float): Torque about the body x-axis (yaw) in Newton-meters.
            pitchTorque_Nm (float): Torque about the body y-axis (pitch) in Newton-meters.
            rollTorque_Nm (float): Torque about the body z-axis (roll) in Newton-meters.
        """
        # Decoupled Euler's equation: alpha = torque / I (per axis, no omega x I*omega term)
        yawAcc_rps2   = yawTorque_Nm / self.Ix_kgm2
        pitchAcc_rps2 = pitchTorque_Nm / self.Iy_kgm2
        rollAcc_rps2  = rollTorque_Nm / self.Iz_kgm2

        # Integrate angular velocity (forward Euler)
        self.yawVel_rps   += yawAcc_rps2 * self.simTimeStep
        self.pitchVel_rps += pitchAcc_rps2 * self.simTimeStep
        self.rollVel_rps  += rollAcc_rps2 * self.simTimeStep

        # Quaternion kinematics: q_dot = 0.5 * q (x) [0, omega_body]
        w, x, y, z = self.q
        wx, wy, wz = self.yawVel_rps, self.pitchVel_rps, self.rollVel_rps

        qDot = 0.5 * np.array([
            -x * wx - y * wy - z * wz,
            w * wx + y * wz - z * wy,
            w * wy - x * wz + z * wx,
            w * wz + x * wy - y * wx,
        ])

        self.q = self.q + qDot * self.simTimeStep
        self.q = self.q / np.linalg.norm(self.q)  # renormalize — drifts every step otherwise

    def _eulerFromQuat(self) -> tuple[float, float, float]:
        """
        Convert the orientation quaternion self.q (scalar-first [w, x, y, z]) into
        absolute Euler angles matching this rocket's axis convention:
        body x = yaw, body y = pitch, body z = roll.

        Uses an intrinsic x-y-z rotation sequence. Note: like any 3-parameter Euler
        angle extraction, this can hit gimbal lock at certain attitudes (here, when
        the pitch angle approaches +/-90 deg) -- self.q itself has no such singularity,
        only this derived representation does.

        Returns:
            tuple[float, float, float]: (yaw_rad, pitch_rad, roll_rad)
        """
        w, x, y, z = self.q
        r = Rotation.from_quat([x, y, z, w])  # scipy wants scalar-last order
        yaw_rad, pitch_rad, roll_rad = r.as_euler('xyz', degrees=False)
        return yaw_rad, pitch_rad, roll_rad

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