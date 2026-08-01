from typing import Callable
import random
import math

from scipy.spatial.transform import Rotation
import numpy as np
import pandas as pd

from Controls import Controls

class Rocket:
    """
    Simulation engine and state container for a single rocket.

    A Rocket owns its full flight state (position, velocity, orientation,
    and angular rates) and advances that state one timestep at a time via
    `sim()`. Each call to `sim()` polls every attached `Controls` object for
    a torque tuple, sums the torques across all controls, applies them to
    update angular velocity and orientation, and refreshes environment-driven
    values (airspeed components, air density) from `simData` for the new
    simulation time.

    Axis convention (consistent throughout this class and its properties):
        body x = yaw, body y = pitch, body z = roll.

    Units convention:
        Positions in meters (`*_m`), velocities in meters/sec (`*_mps`),
        accelerations in m/s^2 (`*_mps2`), angles in radians (`*_rad`) with
        matching `*_deg` convenience properties, and angular rates in
        radians/sec (`*_rps`) with matching `*_dps` convenience properties.

    Attributes:
        simDataPath (str): Path to a CSV of time-indexed environment/flight
            data (e.g. velocity components, air density) used to drive the
            simulation.
        simData (pd.DataFrame): Loaded contents of `simDataPath`.
        Ix_kgm2, Iy_kgm2, Iz_kgm2 (float): Moments of inertia about the
            yaw, pitch, and roll body axes respectively, in kg*m^2. Must be
            nonzero.
        r_m (float): Rocket body radius in meters.
        length_m (float): Rocket body length in meters.
        mass_kg (float): Rocket mass in kilograms.
        targetFunc (Callable[[float], tuple]): Function mapping simulation
            time (seconds) to a 6-tuple of
            (targetXPos_m, targetYPos_m, targetZPos_m, targetYaw_deg,
            targetPitch_deg, targetRoll_deg) describing the desired
            trajectory at that time.
        simTimeStep (float): Fixed timestep in seconds used to advance
            `simTime` and integrate dynamics each call to `sim()`.
        controls (list[Controls]): Control objects polled each step; their
            returned torques are summed before being applied to the rocket.
        airDensity (float): Current air density, refreshed from `simData`
            each step.
        xPos_m/yPos_m/zPos_m, xVel_mps/yVel_mps/zVel_mps: Position and
            velocity in meters and meters/sec.
        yaw_rad/pitch_rad/roll_rad: Current absolute orientation in radians,
            derived from the internal orientation quaternion `self.q`.
        yawVel_rps/pitchVel_rps/rollVel_rps: Current body-frame angular
            rates in radians/sec.
        targetXPos_m/targetYPos_m/targetZPos_m/targetYaw_rad/
        targetPitch_rad/targetRoll_rad: Desired state at the current
            simulation time, as returned by `targetFunc`.
        xPosError_m/yPosError_m/zPosError_m/yawError_rad/pitchError_rad/
        rollError_rad: Difference between target and actual state,
            recomputed each step.
        simStop (float): Simulation time in seconds at which `running`
            becomes False. Defaults to 15.
        simTime (float): Elapsed simulation time in seconds.
        running (bool): False once `simTime` reaches `simStop`.
    """

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
        """
        Initialize the Rocket and load its environment/flight data.

        Args:
            simDataPath (str): Path to a CSV containing time-indexed
                environment/flight data (must include a 'time' column;
                columns such as 'xVel_mps', 'yVel_mps', 'zVel_mps', and
                'airDensity' are read via `_getSimData`).
            Ix_kgm2 (float): Moment of inertia about the yaw axis, kg*m^2.
                Must not be 0.
            Iy_kgm2 (float): Moment of inertia about the pitch axis, kg*m^2.
                Must not be 0.
            Iz_kgm2 (float): Moment of inertia about the roll axis, kg*m^2.
                Must not be 0.
            r_m (float): Rocket body radius in meters.
            length_m (float): Rocket body length in meters.
            mass_kg (float): Rocket mass in kilograms.
            targetFunc (Callable[[float], tuple]): Function mapping
                simulation time (seconds) to
                (targetXPos_m, targetYPos_m, targetZPos_m, targetYaw_deg,
                targetPitch_deg, targetRoll_deg).
            simTimeStep (float): Fixed timestep in seconds used to advance
                the simulation.
            controls (list[Controls]): Control objects to poll each step.

        Raises:
            ValueError: If any of Ix_kgm2, Iy_kgm2, or Iz_kgm2 is 0.
        """
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
        """
        Reset the rocket to its initial simulation state.

        Zeroes `simTime`, reloads environment data for time 0 via
        `_updateAllSimData()`, resets the orientation quaternion to
        identity ([1, 0, 0, 0]), and zeroes all three angular velocities.
        Does not reset position (`xPos_m`/`yPos_m`/`zPos_m`) or the
        `controls` list.

        Call this before starting a new simulation run with an
        already-constructed Rocket, instead of re-instantiating it.

        Returns:
            None
        """
        self.simTime = 0.0

        self._updateAllSimData()

        self.q = np.array([1.0, 0.0, 0.0, 0.0])

        self.yawVel_rps = 0.0
        self.pitchVel_rps = 0.0
        self.rollVel_rps = 0.0

        return

    def _updateAllSimData(self):
        """
        Refresh velocity components and air density from `simData` for the
        current `simTime`.

        Looks up 'zVel_mps', and 'airDensity' at the nearest available row
        in `simData` via `_getSimData` and assigns them to the corresponding
        instance attributes. Called once per `sim()` step (and by `reset()`)
        rather than integrated from first principles, since this data is
        treated as externally supplied flight/environment input rather 
        than derived state.

        Returns:
            None
        """
        self.zVel_mps = self._getSimData('zVel_mps', self.simTime)

        self.airDensity = self._getSimData('airDensity', self.simTime)

    def _getSimData(self, columnName: str, time: float) -> float:
        """
        Look up a value from `simData` at the row whose 'time' column is
        closest to the given time.

        Does not interpolate between rows — this is a nearest-neighbor
        lookup, so accuracy depends on how finely `simData` is sampled
        relative to `simTimeStep`.

        Args:
            columnName (str): Name of the column to read (e.g. 'xVel_mps',
                'airDensity').
            time (float): Simulation time in seconds to look up.

        Returns:
            float: The value in `columnName` at the nearest row to `time`,
                or None if `columnName` is not present in `simData`.
        """
        if not columnName in self.simData:
            return None

        idx = (self.simData['time'] - time).abs().idxmin()
        return self.simData.loc[idx, columnName]

    def sim(self, **kwargs):
        """
        Advance the simulation by one `simTimeStep`.

        Steps `simTime` forward, refreshes environment data, evaluates
        `targetFunc` for the new time, polls every control in `controls`
        (passing `rocket=self` and any `**kwargs` through to each control's
        `sim()`), sums the returned per-control torque tuples, applies the
        aggregate torque via `_applyTorques`, updates the derived Euler
        angles, and recomputes position/attitude error terms against the
        target state.

        Args:
            **kwargs: Forwarded unchanged to every control's `sim()` call.
                For example, a `Canards` control requires a
                `canardAngle_deg` keyword argument here.

        Note:
            The README describes yaw/pitch dynamics as not yet implemented
            and states that a non-zero yaw or pitch torque should raise an
            error. That guard is not currently present in `_applyTorques`
            or here — yaw/pitch torques are applied the same as roll. If
            you're relying on that documented behavior, treat it as a
            pending TODO rather than existing protection.

        Returns:
            None
        """
        self.simTime += self.simTimeStep
        self.running = self.simTime < self.simStop

        self._updateAllSimData()

        self.targetXPos_m, self.targetYPos_m, self.targetZPos_m, self.targetYaw_deg, self.targetPitch_deg, self.targetRoll_deg = self.targetFunc(self.simTime)

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

    def randomize(self):
        self.rollAngle_rad = random.uniform(-math.pi, math.pi)
        self.rollVelocity_rps = random.uniform(0, 20)

    @property
    def yaw_deg(self) -> float:
        """float: Current absolute yaw in degrees (derived from `yaw_rad`)."""
        return math.degrees(self.yaw_rad)

    @property
    def yawVel_dps(self) -> float:
        """float: Current yaw angular rate in degrees/sec (derived from `yawVel_rps`)."""
        return math.degrees(self.yawVel_rps)

    @property
    def targetYaw_deg(self) -> float:
        """float: Target yaw in degrees (derived from `targetYaw_rad`)."""
        return math.degrees(self.targetYaw_rad)

    @targetYaw_deg.setter
    def targetYaw_deg(self, targetYaw_deg):
        """Set the target yaw from degrees, storing it as `targetYaw_rad`."""
        self.targetYaw_rad = math.radians(targetYaw_deg)

    @property
    def yawError_deg(self) -> float:
        """float: Yaw error (target minus actual) in degrees."""
        return math.degrees(self.yawError_rad)

    @property
    def pitch_deg(self) -> float:
        """float: Current absolute pitch in degrees (derived from `pitch_rad`)."""
        return math.degrees(self.pitch_rad)

    @property
    def pitchVel_dps(self) -> float:
        """float: Current pitch angular rate in degrees/sec (derived from `pitchVel_rps`)."""
        return math.degrees(self.pitchVel_rps)

    @property
    def targetPitch_deg(self) -> float:
        """float: Target pitch in degrees (derived from `targetPitch_rad`)."""
        return math.degrees(self.targetPitch_rad)

    @targetPitch_deg.setter
    def targetPitch_deg(self, targetPitch_deg):
        """Set the target pitch from degrees, storing it as `targetPitch_rad`."""
        self.targetPitch_rad = math.radians(targetPitch_deg)

    @property
    def pitchError_deg(self) -> float:
        """float: Pitch error (target minus actual) in degrees."""
        return math.degrees(self.pitchError_rad)

    @property
    def roll_deg(self) -> float:
        """float: Current absolute roll in degrees (derived from `roll_rad`)."""
        return math.degrees(self.roll_rad)

    @property
    def rollVel_dps(self) -> float:
        """float: Current roll angular rate in degrees/sec (derived from `rollVel_rps`)."""
        return math.degrees(self.rollVel_rps)

    @property
    def targetRoll_deg(self) -> float:
        """float: Target roll in degrees (derived from `targetRoll_rad`)."""
        return math.degrees(self.targetRoll_rad)

    @targetRoll_deg.setter
    def targetRoll_deg(self, targetRoll_deg):
        """Set the target roll from degrees, storing it as `targetRoll_rad`."""
        self.targetRoll_rad = math.radians(targetRoll_deg)

    @property
    def rollError_deg(self) -> float:
        """float: Roll error (target minus actual) in degrees."""
        return math.degrees(self.rollError_rad)