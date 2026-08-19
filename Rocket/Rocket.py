from typing import Callable
import random
import math
import warnings

from scipy.spatial.transform import Rotation
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from Controls import Controls, Force

class Rocket:
    """
    Simulation engine and state container for a single rocket.

    A Rocket owns its full flight state (position, velocity, orientation,
    and angular rates) and advances that state one timestep at a time via
    `sim()`. Each call to `sim()` polls every attached `Controls` object for
    a `list[Force]`, converts each returned `Force` into torque (`r x F`
    about the center of gravity) and net translational force, sums both
    across all controls and all forces, applies the resulting torque to
    update angular velocity and orientation, converts the net force to
    acceleration, and refreshes environment-driven values (airspeed
    components, air density) from `simData` for the new simulation time.

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
        CG_m (float): The Center of Gravity measured from the top of the rocket in meters
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
        xForce_N/yForce_N/zForce_N (float): Net body-frame force left over
            each step after each control's force has had its torque
            contribution extracted (`_torqueFromForce`/
            `Force.decomposeForce`). Recomputed every `sim()` call and
            converted to world-frame acceleration by `_applyForces`, but
            not yet integrated into velocity or position.
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
        GRAVITY_MPS2 (float): Standard gravity, 9.80665 m/s^2, applied
            along world-frame -z in `_applyForces` every step. See the
            IMPORTANT INVARIANT note on `_applyForces` for the assumption
            (rocket starts vertical) that makes this direction correct.
        df (pd.DataFrame): Time-indexed log of this rocket's state at every
            past `sim()` step (position, velocity, acceleration,
            orientation, angular rates, target state, and error terms --
            see the columns assigned in `sim()`), indexed by `time_s`.
            Rebuilt empty by `reset()`. Consumed by `plotRoll()`.
    """

    simDataPath:    str
    simData: pd.DataFrame

    Ix_kgm2:    float
    Iy_kgm2:    float
    Iz_kgm2:    float
    CG_m:       float
    r_m:        float
    length_m:   float
    mass_kg:    float

    targetFunc: Callable[[float], tuple[float, float, float, float, float, float]]
    simTimeStep: float
    controls: list[Controls]

    airDensity: float

    xPos_m    = 0.0
    xVel_mps  = 0.0
    xAcc_mps2 = 0.0

    yPos_m    = 0.0
    yVel_mps  = 0.0
    yAcc_mps2 = 0.0

    zPos_m    = 0.0
    zVel_mps  = 0.0
    zAcc_mps2 = 0.0

    # Net body-frame force left over after torque has been extracted from
    # each control's force via `_torqueFromForce`/`Force.decomposeForce`
    # each step. Converted to world-frame acceleration by `_applyForces`,
    # but not yet integrated into velocity/position.
    xForce_N = 0.0
    yForce_N = 0.0
    zForce_N = 0.0

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

    # Standard gravity, applied along world-frame -z in _applyForces.
    GRAVITY_MPS2 = 9.80665

    # Column list only (read-only, safe to share at class scope -- never
    # mutated). The DataFrame itself is NOT created here: `df` is a true
    # instance attribute created fresh in __init__/reset() below. A
    # class-level `df = pd.DataFrame(...)` assignment would create ONE
    # shared DataFrame object across every Rocket instance (see the same
    # fix applied to Canards/ReactionWheel, and TODO.md).
    _DF_COLUMNS = [
        "time_s", 
        "xPos_m", "xVel_mps", "xAcc_mps2",
        "yPos_m", "yVel_mps", "yAcc_mps2",
        "zPos_m", "zVel_mps", "zAcc_mps2",
        "yaw_rad", "yawVel_rps",
        "pitch_rad", "pitchVel_rps",
        "roll_rad", "rollVel_rps",
        "targetXPos_m", "targetYPos_m", "targetZPos_m",
        "targetYaw_rad", "targetPitch_rad", "targetRoll_rad",
        "xPosError_m", "yPosError_m", "zPosError_m",
        "yawError_rad", "pitchError_rad", "rollError_rad"
    ]

    df: pd.DataFrame

    def __init__(self, simDataPath: str, 
        Ix_kgm2: float, Iy_kgm2: float, Iz_kgm2: float, CG_m: float,
        r_m: float, length_m: float, mass_kg:float, 
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
            CG_m (float): The Center of Gravity measured from the top of the rocket in meters
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
            ValueError: If any of Ix_kgm2, Iy_kgm2, Iz_kgm2, or mass_kg is 0.
        """
        self.simDataPath = simDataPath
        self.simData = pd.read_csv(self.simDataPath)

        if Ix_kgm2 == 0: raise ValueError("Ix_kgm2 must not be 0!")
        if Iy_kgm2 == 0: raise ValueError("Iy_kgm2 must not be 0!")
        if Iz_kgm2 == 0: raise ValueError("Iz_kgm2 must not be 0!")
        if mass_kg == 0: raise ValueError("mass_kg must not be 0!")

        self.Ix_kgm2  = Ix_kgm2
        self.Iy_kgm2  = Iy_kgm2
        self.Iz_kgm2  = Iz_kgm2
        self.CG_m     = CG_m
        self.r_m      = r_m
        self.length_m = length_m
        self.mass_kg    = mass_kg

        # _updateAllSimData overwrites CG_m/Ix_kgm2/Iy_kgm2/Iz_kgm2 from
        # simData every step if the matching column exists, silently
        # ignoring the constructor value passed above after the first
        # sim() call. Warn here (once, at construction) rather than let
        # that happen with no indication -- e.g. a CSV column left over
        # from copy-pasting a different scenario's data file.
        for _shadowedArg, _column in (
            ("CG_m", "CG_m"), ("Ix_kgm2", "Ix_kgm2"),
            ("Iy_kgm2", "Iy_kgm2"), ("Iz_kgm2", "Iz_kgm2")
        ):
            if _column in self.simData:
                warnings.warn(
                    f"simData ({self.simDataPath}) contains a '{_column}' "
                    f"column, which will override the constructor argument "
                    f"'{_shadowedArg}' every sim() step starting from the "
                    f"first call -- the value passed to __init__ is only "
                    f"used as the very first frame's fallback before that.",
                    stacklevel=2
                )

        self.targetFunc = targetFunc
        self.simTimeStep = simTimeStep

        self.controls = controls

        self.airDensity = 0.0

        self.simTime = 0.0
        self.running = True

        self.q = np.array([1.0, 0.0, 0.0, 0.0])

        self.df = pd.DataFrame(columns=self._DF_COLUMNS)
        self.df = self.df.set_index("time_s")

    def reset(self):
        """
        Reset the rocket to its initial simulation state.

        Zeroes `simTime`, reloads environment data for time 0 via
        `_updateAllSimData()`, resets the orientation quaternion to
        identity ([1, 0, 0, 0]), and zeroes all three angular velocities.
        Does not reset position (`xPos_m`/`yPos_m`/`zPos_m`), velocity
        (`xVel_mps`/`yVel_mps`/`zVel_mps`), net force (`xForce_N`/
        `yForce_N`/`zForce_N`), acceleration (`xAcc_mps2`/`yAcc_mps2`/
        `zAcc_mps2`), or the `controls` list. This is a deliberate
        decision, not an oversight: translational integration (velocity/
        position from acceleration) isn't wired up yet (see project
        TODO.md), so there's no single correct answer yet for what a
        "reset" should mean for that state -- e.g. whether a second
        `reset()`+run on the same instance should carry over leftover
        velocity from a previous run, or start clean. Revisit this
        decision as part of implementing that integration, rather than
        guessing at it here in isolation beforehand.

        NOTE: resetting `self.q` to identity encodes the assumption that
        the rocket starts pointing straight up (body z aligned with world
        z) -- see the IMPORTANT INVARIANT note in `_applyForces` for why
        this specific choice is what makes applying gravity along world -z
        correct. Do not repurpose the identity quaternion here without
        also revisiting that note.

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

        self.df = pd.DataFrame(columns=self.df.columns)
        self.df.index.name = "time_s"

        return

    def sim(self, **kwargs):
        """
        Advance the simulation by one `simTimeStep`.

        Steps `simTime` forward, refreshes environment data, evaluates
        `targetFunc` for the new time, polls every control in `controls`
        (passing `rocket=self` and any `**kwargs` through to each control's
        `sim()`) for a `list[Force]`. For every returned `Force`, converts
        it to torque about the center of gravity (`_torqueFromForce`, via
        `r x F`) and its own translational contribution
        (`Force.decomposeForce`), summing both across every force from
        every control. Applies the net translational force via
        `_applyForces` (computing world-frame acceleration, not yet
        integrated into velocity/position -- see `TODO.md`), THEN applies
        the aggregate torque via `_applyTorques` (updating angular velocity
        and orientation) -- this order is deliberate: `_applyForces` must
        rotate this step's force using the rocket's attitude from the
        START of the step, before `_applyTorques` updates that attitude,
        or the two would mix old and new attitude within a single
        timestep. Finally updates the derived Euler angles and recomputes
        position/attitude error terms against the target state.

        Args:
            **kwargs: Forwarded unchanged to every control's `sim()` call.
                For example, a `Canards` control requires a
                `canardAngle_deg` keyword argument here.

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

        xForce_N = 0.0
        yForce_N = 0.0
        zForce_N = 0.0

        for control in self.controls:
            for force in control.sim(rocket=self, **kwargs):
                yT, pT, rT = self._torqueFromForce(force)
                yawTorque_Nm   += yT
                pitchTorque_Nm += pT
                rollTorque_Nm  += rT

                fx, fy, fz = force.decomposeForce()
                xForce_N += fx
                yForce_N += fy
                zForce_N += fz

        # Net body-frame force left over after torque has been extracted.
        self.xForce_N = xForce_N
        self.yForce_N = yForce_N
        self.zForce_N = zForce_N

        # Must run before _applyTorques: _applyForces rotates this step's
        # net force using the CURRENT self.q, i.e. the rocket's attitude
        # at the START of this step. If this ran after _applyTorques, it
        # would rotate this step's force using an attitude that already
        # includes this step's own rotation -- mixing old and new attitude
        # within a single timestep.
        self._applyForces(xForce_N, yForce_N, zForce_N)

        # Per-axis MMOI used for torque -> angular acceleration must
        # include any internal spinning mass a control carries (e.g.
        # ReactionWheel's flywheel), not just the passive body structure
        # in Ix_kgm2/Iy_kgm2/Iz_kgm2 -- see Controls.additionalYaw/Pitch/
        # RollInertia_kgm2 and ReactionWheel's class docstring for the
        # conservation-of-angular-momentum reasoning. Computed fresh each
        # step (not accumulated into self.Ix_kgm2 etc.) so those attributes
        # keep meaning "structure only".
        effectiveIx_kgm2 = self.Ix_kgm2 + sum(c.additionalYawInertia_kgm2 for c in self.controls)
        effectiveIy_kgm2 = self.Iy_kgm2 + sum(c.additionalPitchInertia_kgm2 for c in self.controls)
        effectiveIz_kgm2 = self.Iz_kgm2 + sum(c.additionalRollInertia_kgm2 for c in self.controls)

        self._applyTorques(yawTorque_Nm, pitchTorque_Nm, rollTorque_Nm, effectiveIx_kgm2, effectiveIy_kgm2, effectiveIz_kgm2)

        self.yaw_rad, self.pitch_rad, self.roll_rad = self._eulerFromQuat()

        self.posXError_m = self.targetXPos_m - self.xPos_m
        self.posYError_m = self.targetYPos_m - self.yPos_m
        self.posZError_m = self.targetZPos_m - self.zPos_m

        self.yawError_rad   = self.targetYaw_rad - self.yaw_rad
        self.pitchError_rad = self.targetPitch_rad - self.pitch_rad
        self.rollError_rad  = self.targetRoll_rad - self.roll_rad
        
        self.df.loc[self.simTime] = {
            "time_s":           self.simTime,
            "xPos_m":           self.xPos_m,
            "xVel_mps":         self.xVel_mps,
            "xAcc_mps2":        self.xAcc_mps2,
            "yPos_m":           self.yPos_m,
            "yVel_mps":         self.yVel_mps,
            "yAcc_mps2":        self.yAcc_mps2,
            "zPos_m":           self.zPos_m,
            "zVel_mps":         self.zVel_mps,
            "zAcc_mps2":        self.zAcc_mps2,
            "yaw_rad":          self.yaw_rad,
            "yawVel_rps":       self.yawVel_rps,
            "pitch_rad":        self.pitch_rad,
            "pitchVel_rps":     self.pitchVel_rps,
            "roll_rad":         self.roll_rad,
            "rollVel_rps":      self.rollVel_rps,
            "targetXPos_m":     self.targetXPos_m,
            "targetYPos_m":     self.targetYPos_m,
            "targetZPos_m":     self.targetZPos_m,
            "targetYaw_rad":    self.targetYaw_rad,
            "targetPitch_rad":  self.targetPitch_rad,
            "targetRoll_rad":   self.targetRoll_rad,
            "xPosError_m":      self.xPosError_m,
            "yPosError_m":      self.yPosError_m,
            "zPosError_m":      self.zPosError_m,
            "yawError_rad":     self.yawError_rad,
            "pitchError_rad":   self.pitchError_rad,
            "rollError_rad":    self.rollError_rad
        }

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

        if "zVel_mps" in self.simData:   self.zVel_mps   = self._getSimData('zVel_mps', self.simTime)
        if "airDensity" in self.simData: self.airDensity = self._getSimData('airDensity', self.simTime)

        if "CG_m" in self.simData:    self.CG_m    = self._getSimData('CG_m', self.simTime)
        if "mass_kg" in self.simData: self.mass_kg = self._getSimData('mass_kg', self.simTime)
        if "Ix_kgm2" in self.simData: self.Ix_kgm2 = self._getSimData('Ix_kgm2', self.simTime)
        if "Iy_kgm2" in self.simData: self.Iy_kgm2 = self._getSimData('Iy_kgm2', self.simTime)
        if "Iz_kgm2" in self.simData: self.Iz_kgm2 = self._getSimData('Iz_kgm2', self.simTime)

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

        Raises:
            KeyError: If `columnName` is not present in `simData`. Every
                current call site (`_updateAllSimData`) already guards this
                with `if columnName in self.simData` first, since not every
                column (e.g. `CG_m`, `Ix_kgm2`) is expected to be present
                in every scenario's CSV -- this raises rather than
                returning `None` so a *future* caller that skips that
                check fails loudly instead of silently propagating `None`
                into downstream arithmetic.

        Returns:
            float: The value in `columnName` at the nearest row to `time`.
        """
        if not columnName in self.simData:
            raise KeyError(
                f"'{columnName}' is not a column in simData (loaded from "
                f"{self.simDataPath}) -- check for it with "
                f"`if columnName in self.simData` before calling "
                f"_getSimData if it's expected to be optional."
            )

        idx = (self.simData['time'] - time).abs().idxmin()
        return self.simData.loc[idx, columnName]

    def _torqueFromForce(self, force: Force) -> tuple[float, float, float]:
        """
        Convert one applied Force into the torque it generates about the
        rocket's center of gravity.

        Uses `torque = r x F`, where `r` is `force.location_m` relative to
        the CG. `location_m`'s z-component and `CG_m` are both measured
        from the same reference point along the body, so
        `force.location_m[2] - CG_m` gives the lever arm along the body's
        z-axis; X/Y offsets need no such adjustment since the CG is assumed
        to sit on the body centerline.

        Because a control returns one `Force` per distinct application
        point (see `Controls.Force`), a control made of several
        force-generating surfaces (e.g. `Canards` with several fins placed
        symmetrically around the body) can have its individual per-surface
        torques summed here *before* any translational cancellation, which
        correctly produces a net torque (e.g. pure roll) even when those
        same forces cancel to zero net translational force -- a pure
        couple that a single aggregate (force, location) pair could not
        represent.

        Args:
            force (Force): One applied force, with its own `vector_N` and
                `location_m` in the body frame.

        Returns:
            tuple[float, float, float]: (yawTorque_Nm, pitchTorque_Nm,
                rollTorque_Nm) generated by this force about the CG.
        """
        fx, fy, fz = force.vector_N

        rx, ry, rz = force.location_m
        rz -= self.CG_m

        yawTorque_Nm   = ry * fz - rz * fy
        pitchTorque_Nm = rz * fx - rx * fz
        rollTorque_Nm  = rx * fy - ry * fx

        return (yawTorque_Nm, pitchTorque_Nm, rollTorque_Nm)

    def _applyForces(self, xForce_N: float, yForce_N: float, zForce_N: float) -> None:
        """
        Convert the net body-frame force this step into world-frame
        acceleration.

        The net force accumulated in `sim()` (via `Force.decomposeForce`) is
        in the body frame -- it rotates with the rocket. Position/velocity
        are meant to be in the world frame, so the force is first rotated
        into the world frame using the CURRENT `self.q` (see the note in
        `sim()` on why this must run before `_applyTorques` updates `q`),
        then converted to acceleration via Newton's second law (`a = F/m`).

        Earth's gravity (`-GRAVITY_MPS2` along world z) is then added,
        since it acts on the rocket regardless of any control-generated
        force. This assumes the world frame's z-axis is vertical (up).

        IMPORTANT INVARIANT: this is only physically correct if the
        rocket's body z-axis (its long/roll axis) is aligned with world z
        at simulation start -- i.e. the rocket starts pointing straight up
        (a vertical rail launch). This holds today only as a consequence
        of `reset()` setting `self.q` to the identity quaternion, which
        means "body frame == world frame at t=0" by construction; nothing
        else in this class checks or enforces it. If a non-vertical
        starting attitude is ever needed, `self.q` must be initialized to
        the rotation that aligns the rocket's actual starting orientation
        with the world frame -- gravity's own direction below does not
        change, but the identity-quaternion assumption that makes it
        correct would need to be replaced.

        Note: this only computes and stores `xAcc_mps2/yAcc_mps2/
        zAcc_mps2` -- it does NOT integrate them into velocity or
        position. That integration is a deliberately separate, future
        step.

        Args:
            xForce_N (float): Net body-frame x-force (yaw axis) in
                Newtons, summed across all controls' forces this step.
            yForce_N (float): Net body-frame y-force (pitch axis) in
                Newtons, summed across all controls' forces this step.
            zForce_N (float): Net body-frame z-force (roll/length axis) in
                Newtons, summed across all controls' forces this step.

        Returns:
            None
        """
        w, x, y, z = self.q
        rotation = Rotation.from_quat([x, y, z, w])  # scipy wants scalar-last order
        xForce_world_N, yForce_world_N, zForce_world_N = rotation.apply([xForce_N, yForce_N, zForce_N])

        self.xAcc_mps2 = xForce_world_N / self.mass_kg
        self.yAcc_mps2 = yForce_world_N / self.mass_kg
        self.zAcc_mps2 = zForce_world_N / self.mass_kg

        self.zAcc_mps2 -= self.GRAVITY_MPS2

    def _applyTorques(self, yawTorque_Nm: float, pitchTorque_Nm: float, rollTorque_Nm: float,
        Ix_kgm2: float, Iy_kgm2: float, Iz_kgm2: float
    ) -> None:
        """
        Apply torques given in the rocket's body reference frame to update angular velocity
        and integrate the absolute orientation quaternion.

        Angular acceleration is computed per-axis via the full Euler rigid-body rotation
        equation (I*omega_dot + omega x I*omega = torque), including the gyroscopic
        cross-coupling term between axes -- unlike a decoupled `torque / I` approximation,
        this correctly captures the tendency of a spinning body to precess when torqued about
        an axis other than its spin axis. This resulting body-frame angular velocity is then
        used to propagate the orientation quaternion self.q via the standard quaternion
        kinematic equation q_dot = 0.5 * q (x) [0, omega], integrated with a forward-Euler
        step and renormalized to counteract numerical drift.

        Axis convention: body x = yaw, body y = pitch, body z = roll.

        Note: the cross-coupling term above uses the same effective Ix_kgm2/Iy_kgm2/Iz_kgm2
        passed in for the direct torque/I division -- i.e. it includes any control's
        additional spin inertia (see Rocket.sim()). This treats a spinning internal mass
        (e.g. ReactionWheel's flywheel) as rigidly co-rotating with the body for the purpose
        of this equation. A wheel spinning fast relative to the body also has its own
        gyroscopic precession effect on the *rocket* (angular momentum stored in the wheel
        resists changes to the rocket's yaw/pitch), which is a distinct, more advanced
        phenomenon not modeled here -- flagged as a known simplification, not silently
        ignored.

        Args:
            yawTorque_Nm (float): Torque about the body x-axis (yaw) in Newton-meters.
            pitchTorque_Nm (float): Torque about the body y-axis (pitch) in Newton-meters.
            rollTorque_Nm (float): Torque about the body z-axis (roll) in Newton-meters.
            Ix_kgm2 (float): Effective moment of inertia about the yaw axis this step
                (structure Ix_kgm2 plus any control's additionalYawInertia_kgm2).
            Iy_kgm2 (float): Effective moment of inertia about the pitch axis this step
                (structure Iy_kgm2 plus any control's additionalPitchInertia_kgm2).
            Iz_kgm2 (float): Effective moment of inertia about the roll axis this step
                (structure Iz_kgm2 plus any control's additionalRollInertia_kgm2).
        """
        # Full Euler's equation per axis: alpha = (torque - omega x I*omega) / I
        yawAcc_rps2   = (yawTorque_Nm   - (Iz_kgm2 - Iy_kgm2) * self.pitchVel_rps * self.rollVel_rps) / Ix_kgm2
        pitchAcc_rps2 = (pitchTorque_Nm - (Ix_kgm2 - Iz_kgm2) * self.rollVel_rps  * self.yawVel_rps)   / Iy_kgm2
        rollAcc_rps2  = (rollTorque_Nm  - (Iy_kgm2 - Ix_kgm2) * self.yawVel_rps   * self.pitchVel_rps) / Iz_kgm2

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

    def plotRoll(self, ax1:any=None, ax2:any=None) -> None:
        """
        Plots the recorded sim data

        Args:
            ax1 (any, optional): First External Plot Axis. Defaults to None.
            ax2 (any, optional): Second External Plot Axis. Defaults to None.
        """

        importedAxis = True

        if ax1 is None or ax2 is None:
            importedAxis = False
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), sharey=False)
            
        self.df['roll_deg']       = self.df.apply(lambda row: math.degrees(row.roll_rad), axis=1)
        self.df['targetRoll_deg'] = self.df.apply(lambda row: math.degrees(row.targetRoll_rad), axis=1)
        self.df['rollError_deg']  = self.df.apply(lambda row: math.degrees(row.rollError_rad), axis=1)
        ax1.plot(self.df.index, self.df['roll_deg'],       color='black', label='Rocket Roll Angle')
        ax1.plot(self.df.index, self.df['targetRoll_deg'], color='blue',  label='Target Roll Angle')
        ax1.plot(self.df.index, self.df['rollError_deg'],  color='red',   label='Roll Error')
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Angle (Degrees)')
        ax1.legend(loc='lower left')
        ax1.grid(True)

        self.df['rollVel_dps'] = self.df.apply(lambda row: math.degrees(row.rollVel_rps), axis = 1)
        ax2.plot(self.df.index, self.df['rollVel_dps'],  color='red',   label='Rocket Roll Velocity')
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Angular Speed (Degrees / Second)')
        ax2.legend(loc='lower left')
        ax2.grid(True)

        if not importedAxis:
            plt.tight_layout()
            plt.show()

        return