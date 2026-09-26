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
            velocity in meters and meters/sec. Each axis is independently
            either "rails" (read directly from `simData`, see
            `_resolveTranslationModes`) or "integrated" (advanced each
            step from `xAcc_mps2`/`yAcc_mps2`/`zAcc_mps2` via
            semi-implicit Euler in `_integrateTranslation`), depending on
            whether `simData` has a `{axis}Pos_m` and/or `{axis}Vel_mps`
            column for that axis. Both a position AND a velocity column
            may be present for the same axis -- both are then read
            directly with no cross-derivation between them, on the
            assumption that a real flight profile's independently
            supplied position and velocity tracks are each accurate on
            their own. `initialPos_m`/`initialVel_mps` (constructor
            arguments) seed the starting value for axes that are NOT
            rails-driven; rails axes get their initial value from
            `simData` itself, at `simTime=0`.
        xForce_N/yForce_N/zForce_N (float): Net body-frame force left over
            each step after each control's force has had its torque
            contribution extracted (`_torqueFromForce`/
            `Force.decomposeForce`). Recomputed every `sim()` call and
            converted to world-frame acceleration by `_applyForces`, then
            consumed by `_integrateTranslation` for whichever axes are not
            rails-driven (see `xPos_m`/etc. below).
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
    # then consumed by `_integrateTranslation` for non-rails axes.
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

    # Per-axis ('x', 'y', 'z' keys) bools: whether simData has a
    # '{axis}Pos_m'/'{axis}Vel_mps' column, resolved once by
    # `_resolveTranslationModes` (called from __init__). See that
    # method's docstring for the four (hasPos, hasVel) combinations this
    # drives in `_updateAllSimData`/`_integrateTranslation`.
    _hasPosCol: dict
    _hasVelCol: dict

    # Per-axis presorted (time, value) numpy array pairs for whichever
    # position/velocity columns are present, used by
    # `_interpTranslationData` (np.interp -- see that method and
    # `_resolveTranslationModes`). Only populated for axes where the
    # corresponding `_hasPosCol`/`_hasVelCol` entry is True.
    _posInterpData: dict
    _velInterpData: dict

    # Whether simData has a 'CG_m' column, and its cached (time, value)
    # interpolation pair if so -- see _resolveTranslationModes for why
    # CG_m specifically is interpolated rather than read via the
    # nearest-neighbor _getSimData used for the other CSV-shadowed
    # constants.
    _hasCGCol: bool
    _cgInterpData: tuple

    # Initial (x, y, z) position/velocity for axes that are NOT
    # rails-driven, set from the `initialPos_m`/`initialVel_mps`
    # constructor arguments and consumed by `reset()`. Ignored for any
    # axis where `_hasPosCol`/`_hasVelCol` is True, since rails axes get
    # their initial value from `simData` itself at simTime=0 instead.
    _initialPos_m: tuple
    _initialVel_mps: tuple

    df: pd.DataFrame

    def __init__(self, simDataPath: str, 
        Ix_kgm2: float, Iy_kgm2: float, Iz_kgm2: float, CG_m: float,
        r_m: float, length_m: float, mass_kg:float, 
        targetFunc: Callable[[float], tuple[float, float, float, float, float, float]], 
        simTimeStep: float, controls: list[Controls],
        initialPos_m: tuple[float, float, float] = (0.0, 0.0, 0.0),
        initialVel_mps: tuple[float, float, float] = (0.0, 0.0, 0.0)
    ):
        """
        Initialize the Rocket and load its environment/flight data.

        Args:
            simDataPath (str): Path to a CSV containing time-indexed
                environment/flight data (must include a 'time' column;
                columns such as 'airDensity'/'mass_kg'/'Ix_kgm2'/
                'Iy_kgm2'/'Iz_kgm2' are read via `_getSimData`
                (nearest-neighbor); 'CG_m' and
                'xPos_m'/'yPos_m'/'zPos_m'/'xVel_mps'/'yVel_mps'/
                'zVel_mps' are read via `_interpTranslationData` (linear
                interpolation) if present -- see
                `_resolveTranslationModes`).
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
            initialPos_m (tuple[float, float, float], optional): Starting
                (x, y, z) position in meters, in world frame, used by
                `reset()` for whichever axes are NOT rails-driven (see
                `_resolveTranslationModes`). Ignored for rails axes --
                those get their initial value from `simData` at
                simTime=0 instead. Defaults to (0.0, 0.0, 0.0).
            initialVel_mps (tuple[float, float, float], optional): Same as
                `initialPos_m`, but for starting velocity. Defaults to
                (0.0, 0.0, 0.0).

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

        self._initialPos_m  = initialPos_m
        self._initialVel_mps = initialVel_mps

        # _updateAllSimData overwrites CG_m/Ix_kgm2/Iy_kgm2/Iz_kgm2/mass_kg
        # from simData every step if the matching column exists, silently
        # ignoring the constructor value passed above after the first
        # sim() call. Warn here (once, at construction) rather than let
        # that happen with no indication -- e.g. a CSV column left over
        # from copy-pasting a different scenario's data file.
        for _shadowedArg, _column in (
            ("CG_m", "CG_m"), ("Ix_kgm2", "Ix_kgm2"),
            ("Iy_kgm2", "Iy_kgm2"), ("Iz_kgm2", "Iz_kgm2"),
            ("mass_kg", "mass_kg")
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

        # Resolves _hasPosCol/_hasVelCol per axis and presorts/caches any
        # interpolation arrays for translation rails columns that are
        # present -- see method docstring. Done once here (not re-checked
        # every step), same reasoning as TVC's thrust-curve presort.
        self._resolveTranslationModes()

        self.q = np.array([1.0, 0.0, 0.0, 0.0])

        self.df = pd.DataFrame(columns=self._DF_COLUMNS)
        self.df = self.df.set_index("time_s")

    def reset(self):
        """
        Reset the rocket to its initial simulation state.

        Zeroes `simTime`, reloads environment data for time 0 via
        `_updateAllSimData()` (which also sets `xPos_m`/`yPos_m`/`zPos_m`
        and `xVel_mps`/`yVel_mps`/`zVel_mps` directly for any rails axis,
        interpolated at simTime=0), resets the orientation quaternion to
        identity ([1, 0, 0, 0]), and zeroes all three angular velocities.

        For axes that are NOT rails-driven (see `_resolveTranslationModes`
        / `_hasPosCol`/`_hasVelCol`), position and velocity are seeded from
        the `initialPos_m`/`initialVel_mps` constructor arguments instead
        -- this is what makes a second `reset()`+run on the same instance
        reproducible rather than carrying over leftover state from a
        previous run. Net force (`xForce_N`/`yForce_N`/`zForce_N`) and
        acceleration (`xAcc_mps2`/`yAcc_mps2`/`zAcc_mps2`) are NOT reset
        here since both are fully recomputed from scratch at the start of
        every `sim()` step before they're read anywhere, so a stale value
        here would never actually be observed.

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

        # _updateAllSimData (above) already set position/velocity directly
        # for any rails axis (interpolated at simTime=0). Only seed the
        # remaining non-rails state here, from the constructor's
        # initialPos_m/initialVel_mps -- overwriting a rails axis here
        # would stomp the simData-derived t=0 value that was just set.
        # NOTE: velocity is seeded from initVel only when the axis has
        # NEITHER column -- a position-only rails axis (`hasPos` True,
        # `hasVel` False) still derives its velocity from the position
        # interpolator in `_updateAllSimData` and must NOT be overwritten
        # here, even though it has no velocity column of its own.
        for axis, posAttr, velAttr, initPos, initVel in (
            ('x', 'xPos_m', 'xVel_mps', self._initialPos_m[0], self._initialVel_mps[0]),
            ('y', 'yPos_m', 'yVel_mps', self._initialPos_m[1], self._initialVel_mps[1]),
            ('z', 'zPos_m', 'zVel_mps', self._initialPos_m[2], self._initialVel_mps[2]),
        ):
            if not self._hasPosCol[axis]:
                setattr(self, posAttr, initPos)
            if not self._hasVelCol[axis] and not self._hasPosCol[axis]:
                setattr(self, velAttr, initVel)

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

        Fixed step order, every call: (1) refresh environment/CSV-driven
        state, (2) poll every attached control, (3) apply the resulting
        physics. Concretely:

        Steps `simTime` forward, refreshes environment data via
        `_updateAllSimData()` (`zVel_mps`, `airDensity`, and any
        CSV-driven `CG_m`/`Ix_kgm2`/`Iy_kgm2`/`Iz_kgm2`/`mass_kg` columns),
        evaluates `targetFunc` for the new time, THEN polls every control
        in `controls` (passing `rocket=self` and any `**kwargs` through to
        each control's `sim()`) for a `list[Force]`. Because environment
        data is refreshed before controls are polled, a control reading
        `rocket.CG_m` (or the other CSV-driven attributes above) during its
        own `sim()` sees THIS step's value, not last step's -- e.g. a TVC
        control computing gimbal-relative-to-CG geometry can rely on this.
        `rocket.q` (attitude), by contrast, is NOT yet updated at this
        point -- it still reflects the end of the PREVIOUS step, since
        `_applyTorques` (which updates it) doesn't run until after every
        control has been polled -- see below.

        For every `Force` returned by a control, converts it to torque
        about the center of gravity (`_torqueFromForce`, via `r x F`) and
        its own translational contribution (`Force.decomposeForce`),
        summing both across every force from every control. Applies the
        net translational force via `_applyForces` (computing world-frame
        acceleration), THEN advances position/velocity for this step via
        `_integrateTranslation` (per-axis rails-vs-integrated -- see that
        method's docstring), THEN applies the aggregate torque via
        `_applyTorques` (updating `rocket.q` and angular velocity) -- the
        `_applyForces` / `_applyTorques` order is deliberate:
        `_applyForces` must rotate this step's force using the rocket's
        attitude from the START of the step, before `_applyTorques`
        updates that attitude, or the two would mix old and new attitude
        within a single timestep. `_integrateTranslation` only needs this
        step's already-computed acceleration, so its position relative to
        `_applyTorques` doesn't matter -- it's placed right after
        `_applyForces` for locality. Finally updates the derived Euler
        angles and recomputes position/attitude error terms against the
        target state.

        Args:
            **kwargs: Forwarded unchanged to every control's `sim()` call.
                For example, a `Canards` control requires a
                `canardAngle_deg` keyword argument here (or a prefixed
                variant -- see `Controls.kwargPrefix` -- if more than one
                instance of the same control class is attached).

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

        # Advances xPos_m/yPos_m/zPos_m (and, for non-rails axes,
        # xVel_mps/yVel_mps/zVel_mps) by one step -- see method docstring
        # for the per-axis rails-vs-integrated behavior and the
        # rails-vs-physics-force conflict warning it performs. Needs
        # nothing from _applyTorques, so its position relative to that
        # call doesn't matter; placed here for locality with _applyForces,
        # whose xAcc_mps2/yAcc_mps2/zAcc_mps2 output it consumes.
        self._integrateTranslation()

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

        self.xPosError_m = self.targetXPos_m - self.xPos_m
        self.yPosError_m = self.targetYPos_m - self.yPos_m
        self.zPosError_m = self.targetZPos_m - self.zPos_m

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
        Refresh environment data and translational rails state from
        `simData` for the current `simTime`.

        Looks up 'airDensity' at the nearest available row in `simData`
        via `_getSimData` (nearest-neighbor) and assigns it to the
        corresponding instance attribute. Called once per `sim()` step
        (and by `reset()`) rather than integrated from first principles,
        since this data is treated as externally supplied flight/
        environment input rather than derived state.

        Also optionally refreshes `CG_m`/`Ix_kgm2`/`Iy_kgm2`/`Iz_kgm2`/
        `mass_kg` from matching `simData` columns, if present -- letting a
        scenario model these as changing over the burn (e.g. propellant
        depletion shifting CG and reducing mass together) rather than
        fixed constants. See the CSV-shadowing warning in `__init__` for
        the corresponding heads-up when a constructor argument is about to
        be overridden this way.

        Finally, for each axis ('x', 'y', 'z') that `_resolveTranslationModes`
        found a `{axis}Pos_m` and/or `{axis}Vel_mps` column for, refreshes
        that axis's rails state via `_interpTranslationData` (linear
        interpolation, NOT nearest-neighbor -- see that method's
        docstring for why rails translation data is treated differently
        from the nearest-neighbor lookups above):
            - Both columns present: position AND velocity are each read
              directly, independently, with no cross-derivation.
            - Only position present: position is read directly; velocity
              is derived via a central difference against the position
              interpolator itself (see `_resolveTranslationModes` for why
              this avoids a first-step special case).
            - Only velocity present: velocity is read directly; position
              is NOT touched here -- it's integrated from this rails
              velocity later, in `_integrateTranslation`.
            - Neither present: nothing is touched here at all -- both are
              left for `_integrateTranslation` to advance from computed
              acceleration.
        NOTE: this is a behavior change for `zVel_mps` specifically -- it
        was previously read via the nearest-neighbor `_getSimData` (like
        `airDensity` still is above); it's now folded into this same
        rails-interpolation path for consistency with `xVel_mps`/
        `yVel_mps`, since it's exactly the same kind of quantity (a
        rails-supplied translational axis value). If you're comparing
        against a previous run using a coarsely-sampled `zVel_mps` column,
        expect smoother (interpolated) values now rather than step-wise
        nearest-neighbor ones.

        `CG_m` is ALSO now interpolated (see `_hasCGCol`/`_cgInterpData`,
        cached by `_resolveTranslationModes`) rather than read via
        nearest-neighbor `_getSimData` -- this was a deliberate fix found
        during testing: `CG_m` feeds directly into `_torqueFromForce`'s
        lever arm every step, so a coarsely-sampled `CG_m` column read via
        nearest-neighbor produces a discontinuous torque jump at every
        sample boundary, which can drive an already marginally-stable
        attitude-control loop into a numerical blow-up rather than just
        perturbing it. `mass_kg`/`Ix_kgm2`/`Iy_kgm2`/`Iz_kgm2` are
        deliberately NOT changed -- see `_resolveTranslationModes` for why.

        Returns:
            None
        """

        if "airDensity" in self.simData: self.airDensity = self._getSimData('airDensity', self.simTime)

        if self._hasCGCol:              self.CG_m     = self._interpTranslationData(self._cgInterpData, self.simTime)
        if "Ix_kgm2" in self.simData:   self.Ix_kgm2  = self._getSimData('Ix_kgm2', self.simTime)
        if "Iy_kgm2" in self.simData:   self.Iy_kgm2  = self._getSimData('Iy_kgm2', self.simTime)
        if "Iz_kgm2" in self.simData:   self.Iz_kgm2  = self._getSimData('Iz_kgm2', self.simTime)
        if "mass_kg" in self.simData:   self.mass_kg  = self._getSimData('mass_kg', self.simTime)

        for axis, posAttr, velAttr in (
            ('x', 'xPos_m', 'xVel_mps'),
            ('y', 'yPos_m', 'yVel_mps'),
            ('z', 'zPos_m', 'zVel_mps'),
        ):
            hasPos = self._hasPosCol[axis]
            hasVel = self._hasVelCol[axis]

            if hasPos:
                setattr(self, posAttr, self._interpTranslationData(self._posInterpData[axis], self.simTime))

            if hasVel:
                setattr(self, velAttr, self._interpTranslationData(self._velInterpData[axis], self.simTime))
            elif hasPos:
                # Position-only rails axis: derive velocity via central
                # difference against the position interpolator itself,
                # not a backward difference against last step's stored
                # position -- this needs no special case at simTime=0,
                # since np.interp already clamps queries outside the
                # curve's time range to the first/last sample.
                timeArr, valueArr = self._posInterpData[axis]
                h = self.simTimeStep / 2.0
                posPlus_m  = float(np.interp(self.simTime + h, timeArr, valueArr))
                posMinus_m = float(np.interp(self.simTime - h, timeArr, valueArr))
                setattr(self, velAttr, (posPlus_m - posMinus_m) / (2 * h))

    def _getSimData(self, columnName: str, time: float) -> float:
        """
        Look up a value from `simData` at the row whose 'time' column is
        closest to the given time.

        Does not interpolate between rows — this is a nearest-neighbor
        lookup, so accuracy depends on how finely `simData` is sampled
        relative to `simTimeStep`.

        Args:
            columnName (str): Name of the column to read (e.g.
                'airDensity', 'CG_m').
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

    def _resolveTranslationModes(self) -> None:
        """
        Determine, once (called from `__init__`), whether each
        translational axis ('x', 'y', 'z') is rails-driven or must be
        integrated from computed acceleration, and presort/cache any
        interpolation data needed for the rails case -- including `CG_m`,
        for the reason below.

        For each axis, independently checks whether `simData` has a
        `{axis}Pos_m` column and/or a `{axis}Vel_mps` column, storing the
        result in `_hasPosCol[axis]`/`_hasVelCol[axis]`. Both may be
        present together for the same axis -- deliberately NOT an error:
        a real flight profile can supply an independently measured
        position track and velocity track for the same axis, and each is
        trusted on its own rather than one being derived from the other
        (derivation would only be *less* accurate than two direct
        measurements). See `_updateAllSimData`/`_integrateTranslation` for
        how each of the four (hasPos, hasVel) combinations is actually
        evaluated every step.

        Whichever columns are present get their (time, value) pairs
        presorted by time and cached in `_posInterpData[axis]`/
        `_velInterpData[axis]`, for `_interpTranslationData` to consume
        via `np.interp` every step -- same "sort once at construction"
        pattern as `TVC._thrustTime`/`_thrustMagnitude_N`, and for the
        same reason: `np.interp` requires monotonically increasing
        x-values, and `simData` isn't guaranteed to already be time-sorted.

        ALSO caches `CG_m`'s (time, value) pair the same way, if present,
        for `_updateAllSimData` to read via the same
        `_interpTranslationData` interpolation instead of the
        nearest-neighbor `_getSimData` used for the other CSV-shadowed
        constants (`Ix_kgm2`/`Iy_kgm2`/`Iz_kgm2`/`mass_kg`). This was a
        deliberate fix, not part of the original translation-integration
        design: `CG_m` feeds directly into `_torqueFromForce`'s lever arm
        every step, so a coarsely-sampled `CG_m` column read via
        nearest-neighbor produces a discontinuous torque jump at every
        sample boundary -- a periodic "kick" that can pump energy into an
        already marginally-stable/resonant attitude-control loop (see the
        TVC resonance discussion) rather than just perturbing it,
        eventually overflowing to a degenerate quaternion. `mass_kg`/
        `Ix_kgm2`/`Iy_kgm2`/`Iz_kgm2` are NOT changed here -- they weren't
        implicated in that failure, and changing their lookup behavior
        without a specific reason risks an unrelated behavior change; flag
        for discussion if a similar issue shows up with any of them.

        Called only from `__init__` (not `reset()`) since `simData` itself
        doesn't change between `reset()` calls on the same `Rocket`
        instance -- the resolved modes and cached arrays stay valid for
        the instance's lifetime.

        Returns:
            None
        """
        self._hasPosCol = {}
        self._hasVelCol = {}
        self._posInterpData = {}
        self._velInterpData = {}

        _sorted = self.simData.sort_values('time')
        _sortedTime = _sorted['time'].to_numpy()

        for axis in ('x', 'y', 'z'):
            posCol = f"{axis}Pos_m"
            velCol = f"{axis}Vel_mps"

            hasPos = posCol in self.simData
            hasVel = velCol in self.simData

            self._hasPosCol[axis] = hasPos
            self._hasVelCol[axis] = hasVel

            if hasPos:
                self._posInterpData[axis] = (_sortedTime, _sorted[posCol].to_numpy())

            if hasVel:
                self._velInterpData[axis] = (_sortedTime, _sorted[velCol].to_numpy())

        self._hasCGCol = "CG_m" in self.simData
        if self._hasCGCol:
            self._cgInterpData = (_sortedTime, _sorted["CG_m"].to_numpy())

    def _interpTranslationData(self, interpData: tuple, time: float) -> float:
        """
        Linearly interpolate a translation rails column (position or
        velocity) at the given time, against a presorted (time array,
        value array) pair from `_posInterpData`/`_velInterpData` (see
        `_resolveTranslationModes`).

        Uses `np.interp` directly, same reasoning as `TVC._getThrustData`:
        rails position/velocity is treated as a smooth physical quantity
        (a real trajectory), so interpolating between samples avoids
        introducing artificial steps a nearest-neighbor lookup would
        produce on a coarsely-sampled curve -- contrast with `_getSimData`,
        which stays nearest-neighbor for `airDensity`/`CG_m`/etc.

        Args:
            interpData (tuple): (time array, value array) pair, presorted
                by time -- one entry of `_posInterpData`/`_velInterpData`.
            time (float): Simulation time in seconds to look up.

        Returns:
            float: The linearly-interpolated value at `time`, clamped to
                the first/last curve value outside its time range
                (`np.interp`'s default, no extrapolation -- same behavior
                as `TVC`).
        """
        timeArr, valueArr = interpData
        return float(np.interp(time, timeArr, valueArr))

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

    def _integrateTranslation(self) -> None:
        """
        Advance position (and, for fully-integrated axes, velocity) by
        one `simTimeStep`, independently per axis, according to whichever
        of the four rails/integrated combinations that axis resolved to
        in `_resolveTranslationModes` (`_hasPosCol[axis]`/
        `_hasVelCol[axis]`):

            hasPos (True or False combined with hasVel) -- position was
                already set directly this step by `_updateAllSimData`
                (interpolated from `simData`); nothing to do here.
            hasVel and not hasPos -- position integrates the rails
                velocity that `_updateAllSimData` already set this step:
                `pos += vel * dt`.
            neither -- full physics integration via semi-implicit
                (symplectic) Euler: velocity updates from this step's
                `xAcc_mps2`/`yAcc_mps2`/`zAcc_mps2` (already computed by
                `_applyForces`, gravity included) FIRST, then position
                updates using that NEW velocity (not the pre-update
                velocity) in the same step -- this is what makes it
                "semi-implicit"/symplectic rather than plain forward
                Euler, and is what gives it much better long-run energy
                behavior under a sustained force like gravity.

        Other integration schemes considered, for future reference:
            - Forward Euler (position uses the OLD, pre-update velocity):
                simplest, matches the angular-velocity integration already
                used in `_applyTorques`, but leaks energy over many steps
                -- drifts worse than semi-implicit Euler at the same
                timestep under a sustained force.
            - Velocity Verlet / leapfrog: 2nd-order accurate and still
                symplectic, but needs acceleration evaluated at BOTH the
                start and end of the step -- would require a second
                `_applyForces` call mid-step, which doesn't fit this
                project's fixed "poll controls once per step" structure.
            - RK4: highest per-step accuracy, but needs 4 force
                evaluations per step -- same fit problem as Verlet, and
                overkill for a controls-testing sim rather than a
                high-precision trajectory tool.
            Semi-implicit Euler was chosen since it's the same cost/shape
            as forward Euler (one extra ordering choice, no extra force
            evaluations) while being symplectic.

        Also performs a rails-vs-physics conflict check: for any axis
        that is rails-driven (`hasPos` or `hasVel`) where this step's net
        decomposed force on that axis (`xForce_N`/`yForce_N`/`zForce_N`,
        already summed in `sim()` before `_applyForces` runs) is nonzero
        beyond a small epsilon, emits a `warnings.warn` (run continues --
        this is intentionally a warning, not an error, since rails data
        alongside a control that produces real translational force can be
        a deliberate choice, e.g. rails altitude from a real recorded
        flight while evaluating TVC's pitch/yaw tracking against it,
        accepting that the translational consequence of thrust is
        intentionally not reflected in that axis's rails position). This
        never fires for `Canards`/`ReactionWheel`, since both are designed
        to produce zero net translational force (pure couples -- see
        their class docstrings); it's specifically the TVC-style case
        (real net thrust, rails altitude) this is meant to catch.

        Returns:
            None
        """
        _FORCE_EPSILON_N = 1e-9

        axisConfig = (
            ('x', 'xPos_m', 'xVel_mps', 'xAcc_mps2', self.xForce_N),
            ('y', 'yPos_m', 'yVel_mps', 'yAcc_mps2', self.yForce_N),
            ('z', 'zPos_m', 'zVel_mps', 'zAcc_mps2', self.zForce_N),
        )

        for axis, posAttr, velAttr, accAttr, netForce_N in axisConfig:
            hasPos = self._hasPosCol[axis]
            hasVel = self._hasVelCol[axis]

            if (hasPos or hasVel) and abs(netForce_N) > _FORCE_EPSILON_N:
                warnings.warn(
                    f"Axis '{axis}' is rails-driven (from simData "
                    f"{self.simDataPath}) but a control generated "
                    f"{netForce_N:.6g} N of net translational force on "
                    f"this axis this step (simTime={self.simTime:.4f}s) "
                    f"-- the rails position/velocity will NOT reflect "
                    f"this force, so the simulated and rails-driven "
                    f"trajectories can diverge on this axis.",
                    stacklevel=2
                )

            if hasPos:
                continue  # already set directly by _updateAllSimData

            if hasVel:
                # Rails velocity, integrated position.
                vel = getattr(self, velAttr)
                setattr(self, posAttr, getattr(self, posAttr) + vel * self.simTimeStep)
                continue

            # Neither rails -- full physics integration, semi-implicit
            # Euler (see docstring above for why this order, not forward
            # Euler).
            acc = getattr(self, accAttr)
            newVel = getattr(self, velAttr) + acc * self.simTimeStep
            setattr(self, velAttr, newVel)
            setattr(self, posAttr, getattr(self, posAttr) + newVel * self.simTimeStep)

    def _applyTorques(self, yawTorque_Nm: float, pitchTorque_Nm: float, rollTorque_Nm: float,
        Ix_kgm2: float, Iy_kgm2: float, Iz_kgm2: float
    ) -> None:
        """
        Apply torques given in the rocket's body reference frame to update angular velocity
        and integrate the absolute orientation quaternion.

        Angular acceleration is computed per-axis via the full Euler rigid-body rotation
        equation (I*omega_dot + omega x I*omega = torque), including the gyroscopic
        cross-coupling term between axes, which captures the tendency of a spinning body
        to precess when torqued about an axis other than its spin axis. This resulting
        body-frame angular velocity is then
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

        Uses an intrinsic x-y-z rotation sequence. This can hit gimbal lock at certain
        attitudes (here, when the pitch angle approaches +/-90 deg) -- self.q itself
        has no such singularity, only this derived representation does.

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