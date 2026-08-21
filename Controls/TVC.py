import math

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from .Controls import Controls, MissingControlInputError
from .Force import Force
# NOTE: intentionally an absolute import, not `from .Rocket import Rocket`.
# See the "CANONICAL NOTE ON THIS PROJECT'S CIRCULAR-IMPORT HANDLING" in
# Force.py's class docstring for why -- don't "fix" this to `.Rocket`.
from Rocket import Rocket


class TVC(Controls):
    """
    Thrust Vector Control implementation: computes pitch/yaw (and, for an
    off-centerline mount combined with other TVC instances, roll) torque
    by gimbaling the rocket's thrust vector off the body's roll (z) axis.

    Each step, a commanded pitch and yaw gimbal angle -- relative to the
    body frame, since a real gimbal actuator has no external/inertial
    reference and can only command/measure its own deflection relative to
    the mount it's attached to -- are supplied via
    `sim(gimbalPitch_deg=..., gimbalYaw_deg=...)`. Neither axis can reach
    its commanded angle instantly -- each is rate-limited and clamped
    independently (see the `pitch_rad`/`yaw_rad` setters), modeling a
    physically realistic gimbal actuator rather than an ideal one.

    Thrust magnitude is not commanded directly -- it's sampled each step
    from a time-indexed thrust-curve CSV (`thrustCurvePath`), via the same
    nearest-time lookup convention as `Rocket._getSimData` (no
    interpolation between rows).

    `TVC` returns a single `Force`: the full thrust vector, rotated off
    the body z-axis by the current gimbal deflection, applied at the
    nozzle's fixed body-frame location. This is the first control in the
    project expected to produce nonzero pitch/yaw torque -- exercising the
    gyroscopic cross-coupling in `Rocket._applyTorques` for the first time
    in a real (non-roll-only) maneuver.

    Mount placement (`forceLocationX_m`/`forceLocationY_m`): defaults to
    (0.0, 0.0) -- a single engine on the centerline, which by construction
    produces zero roll torque regardless of gimbal deflection (since
    `Rocket._torqueFromForce`'s `rollTorque_Nm = rx*fy - ry*fx` term is
    zero whenever rx=ry=0). Passing a nonzero mount offset -- e.g. for a
    multi-engine cluster, using several `TVC` instances each with a
    different `kwargPrefix` (see `Controls.kwargPrefix`) at different
    (forceLocationX_m, forceLocationY_m) positions, gimbaled
    independently -- lets a nonzero rx/ry combine with this engine's own
    gimbal-tilted Fx/Fy to produce real roll torque, with no additional
    torque math required anywhere: it falls out of the existing
    centralized `r x F` in `Rocket._torqueFromForce` automatically.

    Rotation convention (motor frame -> body frame): intrinsic rotation,
    yaw (about body x) applied first, then pitch (about the new y). This
    rotation is relative to the body's own x/y axes regardless of this
    engine's mount position -- an engine's gimbal direction does not
    depend on where around the body it's mounted. At zero deflection this
    reduces to the nominal thrust
    vector (0, 0, thrust_N), i.e. thrust pushes the rocket in +z
    (nose-ward), consistent with the `forceLocationZ_m` convention used
    elsewhere in this project (more negative = further aft of the CG
    reference point):

        Fx =  thrust_N * sin(pitch_rad)
        Fy = -thrust_N * cos(pitch_rad) * sin(yaw_rad)
        Fz =  thrust_N * cos(pitch_rad) * cos(yaw_rad)

    Note on translational integration: this control produces a real net
    Fz (thrust along the body's roll axis), but `Rocket._applyForces`
    does not yet integrate net force into velocity/position -- see
    `TranslationIntegrationToDo.md`. TVC's pitch/yaw (and, for an
    off-center mount, roll) *torque* and therefore attitude response is
    fully modeled regardless; only the translational consequence of
    thrust is deferred.

    Attributes:
        thrustCurvePath (str): Path to a time-indexed CSV of thrust
            magnitude (must include a 'time' column and a 'thrust_N'
            column), sampled via nearest-time lookup each step.
        maxAngle_rad (float): Max pitch deflection angle, in radians.
            Shared by yaw unless overridden at construction.
        rateLimit_rps (float): Max pitch angular slew rate, in
            radians/sec. Shared by yaw unless overridden at construction.
        yawMaxAngle_rad (float): Max yaw deflection angle, in radians.
            Equal to `maxAngle_rad` unless overridden at construction.
        yawRateLimit_rps (float): Max yaw angular slew rate, in
            radians/sec. Equal to `rateLimit_rps` unless overridden at
            construction.
        pitch_rad (float): Current actual pitch gimbal angle relative to
            the body frame, in radians (read-only view of `_pitch_rad`;
            set the commanded angle through `sim()`, not directly).
        yaw_rad (float): Current actual yaw gimbal angle relative to the
            body frame, in radians (read-only view of `_yaw_rad`; set the
            commanded angle through `sim()`, not directly).
        df (pd.DataFrame): Pandas DataFrame containing sim data.
    """

    thrustCurvePath: str
    _thrustData:     pd.DataFrame

    maxAngle_rad:     float
    rateLimit_rps:    float
    yawMaxAngle_rad:  float
    yawRateLimit_rps: float

    _pitch_rad: float = 0.0
    _yaw_rad:   float = 0.0
    _dt:        float = 0.0

    # Type annotation only, no assignment -- df is a true instance
    # attribute, created fresh in __init__ below. A class-level
    # `df = pd.DataFrame(...)` assignment here would create ONE shared
    # DataFrame object across every TVC instance (see the same fix
    # applied to Canards/ReactionWheel, and TODO.md).
    df: pd.DataFrame

    def __init__(self, thrustCurvePath: str, forceLocationZ_m: float,
        maxAngle_deg: float, rateLimit_dps: float,
        forceLocationX_m: float = 0.0, forceLocationY_m: float = 0.0,
        yawMaxAngle_deg: float = None, yawRateLimit_dps: float = None,
        kwargPrefix: str = ""
    ):
        """
        Initialize the TVC object.

        Args:
            thrustCurvePath (str): Path to a time-indexed CSV of thrust
                magnitude (a 'time' column and a 'thrust_N' column).
                Sampled each step via linear interpolation against
                'time', since a thrust curve is expected to be a smooth
                continuous function of time and interpolating avoids
                introducing artificial steps. Rows are sorted by 'time'
                once at construction; the CSV itself need not already be
                sorted.
            forceLocationZ_m (float): Nozzle location along the body,
                measured from the same reference point as `Rocket.CG_m`.
            maxAngle_deg (float): Max pitch gimbal deflection, in degrees.
                Applied to yaw as well unless `yawMaxAngle_deg` is given.
            rateLimit_dps (float): Max pitch gimbal slew rate, in
                degrees/sec. Applied to yaw as well unless
                `yawRateLimit_dps` is given.
            forceLocationX_m (float, optional): Engine mount offset from
                the body centerline along x, in meters. Defaults to 0.0
                (centerline mount, zero roll authority on its own -- see
                class docstring for how a nonzero offset combined with
                multiple `TVC` instances enables roll control).
            forceLocationY_m (float, optional): Engine mount offset from
                the body centerline along y, in meters. Defaults to 0.0,
                same reasoning as `forceLocationX_m`.
            yawMaxAngle_deg (float, optional): Max yaw gimbal deflection,
                in degrees. Defaults to `maxAngle_deg` if not given.
            yawRateLimit_dps (float, optional): Max yaw gimbal slew rate,
                in degrees/sec. Defaults to `rateLimit_dps` if not given.
            kwargPrefix (str, optional): See `Controls.kwargPrefix` -- set
                this (e.g. `"engine1_"`) if attaching more than one `TVC`
                instance to the same `Rocket` (e.g. a multi-engine
                vehicle), so each can be commanded independently via
                `rocket.sim(engine1_gimbalPitch_deg=..., ...)` instead of
                colliding on shared `gimbalPitch_deg`/`gimbalYaw_deg`.
        """
        super().__init__("TVC", forceLocationX_m, forceLocationY_m, forceLocationZ_m, kwargPrefix=kwargPrefix)
        
        self.thrustCurvePath = thrustCurvePath
        self._thrustData = pd.read_csv(self.thrustCurvePath)

        if "thrust_N" not in self._thrustData:
            raise KeyError(
                f"'thrust_N' is not a column in the thrust curve data "
                f"(loaded from {self.thrustCurvePath}) -- TVC requires a "
                f"'thrust_N' column indexed by 'time'."
            )

        # Sorted once here (rather than in _getThrustData every step)
        # since np.interp requires its xp array to be monotonically
        # increasing -- the source CSV isn't guaranteed to already be in
        # time order.
        _sorted = self._thrustData.sort_values('time')
        self._thrustTime        = _sorted['time'].to_numpy()
        self._thrustMagnitude_N = _sorted['thrust_N'].to_numpy()

        self.maxAngle_rad  = math.radians(maxAngle_deg)
        self.rateLimit_rps = math.radians(rateLimit_dps)

        self.yawMaxAngle_rad  = math.radians(yawMaxAngle_deg) if yawMaxAngle_deg is not None else self.maxAngle_rad
        self.yawRateLimit_rps = math.radians(yawRateLimit_dps) if yawRateLimit_dps is not None else self.rateLimit_rps

        self._pitch_rad = 0.0
        self._yaw_rad   = 0.0
        self._dt = 0.0


        # Instance-level DataFrame -- see class-level `df` annotation
        # above for why this must be created fresh here rather than
        # shared via a class attribute.
        self.df = pd.DataFrame(columns=["time_s", "pitch_rad", "yaw_rad", "thrust_N"])
        self.df = self.df.set_index("time_s")

    @property
    def pitch_rad(self) -> float:
        """float: The gimbal's current actual pitch angle relative to the body frame, in radians (read-only view of `_pitch_rad`)."""
        return self._pitch_rad

    @pitch_rad.setter
    def pitch_rad(self, angle):
        """
        Set a new commanded pitch gimbal angle, subject to rate limiting
        and max-angle clamping.

        This models a physical gimbal actuator rather than an ideal one:
        the nozzle cannot jump instantly to `angle`. Instead, the change
        from the current `_pitch_rad` to the requested `angle` is limited
        to `rateLimit_rps * self._dt` per call (i.e. per simulation
        step), so `_dt` must be set to the current step's `simTimeStep`
        (done in `sim()`) before assigning here for the rate limit to be
        meaningful. The result is then clamped to +/-`maxAngle_rad`.

        Args:
            angle (float): Requested pitch gimbal angle in radians. May
                be reached gradually over multiple steps rather than
                immediately, depending on `rateLimit_rps` and `_dt`.
        """
        maxChange = self.rateLimit_rps * self._dt

        delta = angle - self._pitch_rad
        delta = max(-maxChange, min(maxChange, delta))

        self._pitch_rad = self._pitch_rad + delta
        self._pitch_rad = max(min(self._pitch_rad, self.maxAngle_rad), -self.maxAngle_rad)

    @property
    def yaw_rad(self) -> float:
        """float: The gimbal's current actual yaw angle relative to the body frame, in radians (read-only view of `_yaw_rad`)."""
        return self._yaw_rad

    @yaw_rad.setter
    def yaw_rad(self, angle):
        """
        Set a new commanded yaw gimbal angle, subject to rate limiting
        and max-angle clamping.

        Identical in behavior to the `pitch_rad` setter, but against
        `yawMaxAngle_rad`/`yawRateLimit_rps` instead of `maxAngle_rad`/
        `rateLimit_rps` -- these are equal to the pitch limits unless
        overridden at construction (see `__init__`).

        Args:
            angle (float): Requested yaw gimbal angle in radians. May be
                reached gradually over multiple steps rather than
                immediately, depending on `yawRateLimit_rps` and `_dt`.
        """
        maxChange = self.yawRateLimit_rps * self._dt

        delta = angle - self._yaw_rad
        delta = max(-maxChange, min(maxChange, delta))

        self._yaw_rad = self._yaw_rad + delta
        self._yaw_rad = max(min(self._yaw_rad, self.yawMaxAngle_rad), -self.yawMaxAngle_rad)

    def sim(self, rocket: Rocket, **kwargs) -> list[Force]:
        """
        Simulate the TVC gimbal for one step and return the resulting
        force.

        Updates the commanded pitch and yaw angles (subject to rate
        limiting, see the `pitch_rad`/`yaw_rad` setters) from
        `kwargs['gimbalPitch_deg']`/`kwargs['gimbalYaw_deg']` (or their
        `kwargPrefix`-prefixed forms -- see `Controls.kwargPrefix`),
        samples the current thrust magnitude from `_thrustData` via
        nearest-time lookup against `rocket.simTime` (same convention as
        `Rocket._getSimData`), then rotates the nominal (0, 0, thrust_N)
        thrust vector into the body frame using the current pitch/yaw
        deflection (intrinsic yaw-then-pitch rotation -- see class
        docstring for the exact formula) and returns it as a single
        `Force` applied at this engine's fixed mount location.

        This does not produce a cancelling couple -- the returned force
        carries real net translational and rotational content by design.

        Args:
            rocket (Rocket): The rocket this control is attached to. Used
                for `rocket.simTimeStep` (to rate-limit the gimbal angle
                changes) and `rocket.simTime` (to sample the thrust
                curve).
            **kwargs: Must include `gimbalPitch_deg` and `gimbalYaw_deg`
                (floats, subject to `kwargPrefix`) -- the commanded
                gimbal deflection angles relative to the body frame, in
                degrees, for this step.

        Raises:
            MissingControlInputError: If `gimbalPitch_deg` or
                `gimbalYaw_deg` (or their prefixed forms) is not present
                in `kwargs`.

        Returns:
            list[Force]: A single-element list containing the rotated
                thrust force applied at this engine's mount location.
        """
        self._dt = rocket.simTimeStep

        self.pitch_rad = math.radians(self._requireKwarg(kwargs, "gimbalPitch_deg"))
        self.yaw_rad   = math.radians(self._requireKwarg(kwargs, "gimbalYaw_deg"))

        thrust_N = self._getThrustData(rocket.simTime)

        Fx =  thrust_N * math.sin(self._pitch_rad)
        Fy = -thrust_N * math.cos(self._pitch_rad) * math.sin(self._yaw_rad)
        Fz =  thrust_N * math.cos(self._pitch_rad) * math.cos(self._yaw_rad)

        self.df.loc[rocket.simTime] = {
            "time_s":    rocket.simTime,
            "pitch_rad": self._pitch_rad,
            "yaw_rad":   self._yaw_rad,
            "thrust_N":  thrust_N
        }

        return [Force(
            vector_N=(Fx, Fy, Fz),
            location_m=(self.forceLocationX_m, self.forceLocationY_m, self.forceLocationZ_m)
        )]

    def _getThrustData(self, time: float) -> float:
        """
        Look up the thrust magnitude at the given time via linear
        interpolation between the two nearest rows of the (time-sorted)
        thrust curve.

        A thrust curve is a smooth physical quantity (chamber pressure
        ramping up/down), so interpolating between sample points avoids
        introducing artificial steps that a nearest-neighbor lookup would
        produce on a coarsely-sampled curve. `numpy.interp` is used directly
        against the presorted
        `_thrustTime`/`_thrustMagnitude_N` arrays (see `__init__`) rather
        than pandas, since `np.interp` requires monotonically increasing
        x-values and doing that sort once at construction is cheaper
        than repeating it every step.

        Args:
            time (float): Simulation time in seconds to look up.

        Returns:
            float: The linearly-interpolated thrust magnitude, in
                Newtons, at `time`. Clamped to the first/last curve value
                for times outside the curve's range (numpy.interp's
                default behavior -- no extrapolation).
        """
        return float(np.interp(time, self._thrustTime, self._thrustMagnitude_N))

    @property
    def pitch_deg(self) -> float:
        """float: The gimbal's current actual pitch angle in degrees (derived from `_pitch_rad`)."""
        return math.degrees(self._pitch_rad)

    @property
    def yaw_deg(self) -> float:
        """float: The gimbal's current actual yaw angle in degrees (derived from `_yaw_rad`)."""
        return math.degrees(self._yaw_rad)

    def plot(self, ax1: any = None, ax2: any = None) -> None:
        """
        Plots the recorded sim data.

        Args:
            ax1 (any, optional): First external plot axis -- pitch/yaw
                gimbal angle vs time. Defaults to None.
            ax2 (any, optional): Second external plot axis -- thrust
                magnitude vs time. Defaults to None.
        """
        importedAxis = True

        if ax1 is None or ax2 is None:
            importedAxis = False
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), sharey=False)

        self.df['pitch_deg'] = self.df.apply(lambda row: math.degrees(row.pitch_rad), axis=1)
        self.df['yaw_deg']   = self.df.apply(lambda row: math.degrees(row.yaw_rad), axis=1)
        ax1.plot(self.df.index, self.df['pitch_deg'], color='black', label='Gimbal Pitch')
        ax1.plot(self.df.index, self.df['yaw_deg'],   color='blue',  label='Gimbal Yaw')
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Angle (Degrees)')
        ax1.legend(loc='lower left')
        ax1.grid(True)

        ax2.plot(self.df.index, self.df['thrust_N'], color='red', label='Thrust')
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Thrust (Newtons)')
        ax2.legend(loc='lower left')
        ax2.grid(True)

        if not importedAxis:
            plt.tight_layout()
            plt.show()

        return