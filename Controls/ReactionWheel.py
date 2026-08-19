import math

import matplotlib.pyplot as plt
import pandas as pd

from .Controls import Controls, MissingControlInputError
from .Force import Force
# NOTE: intentionally an absolute import, not `from .Rocket import Rocket`.
# See the "CANONICAL NOTE ON THIS PROJECT'S CIRCULAR-IMPORT HANDLING" in
# Force.py's class docstring for why -- don't "fix" this to `.Rocket`.
from Rocket import Rocket

class ReactionWheel(Controls):
    """
    Reaction Wheel control implementation: computes roll torque via
    conservation of angular momentum as an internal flywheel is spun up or
    down toward a commanded speed.

    Each step, a target wheel speed -- relative to the rocket body, since a
    real motor controller has no external/inertial reference and can only
    command/measure its own shaft speed relative to the housing it's
    mounted in -- is commanded via `sim(wheelSpeed_deg=...)`. The wheel
    cannot reach that speed instantly -- its angular acceleration is
    limited to `maxAcceleration_rps2`, modeling a physically realistic
    motor rather than an ideal one (see the `speed_rps` setter). The torque
    reacted onto the rocket body is the equal-and-opposite reaction to the
    wheel's own angular acceleration: as the wheel spins up in one
    direction, the rocket body is pushed to spin the other way, so the
    underlying torque is `-I_kgm2 * dOmega_wheel/dt`.

    This torque is a genuine internal couple -- it has no real lever arm,
    unlike an external aerodynamic force offset from the CG (e.g.
    `Canards`). Since `Controls.sim()` returns forces rather than torques,
    `sim()` here represents that couple as two equal-and-opposite tangential
    forces at an arbitrary internal radius (`_COUPLE_RADIUS_M`, via the
    shared `Controls._tangentialForces` helper). This is deliberately
    non-physical -- `_COUPLE_RADIUS_M` doesn't correspond to any real wheel
    dimension, and any positive value reproduces the same net torque
    exactly, since the radius cancels out of `r x F` (a smaller radius just
    means a proportionally larger force magnitude, and vice versa). It's a
    stand-in chosen to keep every `Controls` subclass returning `Force`s
    rather than adding a separate torque-return path solely for this one
    control.

    Only roll torque is currently produced -- the two forces returned by
    `sim()` combine (once processed by `Rocket`) to zero net translational
    force and zero net yaw/pitch torque, leaving only roll -- consistent
    with the project's current focus on roll-only dynamics (see README).

    Conservation of angular momentum for a wheel-on-body system means the
    *total* roll-axis MMOI of the system is the rocket structure's own
    inertia PLUS the wheel's own spin inertia, not the structure's alone --
    the wheel's mass is still physically part of the rocket even while
    spinning relative to it. This control exposes that via
    `additionalRollInertia_kgm2` (set to `I_kgm2` in `__init__`, inherited
    from `Controls`); `Rocket.sim()` sums this across all attached controls
    and adds it to `Iz_kgm2` before converting this control's (and every
    other control's) roll torque into angular acceleration. Omitting this
    addition would systematically overestimate the rocket's roll
    acceleration whenever a `ReactionWheel` is attached.

    Attributes:
        I_kgm2 (float): Mass moment of inertia of the reaction wheel about
            its spin axis, in kg*m^2.
        maxAcceleration_rps2 (float): Maximum angular acceleration of the
            reaction wheel, in radians/sec^2 (converted from the
            `maxAcceleration_dps2` constructor argument).
        speed_rps (float): The wheel's current actual angular speed
            relative to the rocket body, in radians/sec (read-only view of
            `_speed_rps`; set the wheel's commanded speed through `sim()`,
            not directly).
        df (pd.DataFrame): Pandas DataFrame containing sim data.
    """

    I_kgm2: float
    maxAcceleration_rps2: float

    _speed_rps: float = 0.0
    _lastSpeed_rps: float = 0.0
    _dt: float = 0.0

    # Arbitrary, non-physical radius used only to express this control's
    # internal reaction couple as two Forces (see class docstring). Any
    # positive value gives an identical net torque once Rocket resolves
    # r x F, since the radius cancels out exactly.
    _COUPLE_RADIUS_M: float = 1.0

    # Type annotation only, no assignment -- df is a true instance
    # attribute, created fresh in __init__ below. A class-level
    # `df = pd.DataFrame(...)` assignment here would create ONE shared
    # DataFrame object across every ReactionWheel instance, silently
    # mutated in place by every instance's `sim()` via `self.df.loc[...] =`
    # (previously the case here -- see TODO.md).
    df: pd.DataFrame


    def __init__(self, I_kgm2: float, maxAcceleration_dps2: float, startingSpeed_dps: float):
        """
        Initialize the Reaction Wheel object.

        Args:
            I_kgm2 (float): Mass moment of inertia of the reaction wheel
                about its spin axis, in kg*m^2.
            maxAcceleration_dps2 (float): Maximum angular acceleration of
                the reaction wheel, in degrees/sec^2. Stored internally as
                `maxAcceleration_rps2` in radians/sec^2.
            startingSpeed_dps (float): Initial wheel speed at t=0, relative
                to the rocket body, in degrees/sec. Stored internally as
                `_speed_rps` (and `_lastSpeed_rps`) in radians/sec.
        """
        # forceLocation is (0, 0, 0) here since it isn't physically
        # meaningful for this control -- see class docstring on
        # _COUPLE_RADIUS_M for how the reaction torque is represented
        # instead.
        super().__init__("ReactionWheel", 0.0, 0.0, 0.0)

        self.I_kgm2 = I_kgm2

        # The wheel's own spin inertia is part of the total roll-axis MMOI
        # of the rocket+wheel system -- see class docstring. Read by
        # `Rocket.sim()` each step and added to `Iz_kgm2` before computing
        # roll acceleration.
        self.additionalRollInertia_kgm2 = I_kgm2

        self.maxAcceleration_rps2 = math.radians(maxAcceleration_dps2)
        self._speed_rps = math.radians(startingSpeed_dps)
        self._lastSpeed_rps = self._speed_rps  # so the first sim() step sees zero delta, not a huge one

        self._dt = 0.0

        # Instance-level DataFrame -- see class-level `df` annotation above
        # for why this must be created fresh here rather than shared via a
        # class attribute. Deliberately NOT set_index("time_s") here,
        # unlike Canards/Rocket's df -- preserving this class's existing
        # behavior unchanged; only the class-vs-instance scope is fixed.
        self.df = pd.DataFrame(columns=["time_s", "wheelSpeed_rps", "generateTorque_Nm"])

    def sim(self, rocket: Rocket, **kwargs) -> list[Force]:
        """
        Simulate the reaction wheel for one step and return the resulting
        forces.

        Updates the commanded wheel speed (subject to rate limiting, see
        the `speed_rps` setter) from `kwargs['wheelSpeed_deg']` -- this
        speed is relative to the rocket body (see class docstring) --
        computes the roll torque reacted onto the rocket body from the
        wheel's own angular acceleration this step (`-I_kgm2 *
        dOmega_wheel/dt`, per conservation of angular momentum — see class
        docstring), then represents that torque as two equal-and-opposite
        tangential forces via `Controls._tangentialForces` (see class
        docstring on `_COUPLE_RADIUS_M` for why).

        Args:
            rocket (Rocket): The rocket this control is attached to. Used
                for `rocket.simTimeStep`, stored as `_dt` and used both to
                rate-limit the wheel's speed change (in the `speed_rps`
                setter) and to compute the torque from that change here.
            **kwargs: Must include `wheelSpeed_deg` (float) — the commanded
                reaction wheel speed relative to the rocket body, in
                degrees/second, for this step.

        Raises:
            MissingControlInputError: If `wheelSpeed_deg` is not present in
                `kwargs`.

        Returns:
            list[Force]: Two tangential forces which, once resolved by
                `Rocket`, combine to zero net translational force and zero
                net yaw/pitch torque, leaving only the roll torque computed
                this step.
        """

        if not ("wheelSpeed_deg" in kwargs.keys()):
            raise MissingControlInputError(self.controlType, "wheelSpeed_deg")

        # _dt must be set BEFORE calling the speed_rps setter below, since
        # the setter's rate limit (maxAcceleration_rps2 * _dt) uses it.
        self._dt = rocket.simTimeStep
        self.speed_rps = math.radians(kwargs["wheelSpeed_deg"])

        dWheelSpeed_rps = self._speed_rps - self._lastSpeed_rps
        generatedTorque_Nm = -self.I_kgm2 * (dWheelSpeed_rps / self._dt)

        self.df.loc[rocket.simTime] = {
            "time_s": rocket.simTime,
            "wheelSpeed_rps": self._speed_rps,
            "generateTorque_Nm": generatedTorque_Nm
        }

        # Two opposing tangential forces at radius _COUPLE_RADIUS_M each
        # contribute torque = radius * magnitude (see _tangentialForces),
        # so with two forces the magnitude below reproduces
        # generatedTorque_Nm exactly regardless of the (arbitrary) radius.
        magnitude_N = generatedTorque_Nm / (2 * self._COUPLE_RADIUS_M)

        return self._tangentialForces(
            magnitude_N=magnitude_N,
            radius_m=self._COUPLE_RADIUS_M,
            z_m=0.0,
            numForces=2
        )

    @property
    def speed_rps(self) -> float:
        """float: The wheel's current actual angular speed relative to the rocket body, in radians/sec (read-only view of `_speed_rps`)."""
        return self._speed_rps

    @speed_rps.setter
    def speed_rps(self, targetWheelSpeed):
        """
        Set a new commanded wheel speed, subject to rate limiting.

        This models a physical motor rather than an ideal one: the wheel
        cannot jump instantly to `targetWheelSpeed`. Instead, the change
        from the current `_speed_rps` to the requested speed is limited to
        `maxAcceleration_rps2 * self._dt` per call (i.e. per simulation
        step), so `_dt` must already be set to the current step's
        `simTimeStep` before assigning here for the rate limit to be
        meaningful. `_lastSpeed_rps` is recorded before the update so
        `sim()` can compute the actual speed change achieved this step.

        Args:
            targetWheelSpeed (float): Requested wheel angular speed
                relative to the rocket body, in radians/sec. May be
                reached gradually over multiple steps rather than
                immediately, depending on `maxAcceleration_rps2` and `_dt`.
        """
        self._lastSpeed_rps = self._speed_rps

        maxChange = self.maxAcceleration_rps2 * self._dt

        delta = targetWheelSpeed - self._speed_rps
        delta = max(-maxChange, min(maxChange, delta))

        self._speed_rps = self._speed_rps + delta

    @property
    def speed_dps(self) -> float:
        """float: The wheel's current actual angular speed relative to the rocket body, in degrees/sec (derived from `_speed_rps`)."""
        return math.degrees(self._speed_rps)
    
    def plot(self, ax1:any=None, ax2:any=None) -> None:
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

        self.df['wheelSpeed_dps'] = self.df.apply(lambda row: math.degrees(row.wheelSpeed_rps), axis=1)
        ax1.plot(self.df.index, self.df['wheelSpeed_dps'], color='black', label='Wheel Speed')
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Reaction Wheel Angular Speed (Degrees / Second)')
        ax1.legend(loc='lower left')
        ax1.grid(True)

        ax2.plot(self.df.index, self.df['generateTorque_Nm'], color='red', label='Torque Generated by the Reaction Wheel')
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Torque (Newton Meter)')
        ax2.legend(loc='lower left')
        ax2.grid(True)
        
        if not importedAxis:
            plt.tight_layout()
            plt.show()

        return