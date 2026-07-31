import math

from .Controls import Controls
# NOTE: intentionally an absolute import, not `from .Rocket import Rocket`.
# Controls/ and Rocket/ are sibling packages with no shared parent package,
# so a relative import cannot cross that boundary — this requires Rocket/
# to be installed/importable on sys.path (see pyproject.toml packages.find)
# and Rocket/__init__.py to expose the Rocket class. Mixing this up caused
# import errors previously; see project notes/TODO.md before "fixing" this
# to `.Rocket` again.
from Rocket import Rocket

class ReactionWheel(Controls):
    """
    Reaction Wheel control implementation: computes roll torque via
    conservation of angular momentum as an internal flywheel is spun up or
    down toward a commanded speed.

    Each step, a target wheel speed is commanded via
    `sim(wheelSpeed_deg=...)`. The wheel cannot reach that speed instantly —
    its angular acceleration is limited to `maxAcceleration_rps2`, modeling
    a physically realistic motor rather than an ideal one (see the
    `speed_rps` setter). The torque reacted onto the rocket body is the
    equal-and-opposite reaction to the wheel's own angular acceleration:
    as the wheel spins up in one direction, the rocket body is pushed to
    spin the other way, so the returned torque is
    `-I_kgm2 * dOmega_wheel/dt`.

    Only roll torque is currently produced — `sim()` always returns
    (0.0, 0.0, rollTorque_Nm) — consistent with the project's current focus
    on roll-only dynamics (see README).

    Attributes:
        I_kgm2 (float): Mass moment of inertia of the reaction wheel about
            its spin axis, in kg*m^2.
        maxAcceleration_rps2 (float): Maximum angular acceleration of the
            reaction wheel, in radians/sec^2 (converted from the
            `maxAcceleration_dps2` constructor argument).
        speed_rps (float): The wheel's current actual angular speed, in
            radians/sec (read-only view of `_speed_rps`; set the wheel's
            commanded speed through `sim()`, not directly).
    """

    I_kgm2: float
    maxAcceleration_rps2: float

    _speed_rps: float = 0.0
    _lastSpeed_rps: float = 0.0
    _dt: float = 0.0

    def __init__(self, I_kgm2: float, maxAcceleration_dps2: float, startingSpeed_dps: float):
        """
        Initialize the Reaction Wheel object.

        Args:
            I_kgm2 (float): Mass moment of inertia of the reaction wheel
                about its spin axis, in kg*m^2.
            maxAcceleration_dps2 (float): Maximum angular acceleration of
                the reaction wheel, in degrees/sec^2. Stored internally as
                `maxAcceleration_rps2` in radians/sec^2.
            startingSpeed_dps (float): Initial wheel speed at t=0, in
                degrees/sec. Stored internally as `_speed_rps` (and
                `_lastSpeed_rps`) in radians/sec.
        """
        super().__init__("ReactionWheel")

        self.I_kgm2 = I_kgm2
        self.maxAcceleration_rps2 = math.radians(maxAcceleration_dps2)
        self._speed_rps = math.radians(startingSpeed_dps)
        self._lastSpeed_rps = self._speed_rps  # so the first sim() step sees zero delta, not a huge one

        self._dt = 0.0

    def sim(self, rocket: Rocket, **kwargs) -> tuple[float, float, float]:
        """
        Simulate the reaction wheel for one step and return the resulting
        torque.

        Updates the commanded wheel speed (subject to rate limiting, see
        the `speed_rps` setter) from `kwargs['wheelSpeed_deg']`, then
        computes the roll torque reacted onto the rocket body from the
        wheel's own angular acceleration this step
        (`-I_kgm2 * dOmega_wheel/dt`, per conservation of angular
        momentum — see class docstring). Only roll torque is produced;
        yaw and pitch are always returned as 0.0.

        Args:
            rocket (Rocket): The rocket this control is attached to. Used
                for `rocket.simTimeStep`, stored as `_dt` and used both to
                rate-limit the wheel's speed change (in the `speed_rps`
                setter) and to compute the torque from that change here.
            **kwargs: Must include `wheelSpeed_deg` (float) — the commanded
                reaction wheel speed, in degrees/second, for this step.

        Raises:
            TypeError: If `wheelSpeed_deg` is not present in `kwargs`.

        Returns:
            tuple[float, float, float]: (0.0, 0.0, rollTorque_Nm) — the
                roll torque reacted onto the rocket body this step, in Nm.
        """

        if not ("wheelSpeed_deg" in kwargs.keys()):
            raise TypeError("missing 1 required positional argument: 'wheelSpeed_deg'")

        # _dt must be set BEFORE calling the speed_rps setter below, since
        # the setter's rate limit (maxAcceleration_rps2 * _dt) uses it.
        self._dt = rocket.simTimeStep
        self.speed_rps = math.radians(kwargs["wheelSpeed_deg"])

        dWheelSpeed_rps = self._speed_rps - self._lastSpeed_rps
        generatedTorque_Nm = -self.I_kgm2 * (dWheelSpeed_rps / self._dt)

        return (0.0, 0.0, generatedTorque_Nm)

    @property
    def speed_rps(self) -> float:
        """float: The wheel's current actual angular speed in radians/sec (read-only view of `_speed_rps`)."""
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
            targetWheelSpeed (float): Requested wheel angular speed in
                radians/sec. May be reached gradually over multiple steps
                rather than immediately, depending on
                `maxAcceleration_rps2` and `_dt`.
        """
        self._lastSpeed_rps = self._speed_rps

        maxChange = self.maxAcceleration_rps2 * self._dt

        delta = targetWheelSpeed - self._speed_rps
        delta = max(-maxChange, min(maxChange, delta))

        self._speed_rps = self._speed_rps + delta

    @property
    def speed_dps(self) -> float:
        """float: The wheel's current actual angular speed in degrees/sec (derived from `_speed_rps`)."""
        return math.degrees(self._speed_rps)