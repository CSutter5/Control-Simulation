import math

from .Force import Force
# NOTE: intentionally an absolute import, not `from .Rocket import Rocket`.
# Controls/ and Rocket/ are sibling packages with no shared parent package,
# so a relative import cannot cross that boundary — this requires Rocket/
# to be installed/importable on sys.path (see pyproject.toml packages.find)
# and Rocket/__init__.py to expose the Rocket class. Mixing this up caused
# import errors previously; see project notes/TODO.md before "fixing" this
# to `.Rocket` again.
from Rocket import Rocket


class Controls:
    """
    Abstract base class for all rocket control methods (e.g. Canards,
    ReactionWheel).

    A Controls object is polled once per simulation step by `Rocket.sim()`,
    which calls `sim(rocket=self, **kwargs)` on every control in
    `Rocket.controls`, and converts every returned `Force` into torque
    (via `r x F` about the CG) and net translational force, summing both
    across all controls and all forces each control returns. Subclasses
    must override `sim()` to compute and return their own force
    contribution(s); the base implementation always raises
    NotImplementedError.

    To add a new control mechanism:
        1. Subclass `Controls` and call
           `super().__init__(<controlType>, forceLocationX_m, forceLocationY_m, forceLocationZ_m)`.
        2. Override `sim(self, rocket: Rocket, **kwargs)` to return a
           `list[Force]` -- one `Force` per distinct application point/
           direction this control produces this step. A control with a
           single, simple force contribution (e.g. a single reaction
           wheel) can just return a one-element list using its own
           `self.forceLocationX_m/Y_m/Z_m`.
        3. Accept whatever control-specific keyword arguments you need via
           `**kwargs` (e.g. `Canards` expects `canardAngle_deg`) — these are
           passed through unchanged from whatever the caller passes into
           `Rocket.sim(**kwargs)`.

    Raises:
        NotImplementedError: When `sim()` isn't overridden by the child object.

    Attributes:
        controlType (str): The name of the control type ex. Canards, ReactionWheel, etc.
    """

    controlType: str = "Base"

    forceLocationX_m: float
    forceLocationY_m: float
    forceLocationZ_m: float

    def __init__(self, controlType: str, forceLocationX_m: float, forceLocationY_m: float, forceLocationZ_m: float):
        """
        Initialize a control object.

        Args:
            controlType (str): The name of the control type ex. Canards, ReactionWheel, etc.
            forceLocationX_m (float): The location that the force is acting on in the X axis in meters
            forceLocationY_m (float): The location that the force is acting on in the Y axis in meters
            forceLocationZ_m (float): The location that the force is acting on in the Z axis in meters
        """
        self.controlType = controlType

        self.forceLocationX_m = forceLocationX_m
        self.forceLocationY_m = forceLocationY_m
        self.forceLocationZ_m = forceLocationZ_m

    def sim(self, rocket: Rocket, **kwargs) -> list[Force]:
        """
        Compute this control's force contribution(s) for the current step.

        Must be overridden by subclasses; this base implementation always
        raises NotImplementedError. Subclasses should use `rocket`'s
        current state (e.g. `rocket.zVel_mps`, `rocket.simTimeStep`,
        `rocket.airDensity`) and any control-specific values passed via
        `**kwargs` to compute the force(s) this control generates this
        step.

        Args:
            rocket (Rocket): The rocket this control is attached to, giving
                access to current state such as velocity, air density, and
                `simTimeStep`.
            **kwargs: Control-specific keyword arguments forwarded from
                `Rocket.sim(**kwargs)` (e.g. `Canards` requires
                `canardAngle_deg`).

        Raises:
            NotImplementedError: Always, unless overridden by a subclass.

        Returns:
            list[Force]: One `Force` per distinct application point this
                control produces this step (most controls will return a
                single-element list).
        """

        raise NotImplementedError(f"Please implement the sim function in {self.controlType}")

    def _tangentialForces(self, magnitude_N: float, radius_m: float, z_m: float,
        numForces: int, startAngle_rad: float = 0.0
    ) -> list[Force]:
        """
        Build `numForces` forces evenly spaced around a circle of
        `radius_m` in the body's x-y plane, each pointing tangent to that
        circle (perpendicular to its own radius vector) at `z_m` along the
        body.

        For `numForces >= 2`, the resulting set has zero net translational
        force and zero net yaw/pitch torque, leaving only roll torque -- a
        pure couple. This is shared by any control whose physical model is
        several identical forces placed symmetrically around the body
        (`Canards`), and also used by `ReactionWheel` as a deliberately
        non-physical device: a reaction wheel's torque is an internal
        couple with no real lever arm, but returning two such forces at an
        arbitrary `radius_m` reproduces the same net roll torque exactly
        (the radius cancels out of `r x F`), letting it stay within the
        Force-only contract without inventing fake real-world geometry
        elsewhere.

        Args:
            magnitude_N (float): Signed force magnitude shared by every
                generated force.
            radius_m (float): Distance from the body centerline to each
                force's application point. Must be nonzero.
            z_m (float): z-coordinate (along the body) shared by every
                force.
            numForces (int): Number of forces to generate, evenly spaced
                around the circle.
            startAngle_rad (float, optional): Angle of the first force;
                the rest are spaced evenly from there. Defaults to 0.0.

        Returns:
            list[Force]: `numForces` forces, each tangent to the circle at
                its own location.
        """
        forces: list[Force] = []

        for i in range(numForces):
            angle = startAngle_rad + i * (2 * math.pi / numForces)

            x_m = radius_m * math.cos(angle)
            y_m = radius_m * math.sin(angle)

            tangentX = -math.sin(angle)
            tangentY =  math.cos(angle)

            forces.append(Force(
                vector_N=(magnitude_N * tangentX, magnitude_N * tangentY, 0.0),
                location_m=(x_m, y_m, z_m)
            ))

        return forces