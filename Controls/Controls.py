import math

from .Force import Force


class MissingControlInputError(TypeError):
    """
    Raised when a `Controls` subclass's `sim()` is called without a
    keyword argument it requires (e.g. `Canards` without `canardAngle_deg`,
    `ReactionWheel` without `wheelSpeed_deg`).

    Subclassing `TypeError` (rather than introducing an unrelated exception
    type) keeps this catchable by any code already expecting `TypeError`
    from a bad `sim()` call, with a message that makes clear these are
    keyword-only application-level checks rather than interpreter errors.
    """

    def __init__(self, controlType: str, missingKwarg: str):
        """
        Args:
            controlType (str): The `controlType` of the `Controls` subclass
                that raised this (e.g. "Canards", "ReactionWheel").
            missingKwarg (str): The name of the required keyword argument
                that was not present in `sim(**kwargs)`.
        """
        super().__init__(f"{controlType}.sim() is missing required keyword argument '{missingKwarg}'")
# NOTE: intentionally an absolute import, not `from .Rocket import Rocket`.
# See the "CANONICAL NOTE ON THIS PROJECT'S CIRCULAR-IMPORT HANDLING" in
# Force.py's class docstring for why -- don't "fix" this to `.Rocket`.
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

    # Additional moment of inertia (kg*m^2) this control's own internal
    # rotating mass contributes to the rocket's total MMOI about each
    # axis, beyond the passive body structure captured in
    # `Rocket.Ix_kgm2/Iy_kgm2/Iz_kgm2`. Zero by default -- most controls
    # (e.g. `Canards`) have no internal spinning mass of their own. A
    # control that does (e.g. `ReactionWheel`'s flywheel) should override
    # whichever axis attribute matches the axis it spins about in
    # `__init__`. `Rocket.sim()` sums these across all attached controls
    # each step and adds them to the corresponding structure MMOI before
    # computing angular acceleration -- see `Rocket._applyTorques` and the
    # class docstring on `ReactionWheel` for the conservation-of-angular-
    # momentum reasoning behind this.
    additionalYawInertia_kgm2:   float = 0.0
    additionalPitchInertia_kgm2: float = 0.0
    additionalRollInertia_kgm2:  float = 0.0

    # Prepended to every kwarg name this control looks up via `_kwarg()`
    # (e.g. "canard1_" turns a lookup of "canardAngle_deg" into
    # "canard1_canardAngle_deg"). Empty by default -- most scenarios attach
    # only one instance of a given control class, so the plain kwarg name
    # (e.g. "canardAngle_deg") is unambiguous and no prefix is needed. Set
    # this (via the `kwargPrefix` constructor argument) whenever multiple
    # instances of the SAME control class are attached to one `Rocket` --
    # e.g. a fore and an aft `Canards`, or multiple TVC-gimbaled motors --
    # since `Rocket.sim(**kwargs)` forwards one shared kwargs dict to every
    # attached control, and two instances reading the same unprefixed
    # kwarg name would silently receive the same commanded value instead
    # of being commanded independently.
    kwargPrefix: str = ""

    def __init__(self, controlType: str, forceLocationX_m: float, forceLocationY_m: float, forceLocationZ_m: float,
        kwargPrefix: str = ""
    ):
        """
        Initialize a control object.

        Args:
            controlType (str): The name of the control type ex. Canards, ReactionWheel, etc.
            forceLocationX_m (float): The location that the force is acting on in the X axis in meters
            forceLocationY_m (float): The location that the force is acting on in the Y axis in meters
            forceLocationZ_m (float): The location that the force is acting on in the Z axis in meters
            kwargPrefix (str, optional): Prefix this control looks for on
                every kwarg it reads via `_kwarg()`/`_requireKwarg()`
                (e.g. `kwargPrefix="canard1_"` makes this instance read
                `canard1_canardAngle_deg` instead of `canardAngle_deg`).
                Leave as the default `""` unless multiple instances of the
                SAME control class are attached to one `Rocket` -- see the
                `kwargPrefix` class attribute docstring above. Include
                your own separator (e.g. the trailing `_`) in the prefix
                you pass; it's concatenated directly onto the kwarg name.
        """
        self.controlType = controlType

        self.forceLocationX_m = forceLocationX_m
        self.forceLocationY_m = forceLocationY_m
        self.forceLocationZ_m = forceLocationZ_m

        self.kwargPrefix = kwargPrefix

    def _requireKwarg(self, kwargs: dict, name: str):
        """
        Look up a required keyword argument from a control's `sim(**kwargs)`
        call, applying `self.kwargPrefix` if one was set at construction.

        Subclasses should use this (rather than indexing `kwargs[name]` or
        checking `name in kwargs` directly) for every kwarg they require,
        so that setting `kwargPrefix` at construction transparently
        disambiguates multiple instances of the same control class -- see
        the `kwargPrefix` class attribute docstring for why this matters.

        Args:
            kwargs (dict): The `**kwargs` dict passed into this control's
                `sim()`.
            name (str): The UNPREFIXED kwarg name this control wants (e.g.
                `"canardAngle_deg"`) -- `self.kwargPrefix` is prepended
                automatically before the lookup.

        Raises:
            MissingControlInputError: If the (possibly prefixed) kwarg is
                not present in `kwargs`.

        Returns:
            The value of the requested kwarg.
        """
        key = f"{self.kwargPrefix}{name}"

        if key not in kwargs:
            raise MissingControlInputError(self.controlType, key)

        return kwargs[key]

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