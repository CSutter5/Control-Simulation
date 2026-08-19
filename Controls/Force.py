from dataclasses import dataclass


@dataclass
class Force:
    """
    A single force applied at a point on the rocket body.

    Both the vector and the location are expressed in the rocket's body
    frame (body x = yaw axis, body y = pitch axis, body z = roll/length
    axis). `location_m`'s z-component follows the same "measured from the
    same reference point as `Rocket.CG_m`" convention used by
    `forceLocationZ_m` elsewhere in this project.

    A control's `sim()` returns a `list[Force]` rather than a single force
    so that one control can emit several forces at different
    locations/directions in a single step. This matters whenever a control
    is physically made of multiple force-generating surfaces placed at
    different points (e.g. several canard fins around the body): a
    symmetric arrangement can produce a net torque (e.g. pure roll) while
    the net *force* cancels to zero -- a pure couple. That can only be
    represented correctly by summing `r x F` per individual force *before*
    the cancellation happens; a single aggregate (force, location) pair
    can't reproduce a pure couple, since zero net force there would also
    force zero net torque.

    Deliberately kept in its own module with no dependency on `Rocket` or
    `Controls` -- `Controls.py` imports `Rocket`, and `Rocket.py` needs
    `Force`, so if `Force` lived inside `Controls.py` it wouldn't be
    defined yet when that circular chain is mid-import.

    CANONICAL NOTE ON THIS PROJECT'S CIRCULAR-IMPORT HANDLING (referenced
    from `Controls.py`, `Canards.py`, and `ReactionWheel.py` -- update
    here, not there, if this ever changes):
    `Controls/` and `Rocket/` are sibling packages with no shared parent
    package, so `Controls.py`/`Canards.py`/`ReactionWheel.py` import
    `Rocket` with `from Rocket import Rocket` (absolute), NOT
    `from .Rocket import Rocket` (relative) -- a relative import cannot
    cross a sibling-package boundary that has no common parent package to
    resolve through. This requires `Rocket/` to be installed/importable on
    `sys.path` (see `pyproject.toml`'s `packages.find`) and
    `Rocket/__init__.py` to expose the `Rocket` class. Combined with
    `Force` living in its own dependency-free module as described above,
    this is what breaks the `Controls` <-> `Rocket` <-> `Force` import
    cycle. Mixing the absolute/relative import up in any of the three
    files above caused real import errors previously -- see this note
    before "fixing" any of them back to a relative import.

    Attributes:
        vector_N (tuple[float, float, float]): Force vector (xForce_N,
            yForce_N, zForce_N) in Newtons, in the body frame.
        location_m (tuple[float, float, float]): Point of application
            (x_m, y_m, z_m) in the body frame, in meters.
    """

    vector_N: tuple[float, float, float]
    location_m: tuple[float, float, float]

    def decomposeForce(self) -> tuple[float, float, float]:
        """
        Return this force's translational (location-independent)
        contribution.

        For a rigid body, a force's contribution to linear acceleration of
        the CG doesn't depend on where on the body it's applied -- only its
        contribution to rotation does (see `Rocket._torqueFromForce`, which
        does use `location_m`). Lives here rather than on `Rocket` since it
        needs nothing but the force itself -- no rocket state (e.g. `CG_m`)
        is involved.

        Returns:
            tuple[float, float, float]: `vector_N` unchanged -- to be
                summed with other forces into a net body-frame force and
                consumed by `Rocket._applyForces`.
        """
        return self.vector_N