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
    defined yet when that circular chain is mid-import (see project
    notes/TODO.md before moving this back).

    Attributes:
        vector_N (tuple[float, float, float]): Force vector (xForce_N,
            yForce_N, zForce_N) in Newtons, in the body frame.
        location_m (tuple[float, float, float]): Point of application
            (x_m, y_m, z_m) in the body frame, in meters.
    """

    vector_N: tuple[float, float, float]
    location_m: tuple[float, float, float]