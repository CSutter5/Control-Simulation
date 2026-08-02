import math

from scipy.interpolate import LinearNDInterpolator
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from .Controls import Controls
from .Force import Force
# NOTE: intentionally an absolute import, not `from .Rocket import Rocket`.
# Controls/ and Rocket/ are sibling packages with no shared parent package,
# so a relative import cannot cross that boundary — this requires Rocket/
# to be installed/importable on sys.path (see pyproject.toml packages.find)
# and Rocket/__init__.py to expose the Rocket class. Mixing this up caused
# import errors previously; see project notes/TODO.md before "fixing" this
# to `.Rocket` again.
from Rocket import Rocket

class Canards(Controls):
    """
    Canards control implementation: computes roll torque by modeling
    aerodynamic lift on a set of canard fins deflected to a commanded angle.

    Lift is looked up from a CSV-based table of C_L values (indexed by angle
    of attack and velocity) via linear interpolation, giving a single lift
    magnitude shared by every fin (all fins share the same deflection angle
    and see the same flow). The commanded angle is rate-limited and clamped
    to `maxAngle_rad` on every write, modeling a physically realistic
    actuator (finite slew rate and maximum deflection) rather than an ideal
    one.

    `numCanards` fins are modeled as individually placed around the body,
    evenly spaced starting from the angle implied by
    (`forceLocationX_m`, `forceLocationY_m`), at a radius `hypot(forceLocationX_m,
    forceLocationY_m)`. Each fin's force is applied tangent to that circle
    (perpendicular to its own radius vector) rather than along a single
    fixed axis -- this is what makes a symmetric arrangement (numCanards >= 2)
    produce pure roll torque with the net yaw/pitch torque and net
    translational force canceling out, instead of the pitch/yaw contribution
    that a single aggregate application point would otherwise introduce
    (see `Controls.Force` docstring for why this needs a list of forces
    rather than one combined force).

    Attributes:
        airfoilDataPath (str): The path to a table of C_L for different velocities and AOA's
        root_m (float): Root coord of the canards in meters
        tip_m (float): Tip coord of the canards in meters
        span_m (float): Span of the canards in meters
        sweep_m (float): Sweep of the canards in meters
        numCanards (int): Number of canards
        maxAngle_rad (float): Max deflection angle of the canards in radians
        rateLimit_rps (float): Max angular speed of the canards in radians / sec
        updateFreq_hz (float): Update frequency of the canards in hz
        angle_rad (float): The canard current angle in radians
        df (pd.DataFrame): Pandas DataFrame containing sim data.
    """

    airfoilDataPath:    str
    _airfoilData:       pd.DataFrame
    _airfoilPoints:     list[float]
    _airfoilValues:     list[float]
    _liftInterpFunc:    any

    root_m:         float
    tip_m:          float
    span_m:         float
    sweep_m:        float
    _surfaceArea_m2: float

    numCanards: int

    _finDistance_m:        float
    _finAxialPlacement_rad: float

    maxAngle_rad:   float
    rateLimit_rps:  float
    updateFreq_hz:  float

    _angle_rad: float = 0.0
    _dt:        float = 0.0

    df = pd.DataFrame(columns=["time_s", "angle_rad", "generatedLift_n"])

    def __init__(self, airfoilDataPath: str, root_m: float, tip_m: float, span_m: float,
        forceLocationX_m: float, forceLocationY_m: float, forceLocationZ_m: float,
        sweep_m: float, numCanards: int, maxAngle_deg: float, rateLimit_dps: float,
        updateFreq_hz: float
    ):
        """
        Initialize the Canards object

        Args:
            airfoilDataPath (str): The path to a table of C_L for different velocities and AOA's
            forceLocationX_m (float): The location that the force is acting on in the X axis in meters
            forceLocationY_m (float): The location that the force is acting on in the Y axis in meters
            forceLocationZ_m (float): The location that the force is acting on in the Z axis in meters
            root_m (float): Root coord of the canards in meters
            tip_m (float): Tip coord of the canards in meters
            span_m (float): Span of the canards in meters
            sweep_m (float): Sweep of the canards in meters
            numCanards (int): Number of canards
            maxAngle_deg (float): Max deflection angle of the canards in degrees
            rateLimit_dps (float): Max angular speed of the canards in degrees / sec
            updateFreq_hz (float): Update frequency of the canards in hz

        Note:
            (forceLocationX_m, forceLocationY_m) is treated as the position
            of fin #0 in the body's x-y plane; it determines both the
            placement radius (`hypot(forceLocationX_m, forceLocationY_m)`)
            and the starting angle that the remaining `numCanards - 1` fins
            are evenly spaced from around that circle. `forceLocationZ_m`
            is shared by every fin (all fins sit at the same point along
            the body's length).
        """
        super().__init__("Canards", forceLocationX_m, forceLocationY_m, forceLocationZ_m)

        self._finDistance_m = math.hypot(forceLocationX_m, forceLocationY_m)
        self._finAxialPlacement_rad = math.atan2(forceLocationY_m, forceLocationX_m)

        self.airfoilDataPath = airfoilDataPath
        self._airfoilData     = pd.read_csv(self.airfoilDataPath)
        self._airfoilPoints   = self._airfoilData[['alpha', 'Velocity']].values
        self._airfoilValues   = self._airfoilData['CL'].values
        self._liftInterpFunc = LinearNDInterpolator(self._airfoilPoints, self._airfoilValues, fill_value=0.0)

        self.root_m  = root_m
        self.tip_m   = tip_m
        self.span_m  = span_m
        self.sweep_m = sweep_m

        self._surfaceArea_m2 = (self.root_m + self.tip_m) / 2 * self.span_m

        self.numCanards    = numCanards
        self.maxAngle_rad  = math.radians(maxAngle_deg)
        self.rateLimit_rps = math.radians(rateLimit_dps)
        self.updateFreq_hz = updateFreq_hz

        # Init default values
        self._angle_rad = 0.0
        self._dt = 0.0

        self.df = self.df.set_index("time_s")

    @property
    def angle_rad(self) -> float:
        """float: The canard's current actual angle in radians (read-only view of `_angle_rad`)."""
        return self._angle_rad

    @angle_rad.setter
    def angle_rad(self, angle):
        """
        Set a new commanded canard angle, subject to rate limiting and
        max-angle clamping.

        This models a physical actuator rather than an ideal one: the
        canard cannot jump instantly to `angle`. Instead, the change from
        the current `_angle_rad` to the requested `angle` is limited to
        `rateLimit_rps * self._dt` per call (i.e. per simulation step), so
        `_dt` must be set to the current step's `simTimeStep` (done in
        `sim()`) before assigning here for the rate limit to be meaningful.
        The result is then clamped to +/-`maxAngle_rad`.

        Args:
            angle (float): Requested canard angle in radians. May be
                reached gradually over multiple steps rather than
                immediately, depending on `rateLimit_rps` and `_dt`.
        """
        maxChange = self.rateLimit_rps * self._dt # Calculate the max rate of change

        self.angleRate_rps = angle - self._angle_rad # Calculate the target rate of change
        self.angleRate_rps = max(-maxChange, min(maxChange, self.angleRate_rps)) # Clamp the rate of change
        self._angle_rad += self.angleRate_rps

        self._angle_rad = max(min(self._angle_rad, self.maxAngle_rad), -self.maxAngle_rad) # Clamp output

    def sim(self, rocket: Rocket, **kwargs) -> list[Force]:
        """
        Simulate the canards for one step and return the resulting forces.

        Updates the commanded angle (subject to rate limiting, see
        `angle_rad` setter) from `kwargs['canardAngle_deg']`, computes a
        single per-fin lift magnitude for the rocket's current vertical
        velocity and air density via `__calculateFinLift` (every fin shares
        the same deflection angle and flow, so the magnitude is the same
        for all of them), then returns one `Force` per fin: each fin's
        force points tangent to the circle it's placed on (perpendicular to
        its own radius vector from the body centerline), at that fin's own
        location. With `numCanards >= 2` placed symmetrically, these
        per-fin forces cancel to zero net translational force and zero net
        yaw/pitch torque, leaving only roll torque -- see the class
        docstring.

        Args:
            rocket (Rocket): The rocket this control is attached to. Used
                for `rocket.simTimeStep` (to rate-limit the angle change)
                and, via `__calculateFinLift`, `rocket.zVel_mps` and
                `rocket.airDensity`.
            **kwargs: Must include `canardAngle_deg` (float) — the
                commanded canard deflection angle in degrees for this step.

        Raises:
            TypeError: If `canardAngle_deg` is not present in `kwargs`.

        Returns:
            list[Force]: One `Force` per fin, each tangent to the fin's
                placement circle at its own location.
        """

        if not ("canardAngle_deg" in kwargs.keys()):
            raise TypeError("missing 1 required positional argument: 'canardAngle_deg'")

        self.angle_rad = math.radians(kwargs['canardAngle_deg'])
        self._dt = rocket.simTimeStep

        finLift_n  = self.__calculateFinLift(rocket)
        finLift_n *= -1 if self.angle_rad < 0 else 1

        forces = self._tangentialForces(
            magnitude_N=finLift_n,
            radius_m=self._finDistance_m,
            z_m=self.forceLocationZ_m,
            numForces=self.numCanards,
            startAngle_rad=self._finAxialPlacement_rad
        )

        # Logged for `plot()` as a signed magnitude (per-fin lift * fin
        # count), not the net vector force -- with numCanards >= 2 placed
        # symmetrically the net force vector is ~0 by design (see class
        # docstring), so logging that instead would just show a flat line
        # at 0. Kept signed (not abs()'d) so the plot still shows which
        # direction the commanded roll is in.
        self.df.loc[rocket.simTime] = {
            "time_s": rocket.simTime,
            "angle_rad": self._angle_rad,
            "generatedLift_n": finLift_n * self.numCanards
        }

        return forces

    def __calculateFinLift(self, rocket: Rocket) -> float:
        """
        Calculate the lift of the canard for the current AOA and rocket vertical velocity

        Looks up C_L for the current absolute deflection angle (in degrees)
        and the rocket's vertical velocity via the CSV-derived
        `_liftInterpFunc`, then converts it to a lift force using the
        standard dynamic-pressure lift equation
        (L = C_L * 0.5 * v^2 * rho * area).

        Args:
            rocket (Rocket): The rocket that this is attached to

        Returns:
            float: The lift generated by the canard. Returns 0 if the
                angle/velocity pair falls outside the interpolator's known
                range (see NOTE below on the current dead check for this).
        """

        cl = self._liftInterpFunc(abs(math.degrees(self._angle_rad)), rocket.zVel_mps)
        # If the angle and velocity are outside the range of the CSV, we can assume the coefficient of lift is 0
        #   Its either a low AOA or slow speed, its an assumption that may or maynot hold up
        cl = float(cl) if not np.isnan(cl) else 0.0
        
        generatedLift = cl * 0.5 * rocket.zVel_mps**2 * rocket.airDensity * self._surfaceArea_m2

        return generatedLift

    @property
    def angle_deg(self) -> float:
        """float: The canard's current actual angle in degrees (derived from `_angle_rad`)."""
        return math.degrees(self._angle_rad)

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

        self.df['angle_rad'] = self.df.apply(lambda row: math.degrees(row.angle_rad), axis=1)
        ax1.plot(self.df.index, self.df['angle_rad'], color='black', label='Canard Angle')
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Angle (Degrees)')
        ax1.legend(loc='lower left')
        ax1.grid(True)

        ax2.plot(self.df.index, self.df['generatedLift_n'], color='red', label='Force Generated by Canards')
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Force (Newtons)')
        ax2.legend(loc='lower left')
        ax2.grid(True)
        
        if not importedAxis:
            plt.tight_layout()
            plt.show()

        return