import math

from scipy.interpolate import LinearNDInterpolator
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from .Controls import Controls
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
    of attack and velocity) via linear interpolation, then converted to a
    torque about the roll axis using the canard offset from the rocket's
    centerline. The commanded angle is rate-limited and clamped to
    `maxAngle_rad` on every write, modeling a physically realistic actuator
    (finite slew rate and maximum deflection) rather than an ideal one.

    Attributes:
        airfoilDataPath (str): The path to a table of C_L for different velocities and AOA's
        root_m (float): Root coord of the canards in meters
        tip_m (float): Tip coord of the canards in meters
        span_m (float): Span of the canards in meters
        sweep_m (float): Sweep of the canards in meters
        numCanards (float): Number of canards
        offset_m (float): Distance from the center axis to the canards in meters
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

    numCanards: float
    offset_m:   float

    maxAngle_rad:   float
    rateLimit_rps:  float
    updateFreq_hz:  float

    _angle_rad: float = 0.0
    _dt:        float = 0.0

    df = pd.DataFrame(columns=["time_s", "angle_rad", "generateTorque_Nm"])

    def __init__(self, airfoilDataPath: str, root_m: float, tip_m: float, span_m: float,
        sweep_m: float, numCanards: float, offset_m: float, maxAngle_deg: float, rateLimit_dps: float,
        updateFreq_hz: float
    ):
        """
        Initialize the Canards object

        Args:
            airfoilDataPath (str): The path to a table of C_L for different velocities and AOA's
            root_m (float): Root coord of the canards in meters
            tip_m (float): Tip coord of the canards in meters
            span_m (float): Span of the canards in meters
            sweep_m (float): Sweep of the canards in meters
            numCanards (float): Number of canards
            offset_m (float): Distance from the center axis to the canards in meters
            maxAngle_deg (float): Max deflection angle of the canards in degrees
            rateLimit_dps (float): Max angular speed of the canards in degrees / sec
            updateFreq_hz (float): Update frequency of the canards in hz
        """
        super().__init__("Canards")

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
        self.offset_m      = offset_m
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

    def sim(self, rocket: Rocket, **kwargs) -> tuple[float, float, float]:
        """
        Simulate the canards for one step and return the resulting torque.

        Updates the commanded angle (subject to rate limiting, see
        `angle_rad` setter) from `kwargs['canardAngle_deg']`, computes fin
        lift for the rocket's current vertical velocity and air density via
        `__calculateFinLift`, and converts that lift into a roll torque
        using the canard offset and count. Only roll torque is produced;
        yaw and pitch are always returned as 0.0.

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
            tuple[float, float, float]: (0.0, 0.0, rollTorque_Nm) — the
                roll torque generated by the canards this step, in Nm.
        """

        if not ("canardAngle_deg" in kwargs.keys()):
            raise TypeError("missing 1 required positional argument: 'canardAngle_deg'")

        self.angle_rad = math.radians(kwargs['canardAngle_deg'])
        self._dt = rocket.simTimeStep

        generatedTorque_Nm  = self.__calculateFinLift(rocket) * self.offset_m
        generatedTorque_Nm *= -1 if self.angle_rad < 0 else 1
        generatedTorque_Nm *= self.numCanards

        self.df.loc[rocket.simTime] = {
            "time_s": rocket.simTime,
            "angle_rad": self._angle_rad,
            "generateTorque_Nm": generatedTorque_Nm
        }

        return (0.0, 0.0, generatedTorque_Nm)

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

        ax2.plot(self.df.index, self.df['generateTorque_Nm'], color='red', label='Torque Generated by Canards')
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Torque (Newton Meter)')
        ax2.legend(loc='lower left')
        ax2.grid(True)
        
        if not importedAxis:
            plt.tight_layout()
            plt.show()

        return