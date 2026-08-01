## Planning for TVC a implementation

- I need to have sim return torques and forces (adding to Controls class and all children)
    - Quick implementation assumes 1 force location per object
        - The control class holds a location in the body reference frame (need to deicide if its from the tip or tail of the rocket)
        - Then the rocket class will take that force a cross it with a vector from the cg to the force location

- Should store CG location in the rocket to more uniformly calculate torques
    - Maybe the controls should return force vectors at a given location that way only the rocket *really only has to deal with reference frames*
    - CG and MMOI may not be constant.
        - This could be read from a CSV file, where it checks to see if the head cgPosX/Y/Z exist and if it does it changes based on time
        - Same applies for MMOI

- I need to be very very careful with reference frames
- I might have 3 different references frames: Absolute Reference Frame, Body Reference Frame (quat), Motor Reference Frame (i'm thinking Euler cause its only 2 angles relative to body, not roll, pitch then yaw?)
    - I should look to see if there are any libraries that would simplify this, spatialmath-python seems very very promising

- The rocket class will not always be allowed to read "sim data" like velocity, altitude, etc from a CSV it needs to be computed.
- In this case it shouldn't read any data cause the TVC will give forces that will be used to compute the accelerations, velocities, and positions

- Need to implement a full 3d acceleration, velocity, position tracking

- Gravity

- ~~Graphing built into the tool~~ Implemented
    - ~~Need to have the sim tool log all the data~~
    - ~~Multiply functions for different plots~~