//alrights fellas 
//today instead of doing planes and coordinates - we are doing spheres and radians - then we'll come up with synthetic function that's best at approximating things - we'll work on this for a month or two - then we'll approxmiate the function by using math_approx
//we'll be rich guys, be patient please
//consider this plane eqn: normal vector . <x - x0> <y - y0> <z - z0> = 0
//= ax + by + cz + ax0 + by0 + cz0 = 0
//linear is done by doing plane projections - which is fine - but not very good at retaining informations

//Without loss of generality
//assume that we have a sphere function f(x) -> y, 3d euclidean coordinate
//the possibility of the plan that passed through y1, y2, y3 (within a reasonable uncertainty) by circumscribing the coordinate, is more than 1

//Without loss of generality
//assume that we have a linear function f(x) -> y, 3d euclidean coordinate
//the possibility of the plane that passed through y1, and y2 and y3 is 1

//in other words, we can always approxmiate f(x) -> y by having enough circumscribed spheres and expand them and accumulate them
//but we cannot approximate f(x) -> y by having a reasonable amount of linear
