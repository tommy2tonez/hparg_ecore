//alrights fellas 
//today instead of doing planes and coordinates - we are doing spheres and radians - then we'll come up with synthetic function that's best at approximating things - we'll work on this for a month or two - then we'll approxmiate the function by using math_approx
//we'll be rich guys, be patient please
//consider this plane eqn: normal vector . <x - x0> <y - y0> <z - z0> = 0
//=> ax + by + cz + ax00 + by00 + cz00 = 0
//linear is done by doing plane projections - which is fine - but not very good at retaining informations

//Without loss of generality
//assume that we have a linear function f(x) -> y, 3d euclidean coordinate
//the possibility of the plane that passed through y1, and y2 and y3 is 1

//Without loss of generality
//assume that we have a sphere function f(x) -> y, 3d euclidean coordinate
//the possibility of the plan that passed through y1, y2, y3 (within a reasonable uncertainty) by circumscribing the coordinate, is more than 1

//one way to describe this one dimensionally is by discretizing spheres within a sphere
//consider this volumn eqn: 4/3*pi*r^3
//assume our unit sphere is r/2
//then the number of one dimensional spheres is 8
//assume our unit sphere is r/4
//then the number of one dimensional spheres is 64
//assume our function f(x) -> y
//x is the one dimensional pointer
//then y is the coordinate of the pointing sphere

//assume our dimension is x
//assume our sphere dimensional reduction is g(x) -> y
//assume our reverse sphere dimensional reduction is g'(y) -> x
//then our fuzzy original coordinate is g'(f(x))

//alrights - let's talk in terms of radians and for loops (which is integral)
//we have coordinate x y, alpha from 0 - pi
//we have y = r * sin(alpha)
//        x = r * cos(alpha)

//we have coordinate y z, theta from 0 - 2pi
//we have y = r * sin(alpha) * cos(theta)
//we have z = r * sin(alpha) * sin(theta)

//we have coordinate z k, gamma from ...
//we have ...

//you got the idea
//this function of f(alpha, alphamultiplier, theta, thetamultiplier, gamma, gammamultiplier, beta, betamultiplier, r, stride_idx) turns out differentiable and mathematically much more accurate to approx f(x) -> y than linear - we have yet to come up with a synthetic function that is good at this - which is probably not sphere and linear (which are the two loved childs of mathematics)
//the problem of linear is actually not that it is bad at approximating - but it is bad at being a context mixer
//assume a feature vector <v1, v2, v3, v4>, and a linear vector b <l1, l2, l3, l4>
//the result of f . b is not as good being a representor of v1 v2 v3 v4 as sphere projection

//alright - I think Dad got a point
//we need to do one dimensional projection (discretization) then sphere projection
//without loss of generality - we assume sphere is only for compression

//so we have f(x) -> y is one dimensional projection
//and f'(g(x)) -> y is one dimensional projection and sphere projection

//in order words, we just do normal linear sin cos add exp log, etc. then we do sphere expansion
//sphere operation is like linear - only it takes in radians coor as row and multipler and radiuses as columns - this is hard 

//what we actually want is a combinatorial formula to retain information - rows x columns combinatorial eqns 
//linear is plane projections, and now we have oval projections (we want to store the radiuses of the oval as columns, and radians as rows)
//then we have oval expansions - which is another function
//things gonna be tough and bumpy
//we are talking in terms of "without loss of generality" - we'll work on a synthetic solution
//the true formula cannot be described in words but rather through experiments - so we'll get there by instruments and benchmarks