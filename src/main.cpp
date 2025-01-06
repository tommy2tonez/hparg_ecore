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

//alrights - I had another feedback
//instead of doing rowxcolumn - we are doing row x row render rules for fast forward - and tradeoff backward speed
//the problems that we currently have are: - uacm (unordered_accum) operations are too basic - (add, min, max) - do not fully reflect the power of dimensional reductions (which is WLOG described above through sphere projections) - we want synthetic solutions and good approximations to reduce the compute overheads 
//                                         - we probably want oacm (ordered_accum)
//                                         - combinatorial pacm (pair_accum) operations are too basic (linear) - we want spheroids, non-mathematical projections (ideal projections)
//                                         - we are avoiding query optimizations - cublas_x - we want jit or runtime compilations - we probably want shared_process memories (this is hard)

//consider this extreme scenerios - we are dispatching 1x1 tiles - and we are dispatching 8192 x 8192 tiles
//the former scenerio is purely pair operations + path optimizations
//the latter scenerio trade off context distribution for locality (this is only for combinatorial operations + dimension reduction operations) - so we must improve the context distribution of those operations
//alrights - that is first cut optimization
//the second cut optimization is the base operations
//we assume that add, sub, mul, div, and, or, xor, exp, log, sin, cos, inv, neg, dsc (math_approx.py) are sufficient - right, it's sufficient to approx every f(x) - but not sufficient to increase logit information densitiy (which is very important for intellect)
//after getting the two optimizations done, you'd unlock true artificial intelligent - hint, it's hard
//the third cut optimization is discrete math - and logit group training
//the fourth cut optimization is discrete initial values

//(1) and (2) could be done via traditional approaches - says scaled rule of logits - we want to maximize logit density
//(3) and (4) must be done via logit mining - billions, trillions of cores spin to find the sweet spot 

//I wasn't joking when I said people that could solve the problems would rule the world
//those optimizations would take a normal person 10 years of doctoral to solve cleanly, accurately