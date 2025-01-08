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

//alrights - let's talk about how we could do this (increase leaf logit density) - opti stategies
//we are talking in terms of pair operations of 1x1 tiles

//mathematically speaking 1st optimization strategy: log discretization of const initial values of continous oval projection f 
//we have plus/minus - which is y axis offset
//we have multiply/divide - which is y axis expansion/shrink
//we have sin/cos - which is circle projection
//these are the basic operations to approx f of f(x) -> y
//we want to maximize f logit density, f as in f(x) -> y
//we know that everything could be described as oval projections

//if the original sets are sufficient? why does this increase logit density - it's the log discretization

//consider this linear operation f(a, 3) = <a, 3 * a>
//consider this trinity discretized oval operation f(pi, a) = <x, y> = rotate(<cos(pi) * r1, sin(pi) * r2>, a) - r1 and r2 aren't logits, they are logaric discretized values (uint4_t), with the space complexity, without loss of generality, of ~100 = 10 x 10, base 10
//rotate function can be described as distance function + another circle function mapping
//so instead of producing 1 value like linear with one logit value, we are producing 2 values pair (x, y) with a slight cost of 100 base 10 space complexity
//legend says that people discretize value in logarits + tile size in logarits to find the sweet spots between cuda computation locality benefits and logit density 

//2nd optimization strategy: positional suffix array mapping, we discretize the vector x, as in f(x), space -> 1 dimensional grid and store suffix array for n! base 10 space (this is not differentiable and very hard to get right!!! - so we have to use good heuristic approach)
//3rd optimization strategy: recursively deflate tree node computation - we want to deflate node computation by running math_approx (we invent a new operation that is denser)
//4th optimization strategy: random-sequence of logit group training of same influence groups
//5th optimization strategy: discrete operations, rinse and repeat 1,2,3 (this is expensive)
//6th optimization strategy: find an escape velocity and let AI optimizes itself once it is smarter than the coder
//7th optimization strategy: coerce every leaf logit to uint4_t and randomize the seeds 
//8th optimization strategy: plot twist - our oval operation is not actually well defined - it is actually circle + uint4_t linear
//9th optimization strategy: just write the 8 previous suggestions and you'd be successful

//alrights - let's talk about 2nd optimization strategy
//everything starts with an assumption, assume that f(x) groups semantically equivalent context in the euclidian coordinate
//Without loss of generality, assume we 1 dimensionalize the x space as in f(x), recursively split the logit space right in the middle (mergesort way)
//we want to find the recursive resolution for this
//assume the resolution is resolution_sz (resolution is the discretized space of the descendant context space)
//the worst recursion resolution is resolution_sz * resolution_sz
//we probably want to stack them together by using addition operation

//so what is machine learning, really?
//machine learning is another words for grouping + oval continuous projections
//we want to group semantically equivalent context together - and map them by using oval continous projections - we are assuming semantically equivalent context means closer euclidian distance - this is a part of the recursive definition
//every f(x) -> y can be described as f(w(x)) -> y - this is the recursive definition
//w(x) is a bijective "rearrange" function - mapping x -> (discretized n)! space
//f(x) is an oval continous mapping function
//what does this mean precisely? it means that the rearrange algorithm complexity must never precede sorting complexity - precisely n * log(n) complexity - we should be talking in terms of bits - we assume bit is the universally absolute unit (which is not)
//here is the twist - to have absolute accuracy - the output logit must touch every input logit
//this is the principle of neural network

//so how do we change the definition of w(x) 
//w(x) is the rearrange and context redistributor function - responsible for grouping semantically equivalent context together - which is a necessary, not mandatory, build up for combinatorial operations (either linear or whatever_sphere_near operation)
//                                                         - responsible for diffracting uniform influence of every input logit x
//                                                         - w(x) is defined by f(x) - this is where the recursion happens

//the question of using whether radian or euclidean coordinate is probably the famous historic round eathers vs flat earther conspiracy theorists 
//the round earthers can deflate the earth to certain resolution to fuzzy approx the positions of all Earhthic particles
//the flat eathers tilt the plane and fit all beings on the plane 

//I'm trying to formalize a proof that is hard to prove
//it's the nature of y as in f(x) -> y and position(x) - semantic space so we speak - does the semantic space shape like a sphere - or it shapes like a plane? - or best yet - is it a spheroid (which best fits sphere and plane) 
//or is there a synthetic shape - that is neither sphere nor plane that best fits the semantic space?
//what we are really aiming is the uniform distribution of output on the x or logit density of y on x - avg_projected_counter(x)/ delta_x is uniformly distributed over the range

//the round eathers' way of writing sphere is:
//<x.cos(alpha), y.sin(alpha).cos(theta), y.sin(alpha).sin(theta)>

//the flat earthers' way of writing sphere is:
//<x.cos(a1), y.sin(a2).sin(a3), y.sin(a4).sin(a5)>

//to prove the theory, without loss of generality - we can say that circle increases density by passing everything twice (on the y-axis) 
//we can radix the input -> odd and even - if it is even then we train it on the left range
//                                       - if it is odd then we train it on the right range
//so instead of being unforgiven like linear - which position(x) must map x -> the designated region (within certain uncertainty) - circle has two designated regions

//okay - so let's get back to the f(w(x)) -> y and w(x) = f(x1) + f(x2)
//we can assume that w(x) has uniform semantic space
//but we can't assume f(x) has uniform semantic space
//f(x) only has uniform semantic space only when we are doing compression
//alrights, too many words - let's describe this via functions