
#let's start another proof of concept
#in this proof of concept
#we want to approx centrality models by using state expansion approach
#and our moral compass is the time-series prediction of velocity + acceleration + jerk + crack + snap + etc.
#our A * heuristic is the best possible next move in the time-series prediction result
#our input consists of the previous moves + the f'(x) f''(x) f'''(x) f''''(x) at x = 0 - without loss of generality
#we want to explore what happened if we use centrality to improve centrality to improve centrality
#we have a week to work on this

#let's see
#let's start very simple
#we have regex rules of centrality + regex rules of training
#we have a binary input data

#our state is the model
#our adjecent states (the actionables) are the expansions of the regex rules

#we want to do random regex at first - to collect data in terms of state + f'(0), f''(0), f'''(0), f''''(0)

#the "path" algorithm that we want to use is exponential backstep + backtrack algorithm 
#we want to unhinge the "best" possible path - exponentially rewind - and remember to not step in the already explored direction
#the moral compass of which direction to explore is the "time-series" prediction - our time step is state1-state2-state3-state4 etc.

import math
from typing import Callable

#alright let's have a scratch of what centrality means - centrality means uniform distribution of rules + propagation
#                                                      - centrality means propagate the value to the neighbor nodes
#                                                      - centrality means the algorithm must converge

#how about taylor series: taylor series can approx every differential function by knowing the differential values at 0 including f(0), f'(0), f''(0), f'''(0), f''''(0), ... - up to 256 differential orders
#our time-series prediction is actually from f'(0) to another f'(0) to another f'(0) - which is the time step
#the way we train our time-series has <support_line> + punishment based on how far down through the support lines the next move is gonna be

#in the midst of all centrality algorithms, we want to know what the next inching direction is going to be (which is the time-series forcast)
#let's work on this

#let's start very simple - without loss of generality

#we want to have a regex rule for the lhs column (for speed) and the rhs row  - and we multiply the rules for all the rows 
#then we move to something complicated - the idea of an arbitrage row by having a static bijective translation rule
#then we regex the lhs to be a random leaf or self operation (leaf|self)

#wait a second
#everything could be seen as a pair operation - an intercourse of two variables
#how about a three-variables - or n variables intercourse - well that's where taylor-series projection + calibration kicked in (EVERYTHING can be quantified as projection + calibration - which is operator add)
#we want to build our centrality on top of this concept

#-----

#why is taylor series the most important concept - because it is not only suitable for doing one dimensional projection but also n-dimensional projection approx f(x) -> y 
#without loss of generality

#assume we are doing one dimensional projection
#we need to find f(0), f'(0), f''(0), f'''(0) ... up to 256 differential orders

#hmm hold on
#come on babes - we are gonna be rich

#assume we are doing two dimensional projection f(x, y)
#we need to find f(0, 0), f'_wrtx(0, 0), f''_wrtx(0, 0) ... up to 256 differential orders
#we also need to find f(x1, 0), f'_wrty(x1, 0), f''_wrty(x1, 0) ... up to 256 differential orders
#the f(x1, 0), f'_wrty(x1, 0) look familiar - yes - they are one-dimensional projection functions - which we've already solved 

#assume we are doing three dimensional projection f(x, y, z)
#by using induction - taylor-series is probably the most important concept in machine learning - taylor swift is, therefore, very important to approx modern infrastructure

#taylor swift + centrality is probably the proof of concept that we are going to write this week
#yall better hurry with the ballistic missiles before taylor swift smacking your asses 

#let's talk about f(x, y, z) -> <value_1, value_2, value_3>
#we want to find the differential vector (velocity vector) instead of a scalar velocity value 
#what does that mean? that means we want to do taylor series binary projections + hypertraining by using time-series prediction 

#let's work on a very simple concept - the concept of dimensional reduction f(x1, x2, x3, x4, x5, x6, x7, x8, x9, 10) -> <xx1, xx2, xx3> - rotate - reprojection to calibrate + and do a calibration operation on the original matrix (so we have a centrality algorithm)
#we want to work on binary data - we can assume that every operation is "ballistic missile" operation - in the sense of f(x) = <x1, x2, x3> - we want to find original velocity vector, original acceleration vector, original jerk vector, etc. - to approximate where the ballistic missle is going to be                                                                                                                                                               
#swifties 2025 guys - we'll rule the world 

class TaylorValue:

    def __init__(self, value: list[float]):

        self.value = value

class TaylorSeries:

    def __init__(self, series: list[TaylorValue]):

        self.series = series

class TaylorApprox:

    def __init__(self, taylor_series: TaylorSeries, operation: Callable[[list[object]], list[object]]):

        self.taylor_series  = taylor_series
        self.operation      = operation 

def elemental_add(lhs: list[object], rhs: list[object]) -> list[object]:

    return [lhs[i] + rhs[i] for i in range(len(lhs))]  

def bind_add_operation(lhs: Callable[[list[object]], list[object]], rhs: Callable[[list[object]], list[object]]) -> Callable[[list[object]], list[object]]:

    return lambda x: elemental_add(lhs[x], rhs[x])

def dot_product(lhs: list[object], rhs: list[object]) -> object:

    return sum([e1 * e2 for (e1, e2) in zip(lhs, rhs)])

def get_scalar_value(x: list[object]) -> float:

    return math.sqrt(dot_product(x, x)) 

def get_unit_directional_vector(x: list[object]) -> list[object]:

    sz: float = get_scalar_value(x)
    return [float(e) / sz for e in x] 

def vector_scalar_multiply(lhs: float, rhs: list[object]) -> list[object]:

    return [e * lhs for e in rhs] 

def add_taylor_series(lhs: TaylorSeries, rhs: TaylorSeries) -> TaylorSeries:

    return TaylorSeries(lhs.series + rhs.series) 

def get_initial_function() -> Callable[[list[object]], list[object]]:

    return lambda x: [float(0) for _ in range(len(x))]

def get_taylor_series(in_feature_sz: int, out_feature_sz: int, differential_order_sz: int, differential_step_sz: int) -> TaylorApprox:

    #recall that this can be resoluted by doing in_feature_sz - 1, out_feature_sz
    #

    if in_feature_sz == 0:
        raise Exception() 

    if in_feature_sz == 1:
        sum_function: Callable  = get_initial_function()
        sum_value_list  = []

        #what im confused is x being the cursor in the vector space - what does that even mean? it means x is the time - we have original vector positions - which is d/dt, d^2/dt, d^3/dt, d^4/dt 
        #s + vt + 1/2*a*t^2 + 1/6*j*t^3 + ...
        #this is precisely the rocket problem
        #i'll be right back

        for i in range(0 , differential_order_sz, differential_step_sz):
            tmp_value_list: list[TaylorValue]   = generate_initial_taylor_value(out_feature_sz)
            operation                           = lambda x: vector_scalar_multiply(1 / math.factorial(i) * (get_scalar_value(tmp_value_list) * x ** i), get_unit_directional_vector(tmp_value_list))
            sum_function                        = bind_add_operation(sum_function, operation)
            sum_value_list                      + sum_value_list + tmp_value_list

        return TaylorApprox(TaylorSeries(sum_value_list), sum_function)

    taylor_series           = TaylorSeries()
    sum_function: Callable  = get_initial_function()

    for i in range(0, differential_order_sz, differential_step_sz):
        derivative_approx_operation: TaylorApprox   = get_taylor_series(in_feature_sz - 1, out_feature_sz, differential_order_sz, differential_step_sz)
        taylor_series: TaylorApprox                 = add_taylor_series(taylor_series, derivative_approx_operation.taylor_series)
        operation                                   = lambda x: vector_scalar_multiply(1 / math.factorial(i) * (get_scalar_value(derivative_approx_operation.operation(x[:-1])) * x[-1] ** i), get_unit_directional_vector(derivative_approx_operation.operation(x[:-1]))) 
        sum_function                                = bind_add_operation(sum_function, operation)

    return TaylorApprox(taylor_series, sum_function)

def newton_approx(f: Callable[[float], float], x: int, newton_iteration_sz: int) -> float:

    pass  

def train(approximator: TaylorApprox, instrument: Callable[[list[object]], list[object]]):
    pass 

def e(x: int) -> float:

    rs = float()

    for i in range(0, 256, 1):
        rs += 1 / math.factorial(i) * (x**i)

    return rs

print(e(12))