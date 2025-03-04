from typing import Callable
import math 
import random
import copy

class TaylorValue:

    def __init__(self, value: float):

        self.value = value

class TaylorSeries:

    def __init__(self, series: list[TaylorValue]):

        self.series = series

class TaylorApprox:

    def __init__(self, taylor_series: TaylorSeries, operation: Callable[[float], float]):

        self.taylor_series  = taylor_series
        self.operation      = operation 

def bind_add_operation(lhs: Callable[[float], float], rhs: Callable[[float], float]) -> Callable[[float], float]:

    return lambda x: lhs(x) + rhs(x)

def get_taylor_series(differential_order_sz: int, differential_step_sz: int) -> TaylorApprox:

    taylor_series           = TaylorSeries([])
    # sum_function: Callable  = lambda x: 0
    sum_list: list          = []

    for i in range(0, differential_order_sz, differential_step_sz):
        random_value            = 0
        container: TaylorValue  = TaylorValue(random_value)
        sum_list                += [container]
        taylor_series.series.append(container)

    def sum_function(x: float):
        rs = float()

        for i in range(len(sum_list)):
            rs += float(1) / math.factorial(i) * sum_list[i].value * (x ** i)

        return rs

    return TaylorApprox(taylor_series, sum_function)

def newton_approx(operation: Callable[[float], float], iteration_sz: int, initial_x: float, a: float = 0.001) -> tuple[float, float]:

    cur_x       = initial_x
    min_y       = operation(cur_x)
    cand_x      = cur_x 
    epsilon     = float(0.01)
    
    for _ in range(iteration_sz):
        cur_y   = operation(cur_x)
        
        if (cur_y < min_y):
            cand_x  = cur_x
            min_y   = cur_y

        a_y     = operation(cur_x + a)
        slope   = (a_y - cur_y) / a

        # print("slope", slope)

        if (abs(slope) < epsilon):
            break 

        cur_x   -= cur_y / slope

    return cand_x, min_y

def discretize(first: float, last: float, discretization_sz: int) -> list[float]:

    width: float    = (last - first) / discretization_sz
    rs: list[float] = []
    
    
    for i in range(discretization_sz):
        rs += [first + (i * width)]
    
    return rs

def calc_deviation(lhs: Callable[[float], float], rhs: Callable[[float], float], x_range: int, discretization_sz: int) -> float:

    discrete_value_arr: list[float] = discretize(0, x_range, discretization_sz)
    sqr_sum: float                  = sum([(lhs(x) - rhs(x)) ** 2 for x in discrete_value_arr])
    denom: float                    = float(len(discrete_value_arr))
    normalized: float               = math.sqrt(sqr_sum / denom)

    return normalized

#alright - we want to be able to approx EVERYTHING today - including exp - sin - cos - linear - sqrt - or - xor - etc.
#let's see what's wrong
#the first thing is the differential order - which heavily affect the approximation
#the second thing is the dynamic "collaboration" of the slopes - which turns thing up down and around 
#is there a solution to solve this problem? hmm...
#it seems like a dynamic programming problem
#alright - what we actually CARE is the direction of the "newton slope" - in this very example - we only move in one direction - which is dx or dy or dz
#what happens if we move in the direction of <1, 1, 1> without loss of generality - and "approx" the gradient?
#let's make things complicated - we discretize all directional unit vector
#and choose the "right" way to inch our rocket into

def taylor_series_to_value_arr(series: TaylorSeries) -> list[float]:

    return [e.value for e in series.series]

def write_taylor_series_value(series: TaylorSeries, value: list[float]):

    for i in range(len(series.series)):
        series.series[i].value = value[i] 

def get_taylor_series_size(series: TaylorSeries) -> int:

    return len(series.series) 

def add_vector(lhs: list[float], rhs: list[float]) -> list[float]:
    
    return [e + e1 for (e, e1) in zip(lhs, rhs)]

def scalar_multiply_vector(c: float, rhs: list[float]) -> list[float]:
    
    return [c * e for e in rhs]

def random_0(sz: int) -> float:

    dice = random.randrange(0, sz)
    
    if dice == 0:
        return float(0)
    
    return float(1)

def random_sign(sz: int) -> float:

    dice = random.randrange(0, sz)

    if dice == 0:
        return float(-1)

    return float(1) 

def get_random_vector(dimension_sz: int) -> list[float]:

    return [random.random() * random_0(4) * random_sign(4) for _ in range(dimension_sz)]

def dot_product(lhs: list[float], rhs: list[float]) -> float:

    return sum([e * e1 for (e, e1) in zip(lhs, rhs)]) 

def to_scalar_value(vector: list[float]) -> float:

    return math.sqrt(dot_product(vector, vector)) 

def get_directional_vector(vector: list[float]) -> list[float]:

    vector_scalar_value = to_scalar_value(vector)    
    return [e / max(vector_scalar_value, 0.001) for e in vector]

def train(approximator: TaylorApprox, instrument: Callable[[float], float], training_epoch_sz: int, directional_optimization_sz: int, x_range: int, discretization_sz: int):

    newton_iteration_sz             = 4
    grad_dimension_sz: list[float]  = get_taylor_series_size(approximator.taylor_series)

    for _ in range(training_epoch_sz):
        inching_direction: list[float]          = [float(0)] * grad_dimension_sz
        inching_direction_multiplier: float     = float(0) 
        inching_direction_value: float          = None

        for __ in range(directional_optimization_sz):
            random_vec: list[float]         = get_random_vector(grad_dimension_sz)
            directional_vec: list[float]    = random_vec
            newton_exp_base                 = random.random() * 10
            newton_discretization_sz        = 10

            def newton_approx_func(multiplier: float):
                previous_value: list[float] = taylor_series_to_value_arr(approximator.taylor_series)
                copied_value: list[float]   = copy.deepcopy(previous_value)
                adjusted_value: list[float] = add_vector(copied_value, scalar_multiply_vector(multiplier, directional_vec))

                write_taylor_series_value(approximator.taylor_series, adjusted_value)
                rs: float = calc_deviation(approximator.operation, instrument, x_range, discretization_sz)
                write_taylor_series_value(approximator.taylor_series, previous_value)

                return rs

            for j in range(newton_discretization_sz):
                exp_offset              = (newton_exp_base ** j) - 1
                (est_new_multiplier, y) = newton_approx(newton_approx_func, newton_iteration_sz, exp_offset)

                if inching_direction_value == None or y < inching_direction_value:
                    inching_direction               = directional_vec
                    inching_direction_multiplier    = est_new_multiplier
                    inching_direction_value         = y

        if inching_direction_value == None:
            continue

        print(inching_direction_value)

        current_value: list[float]  = taylor_series_to_value_arr(approximator.taylor_series) 
        adjusted_value: list[float] = add_vector(current_value, scalar_multiply_vector(inching_direction_multiplier, inching_direction))

        previous_deviation          = calc_deviation(approximator.operation, instrument, x_range, discretization_sz)

        if (previous_deviation > inching_direction_value):
            write_taylor_series_value(approximator.taylor_series, adjusted_value)
            print("update", inching_direction_value)

def main():

    #something went wrong
    #let me show them mfs the real power of taylor fast + electrical engineering designs
    #legend says this algorithm still runs 1000 years later
    #well... it's a fission operation - we specialize in rocket + nuke
    #I just met my brother in my dream - that was a very happy event
    #I wanted to show my brother how far we have come in our rocket science
    #this is like level 1 of our hacking career
    #"you can't organize an ostrich race with just one ostrich" - Prince of Persia
    #you can't make a bomb with just one Uranium
    #that's the truth - because we are in a "golf game" of finding the coordinates 
    #in real life - most of the time, deviation is not even a clue - you might just get one big blackhole and nothing at all on the surroundings (path is not suitable in this situation - we are optimizing things that are not quantifiable)
    #this is why we need fission - to open up a chain-reaction of "where the golf balls might have been" - and try to find the final destination 
    #exponential direction discretization is the correct approach - we also need to add <random_discretization> or chaotics of HUGO's "broista" to increase the randomness of the rocket projections

    #I was thinking of cosine simularity, HUGO, and newton approx - if we make a one shot on a par 3 - we don't even need gradient descend
    #it seems like this is a fission + mining operation - but it is not a fission operation - more like a centrality + fission operation - because our resource is finite
    #let's hope guys - I have a good feeling that we are going to make this
    #we dont really know how that works so well - for every function - we just know that this is dangerous if weaponized in this current planet of the chimp technology - we can break through every possible security layers and access data in an instant

    #now tell me what's the difference between f(x), f(x, y), f(x, y, z)?
    #it's the euclidean locality - x, z and y makes more sense than x,y and z
    #this is a very important concept for training - we want to regroup these guys to have faster training time
    #apart from that - I dont think there is a difference between f(x), f(x,y), x(x, y, z)
    #we'll find a way
    #it's not that difficult to understand things - you need to be able to listen to the music and find your true compass of logic - not what the school taught you

    #what happens with our golf course?
    #in this case it converges too slow - we need to hypercharge the ball to redirect into the correct direction + magnitude - how to? by using centrality time-series forcast based on previous results (when we reach the differential order of 64 or 128 - the random projection is fuzzy - we need to have a nagivator) - recall that we want to maximize the delta
    #we still want to use exponential discretization + randomization - its just that we use another layer of machine learning to predict our difference derivatives
    #remember the 10 steps towards the Sun - we have exactly 10 steps to step out of the bunker
    #what does this mean? it means that if we get this proof of concept correctly (this week) - we can freely walk in the system - HTTPs read + JWT fake + sessions read + virtual machine decodings + etc
    #this can escalate into a browser compromise (which is heavily built on top of these security metrics)
    #which escalates into an OS compromise (alright - im not being correct here - browser is more important than OS - because it is contagious - OS is not - we want to compromise browser from compromising OS, not the opposite)

    #I was thinking about a navigator to "hint" our exponential steps - yet I can't quantify the problem - because this is a hard problem to quantify
    #taylor projection is probably the best projection we could ever do - this must be combined with centrality to offset the cost of <new_words> - we'll talk about this later - things can be all taylor-approxed
    #goal is actually not to "replace" centrality - but to do be a better "linear" to do centrality
    #the flop is shit but it's the best we've got for years
    #we still probably need to implement navigators

    #let's see what Maxwell has to tell
    #the law of flux (external system)
    #the law of no flux (internal system)
    #the law of external and internal system collaboration - torque

    #the problem is the up down and around of the taylor series which creates anomaly <black_holes>
    #I've thought long and hard - there is no better way than to do exponential steps - again - random exponential base + 10 steps is the true way
    #well we need to rid of the exclusive thinking and think that these solutions might not cancel out each other - we can implement a forcast to skew the distribution and a random projection for completeness of algorithm

    #what precisely are we optimizing | mining?
    #we want to make sure that our solution is complete - such that there exists a solution by running this algorithm forever
    #we want to make sure that our "focus" is well spent - such means that the focusing area of optimization must be reasonably skewed with respect to the overall picture (important) - we need to work on this - i'll be right back
    #we want centrality because our resource is finite - we keep our pool of best fit candidates and maintain our pool of best fit candidates - like SPY 500
    #because of those reasons, this algorithm is radixed as a mining operation + centrality operation + fissing operation (we do "fission" by random projection around the focusing area of optimizations)

    approxer: TaylorApprox  = get_taylor_series(8, 1)

    def sqrt_func(x: float):

        return math.sqrt(x)

    # write_taylor_series_value(approxer.taylor_series, [0, 1, 0, -1, 0, 1, 0, -1])
    # write_taylor_series_value(approxer.taylor_series, [0, 1, 0, -1, 0, 1, 0, -1, 0, 1, 0, -1, 0, 1, 0, -1])
    train(approxer, sqrt_func, 1 << 12, 16, 64, 32) #if this could run - it passes my turing test of approximation
    # # # approxer.taylor_series.series[0].value = 01
    # # # approxer.taylor_series.series[1].value = 0.5

    print(approxer.operation(2))
    print(calc_deviation(approxer.operation, sqrt_func, 64, 32))

    for i in range(len(approxer.taylor_series.series)):
        print(approxer.taylor_series.series[i].value)

    # # print()
    # # print(approxer.operation(1))

main()