from typing import Callable
import math 
import random
import copy
import sys
from decimal import * 
from typing import Protocol
import functools
import matplotlib.pyplot as plt

def taylor_projection(f: list[float], x: float) -> float:

    if len(f) == 0:
        return float(0)

    return sum([1 / math.factorial(i) * f[i] * (x ** i) for i in range(len(f))]) 

class TaylorOperation(object):

    def __init__(self, taylor_series_value: list[float]):

        self.taylor_series_value = taylor_series_value

    def __call__(self, x: float):

        return taylor_projection(self.taylor_series_value, x)

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
    sum_list: list          = []

    for _ in range(0, differential_order_sz, differential_step_sz):
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

def get_slope(f: Callable[[float], float], x: int, derivative_order: int, a: float = 0.000001) -> float:

    if derivative_order == 0:
        return f(x)

    return (get_slope(f, x + a, derivative_order - 1) - get_slope(f, x, derivative_order - 1)) / a  

#recall our 4th order zeros approx
#(x+a)(x+b)(x+c)(x+d) = 0

#3rd order zeros approx
#(x+a)(x+b)(x+c) = 0

#2nd order zeros approx
#(x+a)(x+b) = 0

#alright - what's the advanced way of doing things? - there isnt except for linear algebra - 4 zeros - 4 eqns, each polynomial order has bijective map with a coefficient
#this needs high precision of numerical stability - because real life zeros est is not like theoretical approximation

#lets assume we are approxing 0s
def newton_approx(operation: Callable[[float], float], iteration_sz: int, initial_x: float, a: float = 0.001) -> tuple[float, float]:

    #this is not correctly implemented
    cur_x       = initial_x
    min_y       = abs(operation(cur_x))
    cand_x      = cur_x 
    epsilon     = float(0.01)

    for _ in range(iteration_sz):
        cur_y   = operation(cur_x)

        if (abs(cur_y) < min_y):
            cand_x  = cur_x
            min_y   = cur_y

        a_y     = operation(cur_x + a)
        slope   = (a_y - cur_y) / a

        if (abs(slope) < epsilon):
            break 

        cur_x   -= cur_y / slope

    return cand_x, min_y

def tom_approx(operation: Callable[[float], float], iteration_sz: int, initial_x: float, a: float = 0.001) -> tuple[float, float]:

    s0      = get_slope(operation, initial_x, 0)
    v       = get_slope(operation, initial_x, 1)
    accel   = get_slope(operation, initial_x, 2)
    
    epsilon = float(0.01)
    a       = 1/2 * accel
    b       = v
    c       = s0

    if abs(a) < epsilon:
        return newton_approx(operation, iteration_sz, initial_x)

    delta   = b ** 2 - 4 * a * c

    if delta > 0:
        x1  = (-b + math.sqrt(delta)) / (2*a)
        x2  = (-b - math.sqrt(delta)) / (2*a)

        (x1, y1)    = newton_approx(operation, iteration_sz, x1, a)
        (x2, y2)    = newton_approx(operation, iteration_sz, x2, a)

        if abs(y1) < abs(y2):
            return x1, y1

        return x2, y2

    x = -b / (2*a)

    return newton_approx(operation, iteration_sz, x, a)

#this is appropriate for floating operation because the uncertainty for float is only good for s0, v, a, at most j
#moving <beyond> float would require eqn_solver to solve up to 8th derivative order (because this is a realistic number without compromsing speed)

def tom_approx2(operation: Callable[[float], float], iteration_sz: int, initial_x: float, a: float = 0.001) -> tuple[float, float]:

    NEWTON_ITER_SZ: int = 3 

    if iteration_sz == 0:
        return newton_approx(operation, NEWTON_ITER_SZ, initial_x, a)

    s0      = get_slope(operation, initial_x, 0)
    v       = get_slope(operation, initial_x, 1)
    accel   = get_slope(operation, initial_x, 2)
    
    epsilon = float(0.01)
    a       = float(1)/2 * accel
    b       = v
    c       = s0

    if abs(a) < epsilon:
        return newton_approx(operation, NEWTON_ITER_SZ, initial_x)

    delta   = b ** 2 - 4 * a * c

    if delta > 0:
        x1  = (-b + math.sqrt(delta)) / (2*a)
        x2  = (-b - math.sqrt(delta)) / (2*a)

        (x1, y1)    = tom_approx2(operation, iteration_sz - 1, x1, a)
        (x2, y2)    = tom_approx2(operation, iteration_sz - 1, x2, a)

        if abs(y1) < abs(y2) and abs(y1) < abs(s0):
            return x1, y1
        
        if abs(y2) < abs(s0):
            return x2, y2
        
        return initial_x, s0

    x           = -b / (2*a)
    (x1, y1)    = tom_approx2(operation, iteration_sz - 1, x, a)

    if abs(y1) < abs(s0):
        return x1, y1

    return initial_x, s0

def stable_approx(operation: Callable[[float], float], iteration_sz: int, initial_x: float, a: float = 0.001) -> tuple[float, float]:

    return tom_approx2(operation, 3, initial_x, a)

#I think this is the best way to solve the problem
def get_left_right_closest(e_arr: list[float], pos_arr: list[float], noise: float = 0.02) -> list[float]:

    if len(pos_arr) == 0:
        return []

    left_most_pos: float        = min(pos_arr)
    right_most_pos: float       = max(pos_arr)

    e_left_cand_list: float     = [(left_most_pos - e, i) for (i, e) in enumerate(e_arr)]
    e_right_cand_list: float    = [(e - right_most_pos, i) for (i, e) in enumerate(e_arr)]

    filtered_left_cand          = list(filter(lambda x: x[0] > 0, e_left_cand_list))
    filtered_right_cand         = list(filter(lambda x: x[0] > 0, e_right_cand_list))

    if len(filtered_left_cand) != 0:
        filtered_left_cand  = [e_arr[min(filtered_left_cand)[1]] * (float(1) - random.random() * noise)] #assume the event is the sole event to trigger the derivatves' sign flip sequence - noise must be considered

    if len(filtered_right_cand) != 0:
        filtered_right_cand = [e_arr[min(filtered_right_cand)[1]] * (float(1) + random.random() * noise)] #assume the event is the sole event to trigger the derivatives' sign flip sequence - noise must be considered

    return filtered_left_cand + filtered_right_cand

#I'll try to optimize this algorithm for the next day - this is the very important iterative algorithm
def newton_approxx(operation: Callable[[float], float], iteration_sz: int, initial_x: float, differential_order_sz: int = 4, a: float = 0.00001) -> tuple[float, float]:

    current_x: list[float]              = [initial_x]
    base_newton_iteration_sz: int       = 2
    total_projection_arr: list          = []
    derivative_local_minmax_sampling_sz = 3 
    total_cand_arr: list[float]         = [] 

    for _ in range(iteration_sz):
        scope_differential_projection_arr = []

        for x in current_x:
            local_differential_projection_arr = []

            for differential_order in range(differential_order_sz):
                func                                = lambda x: get_slope(operation, x, differential_order, a)
                (projected_x, deviation)            = stable_approx(func, base_newton_iteration_sz, x, a)
                scope_differential_projection_arr   +=  [(projected_x, deviation)]
                local_differential_projection_arr   +=  [(projected_x, deviation)]

            if len(local_differential_projection_arr) != 0:
                total_projection_arr    += [local_differential_projection_arr]

        total_cand_arr  += scope_differential_projection_arr
        current_x       = get_left_right_closest([e[0] for e in total_cand_arr], current_x)

    if len(total_projection_arr) == 0:
        return stable_approx(operation, iteration_sz, initial_x)

    cand_list   = []

    for i in range(len(total_projection_arr)):
        for j in range(min(len(total_projection_arr[i]), derivative_local_minmax_sampling_sz)):
            cand_list += [(abs(operation(total_projection_arr[i][j][0])), total_projection_arr[i][j][0])] 

    min_y, candidate_x = min(cand_list)

    return candidate_x, min_y

def discretize(first: float, last: float, discretization_sz: int) -> list[float]:

    width: float    = (last - first) / discretization_sz
    rs: list[float] = []
    
    
    for i in range(discretization_sz):
        rs += [first + (i * width)]
    
    return rs

def decimal_discretize(first: Decimal, last: Decimal, discretization_sz: int) -> list[Decimal]:

    width: Decimal    = (last - first) / discretization_sz
    rs: list[Decimal] = []
    
    for i in range(discretization_sz):
        rs += [first + (i * width)]
    
    return rs

def calc_deviation(lhs: Callable[[float], float], rhs: Callable[[float], float], x_range: int, discretization_sz: int) -> float:

    discrete_value_arr: list[float] = discretize(0, x_range, discretization_sz)
    sqr_sum: float                  = sum([(lhs(x) - rhs(x)) ** 2 for x in discrete_value_arr]) #i was wrong guys - x^2 is the way - we are not very sure - we really dont know - there is no proof yet of why this works so profoundly + magnificiently
    denom: float                    = float(len(discrete_value_arr))
    normalized: float               = math.sqrt(sqr_sum / denom)

    return normalized

def taylor_series_to_value_arr(series: TaylorSeries) -> list[float]:

    return [e.value for e in series.series]

def write_taylor_series_value(series: TaylorSeries, value: list[float]):

    for i in range(len(series.series)):
        series.series[i].value = value[i] 

def get_taylor_series_size(series: TaylorSeries) -> int:

    return len(series.series) 

def add_vector(lhs: list[float], rhs: list[float]) -> list[float]:
    
    return [e + e1 for (e, e1) in zip(lhs, rhs)]

def split_vector(vec: list[float], lhs_sz: int) -> tuple[list[float], list[float]]:
    
    lhs: list[float]    = vec[:lhs_sz]
    rhs: list[float]    = vec[lhs_sz:]
    
    return (lhs, rhs)

def sub_vector(lhs: list[float], rhs: list[float]) -> list[float]:

    return [e - e1 for (e, e1) in zip(lhs, rhs)] 

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

def flip_a_coin() -> bool:

    return bool(random.randrange(0, 2)) 

def get_random_vector(dimension_sz: int) -> list[float]:

    return [random.random() * random_0(2) * random_sign(2) for _ in range(dimension_sz)]

def dot_product(lhs: list[float], rhs: list[float]) -> float:

    return sum([e * e1 for (e, e1) in zip(lhs, rhs)]) 

def to_scalar_value(vector: list[float]) -> float:

    return math.sqrt(dot_product(vector, vector)) 

def get_unit_vector(vector: list[float]) -> list[float]:

    sz: float = to_scalar_value(vector)
    return scalar_multiply_vector(float(1) / sz, vector)

def left_shift(vector: list[float]) -> list[float]:

    if len(vector) == 0:
        return []

    return [vector[i + 1] for i in range(len(vector) - 1)] + [float(0)]

def taylor_derivative(vector: list[float]) -> list[float]:

    return left_shift(vector) 

# def taylor_div() 

def unscale_taylor(vector: list[float]) -> list[float]:

    return [vector[i] * math.factorial(i) for i in range(len(vector))] 

def scale_taylor(vector: list[float]) -> list[float]:

    return [vector[i] / math.factorial(i) for i in range(len(vector))] 

def taylor_convolution(lhs: list[float], rhs: list[float], window_sz: int = None) -> list[float]:

    operating_window_sz: int    = len(lhs) + len(rhs)
    unscaled_lhs: list[float]   = unscale_taylor(lhs)
    unscaled_rhs: list[float]   = unscale_taylor(rhs)
    unscaled_rs: list[float]    = [0] * operating_window_sz 

    for i in range(len(lhs)):
        for j in range(len(rhs)):
            multiplier                  = unscaled_lhs[i] * unscaled_rhs[j]
            convoluted_idx              = i + j
            unscaled_rs[convoluted_idx] += multiplier

    if window_sz != None:
        return scale_taylor(unscaled_rs)[:window_sz]

    return scale_taylor(unscaled_rs)

def taylor_plus(lhs: list[float], rhs: list[float]) -> list[float]:

    if (len(lhs) > len(rhs)):
        return taylor_plus(rhs, lhs)

    rs = copy.deepcopy(lhs)

    for i in range(len(lhs)):
        rs[i] += rhs[i]
   
    for i in range(len(rhs) - len(lhs)):
        rs += [rhs[len(lhs) + i]]

    return rs

def get_directional_vector(vector: list[float]) -> list[float]:

    vector_scalar_value = to_scalar_value(vector)    
    return [e / max(vector_scalar_value, 0.001) for e in vector]

def get_leading_dimension(vector: list[float], sz: int) -> list[float]:

    rs = []

    for i in range(len(vector)):
        if i < sz:
            rs += [vector[i]]
        else:
            rs += [float(0)]
    
    return rs


#cos(0) = 1
#0 2 4

def get_sin_taylor(dimension_sz: int) -> list[float]:

    #sin(x) == 0
    #cos(x) == 1 0
    #-sin(x) == 0
    #-cos(x) == -1 1

    return [0 if i % 2 == 0 else 1 if ((i - 1) / 2) % 2 == 0 else -1 for i in range(dimension_sz)] 

def get_cos_taylor(dimension_sz: int) -> list[float]:

    return left_shift(get_sin_taylor(dimension_sz + 1))

def get_const_taylor(dimension_sz: int) -> list[float]:

    func = lambda x: random.randrange(0, 10)
    return [get_slope(func, 0.000001, i) for i in range(dimension_sz)] 

def get_polynomial_taylor(dimension_sz: int) -> list[float]:

    func = lambda x: x ** 2 + x
    return [get_slope(func, 0.000001, i) for i in range(dimension_sz)] 

def get_exp_taylor(dimension_sz: int) -> list[float]:

    return [1] * dimension_sz

def get_gravity_taylor(dimension_sz: int) -> list[float]:

    func = lambda x: 1 / (x ** 2)
    return [get_slope(func, 0.000001, i) for i in range(dimension_sz)] 

def get_sin_gravity_taylor(dimension_sz: int, amplitude: float, frequency: float) -> list[float]:

    func = lambda x: amplitude * math.sin(frequency * x) / (frequency * x)
    return [get_slope(func, 0.000001, i) for i in range(dimension_sz)]

def get_sqsin_gravity_taylor(dimension_sz: int, amplitude: float, frequency: float) -> list[float]:

    func = lambda x: amplitude * (math.sin(frequency * x) / (frequency * x)) ** 2
    return [get_slope(func, 0.000001, i) for i in range(dimension_sz)]

def get_powsin_gravity_taylor(dimension_sz: int, amplitude: float, frequency: float, pow_sz: int) -> list[float]:

    func = lambda x: amplitude * (math.sin(frequency * x) / (frequency * x)) ** pow_sz
    return [get_slope(func, 0.000001, i) for i in range(dimension_sz)]

def get_div_function_taylor(dimension_sz: int, divisor: float):

    func = lambda x: x / divisor
    return [get_slope(func, 0.000001, i) for i in range(dimension_sz)]

def get_cos_gravity_taylor(dimension_sz: int) -> list[float]:

    func = lambda x: math.cos(x) / x
    return [get_slope(func, 0.000001, i) for i in range(dimension_sz)]

def get_minimum_cos_wave(dimension_sz: int) -> list[float]:

    func = lambda x: math.cos(x) / 100
    return [get_slope(func, 0.000001, i) for i in range(dimension_sz)]

def get_minimum_sin_wave(dimension_sz: int) -> list[float]:

    func = lambda x: math.sin(x) / 100
    return [get_slope(func, 0.000001, i) for i in range(dimension_sz)]

def clamp(val: float, min_val: float, max_val: float) -> float:

    if val < min_val:
        return min_val

    if val > max_val:
        return max_val

    return val 

def clamp_taylor(func: list[float], MIN_VALUE: float = -1000.0, MAX_VALUE: float = 1000.0, floating_accuracy_sz: int = 5):

    return [clamp(float(int(e * 10 ** floating_accuracy_sz)) / (10 ** floating_accuracy_sz), MIN_VALUE, MAX_VALUE) for e in func]

def get_x_taylor(dimension_sz: int) -> list[float]:

    func = lambda x: x
    return [get_slope(func, 0.000001, i) for i in range(dimension_sz)]

def get_sqrt_taylor(dimension_sz: int) -> list[float]:

    func = lambda x: math.sqrt(x)
    return [get_slope(func, 0.000001, i) for i in range(dimension_sz)]

def get_inverse_taylor(dimension_sz: int) -> list[float]:

    func = lambda x: 1 / x
    return [get_slope(func, 0.000001, i) for i in range(dimension_sz)]

def get_log_taylor(dimension_sz: int) -> list[float]:

    func = lambda x: math.log(x)
    return [get_slope(func, 0.000001, i) for i in range(dimension_sz)]

def taylor_fog(f: list[float], g: list[float]) -> list[float]:

    func = lambda x: taylor_projection(f, taylor_projection(g, x))
    return [get_slope(func, 0, i) for i in range(max(len(f), len(g)))] 

def get_random_taylor(dimension_sz: int, function_sz: int) -> list[float]:

    func_arr    = [get_cos_taylor, get_exp_taylor, get_log_taylor, get_sin_taylor, get_sqrt_taylor, get_x_taylor, get_polynomial_taylor, get_inverse_taylor, get_const_taylor]

    if function_sz == 0:
        return [float(0)] * dimension_sz 

    if function_sz == 1:
        return func_arr[random.randrange(0, len(func_arr))](dimension_sz)

    lhs_sz  = random.randrange(0, function_sz - 1) + 1
    rhs_sz  = function_sz - lhs_sz

    lhs_f           = get_random_taylor(dimension_sz, lhs_sz)
    rhs_f           = get_random_taylor(dimension_sz, rhs_sz)
    random_value    = random.randrange(0, 3)

    if random_value == 0:
        return taylor_fog(lhs_f, rhs_f)

    if random_value == 1:
        return taylor_plus(lhs_f, rhs_f)

    return taylor_convolution(lhs_f, rhs_f, dimension_sz)

def taylor_values_to_operation(value_arr: list[float]) -> TaylorOperation:

    return TaylorOperation(copy.deepcopy(value_arr)) 

def min_list(a: list[float]) -> float:

    if len(a) == 0:
        return float(0)

    return min(a) 

def avg_list(a: list[float]) -> float:

    if len(a) == 0:
        return float(0)

    return sum(a) / len(a) 

def avg_invsqr_list(a: list[float]) -> float:

    if len(a) == 0:
        return float(0)

    return float(1) / (sum([float(1) / (a[i] ** 4) for i in range(len(a))]) / len(a)) #what's happening? sum(1 / x^2) would approximate the global maxima - which represents the anomaly or our <looking_for> value - 1/those would turn things upside down (minima <-> maxima) - we are looking for global minima - which is the criteria for newton_approx  

def pairwise_multiply_vector(lhs: list[float], rhs: list[float]):

    return [lhs[i] * rhs[i] for i in range(len(lhs))] 

def magnetic_equation(dimension_sz: int) -> list[str]:

    #slicing x
    #f(x)   = <x, x1, x2, x3, ...>
    #|f(x)| = x^2 + <x1, x2, x3, ...> * <x1, x2, x3, ...> = 1
    #<x1, x2, x3, ...> * <x1, x2, x3, ...> = 1 - x^2 = sin(x)

    #x^2 is fixed - detached - so we can call a recursive call

    if dimension_sz == 0:
        return []

    if dimension_sz == 1:
        return ["1"]

    sliced_equation: list[str] = magnetic_equation(dimension_sz - 1)
    return ["cos(x_%s)" % str(dimension_sz)] + ["sin(x_%s)*%s" % (str(dimension_sz), e) for e in sliced_equation] 

def rand_multidimensional_sphere_radian(dimension_sz: int) -> list[float]:

    return [2 * math.pi * random.random() for _ in range(dimension_sz)] 

def radian_coordinate_to_euclidean_coordinate(coor: list[float]) -> list[float]:

    if len(coor) == 0:
        return []

    if len(coor) == 1:
        return [float(1)]

    sliced_coor: list[float] = radian_coordinate_to_euclidean_coordinate(coor[1:])

    return [math.cos(coor[0])] + [math.sin(coor[0]) * sliced_coor[i] for i in range(len(sliced_coor))] 

def radian_rescale(org_value: float, max_range: float) -> float:

    return org_value

#let's see how we could categorize these:

#step: - exponential step
#           - exponential step with uniform random
#      - even step
#           - even step with exponential random
#           - even step with uniform random

#continuous extremes finding strategy:  - ballistic infinite (range)
#                                       - ballistic finite (range)
#                                       - ballistic fission (range)
#                                       - extremes siever
#                                       - one_dimensionalization (melee)  
#                                       - random combination of those strategies

#calibration strategies: - sin | cos calibration
#                        - gravity calibration
#                        - synth waves calibration
#                        - static + engineered model calibration (randomization of the coefficients)
#                        - random model calibration

#one dimensional approximation strategies:  - newton_naive_approx (1st order)
#                                           - newton_improved_approx (2nd order)
#                                           - newton_improved_approx + left_right step within numerical stability 
#                                           - multi-variable eqn solver 

#uncommit of bad commits strategies:        - scheduled uncommits (train falls down the track if the training stops)
#                                           - exact inverse operation of the commit (bullets cancelling out)
#                                           - combination of uncommits (Angelina Jolie's)

#we'll implement those tmr
#out goal today is actually to write this accurately - we'll implement this in C or C++ + massive parallel compute later
#we dont really know what's the best combination - we just know the techniques and let randomization figure things out - we dont have time to TIME every instrument or method

#as you could see - we split the responsibility very clearly 
#we have a space where we want to do continuous finding - we assume this space is calibrated - such the linearity is reasonable (there is no up down sin cos random waves of deviation)

#the methods we have are clear      - ballistic bullet, magnetic bullet, one-dimensionalization bullet (two rotating arms), bullet bag (shot-gun), circumscribing bullet bag
#the steps we have are also clear:  - exponential to emphasize that the <lottery_ticket_number> we are looking for is local
#                                   - linear to emphasize that the <lottery_ticket_number> we are looking for is not local

#if the space is not calibrated, we want: - calibration of space by using engineered models
#                                         - calibration of space by using random models
#                                         - calibration of space by using gravity waves
#                                         - calibration of space by using sin-cos waves (because this is common)

#the one dimensional optimizer we have:   - naive_newton_one_slope
#                                         - naive_newton_two_slope
#                                         - left_right newton_two_slope within numerical stability
#                                         - multi-variable eqn solver

#alright i'll be back after lunch break, we'll make sure to get this working in the afternoon
#we want to be able to approx every continuous function -> we move on to approx synth waves continuous function -> we move on to <synth>esize the real-world data and train our model on such
#it's not gonna be easy fellas
#God spent his life to have this universe
#what's the lottery ticket number, Old Man? i'll figure it out

#what? I have never missed a stock price prediction with this simple Taylor Approximation (every linear function -> TaylorApprox) + sorting techniques?
#remember that stock prediction only works for one stock - in the entire world of stocks
#well - the tech has far advanced from just simply getting the chart data - we read virtual machine data at CPU clock rate to extract finance data now
#in a week - we'll show you the full proof of concept and how that would forever change the stock market (im being very serious - 10000 tickers are linked together and represent every possible catalyst in the world)
#we never miss fellas - remember that you only need to be correct once for every x seconds - you dont need to be correct ALL THE TIME - it's impossible (random + chaotic)  
#trust the process fellas, dont trust the threads + memory orderings (experts only)
#in a week, we'll be releasing the 100% accurate stock prediction model, I might be bluffing, let's see 
#i miss my intern, let's see after the proof of concept if we could reconcile

#it was in the 1993 that people know of taylor series multidimensional projections (1028 bits row - 16 bits cell)
#yet the tech for centrality was limited + there was no compute back then
#we finally had all of those now - what we are missing is an algorithm to run the GMP on cuda 

#they dont have the faintest clue about the circle of money - the law of momentum conservation - the broken string analysis
#let me clue you in - every broken string has it coming - every stock is made of at least 32 strings - SPY string, random-string, trait-string, manipulation-string, world-catalyst string, habit-string (habit day of week, month of year, special occasions), fundamental string, empathy string, greed string, fear string, sector-string, cap-string, historic strings, etc. 
#                   - why is our way of sorting things and spinning around to find deviation is good? because it calibrates VERY GOOD + VERY WELL - it is an absolute technique to cancel out the strings + max-rocket system optimization
#                   - do you think <the catalyst pulls the price string> or <the price string pulls the catalyst string> (I'll let you think on this, things applied 30 years ago do not apply today)

#the most difficult string to guess is ... the random string (to accurately guess the random string, we need to accurately extract the timestamp (nanoseconds-correct) as for time-based random, accurately extract the environment of the exotic random devices - maybe humans)
#in training, we dont really care about the random string (people spent 30 years to randomize these stuffs), we need to calibrate the random string and only control what's predictable - what's in our control

#strings could be intentionally broken or statistically broken, this is when all hell breaks loose, we dont precisely know when the strings would break, why the strings would break, it is random string break?
#we are interested in the intentionally broken (low-cap stocks, penny-stocks, pump-n-dump)
#sometimes - no strings are broken (it is only broken because we are not smart + informed enough) - let this sink in for a moment

#recall our random approximation (it predicts the linearity of the <not_randomized> correctly, yet not the linearity of the random values)
#in large language model, we increase linearity by increasing dimension and duplication of context
#this is the secret sauce got us millionaires back in the days - I'm surprised it still applies
#the thing with stock is you CANT get greedy - IRS + SEC only allows you to win a certain amount of money without being suspicious (I'm talking every citizen can reasonably get 5 millions without being in the radar)

#if you wanna get a job at PandaKoin with us (we are not the more of Morgan Stanley)
#you have to be able to answer this interview question

#if you have one bullet to keep 500 murderers under control, what would you do?
#assume you know the law of physics, momentum conservation, and the law of natural exponential growth - bullet ballistic + magnetic ballistic
#i'm waiting for your answer

def exponential_randomize_frange(frange: float, exp_base: float) -> float:

    exp_first: float    = float(0) #0 for now
    exp_last: float     = math.log(frange, exp_base)
    exp_diff: float     = exp_last - exp_first
    exp_rand: float     = random.random() * exp_diff

    return exp_rand + exp_first

def uniform_randomize_frange(frange: float) -> float:

    return frange * random.random()

class Coordinate:

    def __init__(self, coor_value: list[float]):

        self.coor_value = copy.deepcopy(coor_value)

    def size(self) -> int:

        return len(self.coor_value) 

    def at(self, idx: int) -> float:

        return self.coor_value[idx] 

    def raw(self) -> list[float]:

        return copy.deepcopy(self.coor_value) 

class StepperInterface(Protocol):

    def step(self) -> float:
        ...

    def has_next_step(self) -> bool:
        ... 

    def reset(self):
        ...

class ExponentialStepper:

    def __init__(self, y0: float, exp_base: float, exp_step_sz: int): #uint

        self.y0             = y0
        self.exp_base       = exp_base
        self.exp_step_sz    = exp_step_sz
        self.cur_step_idx   = 0
    
    def step(self) -> float:

        rs                  = self.y0 + math.pow(self.exp_base, self.cur_step_idx)
        self.cur_step_idx   = min(self.exp_step_sz, self.cur_step_idx + 1)

        return rs

    def has_next_step(self) -> bool:

        return self.cur_step_idx < self.exp_step_sz

    def reset(self):

        self.cur_step_idx = 0

class LinearStepper: 

    def __init__(self, y0: float, a: float, step_sz: int):

        self.y0             = y0
        self.a              = a
        self.step_sz        = step_sz
        self.cur_step_idx   = 0

    def step(self) -> float:

        rs                  = self.y0 + self.a * self.cur_step_idx
        self.cur_step_idx   = min(self.step_sz, self.cur_step_idx + 1)

        return rs

    def has_next_step(self) -> bool:

        return self.cur_step_idx < self.step_sz

    def reset(self):

        self.cur_step_idx = 0 

class RandomizerInterface(Protocol):

    def randomize(self) -> float:
        ...

class ExponentialRandomizerInterface:

    def __init__(self, y0: float, frange: float, exp_base: float):
        
        self.y0         = y0
        self.frange     = frange
        self.exp_base   = exp_base

    def randomize(self) -> float:
        
        return exponential_randomize_frange(self.frange, self.exp_base) + self.y0

class UniformRandomizerInterface:

    def __init__(self, y0: float, frange: float):

        self.y0         = y0
        self.frange     = frange

    def randomize(self) -> float:
        
        return uniform_randomize_frange(self.frange) + self.y0

class BallisticDeviceInterface(Protocol):

    def shoot(self, t: float) -> list[Coordinate]:
        ...

class BulletBallisticDevice:

    def __init__(self, coor: Coordinate):

        self.coor = Coordinate(coor.raw()) 

    def shoot(self, t: float) -> list[Coordinate]:

        raw_coor: list[float]       = self.coor.raw()
        scaled_coor: list[float]    = scalar_multiply_vector(t, raw_coor)

        return [Coordinate(scaled_coor)]        

class SphereMagneticBallisticDevice:

    def __init__(self, s0_rad_coor: Coordinate, direction_vec: Coordinate, frequency_coeff: float, r: float):

        self.s0_rad_coor        = Coordinate(s0_rad_coor.raw())
        self.direction_vec      = Coordinate(direction_vec.raw())
        self.frequency_coeff    = frequency_coeff
        self.r                  = r

    def shoot(self, t: float) -> list[Coordinate]:

        raw_s0_rad_coor: list[float]            = self.s0_rad_coor.raw()
        raw_dir_rad_coor: list[float]           = self.direction_vec.raw()

        bullet_rad_coor: list[float]            = add_vector(raw_s0_rad_coor, scalar_multiply_vector(self.frequency_coeff * t, raw_dir_rad_coor))
        bullet_euclid_coor: list[float]         = radian_coordinate_to_euclidean_coordinate(bullet_rad_coor)
        scaled_bullet_euclid_coor: list[float]  = scalar_multiply_vector(self.r, bullet_euclid_coor)

        return [Coordinate(scaled_bullet_euclid_coor)]

class RandomSphereMagneticBallisticDevice(SphereMagneticBallisticDevice):

    def __init__(self, dimension_sz: int, max_r: float, max_frequency_coeff: float):

        s0_rad_coor: list[float]        = rand_multidimensional_sphere_radian(dimension_sz)
        directional_vec: list[float]    = get_random_vector(dimension_sz)
        r: float                        = uniform_randomize_frange(max_r)
        frequency_coeff: float          = uniform_randomize_frange(max_frequency_coeff)

        super().__init__(Coordinate(s0_rad_coor), Coordinate(directional_vec), frequency_coeff, r)

class SpheroidMagneticBallisticDevice:

    def __init__(self, s0_rad_coor: Coordinate, direction_vec: Coordinate, frequency_coeff: float, oval_shape: Coordinate):

        self.s0_rad_coor        = Coordinate(s0_rad_coor.raw())
        self.direction_vec      = Coordinate(direction_vec.raw())
        self.frequency_coeff    = frequency_coeff
        self.oval_shape         = Coordinate(oval_shape.raw()) 

    def shoot(self, t: float) -> list[Coordinate]:

        raw_s0_rad_coor: list[float]            = self.s0_rad_coor.raw()
        raw_dir_rad_coor: list[float]           = self.direction_vec.raw()

        bullet_rad_coor: list[float]            = add_vector(raw_s0_rad_coor, scalar_multiply_vector(self.frequency_coeff * t, raw_dir_rad_coor))
        bullet_euclid_coor: list[float]         = radian_coordinate_to_euclidean_coordinate(bullet_rad_coor)
        scaled_bullet_euclid_coor: list[float]  = pairwise_multiply_vector(self.oval_shape.raw(), bullet_euclid_coor)

        return [Coordinate(scaled_bullet_euclid_coor)]

class RandomSphroidMagneticBallisticDevice(SpheroidMagneticBallisticDevice):

    def __init__(self, dimension_sz: int, max_r: float, max_frequency_coeff: float):

        s0_rad_coor: list[float]        = rand_multidimensional_sphere_radian(dimension_sz)
        directional_vec: list[float]    = get_random_vector(dimension_sz)
        r: float                        = uniform_randomize_frange(max_r)
        frequency_coeff: float          = uniform_randomize_frange(max_frequency_coeff)
        oval_shape: list[float]         = scalar_multiply_vector(r, get_random_vector(dimension_sz)) 

        super().__init__(Coordinate(s0_rad_coor), Coordinate(directional_vec), frequency_coeff, Coordinate(oval_shape)) 

class RandomTwoArmsBallisticDevice:

    def __init__(self, dimension_sz: int, 
                 max_arm1_length: float, max_arm1_frequency_coeff: float, 
                 max_arm2_length: float, max_arm2_frequency_coeff: float):

        self.rotating_arm1: BallisticDeviceInterface  = RandomSphereMagneticBallisticDevice(dimension_sz, max_arm1_length, max_arm1_frequency_coeff)
        self.rotating_arm2: BallisticDeviceInterface  = RandomSphereMagneticBallisticDevice(dimension_sz, max_arm2_length, max_arm2_frequency_coeff)

    def shoot(self, t: float) -> list[Coordinate]:

        arm1_coor: Coordinate = self.rotating_arm1.shoot(t)[0]
        arm2_coor: Coordinate = self.rotating_arm2.shoot(t)[0]

        return [Coordinate(add_vector(arm1_coor.raw(), arm2_coor.raw()))]

class RandomThreeArmBallisticDevice:

    def __init__(self, dimension_sz: int, 
                 max_arm1_length: float, max_arm1_frequency_coeff: float, 
                 max_arm2_length: float, max_arm2_frequency_coeff: float, 
                 max_arm3_length: float, max_arm3_frequency_coeff: float):

        self.rotating_arm1: BallisticDeviceInterface = RandomSphroidMagneticBallisticDevice(dimension_sz, max_arm1_length, max_arm1_frequency_coeff)
        self.rotating_arm2: BallisticDeviceInterface = RandomSphroidMagneticBallisticDevice(dimension_sz, max_arm2_length, max_arm2_frequency_coeff)
        self.rotating_arm3: BallisticDeviceInterface = RandomSphroidMagneticBallisticDevice(dimension_sz, max_arm3_length, max_arm3_frequency_coeff)

    def shoot(self, t: float) -> list[Coordinate]:

        arm1_coor: Coordinate = self.rotating_arm1.shoot(t)[0]
        arm2_coor: Coordinate = self.rotating_arm2.shoot(t)[0]
        arm3_coor: Coordinate = self.rotating_arm3.shoot(t)[0]

        return [Coordinate(add_vector(add_vector(arm1_coor.raw(), arm2_coor.raw()), arm3_coor.raw()))] 

class StaticPointBagBallisticDevice:

    def __init__(self, point_bag: list[Coordinate]):

        self.point_bag = [Coordinate(point.raw()) for point in point_bag]

    def shoot(self, t: float) -> list[Coordinate]:

        return [Coordinate(e.raw()) for e in self.point_bag]

class UniformRandomStaticPointBagBallisticDevice(StaticPointBagBallisticDevice):

    def __init__(self, dimension_sz: int, bag_sz: int, y_range: float):

        point_bag: list[Coordinate] = [Coordinate(scalar_multiply_vector(y_range, get_random_vector(dimension_sz))) for _ in range(bag_sz)] 
        super().__init__(point_bag)

class UniformRandomStaticSpherePointBagBallisticDevice(StaticPointBagBallisticDevice):

    def __init__(self, dimension_sz: int, bag_sz: int, r: float):

        point_bag: list[Coordinate] = [Coordinate(scalar_multiply_vector(r, radian_coordinate_to_euclidean_coordinate(rand_multidimensional_sphere_radian(dimension_sz)))) for _ in range(bag_sz)] 
        super().__init__(point_bag)

class CircumscribingStaticPointBagBallisticDevice:

    def __init__(self, point_bag: list[Coordinate], frequency_coeff: float):

        self.ballistic          = StaticPointBagBallisticDevice(point_bag)
        self.frequency_coeff    = frequency_coeff

    def shoot(self, t: float) -> list[Coordinate]:

        point_bag: list[Coordinate] = self.ballistic.shoot(t)
        return [Coordinate(scalar_multiply_vector(self.frequency_coeff * t, point.raw())) for point in point_bag] 

class ChainedBallisticDevice:

    def __init__(self, ballistic_device_arr: list[BallisticDeviceInterface]):

        self.ballistic_device_arr: list[BallisticDeviceInterface] = ballistic_device_arr

    def shoot(self, t: float) -> list[Coordinate]:

        rs: list[list[Coordinate]] = []

        for i in range(len(self.ballistic_device_arr)):
            rs += [self.ballistic_device_arr[i].shoot(t)]

        return self._combinatorial_reduce_add(rs)

    def _combinatorial_reduce_add(self, inp: list[list[Coordinate]]) -> list[Coordinate]:

        if len(inp) == 0:
            return []

        def _internal_reduce(lhs: list[Coordinate], rhs: list[Coordinate]) -> list[Coordinate]:

            rs: list[Coordinate] = []

            for e_lhs in lhs:
                for e_rhs in rhs:
                    e_rs    = add_vector(e_lhs.raw(), e_rhs.raw()) 
                    rs      += [Coordinate(e_rs)]

            return rs

        return functools.reduce(_internal_reduce, inp[1:], inp[0])

def get_random_twoarms_ballistic_device(dimension_sz: int, _range: float) -> BallisticDeviceInterface:

    arm1_frequency_coeff: float = (random.random() * 10) ** random.randrange(10)
    arm2_frequency_coeff: float = (random.random() * 10) ** random.randrange(10)

    return RandomTwoArmsBallisticDevice(dimension_sz, 
                                        _range, arm1_frequency_coeff,
                                        _range, arm2_frequency_coeff)

def get_random_threearms_ballistic_device(dimension_sz: int, _range: float) -> BallisticDeviceInterface:

    arm1_frequency_coeff: float = (random.random() * 10) ** random.randrange(10)
    arm2_frequency_coeff: float = (random.random() * 10) ** random.randrange(10)
    arm3_frequency_coeff: float = (random.random() * 10) ** random.randrange(10)

    return RandomThreeArmBallisticDevice(dimension_sz, 
                                         _range, arm1_frequency_coeff, 
                                         _range, arm2_frequency_coeff,
                                         _range, arm3_frequency_coeff)

def get_random_circumscribing_ballistic_device(dimension_sz: int, _range: float, bag_sz_range: int = 16, bag_sz_min: int = 1) -> BallisticDeviceInterface:

    bag_sz: int                 = min(random.randrange(bag_sz_range), bag_sz_min)
    radius: int                 = uniform_randomize_frange(_range)
    frequency_coeff: float      = (random.random() * 10) ** random.randrange(10)
    point_bag: list[Coordinate] = [Coordinate(scalar_multiply_vector(radius, radian_coordinate_to_euclidean_coordinate(rand_multidimensional_sphere_radian(dimension_sz)))) for _ in range(bag_sz)] 

    return CircumscribingStaticPointBagBallisticDevice(point_bag, frequency_coeff)

def get_random_point_bag_ballistic_device(dimension_sz: int, _range: float, bag_sz_range: int = 16, bag_sz_min: int = 1) -> BallisticDeviceInterface:

    bag_sz: int = min(random.randrange(bag_sz_range), bag_sz_min)
    radius: int = uniform_randomize_frange(_range)

    return UniformRandomStaticPointBagBallisticDevice(dimension_sz, bag_sz, radius)

def get_random_melee_ballistic_device(dimension_sz: int, _range: float) -> BallisticDeviceInterface:

    device_sz_range: int    = 3
    device_sz_min: int      = 1
    device_sz: int          = max(random.randrange(device_sz_range), device_sz_min)

    device_list: list[BallisticDeviceInterface]                                     = []
    random_melee_device_gen: list[Callable[[int, float], BallisticDeviceInterface]] = [get_random_twoarms_ballistic_device, get_random_threearms_ballistic_device, get_random_circumscribing_ballistic_device, get_random_point_bag_ballistic_device]

    for _ in range(device_sz):
        idx = random.randrange(0, len(random_melee_device_gen))
        device: BallisticDeviceInterface = random_melee_device_gen[idx](dimension_sz, _range)
        device_list += [device]

    return ChainedBallisticDevice(device_list) 

def get_random_bullet_ballistic_device(dimension_sz: int) -> BallisticDeviceInterface:
    
    return BulletBallisticDevice(Coordinate(get_random_vector(dimension_sz)))

def get_random_spheremagnetic_ballistic_device(dimension_sz: int, _range: float) -> BallisticDeviceInterface:
    
    frequency_coeff: float = (random.random() * 10) ** random.randrange(10)
    return RandomSphereMagneticBallisticDevice(dimension_sz, _range, frequency_coeff)

def get_random_spheroidmagnetic_ballistic_device(dimension_sz: int, _range: float) -> BallisticDeviceInterface:

    frequency_coeff: float = (random.random() * 10) ** random.randrange(10)
    return RandomSphroidMagneticBallisticDevice(dimension_sz, _range, frequency_coeff) 

def get_random_range_ballistic_device(dimension_sz: int, magnetic_radius: float) -> BallisticDeviceInterface:
    
    infinite_ballistic_gen: list    = [get_random_bullet_ballistic_device]
    finite_ballistic_gen: list      = [get_random_spheremagnetic_ballistic_device, get_random_spheroidmagnetic_ballistic_device]

    inifite_device: BallisticDeviceInterface    = infinite_ballistic_gen[random.randrange(0, len(infinite_ballistic_gen))](dimension_sz)
    finite_device: BallisticDeviceInterface     = finite_ballistic_gen[random.randrange(0, len(finite_ballistic_gen))](dimension_sz, magnetic_radius)

    return ChainedBallisticDevice([inifite_device, finite_device])

def get_random_rangemelee_ballistic_device(dimension_sz: int, magnetic_radius: float, melee_range: float) -> BallisticDeviceInterface:

    range_device: BallisticDeviceInterface = get_random_range_ballistic_device(dimension_sz, magnetic_radius)
    melee_device: BallisticDeviceInterface = get_random_melee_ballistic_device(dimension_sz, melee_range)

    return ChainedBallisticDevice([range_device, melee_device])

def get_random_ballistic_device(dimension_sz: int) -> BallisticDeviceInterface:

    magnetic_radius: float  = (random.random() * 10) ** random.randrange(0, 10)
    melee_range: float      = (random.random() * 10) ** random.randrange(0, 10)
    rand_value: int         = random.randrange(0, 3)

    if rand_value == 0:
        return get_random_melee_ballistic_device(dimension_sz, melee_range)

    if rand_value == 1:
        return get_random_range_ballistic_device(dimension_sz, magnetic_radius)

    return get_random_rangemelee_ballistic_device(dimension_sz, magnetic_radius, melee_range)  

class CalibrationDeviceInterface(Protocol):

    def calibrate(self, coor: Coordinate) -> Coordinate:
        ...

#af(g(x))
class RandomModelCalibrationDevice:

    def __init__(self, dimension_sz: int, random_func_sz: int, a_dimension_sz: int):

        self.taylor_model   = get_random_taylor(dimension_sz, random_func_sz)
        self.a_dimension_sz = a_dimension_sz

    def calibrate(self, coor: Coordinate) -> Coordinate:

        a, b = split_vector(coor.raw(), self.a_dimension_sz)
        return Coordinate(taylor_convolution(a, taylor_fog(self.taylor_model, b)))

class SynthWaveCalibrationDevice:

    def __init__(self, dimension_sz: int, wave_amplitude: float, wave_frequency: float, pow_sz: int, a_dimension_sz: int):

        self.taylor_model   = get_powsin_gravity_taylor(dimension_sz, wave_amplitude, wave_frequency, pow_sz)
        self.a_dimension_sz = a_dimension_sz

    def calibrate(self, coor: Coordinate) -> Coordinate:

        a, b = split_vector(coor.raw(), self.a_dimension_sz)
        return Coordinate(taylor_convolution(a, taylor_fog(self.taylor_model, b)))

class MaxwellCalibrationDevice:

    def __init__(self, dimension_sz: int, a_dimension_sz: int):
        
        self.taylor_model   = get_sin_taylor(dimension_sz)
        self.a_dimension_sz = a_dimension_sz
    
    def calibrate(self, coor: Coordinate) -> Coordinate:

        a, b = split_vector(coor.raw(), self.a_dimension_sz)
        return Coordinate(taylor_convolution(a, taylor_fog(self.taylor_model, b)))

class NoCalibrationDevice:

    def calibrate(self, coor: Coordinate) -> Coordinate:

        return coor

def get_random_model_calibration_device(dimension_sz: int, random_func_range: int = 4, a_dimension_sz_range: int = 4) -> CalibrationDeviceInterface:
    
    random_func_sz: int = random.randrange(0, random_func_range)
    a_dimension_sz: int = random.randrange(0, a_dimension_sz_range)

    return RandomModelCalibrationDevice(dimension_sz, random_func_sz, a_dimension_sz)

def get_random_synthwave_calibration_device(dimension_sz: int, wave_amplitude_range: float = float(16), wave_frequency_range: float = float(16), pow_sz_range: int = 8, a_dimension_sz_range: int = 4) -> CalibrationDeviceInterface:
    
    wave_amp: float     = random.random() * wave_amplitude_range
    wave_freq: float    = random.random() * wave_frequency_range
    pow_sz: int         = random.randrange(pow_sz_range)
    a_dimension_sz: int = random.randrange(0, a_dimension_sz_range)
    
    return SynthWaveCalibrationDevice(dimension_sz, wave_amp, wave_freq, pow_sz, a_dimension_sz)

def get_random_maxwell_calibration_device(dimension_sz: int, a_dimension_sz_range: int = 4) -> CalibrationDeviceInterface:

    a_dimension_sz: int = random.randrange(a_dimension_sz_range)
    
    return MaxwellCalibrationDevice(dimension_sz, a_dimension_sz)  

def get_random_calibration_device(dimension_sz: int) -> CalibrationDeviceInterface:

    random_calibration_gen = [get_random_model_calibration_device, get_random_synthwave_calibration_device, get_random_maxwell_calibration_device]
    idx: int = random.randrange(0, len(random_calibration_gen))

    return random_calibration_gen[idx](dimension_sz)

class DeviationCalculatorInterface(Protocol):

    def deviation(self, f: Callable[[float], float], instrument: Callable[[float], float]) -> float:
        ... 

class MeanSquareDeviationCalculator:

    def __init__(self, point_arr: list[float]):

        self.point_arr = copy.deepcopy(point_arr)

    def deviation(self, f: Callable[[float], float], instrument: Callable[[float], float]) -> float:

        if len(self.point_arr) == 0:
            return float(0)

        sqr_sum: float      = sum([(f(x) - instrument(x)) ** 2 for x in self.point_arr])
        denom: float        = float(len(self.point_arr))
        normalized: float   = sqr_sum / denom

        return normalized

class DecimalMeanSquareDeviationCalculator:

    def __init__(self, point_arr: list[Decimal]):

        self.point_arr = copy.deepcopy(point_arr)

    def deviation(self, f: Callable[[Decimal], Decimal], instrument: Callable[[Decimal], Decimal]) -> Decimal:

        if len(self.point_arr) == 0:
            return Decimal(0)

        sqr_sum: Decimal    = sum([(f(x) - instrument(x)) ** 2 for x in self.point_arr])
        denom: Decimal      = Decimal(len(self.point_arr))
        normalized: Decimal = sqr_sum / denom

        return normalized 

class DiscreteMeanSquareDeviationCalculator(MeanSquareDeviationCalculator):

    def __init__(self, first: float, last: float, discretization_sz: int):

        super().__init__(discretize(first, last, discretization_sz))

class DiscreteDecimalMeanSquareDeviationCalculator(DecimalMeanSquareDeviationCalculator):

    def __init__(self, first: Decimal, last: Decimal, discretization_sz: int):

        super().__init__(decimal_discretize(first, last, discretization_sz)) 

def get_msqr_deviation_calculator(first: float, last: float, discretization_sz: int) -> DeviationCalculatorInterface:

    return DiscreteMeanSquareDeviationCalculator(first, last, discretization_sz)  

def get_decimal_msqr_deviation_calculator(first: Decimal, last: Decimal, discretization_sz: int) -> DiscreteDecimalMeanSquareDeviationCalculator:

    return DiscreteDecimalMeanSquareDeviationCalculator(first, last, discretization_sz) 

class NewtonOptimizerInterface(Protocol):

    def optimize(self, f: Callable[[float], float], x0: float) -> float:
        ...

class TwoOrderStepNewtonOptimizer:

    def __init__(self, stepper: StepperInterface, iteration_sz: int = 4, a: float = 0.001):

        self.stepper        = stepper
        self.iteration_sz   = iteration_sz
        self.a              = a

    def optimize(self, f: Callable[[float], float], x0: float) -> float:

        x_cursor: float                         = x0
        x_cand_list: list[tuple[float, float]]  = [(f(x0), x0)]

        while self.stepper.has_next_step():
            x_cursor    = x_cursor + self.stepper.step()
            (x_cand, y) = tom_approx2(f, self.iteration_sz, x_cursor, self.a)
            x_cand_list += [(abs(y), x_cand)]

        self.stepper.reset()
    
        return min(x_cand_list)[1]  

class URLinearTwoOrderNewtonOptimizer(TwoOrderStepNewtonOptimizer):

    def __init__(self, y0_first: float = float(0), y0_last: float = float(0), 
                 a_abs_range: float = float(10), a_abs_range_min: float = 0.001,
                 step_sz_range: int = 32, step_sz_min: int = 1,
                 iteration_sz: int = 4, derivative_offset: float = 0.001):

        y0: float       = uniform_randomize_frange(y0_last - y0_first) + y0_first
        signness: float = float(-1) if flip_a_coin() else float(1)
        a: float        = max(uniform_randomize_frange(a_abs_range), a_abs_range_min) * signness 
        step_sz: int    = max(step_sz_min, random.randrange(0, step_sz_range))

        super().__init__(LinearStepper(y0, a, step_sz), iteration_sz, derivative_offset)

class ERTwoOrderNewtonOptimizer(TwoOrderStepNewtonOptimizer):

    def __init__(self, y0_first: float = float(0), y0_last: float = float(0),
                 exp_base_range: float = float(10), exp_base_range_min: float = 1,
                 exp_step_range: int = 10, exp_step_min: int = 1,
                 iteration_sz: int = 4, derivative_offset: float = 0.001):

        y0: float       = uniform_randomize_frange(y0_last - y0_first) + y0_first
        exp_base: float = max(uniform_randomize_frange(exp_base_range), exp_base_range_min)
        exp_step: int   = max(random.randrange(0, exp_step_range), exp_step_min)

        super().__init__(ExponentialStepper(y0, exp_base, exp_step), iteration_sz, derivative_offset)

def get_random_linear_twoorder_newton_optimizer() -> NewtonOptimizerInterface:

    a_abs_range: float = (random.random() * 10) ** random.randrange(0, 10)
    return URLinearTwoOrderNewtonOptimizer(0, 0, a_abs_range)

def get_random_exponential_twoorder_newton_optimizer() -> NewtonOptimizerInterface:

    return ERTwoOrderNewtonOptimizer() 

def get_random_twoorder_newton_optimizer() -> NewtonOptimizerInterface:

    if flip_a_coin():
        return get_random_linear_twoorder_newton_optimizer()
    else:
        return get_random_exponential_twoorder_newton_optimizer() 

class OneDimensionalFunctionInterface(Protocol):

    def __call__(self, x: float) -> float:
        ...

class TaylorSeriesFunctionizer:

    def __init__(self, taylor_series: list[float]):

        self.taylor_series = taylor_series

    def __call__(self, x: float) -> float:

        return taylor_projection(self.taylor_series, x)

class DecimalTaylorSeriesFunctionizer:

    def __init__(self, taylor_series: list[Decimal]):
        
        self.taylor_series = taylor_series
    
    def __call__(self, x: Decimal) -> Decimal:

        return decimal_taylor_compute(self.taylor_series, x) 

def ballistic_optimize(taylor_model: list[float], ballistic_device: BallisticDeviceInterface,
                       instrument: Callable[[float], float], instrument_x_range: float, instrument_discretization_sz: int) -> tuple[list[float], float]:

    deviation_calculator: DeviationCalculatorInterface  = get_msqr_deviation_calculator(float(0), instrument_x_range, instrument_discretization_sz)
    optimizer: NewtonOptimizerInterface                 = get_random_twoorder_newton_optimizer()
    x0: float                                           = float(0)

    def _deviation_negotiator(t: float) -> float:

        ballistic_coor_list: list[Coordinate]   = ballistic_device.shoot(t)
        calibrated_coor_list: list[Coordinate]  = [Coordinate(add_vector(taylor_model, coor.raw())) for coor in ballistic_coor_list] 
        deviation_list: list[float]             = []

        for taylor_model_coor in calibrated_coor_list:
            taylor_function: OneDimensionalFunctionInterface = TaylorSeriesFunctionizer(taylor_model_coor.raw())
            deviation: float    = deviation_calculator.deviation(taylor_function, instrument)
            deviation_list      += [deviation]

        return avg_invsqr_list(deviation_list)
    
    x: float                                = optimizer.optimize(_deviation_negotiator, x0)
    ballistic_coor_list: list[Coordinate]   = ballistic_device.shoot(x)
    calibrated_coor_list: list[Coordinate]  = [Coordinate(add_vector(taylor_model, coor.raw())) for coor in ballistic_coor_list]
    deviation_list: list[float]             = []

    for taylor_model_coor in calibrated_coor_list:
        taylor_function: OneDimensionalFunctionInterface = TaylorSeriesFunctionizer(taylor_model_coor.raw())
        deviation: float    = deviation_calculator.deviation(taylor_function, instrument)
        deviation_list      += [(deviation, taylor_model_coor.raw())]

    if len(deviation_list) == 0:
        return (taylor_model, sys.float_info.max)

    return (min(deviation_list)[1], min(deviation_list)[0])

def range_ballistic_optimize(taylor_model: list[float],
                             instrument: Callable[[float], float], instrument_x_range: float, instrument_discretization_sz: int,
                             magnetic_range: float = float(16)):
    
    return ballistic_optimize(taylor_model, get_random_range_ballistic_device(len(taylor_model), magnetic_range), instrument, instrument_x_range, instrument_discretization_sz)

def melee_ballistic_optimize(taylor_model: list[float],
                             instrument: Callable[[float], float], instrument_x_range: float, instrument_discretization_sz: int,
                             melee_range: float = float(16)) -> tuple[list[float], float]:

    return ballistic_optimize(taylor_model, get_random_melee_ballistic_device(len(taylor_model), melee_range), instrument, instrument_x_range, instrument_discretization_sz)

def rangemelee_ballistic_optimize(taylor_model: list[float],
                                  instrument: Callable[[float], float], instrument_x_range: float, instrument_discretization_sz: int,
                                  magnetic_range: float = float(16), melee_range: float = float(16)):

    return ballistic_optimize(taylor_model, get_random_rangemelee_ballistic_device(len(taylor_model), magnetic_range, melee_range), instrument, instrument_x_range, instrument_discretization_sz)

#this is not the right approach - we'll improve this later - it's very super complicated
def calibrated_ballistic_optimize(taylor_model: list[float], ballistic_device: BallisticDeviceInterface,
                                  calibration_device: CalibrationDeviceInterface,
                                  instrument: Callable[[float], float], instrument_x_range: float, instrument_discretization_sz: int) -> tuple[list[float], float]:

    deviation_calculator: DeviationCalculatorInterface  = get_msqr_deviation_calculator(float(0), instrument_x_range, instrument_discretization_sz)
    optimizer: NewtonOptimizerInterface                 = get_random_twoorder_newton_optimizer()
    x0: float                                           = float(0)

    def _deviation_negotiator(t: float) -> float:

        ballistic_coor_list: list[Coordinate]   = ballistic_device.shoot(t)
        calibrated_coor_list: list[Coordinate]  = [Coordinate(add_vector(taylor_model, calibration_device.calibrate(coor).raw())) for coor in ballistic_coor_list] 
        deviation_list: list[float]             = []

        for taylor_model_coor in calibrated_coor_list:
            taylor_function: OneDimensionalFunctionInterface = TaylorSeriesFunctionizer(taylor_model_coor.raw())
            deviation: float    = deviation_calculator.deviation(taylor_function, instrument)
            deviation_list      += [deviation]

        return avg_invsqr_list(deviation_list)

    x: float                                = optimizer.optimize(_deviation_negotiator, x0)
    ballistic_coor_list: list[Coordinate]   = ballistic_device.shoot(x)
    calibrated_coor_list: list[Coordinate]  = [Coordinate(add_vector(taylor_model, calibration_device.calibrate(coor).raw() )) for coor in ballistic_coor_list] 
    deviation_list: list[float]             = []

    for taylor_model_coor in calibrated_coor_list:
        taylor_function: OneDimensionalFunctionInterface = TaylorSeriesFunctionizer(taylor_model_coor.raw())
        deviation: float    = deviation_calculator.deviation(taylor_function, instrument)
        deviation_list      += [(deviation, taylor_model_coor.raw())]

    if len(deviation_list) == 0:
        return (taylor_model, sys.float_info.max)

    return (min(deviation_list)[1], min(deviation_list)[0])
    

def calibrated_range_ballistic_optimize(taylor_model: list[float],
                                        instrument: Callable[[float], float], instrument_x_range: float, instrument_discretization_sz: int,
                                        magnetic_range: float = float(16)):
    
    return calibrated_ballistic_optimize(taylor_model,
                                         get_random_range_ballistic_device(len(taylor_model), magnetic_range),
                                         get_random_calibration_device(len(taylor_model)),
                                         instrument, instrument_x_range, instrument_discretization_sz)

def calibrated_melee_ballistic_optimize(taylor_model: list[float],
                                        instrument: Callable[[float], float], instrument_x_range: float, instrument_discretization_sz: int,
                                        melee_range: float = float(16)):
    
    return calibrated_ballistic_optimize(taylor_model,
                                         get_random_melee_ballistic_device(len(taylor_model), melee_range),
                                         get_random_calibration_device(len(taylor_model)),
                                         instrument, instrument_x_range, instrument_discretization_sz)

def calibrated_rangemelee_ballistic_optimize(taylor_model: list[float],
                                             instrument: Callable[[float], float], instrument_x_range: float, instrument_discretization_sz: int,
                                             magnetic_range: float = float(16), melee_range: float = float(16)):
    
    return calibrated_ballistic_optimize(taylor_model,
                                         get_random_rangemelee_ballistic_device(len(taylor_model), magnetic_range, melee_range),
                                         get_random_calibration_device(len(taylor_model)),
                                         instrument, instrument_x_range, instrument_discretization_sz)

def get_initial_taylor_model(derivative_order_sz: int) -> list[float]:

    return [float(0)] * derivative_order_sz 

#the goal is to do educated random - such is the coverage for every possible case must be reasonably distributed
#alright its complicated - we realized that it only works because of agile iterative process of optimization not engineering practices

def train(taylor_model: list[float], 
          instrument: Callable[[float], float], instrument_x_range: int, instrument_discretization_sz: int,
          directional_optimization_sz: int, training_epoch_sz: int) -> list[float]:

    optimizing_taylor_model: list[float]                = copy.deepcopy(taylor_model)
    deviation_calculator: DeviationCalculatorInterface  = get_msqr_deviation_calculator(float(0), instrument_x_range, instrument_discretization_sz) 

    for _ in range(training_epoch_sz):
        round_rs: list[tuple[list[float], float]] = []

        for __ in range(directional_optimization_sz):
            random_value: int = random.randrange(0, 6)

            try:
                if random_value == 0:
                    (new_taylor_model, deviation_hint) = range_ballistic_optimize(optimizing_taylor_model, instrument, instrument_x_range, instrument_discretization_sz, (random.random() * 10) ** random.randrange(0, 10))
                    round_rs += [(deviation_hint, new_taylor_model)]
                elif random_value == 1:
                    (new_taylor_model, deviation_hint) = melee_ballistic_optimize(optimizing_taylor_model, instrument, instrument_x_range, instrument_discretization_sz, (random.random() * 10) ** random.randrange(0, 10))
                    round_rs += [(deviation_hint, new_taylor_model)]
                elif random_value == 2:
                    (new_taylor_model, deviation_hint) = rangemelee_ballistic_optimize(optimizing_taylor_model, instrument, instrument_x_range, instrument_discretization_sz, (random.random() * 10) ** random.randrange(0, 10), (random.random() * 10) ** random.randrange(0, 10))
                    round_rs += [(deviation_hint, new_taylor_model)]
                elif random_value == 3:
                    (new_taylor_model, deviation_hint) = calibrated_range_ballistic_optimize(optimizing_taylor_model, instrument, instrument_x_range, instrument_discretization_sz, (random.random() * 10) ** random.randrange(0, 10))
                    round_rs += [(deviation_hint, new_taylor_model)]
                elif random_value == 4:
                    (new_taylor_model, deviation_hint) = calibrated_melee_ballistic_optimize(optimizing_taylor_model, instrument, instrument_x_range, instrument_discretization_sz, (random.random() * 10) ** random.randrange(0, 10))
                    round_rs += [(deviation_hint, new_taylor_model)]
                elif random_value == 5:
                    (new_taylor_model, deviation_hint) = calibrated_rangemelee_ballistic_optimize(optimizing_taylor_model, instrument, instrument_x_range, instrument_discretization_sz, (random.random() * 10) ** random.randrange(0, 10), (random.random() * 10) ** random.randrange(0, 10))
                    round_rs += [(deviation_hint, new_taylor_model)]

            except Exception as e:
                print(e)

        if len(round_rs) == 0:
            continue

        (best_deviation_hint, cand_taylor_model)    = min(round_rs)
        best_deviation_real                         = deviation_calculator.deviation(TaylorSeriesFunctionizer(cand_taylor_model), instrument)
        current_deviation                           = deviation_calculator.deviation(TaylorSeriesFunctionizer(optimizing_taylor_model), instrument)

        if best_deviation_real < current_deviation:
            optimizing_taylor_model = cand_taylor_model
            print("update", best_deviation_real)
        else:
            print("keep", current_deviation)

    return optimizing_taylor_model

#what we know is that Taylor Series is only good for:
#(1): plus
#(2): multiply
#(3): trig
#(4): exponential
#(5): subtract

#we must stay in the territory in order to get stable <root_finding> by using differential methods   
#what we are going to do today is finding the very high order root, to be specific 1 << 20 -> 1 << 30 newton root + using one dimensionalization, circumscribing rotating magnetic, or ballistic to find our answer for global extremes

def decimal_sqrt(x: Decimal, x0: Decimal = Decimal(1), iteration_sz: int = 64) -> Decimal:

    rs: Decimal     = Decimal(0)
    coeff: Decimal  = Decimal(1)
    exp: Decimal    = Decimal(-1)

    for i in range(iteration_sz):
        a       = coeff * (x0 ** exp)
        coeff   *= exp
        exp     -= 1
        rs      += Decimal(1) / math.factorial(i) * a * (x ** i)

    return rs

def decimal_e(x: Decimal, iteration_sz: int = 1024) -> Decimal:

    #f0 + f'(0) * x + 1/2 * f''(0) * x^2

    rs: Decimal = Decimal(0)

    for i in range(iteration_sz):
        rs += Decimal(1) / math.factorial(i) * Decimal(x)**i  

    return rs 

def decimal_sin(x: Decimal, iteration_sz: int = 256) -> Decimal:

    #alright let's see - sin(x), cos(x), -sin(x), -cos(x)
    #0 1 0 -1 0 1 0 -1
    #so it is 

    coeff: int      = 1
    rs: Decimal     = Decimal(0)

    for i in range(iteration_sz):
        signness:int    = 1 if i % 2 == 0 else -1  
        idx: int        = i * 2 + 1 
        rs              += Decimal(1) / math.factorial(idx) * Decimal(signness * coeff) * (x ** idx)

    return rs

def decimal_cos(x: Decimal, iteration_sz: int = 256) -> Decimal:

    #cos = sin(90 - x) yet I want to implement taylor series for this
    #cos(x), -sin(x), -cos(x), sin(x), cos(x)
    #1, 0, -1, 0, 1

    coeff: int      = 1
    rs: Decimal     = Decimal(0)

    for i in range(iteration_sz):
        signness        = 1 if i % 2 == 0 else -1
        idx             = i * 2
        pow: Decimal    = x ** Decimal(idx) if idx != 0 else Decimal(1)
        rs              += Decimal(1) / math.factorial(idx) * Decimal(signness * coeff) * pow

    return rs

def decimal_synth_wave(x: Decimal, iteration_sz: int = 64) -> Decimal:

    return decimal_sin(x - Decimal(16), iteration_sz) / (x - Decimal(16))

def decimal_get_slope(f: Callable[[Decimal], Decimal], x: Decimal, derivative_order: int, a: Decimal = Decimal(0.0000001)):

    if derivative_order == 0:
        return f(x)

    return (decimal_get_slope(f, x + a, derivative_order - 1, a) - decimal_get_slope(f, x, derivative_order - 1, a)) / a

def decimal_taylorize(f: Callable[[Decimal], Decimal], derivative_order_sz: int, a: Decimal = Decimal(0.0000001)) -> list[Decimal]:

    return [decimal_get_slope(f, Decimal(0.01), i, a) for i in range(derivative_order_sz)]

def decimal_taylor_compute(coeff_arr: list[Decimal], x: Decimal) -> Decimal:

    rs: Decimal = Decimal(0)

    for i in range(len(coeff_arr)):

        factorial: Decimal  = Decimal(1) / Decimal(math.factorial(i))
        pow: Decimal        = x ** Decimal(i) if i != 0 else Decimal(1)
        rs                  += factorial * coeff_arr[i] * pow


    return rs

def taylorize(f: Callable[[float], float], derivative_order_sz: int, a: float = 0.0001):

    return [get_slope(f, 0, i, a) for i in range(derivative_order_sz)]

#

def decimal_get_slopes(f: Callable[[Decimal], Decimal], x: Decimal, derivative_order_sz: int, a: Decimal = Decimal(0.000000001)) -> list[Decimal]:

    def _get_derivative(point_arr: list[tuple[Decimal, Decimal]]) -> list[tuple[Decimal, Decimal]]:
        
        if len(point_arr) == 0:
            return []
        
        iteration_sz: int                   = len(point_arr) - 1
        rs: list[tuple[Decimal, Decimal]]   = []

        for i in range(iteration_sz):
            current_x, current_y    = point_arr[i]
            next_x, next_y          = point_arr[i + 1]
            delta_x: Decimal        = next_x - current_x
            delta_y: Decimal        = next_y - current_y
            slope: Decimal          = delta_y / delta_x

            rs                      += [(current_x, slope)]

        return rs 

    collected_points: list[Decimal] = []
    rs: list[Decimal]               = []

    for i in range(derivative_order_sz):
        current_x: Decimal  = x + a * i
        collected_points    += [(current_x, f(current_x))]

    for i in range(derivative_order_sz):
        rs                  += [collected_points[0][1]]
        collected_points    = _get_derivative(collected_points)

    return rs

def decimal_radian_coordinate_to_euclidean_coordinate(coor: list[Decimal]) -> list[Decimal]:

    if len(coor) == 0:
        return []

    if len(coor) == 1:
        return [Decimal(1)]

    sliced_coor: list[Decimal] = decimal_radian_coordinate_to_euclidean_coordinate(coor[1:])

    return [decimal_cos(coor[0])] + [decimal_sin(coor[0]) * sliced_coor[i] for i in range(len(sliced_coor))] 

distance_vec: list[float] = []

def main():

    #let's talk about finite sequences, infinite sequences, the convolution of infinite sequence + finite sequence Taylor Series
    #the calibrated space - the cue in the calibrated space (we need to move the cue more than once to achieve higher accuracy)
    #the finite sequence as we know it is our current Taylor Series
    #the infinite sequence as we know it is sin, cos, 1/x, sqrt(x), log(x), e^x
    #what is the general form of infinite sequence?
    #we probably don't want to <calibrate> our finite space of Taylor Series to achieve infinite pattern result (it's ... not smart)
    #now consider this, assume everything can be represented as <1, 0, -1, 0, 1, 0, -1...>
    #we want to construct EVERY possible Taylor Series by using such sequence, how would we do that? this is a dynamic programming problem of left + right shift of sin | cos (trig) Taylor Series 
    #... or do we even need a shift? is it a good ol' induction + dynamic programming problem of space construction? 
    #assume we are moving from left to right, the left space is fully constructed, now construct the current space using sin or cos wave 
    #there are three particular infinite sequences that we want to know

    #the diverging sequence (e^x, etc.) - we dont care about this
    #the converging sequence (1/x, ln(x), log(x))
    #the neither sequence (sin + cos)
    
    #now what if the infinite sequence pattern is extractable?
    #what's our smart move to approx every possible extractable infinite sequence? 
    #can we reasonably prove that every extractable infinite sequence must be the product of a convolution of a finite sequence and an infinite sequence?
    #we want to work on this theory today and tomorrow

    #I've spent yesterday thinking
    #its called Taylor Series compression

    #recall that our semantic space is finite, we choose what to include in the semantic space
    #what's the difference between choosing 4 order Taylor derivatives vs sin, cos, exp + their coefficients? there aren't differences in terms of shape variety, but there are differences in terms of likelyhood of shapes that are going to happen   

    #alright - consider this x1*sin(x) + x2*e^x + x3*1/x
    #>x1, x2, x3> in this case bijectively relates to the result, what the difference between this and the 3 Taylor derivative orders?
    #there is not a difference in terms of shape variety - yet there is a difference in terms of preferred pool of shapes 
    #<x1,x2,x3> in this case is a form of Taylor Series compression, because it compresses infinite Taylor patterns into finite variables (x1, x2, x3)
    #Taylor Series is a complete continuity compression method if the derivative order is to approach infinity

    #we dont have that infinite storage + compute, so we must map our domain space to another space where we could represent the likely <preferred_pool_of_shapes> better
    #this poses a problem of numerical stability, assume we have found our Taylor Series, we know that the Taylor Series if added to the original Taylor Series would bring the deviation -> 0.01, without loss of generality
    #yet our compression method would <float> the deviation -> 0.02, 0.03
    #alright, do we want to find the Taylor Series and compress it or we find the coordinate in the already compressed domain? The latter guarantees numerical stability, because we are operating on the ground truth (yet requires the domain <-> range to be continuous)
    #this means that multiple <semantic_domain space> are required, this sounds a lot like models and their trig functions (sin, cos, tan, etc.), exps, etc., because it really is, just in a generalized form

    #why Taylor Series? because Taylor Series allows us to do iterative calibration, by taking deviation with respect to the instrument
    #Taylor Series is continuity compression complete
    #Taylor Series is good, yet we need to float + map the domain space of <x, x1, x2> as in <derivative_order_0th, derivative_order_1st, derivative_order_2nd> -> another engineered domain space
    #if there exists a better method, it must not be logically less complex than doing Taylor Series compression
        #proof by contradiction, assume it is logically less complex than TaylorSeries compression and it is continuity comnpression complete
        #=> there must be a way to <map> the solution to our Taylor Series ...
        #so the process can be a subprocess of the Taylor compression
        #=> Taylor compression has equal logical complexity
        #=> contradicts to the orginal statement

    #I was thinking, what if we do a preface of data to analyze the best semantic space
    #its actually complex, with the centrality algorithm (x = x + f(x)), things move around really quickly
    #we'll dwell on this today

    #compression is intellect, is our finite pool of Taylor Series representing finite infinite pool of Taylor Series?   
    #we dont really know the butterfly effects, yet we must get the basic of continuity compression right

    #Taylor Series has one drawback, however, it is the infinity trap 
    #it seems like continuity discontinues at infinity
    #we have to <hack> this by using L'Hospital rule 
    #recall that if f(x) and g(x) are to both approach infinity or 0s then we could take the derivative of f(x) and g(x) to calculate the result
    #goal is to turn tan(x) -> sin(x) and cos(x) to avoid infinity traps, by compute them separately and find the division of the lhs and rhs
    #so is the Taylor Series sufficient or is it not, it is sufficient in the space where there is no infinity, and we have to find such space to do Taylor Series compression
    #now is the confusing part - do we analyze the space or we train the space? if we are training the space then is Taylor Series alone sufficient for continuity compression, given that we place appropriate division operations?
    #now I know the meaning of divider, it is the best security measure for continuous space
    #what happens at the singularities, the black holes? I think a division happens fellas

    #we dont have anything to do today so let's improve our newton approx
    #I'll be right back

    #alright - let's move to Decimal first thing first - we want to approx global extreme of this function - let's see how many floating accuracy is required
    #something is very off
    #we cant keep the numerical stability
    #alright - we are doing compression via differential methods - and the numerical precision represents the volume of compressing data - this is the reason why we need explosive rats - or exponential step  

    #alright we dont know what's happening - we dont care what's happening
    #but we know for sure - that there is a trade off between <numerical precision + numerical stability + locality compression> vs runtime
    #as an engineer our job is not to question or research but to implement a parameterizable version so we could calibrate it

    #ideally - the only reason we want to increase the numerical precision is to do high order derivative extraction so we could approx <global_extremes> better
    #approxing <global_extremes> better is a question of whether doing newton_approxx within the operating window of the numerical precision is better or iteratively jump into the next pool is better   
    #recall our newton_approxx method - this is the very fine way to approx (we want to make sure that our numerical precision is within the delta step) - or we have to make sure that it is within the numerical precision (in other words, numerical precision is our true max_delta_step, or min(acceptable_numeric_last, heuristic_last))
    #so... what's the point? that the numerical precision increases our <AD_range> - yet very costly - there is a practical sweet spot between the infinity range of newton approx and the cost of running the thing
    #yet this is very crucial for the oval circumscribe + sphere circumscribe operation - we are taking the derivative to approx 1 << 30 grids all at once 

    #I tried to explain to Dad that it is not f(x,y,z) etc. for x,y,z being the three words and f being the next word
    #we have moved to centrality long ago because centrality can sort things VERY FAST - and bring relevant things together (y as in f(x) -> y) so we can have smoother linearity (which is our goal)
    #the best way to do thing is the best of both worlds - we want the sortability of centrality and the flexibility of taylor_approximation
    #2 years from now - we would realize that it's the logit density that makes the difference (f(x) -> y 99% accurate) (f1(x) -> y also 99% accurate) yet the f(x) is 2 times denser than the f1(x) - so f(x) is smarter 

    #how we would implement this thing is actually the worth mentioning problem
    #how we would implement this efficiently is actually another billion dollar problem
    #i'll try guys

    #in this taylor_fast.py we have only answered 5% of the problem - which is the heading direction
    #there is distributed compute + cuda compute + network transportation + parameterized numerical stability + parameterized numerical precision + virtual machine data extraction (virtual machine extraction frequency) + data ingestion + etc.
    #if a company heads in this direction and implements this correctly (I dont care the hows and the practices) - they would be worth at least $100B - that's the least

    #what are we missing?
    #we have EVERYTHING - yet approxing trig is too costly - how about we add trig into our taylor approximation? - this is actually debatable
    #alright let's attempt to improve our trig ballistic optimization
    #what really happens is floating problem - float is only good for approxing at most 3 order derivatives, best 2 order derivatives, that's precisely why our tom_approx2 works very very well 
    #our negotiator is too curvy - we are only allowed to <see> up to the numerical stability, exceeding the numerical stability would result in a failure of deviation space reconstruction 
    #how could we offset this? by making sure that the sphere points are reasonably spread-outted
    #we'll work on this problem tmr, we'd want to make sure that we can approx things fast, under 60 seconds, we, then, will move on to multivariate

    #we'll talk about how this thing fits in our core later, it's a billion dollar question, i'm being very serious

    #alright let's increase the frequency + improve numerical stability
    #if we are to increase the frequency of subarm to infinity with respect to the other arm, we should be able to reduce the iterating dimension -> the other arm 
    #this requires us to collect d/dx, d2/dx ... up to the numerical stability allowance
    #let's implement this
    #we are to see if we are to increase the frequency to infinity, what is the required numerical stability to approx our deviation space  
    #what we are missing are the directional backprop + higher order of derivatives + numerical stability + Taylor Series' compression space + GMP on cuda
    #how does this translate to our billion dollar framework, it is a shit ton of output + input
    #we'll get to the bottom of this numerical stability, two order + step or multiorder eqn solver tmr - it's a very important topic

    #its complicated, its about Taylor Series convergence, such is the scaling factor of Taylor Series must outrun the polynomial order
    #we have two types of Taylor Series
    #finite Taylor Series
    #finite infinite Taylor Series

    #finite infinite needs a <translation_layer> to operate on a finite domain space (Interstellar, docking space ship)
    #such finite domain space is "linear", a.k.a Taylor Series finite

    #consider this example
    #<x1, x2, x3>
    #x1 * cos(x) + x2 * sin(x) + x3 * e^x
    #<x1, x2, x3> is Taylor linear because it could be rewritten as x1 * cos(x), x2 * sin(x), x3 * e^x
    #<we are money monkey, we dont have time to research this, we'll be back, our main focus for the proof of concept is directional backpropagation + multivariate Taylor Series as an immediate substitution for every linear operation + training based on synth-wave deviation space> 
    #that's precisely why yall failed, because yall never ask questions and keep implementing bad stuff

    #I was trying to generalize the finite infinite (compressing the Taylor Series) only to realize the obvious truth, it is pattern extraction, for every pattern extraction requires an intellectual move
    #the question now becomes how exactly do we do this?
    #how could we provide the basic storage for finite infinite and let the neural network learn the way to store such information?
    #finite infinite also has two categories: - computable on the Taylor Series original space
    #                                         - not computable on the Taylor Series original space (higher polynomial results outrun their scaling coefficients)

    #can we prove that every finite infinite is a product of add/sub/mul/div of finite Taylor Series(s)?
    #its hard to prove, however, we can unprove this by using convolution law of Taylor Series, finite * finite = finite
    #so that was not a good approach
    #can we "find" the exotic infinite patterns by using <Bernoulli database collector> or <antivirus> database?

    #recall that every function is h(x), f(g(x))
    #a naive compression function job is just simply storing the enumerated h, f, g along with finite Taylor Series(s)
    #so we are talking about enumeration of exotic infinite patterns, regex law of function making, and the base rule being finite Taylor Series float numbers

    #so in our glass of water, being the domain space, we have the content of sin(2x) + sin(3x) + sin(4x) without loss of generality
    #the only thing being concerned is choosing the right collection of glasses (a glass for trig, a glass for rational polynomials, a glass for N polynomials, a glass for whatever it might come)
    #<adding> them together -> offload the responsibility to the optimization engine 

    #I was thinking of the interchangability of Taylor Series naive polynomial compression and our collection of glasses compression
    #it's complicated, we'd definitely break numerical stability and probably lose training friction if not being careful  

    #recall the 6 infinity stones: time, power, reality, space, mind, soul 
    #what is the mind stone?
    #mind is the ability to <mood_decide> the next best move
    #mind stone is a part of stock optimization, for mood levers are stock charts, and we are to move the <soul_cursor> in the direction that satisfies the mood 

    #what is the soul stone?
    #soul is a sliding window buffer of context, roughly 1GB of sliding window buffer
    #mind stone would find the information that is temporally best appropriate to <finite_push_back> to the sliding_window_buffer

main()