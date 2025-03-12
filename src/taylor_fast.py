from typing import Callable
import math 
import random
import copy
import sys
from decimal import * 
from typing import Protocol
import functools

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

def tom_approx2(operation: Callable[[float], float], iteration_sz: int, initial_x: float, a: float = 0.001) -> tuple[float, float]:

    NEWTON_ITER_SZ: int = 3 

    if iteration_sz == 0:
        return newton_approx(operation, NEWTON_ITER_SZ, initial_x, a)

    s0      = get_slope(operation, initial_x, 0)
    v       = get_slope(operation, initial_x, 1)
    accel   = get_slope(operation, initial_x, 2)
    
    epsilon = float(0.01)
    a       = 1/2 * accel
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

    try:
        return tom_approx2(operation, 3, initial_x, a)
    except:
        return initial_x, operation(initial_x) 

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

    return stable_approx(operation, iteration_sz, initial_x, a)

    # current_x: list[float]              = [initial_x]
    # base_newton_iteration_sz: int       = 2
    # total_projection_arr: list          = []
    # derivative_local_minmax_sampling_sz = 3 
    # total_cand_arr: list[float]         = [] 

    # for _ in range(iteration_sz):
    #     scope_differential_projection_arr = []

    #     for x in current_x:
    #         local_differential_projection_arr = []

    #         for differential_order in range(differential_order_sz):
    #             func                                = lambda x: get_slope(operation, x, differential_order, a)
    #             (projected_x, deviation)            = stable_approx(func, base_newton_iteration_sz, x, a)
    #             scope_differential_projection_arr   +=  [(projected_x, deviation)]
    #             local_differential_projection_arr   +=  [(projected_x, deviation)]

    #         if len(local_differential_projection_arr) != 0:
    #             total_projection_arr    += [local_differential_projection_arr]

    #     total_cand_arr  += scope_differential_projection_arr
    #     current_x       = get_left_right_closest([e[0] for e in total_cand_arr], current_x)

    # if len(total_projection_arr) == 0:
    #     return stable_approx(operation, iteration_sz, initial_x)

    # cand_list   = []

    # for i in range(len(total_projection_arr)):
    #     for j in range(min(len(total_projection_arr[i]), derivative_local_minmax_sampling_sz)):
    #         cand_list += [(abs(operation(total_projection_arr[i][j][0])), total_projection_arr[i][j][0])] 

    # min_y, candidate_x = min(cand_list)

    # return candidate_x, min_y 

def discretize(first: float, last: float, discretization_sz: int) -> list[float]:

    width: float    = (last - first) / discretization_sz
    rs: list[float] = []
    
    
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

    def __init__(self, s0_rad_coor: Coordinate, direction_vec: Coordinate, r: float):

        self.s0_rad_coor    = Coordinate(s0_rad_coor.raw())
        self.direction_vec  = Coordinate(direction_vec.raw())
        self.r              = r

    def shoot(self, t: float) -> list[Coordinate]:

        raw_s0_rad_coor: list[float]            = self.s0_rad_coor.raw()
        raw_dir_rad_coor: list[float]           = self.direction_vec.raw()

        bullet_rad_coor: list[float]            = add_vector(raw_s0_rad_coor, scalar_multiply_vector(t, raw_dir_rad_coor))
        bullet_euclid_coor: list[float]         = radian_coordinate_to_euclidean_coordinate(bullet_rad_coor)
        scaled_bullet_euclid_coor: list[float]  = scalar_multiply_vector(self.r, bullet_euclid_coor)

        return [Coordinate(scaled_bullet_euclid_coor)]

class RandomSphereMagneticBallisticDevice(SphereMagneticBallisticDevice):

    def __init__(self, dimension_sz: int, r: float):

        s0_rad_coor: list[float]        = rand_multidimensional_sphere_radian(dimension_sz)
        directional_vec: list[float]    = get_random_taylor(dimension_sz)

        super().__init__(s0_rad_coor, directional_vec, r)

class SpheroidMagneticBallisticDevice:

    def __init__(self, s0_rad_coor: Coordinate, direction_vec: Coordinate, oval_shape: Coordinate):

        self.s0_rad_coor    = Coordinate(s0_rad_coor.raw())
        self.direction_vec  = Coordinate(direction_vec.raw())
        self.oval_shape     = Coordinate(oval_shape.raw()) 

    def shoot(self, t: float) -> list[Coordinate]:
        
        raw_s0_rad_coor: list[float]            = self.s0_rad_coor.raw()
        raw_direction_vec: list[float]          = self.direction_vec.raw()

        bullet_rad_coor: list[float]            = add_vector(raw_s0_rad_coor, scalar_multiply_vector(t, raw_direction_vec))
        bullet_euclid_coor: list[float]         = radian_coordinate_to_euclidean_coordinate(bullet_rad_coor)
        scaled_bullet_euclid_coor: list[float]  = pairwise_multiply_vector(self.oval_shape.raw(), bullet_euclid_coor)

        return [Coordinate(scaled_bullet_euclid_coor)]

class TwoArmsBallisticDevice:

    def __init__(self, dimension_sz: int, arm1_length: float, arm2_length: float):

        self.rotating_arm1: BallisticDeviceInterface  = RandomSphereMagneticBallisticDevice(dimension_sz, arm1_length)
        self.rotating_arm2: BallisticDeviceInterface  = RandomSphereMagneticBallisticDevice(dimension_sz, arm2_length)

    def shoot(self, t: float) -> list[Coordinate]:

        arm1_coor: Coordinate = self.rotating_arm1.shoot(t)[0]
        arm2_coor: Coordinate = self.rotating_arm2.shoot(t)[0]

        return [Coordinate(add_vector(arm1_coor.raw(), arm2_coor.raw()))]

class StaticPointBagBallisticDevice:

    def __init__(self, point_bag: list[Coordinate]):
        
        self.point_bag = [Coordinate(point.raw()) for point in point_bag]

    def shoot(self, t: float) -> list[Coordinate]:
        
        return [Coordinate(e.raw()) for e in self.point_bag]

class UniformRandomStaticPointBagBallisticDevice(StaticPointBagBallisticDevice):

    def __init__(self, dimension_sz: int, bag_sz: int, y_range: float):

        point_bag: list[Coordinate] = [scalar_multiply_vector(y_range, get_random_vector(dimension_sz)) for _ in range(bag_sz)] 
        super().__init__(point_bag)

class UniformRandomStaticSpherePointBagBallisticDevice(StaticPointBagBallisticDevice):

    def __init__(self, dimension_sz: int, bag_sz: int, r: float):

        point_bag: list[Coordinate] = [scalar_multiply_vector(r, radian_coordinate_to_euclidean_coordinate(rand_multidimensional_sphere_radian(dimension_sz))) for _ in range(bag_sz)] 
        super().__init__(point_bag)

class CircumscribingStaticPointBagBallisticDevice:

    def __init__(self, dimension_sz: int, bag_sz: int, r: float):

        self.ballistic = UniformRandomStaticSpherePointBagBallisticDevice(dimension_sz, bag_sz, r) 

    def shoot(self, t: float) -> list[Coordinate]:

        point_bag: list[Coordinate] = self.ballistic(t)
        return [Coordinate(scalar_multiply_vector(t, point.raw())) for point in point_bag] 

class ChainedBallisticDevice:

    def __init__(self, ballistic_device_arr: list[BallisticDeviceInterface]):

        self.ballistic_device_arr = ballistic_device_arr

    def shoot(self, t: float) -> list[Coordinate]:

        rs: list[list[Coordinate]] = []

        for i in range(len(self.ballistic_device_arr)):
            rs += [self.ballistic_device_arr[i].shoot(t)]

        return self._combinatorial_reduce_add(rs)

    def _combinatorial_reduce_add(inp: list[list[Coordinate]]) -> list[Coordinate]:

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

def get_random_melee_ballistic_device() -> BallisticDeviceInterface:
    pass

def get_random_range_ballistic_device() -> BallisticDeviceInterface:
    pass

def get_random_ballistic_device() -> BallisticDeviceInterface:
    pass 

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
        return taylor_convolution(a, taylor_fog(self.taylor_model, b))

class SynthWaveCalibrationDevice:

    def __init__(self, dimension_sz: int, wave_amplitude: float, wave_frequency: float, pow_sz: int, a_dimension_sz: int):

        self.taylor_model   = get_powsin_gravity_taylor(dimension_sz, wave_amplitude, wave_frequency, pow_sz)
        self.a_dimension_sz = a_dimension_sz

    def calibrate(self, coor: Coordinate) -> Coordinate:

        a, b = split_vector(coor.raw(), self.a_dimension_sz)
        return taylor_convolution(a, taylor_fog(self.taylor_model, b))

class MaxwellCalibrationDevice:

    def __init__(self, dimension_sz: int, a_dimension_sz: int):
        
        self.taylor_model   = get_sin_taylor(dimension_sz)
        self.a_dimension_sz = a_dimension_sz
    
    def calibrate(self, coor: Coordinate) -> Coordinate:

        a, b = split_vector(coor.raw(), self.a_dimension_sz)
        return taylor_convolution(a, taylor_fog(self.taylor_model, b))

class NoCalibrationDevice:

    def calibrate(self, coor: Coordinate) -> Coordinate:

        return coor

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
        normalized: float   = math.sqrt(sqr_sum / denom)

        return normalized

class DiscreteMeanSquareDeviationCalculator(MeanSquareDeviationCalculator):

    def __init__(self, first: float, last: float, discretization_sz: int):

        super().__init__(discretize(first, last, discretization_sz))

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
            (x_cand, y) = stable_approx(f, self.iteration_sz, x_cursor, self.a)
            x_cand_list += [(y, x_cand)]

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

class ExponentialTwoOrderNewtonOptimizer(TwoOrderStepNewtonOptimizer):

    def __init__(self, y0_first: float = float(0), y0_last: float = float(0),
                 exp_base_range: float = float(10), exp_base_range_min: float = 1,
                 exp_step_range: int = 10, exp_step_min: int = 1,
                 iteration_sz: int = 4, derivative_offset: float = 0.001):

        y0: float       = uniform_randomize_frange(y0_last - y0_first) + y0_first
        exp_base: float = max(uniform_randomize_frange(exp_base_range), exp_base_range_min)
        exp_step: int   = max(random.randrange(0, exp_step_range), exp_step_min)

        super().__init__(ExponentialStepper(y0, exp_base, exp_step), iteration_sz, derivative_offset)

class OneDimensionalFunctionInterface(Protocol):

    def __call__(self, x: float) -> float:
        ...

class SingleBallisticDeviationEncapsulator:

    def __init__(self, deviation_calculator: DeviationCalculatorInterface, f: Callable[[float], Callable[[float], float]], instrument: Callable[[float], float]):

        self.deviation_calculator   = deviation_calculator
        self.f_producer             = f
        self.instrument             = instrument

    def __call__(self, x: float) -> float:

        f = self.f_producer(x)
        return self.deviation_calculator.deviation(f, self.instrument)

class BatchBallisticDeviationInvSqrNegotiator:

    def __init__(self, deviation_calculator: DeviationCalculatorInterface, f: Callable[[float], list[Callable[[float], float]]], instrument: Callable[[float], float]):

        self.deviation_calculator   = deviation_calculator
        self.f                      = f
        self.instrument             = instrument

    def __call__(self, x: float) -> float:

        callable_list: list[Callable[[float], float]] = self.f(x)
        deviation_list: list[float] = []

        for callable in callable_list:
            deviation_list += [self.deviation_calculator.deviation(callable, self.instrument)]

        return avg_invsqr_list(deviation_list)

def random_optimize(taylor_model: list[float], seed: Coordinate, instrument: Callable[[float], float], x_range: int, discretization_sz: int) -> list[float]:

    pass

    # ballistic_device: BallisticDeviceInterface                          = get_random_ballistic_device() #we dont know the params
    # deviation_calculator: DeviationCalculatorInterface                  = DiscreteMeanSquareDeviationCalculator(0, x_range, discretization_sz)
    # calibrator: CalibrationDeviceInterface                              = get_random_calibration_device()

    # encapsulated_deviation_calculator: OneDimensionalFunctionInterface  = DeviationEncapsulator(deviation_calculator, BallisticTaylorizer(), instrument)
    # optimizer: NewtonOptimizerInterface                                 = get_random_newton_optimizer()

#

def rotating_multiarm_optimization(approximator: TaylorApprox, instrument: Callable[[float], float], x_range: int, discretization_sz: int):

    MAX_ARM_SZ: int                             = 8
    MAX_RADIUS_EXPONENTIAL_SZ: int              = 10
    MAX_ARM_RADIUS: float                       = float(2)
    MAX_FREQUENCY_COEFF: float                  = 32
    dimension_sz                                = get_taylor_series_size(approximator.taylor_series)    
    arm_sz: int                                 = random.randrange(MAX_ARM_SZ) + 1 

    arm_radius_arr                              = [(random.random() * MAX_ARM_RADIUS) ** random.randrange(0, MAX_RADIUS_EXPONENTIAL_SZ) for _ in range(arm_sz)]
    arm_starting_radian_arr                     = [rand_multidimensional_sphere_radian(dimension_sz) for _ in range(arm_sz)]
    arm_rotating_vec_arr                        = [get_random_vector(dimension_sz) for _ in range(arm_sz)]
    arm_frequency_coeff_arr                     = [random.random() * MAX_FREQUENCY_COEFF  for _ in range(arm_sz)]

    t_exponential_base                          = random.random() * 10
    t_discretization_sz                         = 10
    newton_iteration_sz                         = 4

    t                                           = None
    deviation                                   = None

    def newton_approx_func(t: float):
        arm_tip_coordinate: list[float] = [float(0)] * dimension_sz 

        for i in range(arm_sz):
            arm_current_radian      = add_vector(arm_starting_radian_arr[i], scalar_multiply_vector(arm_frequency_coeff_arr[i] * t, arm_rotating_vec_arr[i]))
            arm_current_coordinate  = scalar_multiply_vector(arm_radius_arr[i], radian_coordinate_to_euclidean_coordinate(arm_current_radian))
            arm_tip_coordinate      = add_vector(arm_tip_coordinate, arm_current_coordinate)

        copied_value: list[float]       = copy.deepcopy(taylor_series_to_value_arr(approximator.taylor_series))
        adjusted_value: list[float]     = add_vector(copied_value, arm_tip_coordinate)

        rs: float = calc_deviation(taylor_values_to_operation(adjusted_value), instrument, x_range, discretization_sz)

        return rs

    for j in range(t_discretization_sz):
        exp_offset  = radian_rescale((t_exponential_base ** j) - 1, t_exponential_base ** t_discretization_sz)
        (new_t, y)  = newton_approxx(newton_approx_func, newton_iteration_sz, exp_offset)

        if deviation == None or y < deviation:
            deviation   = y
            t           = new_t

    if deviation == None or t == None:
        return ([float(0)] * dimension_sz, float(0))  

    arm_tip_coordinate: list[float] = [float(0)] * dimension_sz 

    for i in range(arm_sz):
        arm_current_radian      = add_vector(arm_starting_radian_arr[i], scalar_multiply_vector(arm_frequency_coeff_arr[i] * t, arm_rotating_vec_arr[i]))
        arm_current_coordinate  = scalar_multiply_vector(arm_radius_arr[i], radian_coordinate_to_euclidean_coordinate(arm_current_radian))
        arm_tip_coordinate      = add_vector(arm_tip_coordinate, arm_current_coordinate)

    directional_vec = arm_tip_coordinate

    return directional_vec, deviation

def rotating_twoarm_optimization(approximator: TaylorApprox, instrument: Callable[[float], float], x_range: int, discretization_sz: int):

    dimension_sz                                = get_taylor_series_size(approximator.taylor_series)    
    arm1_radius                                 = (random.random() * 2) ** random.randrange(0, 10)
    arm1_starting_radian                        = rand_multidimensional_sphere_radian(dimension_sz)
    arm1_rotating_vec                           = get_random_vector(dimension_sz)
    arm1_frequency_coeff                        = random.random()

    arm2_radius                                 = arm1_radius #two arms with r1 == r2 should suffice - assume there exists a point within r - then the sphere whose origin is the point must intersect with the original sphere 
    arm2_starting_radian                        = rand_multidimensional_sphere_radian(dimension_sz)
    arm2_rotating_vec                           = get_random_vector(dimension_sz)
    arm2_frequency_coeff                        = random.random()

    t_exponential_base                          = random.random() * 10
    t_discretization_sz                         = 10
    newton_iteration_sz                         = 4

    t                                           = None
    deviation                                   = None

    def newton_approx_func(t: float):
        arm1_current_radian         = add_vector(arm1_starting_radian, scalar_multiply_vector(arm1_frequency_coeff * t, arm1_rotating_vec))
        arm1_current_coordinate     = scalar_multiply_vector(arm1_radius, radian_coordinate_to_euclidean_coordinate(arm1_current_radian))

        arm2_current_radian         = add_vector(arm2_starting_radian, scalar_multiply_vector(arm2_frequency_coeff * t, arm2_rotating_vec))
        arm2_current_coordinate     = scalar_multiply_vector(arm2_radius, radian_coordinate_to_euclidean_coordinate(arm2_current_radian))

        arm_tip_coordinate          = add_vector(arm1_current_coordinate, arm2_current_coordinate)

        copied_value: list[float]       = copy.deepcopy(taylor_series_to_value_arr(approximator.taylor_series))
        adjusted_value: list[float]     = add_vector(copied_value, arm_tip_coordinate)

        rs: float = calc_deviation(taylor_values_to_operation(adjusted_value), instrument, x_range, discretization_sz)

        return rs

    for j in range(t_discretization_sz):
        exp_offset  = radian_rescale((t_exponential_base ** j) - 1, t_exponential_base ** t_discretization_sz)
        (new_t, y)  = newton_approxx(newton_approx_func, newton_iteration_sz, exp_offset)

        if deviation == None or y < deviation:
            deviation   = y
            t           = new_t

    if deviation == None or t == None:
        return ([float(0)] * dimension_sz, float(0))  

    arm1_current_radian         = add_vector(arm1_starting_radian, scalar_multiply_vector(arm1_frequency_coeff * t, arm1_rotating_vec))
    arm1_current_coordinate     = scalar_multiply_vector(arm1_radius, radian_coordinate_to_euclidean_coordinate(arm1_current_radian))

    arm2_current_radian         = add_vector(arm2_starting_radian, scalar_multiply_vector(arm2_frequency_coeff * t, arm2_rotating_vec))
    arm2_current_coordinate     = scalar_multiply_vector(arm2_radius, radian_coordinate_to_euclidean_coordinate(arm2_current_radian))

    arm_tip_coordinate          = add_vector(arm1_current_coordinate, arm2_current_coordinate)
    directional_vec             = arm_tip_coordinate

    return directional_vec, deviation

def rotating_hand_optimization(approximator: TaylorApprox, instrument: Callable[[float], float], x_range: int, discretization_sz: int):

    dimension_sz                                = get_taylor_series_size(approximator.taylor_series)    
    arm1_radius                                 = (random.random() * 2) ** random.randrange(0, 10)
    arm1_starting_radian                        = rand_multidimensional_sphere_radian(dimension_sz)
    arm1_rotating_vec                           = get_random_vector(dimension_sz)
    arm1_frequency_coeff                        = random.random()

    arm2_radius                                 = arm1_radius
    arm2_starting_radian                        = rand_multidimensional_sphere_radian(dimension_sz)
    arm2_rotating_vec                           = get_random_vector(dimension_sz)
    arm2_frequency_coeff                        = random.random()

    integral_sampling_sz: int                   = 8
    random_coor: list[list[float]]              = [rand_multidimensional_sphere_radian(dimension_sz) for _ in range(integral_sampling_sz)]
    oval_ishape: list[float]                    = get_random_vector(dimension_sz)  
    oval_scale_multiplier: float                = (random.random() * 2) ** random.randrange(0, 10)
    oval_shape: list[float]                     = scalar_multiply_vector(oval_scale_multiplier, oval_ishape)

    t_exponential_base                          = random.random() * 10
    t_discretization_sz                         = 10
    newton_iteration_sz                         = 4

    t                                           = None
    deviation                                   = None

    def newton_approx_func(t: float):
        arm1_current_radian             = add_vector(arm1_starting_radian, scalar_multiply_vector(arm1_frequency_coeff * t, arm1_rotating_vec))
        arm1_current_coordinate         = scalar_multiply_vector(arm1_radius, radian_coordinate_to_euclidean_coordinate(arm1_current_radian))

        arm2_current_radian             = add_vector(arm2_starting_radian, scalar_multiply_vector(arm2_frequency_coeff * t, arm2_rotating_vec))
        arm2_current_coordinate         = scalar_multiply_vector(arm2_radius, radian_coordinate_to_euclidean_coordinate(arm2_current_radian))

        arm_tip_coordinate              = add_vector(arm1_current_coordinate, arm2_current_coordinate)
        copied_value: list[float]       = copy.deepcopy(taylor_series_to_value_arr(approximator.taylor_series))
        abs_armtip_coordinate           = add_vector(copied_value, arm_tip_coordinate)

        deviation_list: list[float]     = []

        for i in range(integral_sampling_sz):
            relative_finger_coordinate: list[float] = pairwise_multiply_vector(oval_shape, radian_coordinate_to_euclidean_coordinate(random_coor[i]))
            absolute_finger_coordinate: list[float] = add_vector(abs_armtip_coordinate, relative_finger_coordinate)
            deviation_list                          += [calc_deviation(taylor_values_to_operation(absolute_finger_coordinate), instrument, x_range, discretization_sz)]

        return avg_invsqr_list(deviation_list)

    for j in range(t_discretization_sz):
        exp_offset  = radian_rescale((t_exponential_base ** j) - 1, t_exponential_base ** t_discretization_sz)
        (new_t, y)  = newton_approxx(newton_approx_func, newton_iteration_sz, exp_offset)

        if deviation == None or y < deviation:
            deviation   = y
            t           = new_t

    if deviation == None or t == None:
        return ([float(0)] * dimension_sz, float(0))  

    deviation_list: list                = []

    arm1_current_radian                 = add_vector(arm1_starting_radian, scalar_multiply_vector(arm1_frequency_coeff * t, arm1_rotating_vec))
    arm1_current_coordinate             = scalar_multiply_vector(arm1_radius, radian_coordinate_to_euclidean_coordinate(arm1_current_radian))

    arm2_current_radian                 = add_vector(arm2_starting_radian, scalar_multiply_vector(arm2_frequency_coeff * t, arm2_rotating_vec))
    arm2_current_coordinate             = scalar_multiply_vector(arm2_radius, radian_coordinate_to_euclidean_coordinate(arm2_current_radian))

    arm_tip_coordinate                  = add_vector(arm1_current_coordinate, arm2_current_coordinate)
    copied_value: list[float]           = copy.deepcopy(taylor_series_to_value_arr(approximator.taylor_series))
    abs_armtip_coordinate: list[float]  = add_vector(copied_value, arm_tip_coordinate)

    for i in range(integral_sampling_sz):
        relative_finger_coordinate: list[float] = pairwise_multiply_vector(oval_shape, radian_coordinate_to_euclidean_coordinate(random_coor[i]))
        absolute_finger_coordinate: list[float] = add_vector(abs_armtip_coordinate, relative_finger_coordinate)
        deviation_list                          += [(calc_deviation(taylor_values_to_operation(absolute_finger_coordinate), instrument, x_range, discretization_sz), absolute_finger_coordinate)]

    (min_deviation, abs_coordinate) = min(deviation_list)
    
    return sub_vector(abs_coordinate, copied_value), min_deviation

def magnetic_random_optimization(approximator: TaylorApprox, instrument: Callable[[float], float], x_range: int, discretization_sz: int):

    dimension_sz                                = get_taylor_series_size(approximator.taylor_series)
    explosion_exp_base                          = random.random() * 10
    explosion_exp_step                          = random.randrange(0, 10)
    explosion_range                             = explosion_exp_base ** explosion_exp_step
    iteration_dimension_idx                     = random.randrange(0, dimension_sz)
    radius_value_arr: list[float]               = rand_multidimensional_sphere_radian(dimension_sz)
    radius_value_arr[iteration_dimension_idx]   = 0

    t_exponential_base                          = random.random() * 10
    t_discretization_sz                         = 10
    newton_iteration_sz                         = 4

    t                                           = None
    deviation                                   = None
    directional_vec                             = None

    def newton_approx_func(t: float):
        tmp_radius_value_arr                            = copy.deepcopy(radius_value_arr)
        tmp_radius_value_arr[iteration_dimension_idx]   = t
        taylor_directional_arr: list[float]             = scalar_multiply_vector(explosion_range, radian_coordinate_to_euclidean_coordinate(tmp_radius_value_arr))
        previous_value: list[float]                     = taylor_series_to_value_arr(approximator.taylor_series)
        copied_value: list[float]                       = copy.deepcopy(previous_value)
        adjusted_value: list[float]                     = add_vector(copied_value, taylor_directional_arr)

        rs: float = calc_deviation(taylor_values_to_operation(adjusted_value), instrument, x_range, discretization_sz)

        return rs

    for j in range(t_discretization_sz):
        exp_offset  = radian_rescale((t_exponential_base ** j) - 1, t_exponential_base ** t_discretization_sz)
        (new_t, y)  = newton_approxx(newton_approx_func, newton_iteration_sz, exp_offset)

        if deviation == None or y < deviation:
            deviation   = y
            t           = new_t

    if (deviation != None and t != None):
        tmp_radius_value_arr                            = copy.deepcopy(radius_value_arr)
        tmp_radius_value_arr[iteration_dimension_idx]   = t
        taylor_directional_arr: list[float]             = scalar_multiply_vector(explosion_range, radian_coordinate_to_euclidean_coordinate(tmp_radius_value_arr))
        directional_vec                                 = taylor_directional_arr

    return (directional_vec, deviation)

def magnetic_oval_random_optimization(approximator: TaylorApprox, instrument: Callable[[float], float], x_range: int, discretization_sz: int):

    dimension_sz                                = get_taylor_series_size(approximator.taylor_series)
    explosion_exp_base                          = random.random() * 10
    explosion_exp_step                          = random.randrange(0, 10)
    explosion_range                             = explosion_exp_base ** explosion_exp_step
    iteration_dimension_idx                     = random.randrange(0, dimension_sz)
    radius_value_arr: list[float]               = rand_multidimensional_sphere_radian(dimension_sz)
    radius_value_arr[iteration_dimension_idx]   = 0

    oval_direction: list[float]                 = get_random_vector(dimension_sz)
    adjusted_oval_direction: list[float]        = scalar_multiply_vector(explosion_range, oval_direction) 

    t_exponential_base                          = random.random() * 10
    t_discretization_sz                         = 10
    newton_iteration_sz                         = 4

    t                                           = None
    deviation                                   = None
    directional_vec                             = None

    def newton_approx_func(t: float):
        tmp_radius_value_arr                            = copy.deepcopy(radius_value_arr)
        tmp_radius_value_arr[iteration_dimension_idx]   = t
        taylor_directional_arr: list[float]             = pairwise_multiply_vector(adjusted_oval_direction, radian_coordinate_to_euclidean_coordinate(tmp_radius_value_arr))
        previous_value: list[float]                     = taylor_series_to_value_arr(approximator.taylor_series)
        copied_value: list[float]                       = copy.deepcopy(previous_value)
        adjusted_value: list[float]                     = add_vector(copied_value, taylor_directional_arr)

        rs: float = calc_deviation(taylor_values_to_operation(adjusted_value), instrument, x_range, discretization_sz)

        return rs

    for j in range(t_discretization_sz):
        exp_offset  = radian_rescale((t_exponential_base ** j) - 1, t_exponential_base ** t_discretization_sz)
        (new_t, y)  = newton_approxx(newton_approx_func, newton_iteration_sz, exp_offset)

        if deviation == None or y < deviation:
            deviation   = y
            t           = new_t

    if (deviation != None and t != None):
        tmp_radius_value_arr                            = copy.deepcopy(radius_value_arr)
        tmp_radius_value_arr[iteration_dimension_idx]   = t
        taylor_directional_arr: list[float]             = pairwise_multiply_vector(adjusted_oval_direction, radian_coordinate_to_euclidean_coordinate(tmp_radius_value_arr))
        directional_vec                                 = taylor_directional_arr

    return (directional_vec, deviation)

def oval_circumscribe_optimization(approximator: TaylorApprox, instrument: Callable[[float], float], x_range: int, discretization_sz: int):

    dimension_sz                                = get_taylor_series_size(approximator.taylor_series)
    
    integral_sampling_sz                        = 16
    newton_iteration_sz                         = 4

    r_exponential_base                          = random.random() * 10
    r_discretization_sz                         = 10

    r_deviation                                 = None
    r                                           = None

    random_radian_arr                           = [rand_multidimensional_sphere_radian(dimension_sz) for _ in range(integral_sampling_sz)]
    initial_radius_vec                          = get_random_vector(dimension_sz)

    def newton_radius_approx_func(r: float):
        copied_value: list[float]   = copy.deepcopy(taylor_series_to_value_arr(approximator.taylor_series))
        deviation_list: list[float] = []

        for i in range(integral_sampling_sz):
            taylor_directional_arr: list[float]     = pairwise_multiply_vector(scalar_multiply_vector(r, initial_radius_vec), radian_coordinate_to_euclidean_coordinate(random_radian_arr[i]))
            adjusted_value: list[float]             = add_vector(copied_value, taylor_directional_arr)
            deviation: float                        = calc_deviation(taylor_values_to_operation(adjusted_value), instrument, x_range, discretization_sz)
            deviation_list                          += [deviation]

        avg_deviation: float = avg_invsqr_list(deviation_list) #alright what's our aim ? we want global extremes - what is the differentiable version of the function ? log? 1/x - we'll settle for (1/x)^2 for now 

        return avg_deviation

    for i in range(r_discretization_sz):
        exp_offset          = (r_exponential_base ** i) - 1
        (new_r, new_r_dev)  = newton_approxx(newton_radius_approx_func, newton_iteration_sz, exp_offset)

        if r_deviation == None or new_r_dev < r_deviation:
            r_deviation = new_r_dev
            r           = new_r

    if r_deviation == None or r == None:
        return ([float(0)] * dimension_sz, float(0))  

    deviation_rad_arr: list[float]  = []

    for i in range(len(random_radian_arr)):
        cur_radian_vec: list[float]         = random_radian_arr[i]
        taylor_directional_arr: list[float] = pairwise_multiply_vector(scalar_multiply_vector(r, initial_radius_vec), radian_coordinate_to_euclidean_coordinate(cur_radian_vec))
        adjusted_value: list[float]         = add_vector(taylor_series_to_value_arr(approximator.taylor_series), taylor_directional_arr)
        deviation: float                    = calc_deviation(taylor_values_to_operation(adjusted_value), instrument, x_range, discretization_sz)
        deviation_rad_arr                   += [(deviation, taylor_directional_arr)]

    return tuple(list(min(deviation_rad_arr))[::-1]) 

def sphere_circumscribe_optimization(approximator: TaylorApprox, instrument: Callable[[float], float], x_range: int, discretization_sz: int):

    dimension_sz                                = get_taylor_series_size(approximator.taylor_series)
    
    integral_sampling_sz                        = 16
    newton_iteration_sz                         = 4

    r_exponential_base                          = random.random() * 10
    r_discretization_sz                         = 10

    r_deviation                                 = None
    r                                           = None

    random_radian_arr                           = [rand_multidimensional_sphere_radian(dimension_sz) for _ in range(integral_sampling_sz)]

    def newton_radius_approx_func(r: float):
        copied_value: list[float]   = copy.deepcopy(taylor_series_to_value_arr(approximator.taylor_series))
        deviation_list: list[float] = []

        for i in range(integral_sampling_sz):
            taylor_directional_arr: list[float]     = scalar_multiply_vector(r, radian_coordinate_to_euclidean_coordinate(random_radian_arr[i]))
            adjusted_value: list[float]             = add_vector(copied_value, taylor_directional_arr)
            deviation: float                        = calc_deviation(taylor_values_to_operation(adjusted_value), instrument, x_range, discretization_sz)
            deviation_list                          += [deviation]

        avg_deviation: float = avg_invsqr_list(deviation_list) #alright what's our aim ? we want global extremes - what is the differentiable version of the function ? log? 1/x - we'll settle for (1/x)^2 for now 

        return avg_deviation

    for i in range(r_discretization_sz):
        exp_offset          = (r_exponential_base ** i) - 1
        (new_r, new_r_dev)  = newton_approxx(newton_radius_approx_func, newton_iteration_sz, exp_offset)

        if r_deviation == None or new_r_dev < r_deviation:
            r_deviation = new_r_dev
            r           = new_r

    if r_deviation == None or r == None:
        return ([float(0)] * dimension_sz, float(0))  

    deviation_rad_arr: list[float]  = []

    for i in range(len(random_radian_arr)):
        cur_radian_vec: list[float]         = random_radian_arr[i]
        taylor_directional_arr: list[float] = scalar_multiply_vector(r, radian_coordinate_to_euclidean_coordinate(cur_radian_vec))
        adjusted_value: list[float]         = add_vector(taylor_series_to_value_arr(approximator.taylor_series), taylor_directional_arr)
        deviation: float                    = calc_deviation(taylor_values_to_operation(adjusted_value), instrument, x_range, discretization_sz)
        deviation_rad_arr                   += [(deviation, taylor_directional_arr)]

    return tuple(list(min(deviation_rad_arr))[::-1]) 

def photon_random_optimization(approximator: TaylorApprox, instrument: Callable[[float], float], x_range: int, discretization_sz: int):

    dimension_sz                                = get_taylor_series_size(approximator.taylor_series)
    directional_vec: list[float]                = get_random_vector(dimension_sz)
    newton_exp_base                             = random.random() * 10
    newton_discretization_sz                    = 10
    newton_iteration_sz                         = 8

    explosion_exp_base                          = random.random() * 10
    explosion_exp_step                          = random.randrange(0, 10)
    explosion_range                             = explosion_exp_base ** explosion_exp_step
    iteration_dimension_idx                     = random.randrange(0, dimension_sz)
    radius_value_arr: list[float]               = rand_multidimensional_sphere_radian(dimension_sz)
    radius_value_arr[iteration_dimension_idx]   = 0
    random_alpha                                = random.random() 

    oval_direction: list[float]                 = get_random_vector(dimension_sz)
    adjusted_oval_direction: list[float]        = scalar_multiply_vector(explosion_range, oval_direction) 

    deviation                                   = None

    def newton_approx_func(multiplier: float):
        previous_value: list[float]                     = taylor_series_to_value_arr(approximator.taylor_series)
        copied_value: list[float]                       = copy.deepcopy(previous_value)

        segment_vector: list[float]                     = scalar_multiply_vector(multiplier, directional_vec)

        tmp_radius_value_arr                            = copy.deepcopy(radius_value_arr)
        tmp_radius_value_arr[iteration_dimension_idx]   = random_alpha * multiplier
        magnetic_vector: list[float]                    = pairwise_multiply_vector(adjusted_oval_direction, radian_coordinate_to_euclidean_coordinate(tmp_radius_value_arr))
        delta_vector: list[float]                       = add_vector(segment_vector, magnetic_vector)

        adjusted_vector: list[float]                    = add_vector(copied_value, delta_vector)

        rs: float = calc_deviation(taylor_values_to_operation(adjusted_vector), instrument, x_range, discretization_sz)

        return rs

    for j in range(newton_discretization_sz):
        exp_offset              = (newton_exp_base ** j) - 1
        (est_new_multiplier, y) = newton_approxx(newton_approx_func, newton_iteration_sz, exp_offset)

        if deviation == None or y < deviation:
            segment_vector: list[float]                     = scalar_multiply_vector(est_new_multiplier, directional_vec)
            tmp_radius_value_arr                            = copy.deepcopy(radius_value_arr)
            tmp_radius_value_arr[iteration_dimension_idx]   = random_alpha * est_new_multiplier
            magnetic_vector: list[float]                    = pairwise_multiply_vector(adjusted_oval_direction, radian_coordinate_to_euclidean_coordinate(tmp_radius_value_arr))
            delta_vector: list[float]                       = add_vector(segment_vector, magnetic_vector)

            deviation                                       = y
            directional_vec                                 = delta_vector

    return (directional_vec, deviation)

def calibrated_random_optimization(approximator: TaylorApprox, instrument: Callable[[float], float], x_range: int, discretization_sz: int):

    random_func_sz              = random.randrange(0, 10) + 1
    dimension_sz                = get_taylor_series_size(approximator.taylor_series)
    x1_directional_vec          = get_leading_dimension(get_random_vector(dimension_sz), 4)
    x2_directional_vec          = get_leading_dimension(get_random_vector(dimension_sz), 4)
    random_vec                  = get_random_taylor(dimension_sz, random_func_sz)
    newton_exp_base             = random.random() * 10
    newton_discretization_sz    = 10
    newton_iteration_sz         = 8
    multiplier                  = None
    deviation                   = None
    directional_vec             = None

    def newton_approx_func(multiplier: float):
        previous_value: list[float]     = taylor_series_to_value_arr(approximator.taylor_series)
        copied_value: list[float]       = copy.deepcopy(previous_value)
        scaled_x1_vec                   = scalar_multiply_vector(multiplier, x1_directional_vec)
        scaled_x2_vec                   = scalar_multiply_vector(multiplier, x2_directional_vec)
        directional_vec                 = taylor_convolution(scaled_x1_vec, taylor_fog(random_vec, scaled_x2_vec)) #sin-cos calibration
        adjusted_value: list[float]     = add_vector(copied_value, directional_vec)

        rs: float = calc_deviation(taylor_values_to_operation(adjusted_value), instrument, x_range, discretization_sz)

        return rs

    for j in range(newton_discretization_sz):
        exp_offset              = (newton_exp_base ** j) - 1
        (est_new_multiplier, y) = newton_approxx(newton_approx_func, newton_iteration_sz, exp_offset)

        if deviation == None or y < deviation:
            deviation   = y
            multiplier  = est_new_multiplier

    if (deviation != None and multiplier != None):
        scaled_x1_vec       = scalar_multiply_vector(multiplier, x1_directional_vec)
        scaled_x2_vec       = scalar_multiply_vector(multiplier, x2_directional_vec)
        directional_vec     = taylor_convolution(scaled_x1_vec, taylor_fog(random_vec, scaled_x2_vec))

    return (directional_vec, deviation)

def random_taylor_optimization(approximator: TaylorApprox, instrument: Callable[[float], float], x_range: int, discretization_sz: int):

    dimension_sz                    = get_taylor_series_size(approximator.taylor_series)
    directional_vec: list[float]    = get_random_vector(dimension_sz)
    newton_exp_base                 = random.random() * 10
    newton_discretization_sz        = 10
    newton_iteration_sz             = 8
    multiplier                      = None
    deviation                       = None

    def newton_approx_func(multiplier: float):
        previous_value: list[float] = taylor_series_to_value_arr(approximator.taylor_series)
        copied_value: list[float]   = copy.deepcopy(previous_value)
        adjusted_value: list[float] = add_vector(copied_value, scalar_multiply_vector(multiplier, directional_vec))

        rs: float = calc_deviation(taylor_values_to_operation(adjusted_value), instrument, x_range, discretization_sz)

        return rs

    for j in range(newton_discretization_sz):
        exp_offset              = (newton_exp_base ** j) - 1
        (est_new_multiplier, y) = newton_approxx(newton_approx_func, newton_iteration_sz, exp_offset)

        if deviation == None or y < deviation:
            deviation   = y
            multiplier  = est_new_multiplier

    if multiplier != None and deviation != None:
        return (scalar_multiply_vector(multiplier, directional_vec), deviation) 

    return (directional_vec, deviation)

def calibrated_maxwell_optimization(approximator: TaylorApprox, instrument: Callable[[float], float], x_range: int, discretization_sz: int):

    dimension_sz                = get_taylor_series_size(approximator.taylor_series)
    x1_directional_vec          = get_leading_dimension(get_random_vector(dimension_sz), 4)
    x2_directional_vec          = get_leading_dimension(get_random_vector(dimension_sz), 4)
    sin_cos_directional_vec     = get_sin_taylor(dimension_sz) if flip_a_coin() else get_cos_taylor(dimension_sz)
    newton_exp_base             = random.random() * 10
    newton_discretization_sz    = 10
    newton_iteration_sz         = 8
    multiplier                  = None
    deviation                   = None

    def newton_approx_func(multiplier: float):
        previous_value: list[float]     = taylor_series_to_value_arr(approximator.taylor_series)
        copied_value: list[float]       = copy.deepcopy(previous_value)
        scaled_x1_vec                   = scalar_multiply_vector(multiplier, x1_directional_vec)
        scaled_x2_vec                   = scalar_multiply_vector(multiplier, x2_directional_vec)
        directional_vec                 = taylor_convolution(scaled_x1_vec, taylor_fog(sin_cos_directional_vec, scaled_x2_vec)) #sin-cos calibration
        adjusted_value: list[float]     = add_vector(copied_value, directional_vec)

        rs: float = calc_deviation(taylor_values_to_operation(adjusted_value), instrument, x_range, discretization_sz)

        return rs

    for j in range(newton_discretization_sz):
        exp_offset              = (newton_exp_base ** j) - 1
        (est_new_multiplier, y) = newton_approxx(newton_approx_func, newton_iteration_sz, exp_offset)

        if deviation == None or y < deviation:
            deviation   = y
            multiplier  = est_new_multiplier

    if (deviation != None and multiplier != None):
        scaled_x1_vec       = scalar_multiply_vector(multiplier, x1_directional_vec)
        scaled_x2_vec       = scalar_multiply_vector(multiplier, x2_directional_vec)
        directional_vec     = taylor_convolution(scaled_x1_vec, taylor_fog(sin_cos_directional_vec, scaled_x2_vec))

    return (directional_vec, deviation)

def calibrated_gravity_optimization(approximator: TaylorApprox, instrument: Callable[[float], float], x_range: int, discretization_sz: int):

    dimension_sz                = get_taylor_series_size(approximator.taylor_series)
    x1_directional_vec          = get_leading_dimension(get_random_vector(dimension_sz), 2)
    x2_directional_vec          = get_leading_dimension(get_random_vector(dimension_sz), 2)
    gravity_directional_vec     = get_gravity_taylor(dimension_sz)
    newton_exp_base             = random.random() * 10
    newton_discretization_sz    = 10
    newton_iteration_sz         = 2
    multiplier                  = None
    deviation                   = None

    def newton_approx_func(multiplier: float):
        previous_value: list[float]     = taylor_series_to_value_arr(approximator.taylor_series)
        copied_value: list[float]       = copy.deepcopy(previous_value)
        scaled_x1_vec                   = scalar_multiply_vector(multiplier, x1_directional_vec)
        scaled_x2_vec                   = scalar_multiply_vector(multiplier, x2_directional_vec)
        directional_vec                 = taylor_convolution(scaled_x1_vec, taylor_fog(gravity_directional_vec, scaled_x2_vec)) #gravity calibration
        adjusted_value: list[float]     = add_vector(copied_value, directional_vec)

        rs: float = calc_deviation(taylor_values_to_operation(adjusted_value), instrument, x_range, discretization_sz)

        return rs

    for j in range(newton_discretization_sz):
        exp_offset              = (newton_exp_base ** j) - 1
        (est_new_multiplier, y) = newton_approxx(newton_approx_func, newton_iteration_sz, exp_offset)

        if deviation == None or y < deviation:
            deviation   = y
            multiplier  = est_new_multiplier

    if (deviation != None and multiplier != None):
        scaled_x1_vec       = scalar_multiply_vector(multiplier, x1_directional_vec)
        scaled_x2_vec       = scalar_multiply_vector(multiplier, x2_directional_vec)
        directional_vec     = taylor_convolution(scaled_x1_vec, taylor_fog(gravity_directional_vec, scaled_x2_vec))

    return (directional_vec, deviation)

def calibrated_sin_gravity_optimization(approximator: TaylorApprox, instrument: Callable[[float], float], x_range: int, discretization_sz: int):

    dimension_sz                = get_taylor_series_size(approximator.taylor_series)
    x1_directional_vec          = get_leading_dimension(get_random_vector(dimension_sz), 2)
    x2_directional_vec          = get_leading_dimension(get_random_vector(dimension_sz), 2)

    wave_amplitude              = (random.random() * 2) ** (random.randrange(0, 10)) * random_sign(2)
    wave_frequency              = (random.random() * 2) ** (random.randrange(0, 10))

    gravity_directional_vec     = get_sin_gravity_taylor(dimension_sz, wave_amplitude, wave_frequency)
    newton_exp_base             = random.random() * 10
    newton_discretization_sz    = 10
    newton_iteration_sz         = 8
    multiplier                  = None
    deviation                   = None

    def newton_approx_func(multiplier: float):
        previous_value: list[float]     = taylor_series_to_value_arr(approximator.taylor_series)
        copied_value: list[float]       = copy.deepcopy(previous_value)
        scaled_x1_vec                   = scalar_multiply_vector(multiplier, x1_directional_vec)
        scaled_x2_vec                   = scalar_multiply_vector(multiplier, x2_directional_vec)
        directional_vec                 = taylor_plus(scaled_x1_vec, taylor_fog(gravity_directional_vec, scaled_x2_vec)) #gravity calibration
        adjusted_value: list[float]     = add_vector(copied_value, directional_vec)

        rs: float = calc_deviation(taylor_values_to_operation(adjusted_value), instrument, x_range, discretization_sz)

        return rs

    for j in range(newton_discretization_sz):
        exp_offset              = (newton_exp_base ** j) - 1
        (est_new_multiplier, y) = newton_approxx(newton_approx_func, newton_iteration_sz, exp_offset)

        if deviation == None or y < deviation:
            deviation   = y
            multiplier  = est_new_multiplier

    if (deviation != None and multiplier != None):
        scaled_x1_vec       = scalar_multiply_vector(multiplier, x1_directional_vec)
        scaled_x2_vec       = scalar_multiply_vector(multiplier, x2_directional_vec)
        directional_vec     = taylor_plus(scaled_x1_vec, taylor_fog(gravity_directional_vec, scaled_x2_vec))

    return (directional_vec, deviation)

def calibrated_sin2_gravity_optimization(approximator: TaylorApprox, instrument: Callable[[float], float], x_range: int, discretization_sz: int):

    dimension_sz                = get_taylor_series_size(approximator.taylor_series)
    x1_directional_vec          = get_leading_dimension(get_random_vector(dimension_sz), 2)
    x2_directional_vec          = get_leading_dimension(get_random_vector(dimension_sz), 2)

    wave_amplitude              = (random.random() * 10) ** (random.randrange(0, 10)) * random_sign(2)
    wave_frequency              = (random.random() * 10) ** (random.randrange(0, 10))

    gravity_directional_vec     = get_sqsin_gravity_taylor(dimension_sz, wave_amplitude, wave_frequency)
    newton_exp_base             = random.random() * 10
    newton_discretization_sz    = 5
    newton_iteration_sz         = 4
    
    x1_multiplier               = None
    x2_multiplier               = None
    deviation                   = None

    def newton_approx_func(x1_multiplier: float, x2_multiplier: float):
        previous_value: list[float]     = taylor_series_to_value_arr(approximator.taylor_series)
        copied_value: list[float]       = copy.deepcopy(previous_value)
        scaled_x1_vec                   = scalar_multiply_vector(x1_multiplier, x1_directional_vec)
        scaled_x2_vec                   = scalar_multiply_vector(x2_multiplier, x2_directional_vec)
        directional_vec                 = taylor_plus(scaled_x1_vec, taylor_fog(gravity_directional_vec, scaled_x2_vec)) #gravity calibration - something is very off here
        adjusted_value: list[float]     = add_vector(copied_value, directional_vec)

        rs: float = calc_deviation(taylor_values_to_operation(adjusted_value), instrument, x_range, discretization_sz)

        return rs

    for j in range(newton_discretization_sz):
        for j1 in range(newton_discretization_sz):
            x1_offset                   = (newton_exp_base ** j) - 1
            x2_offset                   = (newton_exp_base ** j1) - 1
            (new_x1_mul, new_x2_mul, y) = newton2_approx(newton_approx_func, newton_iteration_sz, x1_offset, x2_offset)

            if deviation == None or y < deviation:
                deviation       = y
                x1_multiplier   = new_x1_mul
                x2_multiplier   = new_x2_mul

    if (deviation != None and x1_multiplier != None and x2_multiplier != None):
        scaled_x1_vec       = scalar_multiply_vector(x1_multiplier, x1_directional_vec)
        scaled_x2_vec       = scalar_multiply_vector(x2_multiplier, x2_directional_vec)
        directional_vec     = taylor_plus(scaled_x1_vec, taylor_fog(gravity_directional_vec, scaled_x2_vec))

    return (directional_vec, deviation)

def calibrated_powsin_gravity_optimization(approximator: TaylorApprox, instrument: Callable[[float], float], x_range: int, discretization_sz: int):

    dimension_sz                = get_taylor_series_size(approximator.taylor_series)
    x1_directional_vec          = get_leading_dimension(get_random_vector(dimension_sz), 2)
    x2_directional_vec          = get_leading_dimension(get_random_vector(dimension_sz), 2)

    wave_amplitude              = (random.random() * 2) ** (random.randrange(0, 10)) * random_sign(2)
    wave_frequency              = (random.random() * 2) ** (random.randrange(0, 10))
    pow_sz                      = random.randrange(0, 8) + 1

    gravity_directional_vec     = get_powsin_gravity_taylor(dimension_sz, wave_amplitude, wave_frequency, pow_sz)
    newton_exp_base             = random.random() * 10
    newton_discretization_sz    = 10
    newton_iteration_sz         = 8
    multiplier                  = None
    deviation                   = None

    def newton_approx_func(multiplier: float):
        previous_value: list[float]     = taylor_series_to_value_arr(approximator.taylor_series)
        copied_value: list[float]       = copy.deepcopy(previous_value)
        scaled_x1_vec                   = scalar_multiply_vector(multiplier, x1_directional_vec)
        scaled_x2_vec                   = scalar_multiply_vector(multiplier, x2_directional_vec)
        directional_vec                 = taylor_plus(scaled_x1_vec, taylor_fog(gravity_directional_vec, scaled_x2_vec)) #gravity calibration
        adjusted_value: list[float]     = add_vector(copied_value, directional_vec)

        rs: float = calc_deviation(taylor_values_to_operation(adjusted_value), instrument, x_range, discretization_sz)

        return rs

    for j in range(newton_discretization_sz):
        exp_offset              = (newton_exp_base ** j) - 1
        (est_new_multiplier, y) = newton_approxx(newton_approx_func, newton_iteration_sz, exp_offset)

        if deviation == None or y < deviation:
            deviation   = y
            multiplier  = est_new_multiplier

    if (deviation != None and multiplier != None):
        scaled_x1_vec       = scalar_multiply_vector(multiplier, x1_directional_vec)
        scaled_x2_vec       = scalar_multiply_vector(multiplier, x2_directional_vec)
        directional_vec     = taylor_plus(scaled_x1_vec, taylor_fog(gravity_directional_vec, scaled_x2_vec))

    return (directional_vec, deviation)

def calibrated_cos_gravity_optimization(approximator: TaylorApprox, instrument: Callable[[float], float], x_range: int, discretization_sz: int):

    dimension_sz                = get_taylor_series_size(approximator.taylor_series)
    x1_directional_vec          = get_leading_dimension(get_random_vector(dimension_sz), 2)
    x2_directional_vec          = get_leading_dimension(get_random_vector(dimension_sz), 2)
    gravity_directional_vec     = get_cos_gravity_taylor(dimension_sz)
    newton_exp_base             = random.random() * 10
    newton_discretization_sz    = 10
    newton_iteration_sz         = 8
    multiplier                  = None
    deviation                   = None

    def newton_approx_func(multiplier: float):
        previous_value: list[float]     = taylor_series_to_value_arr(approximator.taylor_series)
        copied_value: list[float]       = copy.deepcopy(previous_value)
        scaled_x1_vec                   = scalar_multiply_vector(multiplier, x1_directional_vec)
        scaled_x2_vec                   = scalar_multiply_vector(multiplier, x2_directional_vec)
        directional_vec                 = taylor_convolution(scaled_x1_vec, taylor_fog(gravity_directional_vec, scaled_x2_vec)) #gravity calibration
        adjusted_value: list[float]     = add_vector(copied_value, directional_vec)

        rs: float = calc_deviation(taylor_values_to_operation(adjusted_value), instrument, x_range, discretization_sz)

        return rs

    for j in range(newton_discretization_sz):
        exp_offset              = (newton_exp_base ** j) - 1
        (est_new_multiplier, y) = newton_approxx(newton_approx_func, newton_iteration_sz, exp_offset)

        if deviation == None or y < deviation:
            deviation   = y
            multiplier  = est_new_multiplier

    if (deviation != None and multiplier != None):
        scaled_x1_vec       = scalar_multiply_vector(multiplier, x1_directional_vec)
        scaled_x2_vec       = scalar_multiply_vector(multiplier, x2_directional_vec)
        directional_vec     = taylor_convolution(scaled_x1_vec, taylor_fog(gravity_directional_vec, scaled_x2_vec))

    return (directional_vec, deviation)

def train(approximator: TaylorApprox, instrument: Callable[[float], float], training_epoch_sz: int, directional_optimization_sz: int, x_range: int, discretization_sz: int):

    grad_dimension_sz: list[float]  = get_taylor_series_size(approximator.taylor_series)
    best_deviation                  = None

    #tmr we'll implement a generalized form to actually work on every set of approximation - then we'll move on to multi-variables
    #Mom talked about calibrations - this is the random generalized calibrations that we chimps could come up with - we'll add more cases for the base optimizations
    #point is if we can't do taylor approx on multi-variables to project the next words (for the reason of locality) - we want to increase the dimensions - and do multi-dimensional projections to do centrality
    #yeah yeah yall argued that it is not a centrality algorithm - but it is a centrality algorithm
    #centrality is finite nodes - edges - and value propagations
    #we'll build from there
    #these are all the methods that I could come up with as of now - we'll be back tmr - its pretty decent - we just need to take a very high derivative order to get this working - it's gonna be GMP on GPU - i'm not telling yall this is easy - yet the idea is to <encode> the <differential value> as much as possible at any random point in our projection by using differential methods (very skewed synth waves)

    #today we are going to see what happen if we increase the numerical stability and take the (1 << 16)th derivative order of the multiarm optimization or circumscribe optimization for that matter
    #if we could successfully approx the global exterme in one shot using the circumscribe optimization (taking the number of sampling -> inf) - it's considered a success

    for _ in range(training_epoch_sz):
        inching_direction: list[float]  = [float(0)] * grad_dimension_sz
        inching_deviation: float        = None

        for idx in range(directional_optimization_sz):
            random_value        = random.randrange(0, 16)
            new_directional_vec = None
            deviation           = None 

            if random_value == 0:
                (new_directional_vec, deviation)    = random_taylor_optimization(approximator, instrument, x_range, discretization_sz)
            elif random_value == 1:
                (new_directional_vec, deviation)    = calibrated_maxwell_optimization(approximator, instrument, x_range, discretization_sz)
            elif random_value == 2:
                (new_directional_vec, deviation)    = calibrated_sin_gravity_optimization(approximator, instrument, x_range, discretization_sz)
            elif random_value == 3:
                (new_directional_vec, deviation)    = calibrated_cos_gravity_optimization(approximator, instrument, x_range, discretization_sz)
            elif random_value == 4:
                (new_directional_vec, deviation)    = calibrated_gravity_optimization(approximator, instrument, x_range, discretization_sz)
            elif random_value == 5:
                (new_directional_vec, deviation)    = calibrated_sin2_gravity_optimization(approximator, instrument, x_range, discretization_sz)
            elif random_value == 6:
                (new_directional_vec, deviation)    = calibrated_powsin_gravity_optimization(approximator, instrument, x_range, discretization_sz)
            elif random_value == 7:
                (new_directional_vec, deviation)    = calibrated_random_optimization(approximator, instrument, x_range, discretization_sz)
            elif random_value == 8:
                (new_directional_vec, deviation)    = magnetic_random_optimization(approximator, instrument, x_range, discretization_sz)
            elif random_value == 9:
                (new_directional_vec, deviation)    = magnetic_oval_random_optimization(approximator, instrument, x_range, discretization_sz)
            elif random_value == 10:
                (new_directional_vec, deviation)    = photon_random_optimization(approximator, instrument, x_range, discretization_sz)
            elif random_value == 11:
                (new_directional_vec, deviation)    = sphere_circumscribe_optimization(approximator, instrument, x_range, discretization_sz)
            elif random_value == 12:
                (new_directional_vec, deviation)    = oval_circumscribe_optimization(approximator, instrument, x_range, discretization_sz) 
            elif random_value == 13:
                (new_directional_vec, deviation)    = rotating_multiarm_optimization(approximator, instrument, x_range, discretization_sz)
            elif random_value == 14:
                (new_directional_vec, deviation)    = rotating_twoarm_optimization(approximator, instrument, x_range, discretization_sz)
            elif random_value == 15:
                (new_directional_vec, deviation)    = rotating_hand_optimization(approximator, instrument, x_range, discretization_sz)

            #there are probably 64 optimization strategies
            #we are a quarter through

            #if we only have one variable

            #I was thinking of (1): circumscribe optimization (range)
            #                  (2): bullet optimization (range)
            #                  (3): nuclear optimization (range) - fission optimization - have original capitals - decay if there is no improvement
            #                  (4): military rocket optimization (range) - launch rocket - explode on site (think of bullet + circumscribe + newton_approxx)
            #                  (5): sievering optimization (range) - detect anomaly by using softmax (or large pow or friends)
            #                  (6): one-dimensionalization optimization (melee) - including two-arms - two-arms + sievering, three-arms, three-arms + sievering

            #thing is we want to build our instrument correctly - such is continuous by the summation of very skewed synth-waves (we are assuming one gay is one gay - there is no glass, game, garden, galaxy, etc.) - and we pour water over the data to extract the taylor model
            #if we could extract the virtual machine data at the cpu frequency - we are not missing a bit of information (yeah - it's that accurate if built correctly)


            if new_directional_vec != None and deviation != None:
                if (inching_deviation == None) or (deviation < inching_deviation):
                    inching_direction   = new_directional_vec
                    inching_deviation   = deviation
                    print(random_value)

        if inching_deviation == None:
            continue

        print(inching_deviation)

        current_value: list[float]  = taylor_series_to_value_arr(approximator.taylor_series) 
        adjusted_value: list[float] = add_vector(current_value, inching_direction)
        write_taylor_series_value(approximator.taylor_series, adjusted_value)
        optimized_deviation         = calc_deviation(approximator.operation, instrument, x_range, discretization_sz)

        if (best_deviation == None) or (best_deviation > optimized_deviation):
            print("update", optimized_deviation)
            best_deviation = optimized_deviation
        else:
            write_taylor_series_value(approximator.taylor_series, current_value)

def decimal_e(x: Decimal, iteration_sz: int = 1024) -> Decimal:

    #f0 + f'(0) * x + 1/2 * f''(0) * x^2

    rs: Decimal = Decimal(0)

    for i in range(iteration_sz):
        rs += Decimal(1) / math.factorial(i) * Decimal(x)**i  

    return rs 

def decimal_sin(x: Decimal, iteration_sz: int = 1024) -> Decimal:

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

def decimal_cos(x: Decimal, iteration_sz: int = 1024) -> Decimal:

    #cos = sin(90 - x) yet I want to implement taylor series for this
    #cos(x), -sin(x), -cos(x), sin(x), cos(x)
    #1, 0, -1, 0, 1

    coeff: int      = 1
    rs: Decimal     = Decimal(0)

    for i in range(iteration_sz):
        signness        = 1 if i % 2 == 0 else -1
        idx             = i * 2
        rs              += Decimal(1) / math.factorial(idx) * Decimal(signness * coeff) * (x ** idx)

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
        rs += Decimal(1) / math.factorial(i) * coeff_arr[i] * (x ** Decimal(i)) 

    return rs

def main():

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

    # getcontext().prec = 128
    # print(getcontext())

    # func = lambda x: decimal_synth_wave(x, 128)
    # taylor_series: list[Decimal] = decimal_taylorize(func, 24)

    # print(taylor_series)

    # # print(decimal_sin(Decimal(math.pi), 16))

    # print(decimal_taylor_compute(taylor_series, Decimal(15.99)))

    # print(math.exp(Decimal(4)))

    # print(decimal_synth_wave(Decimal(15.99)))
    # print(decimal_sin(2))
    # print(synth_wave(Decimal(0)))

    # print(approx_e(4, 40))

    # for i in range(1024)

    # print(Decimal(1 << 3000) + Decimal(0.1))
    # print(sys.float_info)

    approxer: TaylorApprox  = get_taylor_series(5, 1)

    def sqrt_func(x: float):

        return (x-1) * (x-2) * (x-3) * (x-4)

    # print(newton_approxx(sqrt_func, 20, 32, 5))

    train(approxer, sqrt_func, 1 << 13, 16, 8, 64) #we'll move on if this ever reach < 0.01 (alright - it finally reaches 0.008 - this proves that this method is stable - we are happy)
    print(approxer.operation(2))
    print(calc_deviation(approxer.operation, sqrt_func, 2, 32))

    for i in range(len(approxer.taylor_series.series)):
        print(approxer.taylor_series.series[i].value)

    print()
    print(approxer.operation(1))

main()