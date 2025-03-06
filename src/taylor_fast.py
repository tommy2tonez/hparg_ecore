from typing import Callable
import math 
import random
import copy
import sys

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

def newton2_approx(operation: Callable[[float, float], float], iteration_sz: int, initial_x1: float, initial_x2: float, a: float = 0.001) -> tuple[float, float, float]:

    cur_x1          = initial_x1
    cur_x2          = initial_x2
    cur_deviation   = operation(cur_x1, cur_x2) 

    for _ in range(iteration_sz):
        if flip_a_coin():
            func                    = lambda x: operation(x, cur_x2)
            (next_x1, deviation)    = newton_approx(func, iteration_sz, cur_x1, a)
            cur_x1                  = next_x1
            cur_deviation           = deviation
        else:
            func                    = lambda x: operation(cur_x1, x)
            (next_x2, deviation)    = newton_approx(func, iteration_sz, cur_x2, a)
            cur_x2                  = next_x2
            cur_deviation           = deviation

    return cur_x1, cur_x2, cur_deviation

def discretize(first: float, last: float, discretization_sz: int) -> list[float]:

    width: float    = (last - first) / discretization_sz
    rs: list[float] = []
    
    
    for i in range(discretization_sz):
        rs += [first + (i * width)]
    
    return rs

def calc_deviation(lhs: Callable[[float], float], rhs: Callable[[float], float], x_range: int, discretization_sz: int) -> float:

    #what is the correct function?
    #i'm thinking
    #if there is no domain - range importance of value - exponential is the way - this is not a correct statement - because mean sqr emphasizes the importance of higher differential orders

    try:
        discrete_value_arr: list[float] = discretize(0, x_range, discretization_sz)
        sqr_sum: float                  = sum([(lhs(x) - rhs(x)) ** 2 for x in discrete_value_arr]) #i was wrong guys - x^2 is the way - we are not very sure - we really dont know - there is no proof yet of why this works so profoundly + magnificiently
        denom: float                    = float(len(discrete_value_arr))
        normalized: float               = math.sqrt(sqr_sum / denom)

        return normalized
    except:
        return sys.float_info.max

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

#alright - this we need to add damped sin cos waves - because it is not linear enough
#let's see
#e^x
#chaim rules - f(x) = e^x
#d f(x)/dx = e^x * x' 
#

def get_exp_taylor(dimension_sz: int) -> list[float]:

    return [1] * dimension_sz


def get_slope(f: Callable[[float], float], x: int, derivative_order: int, a: float = 0.000001) -> float:

    if derivative_order == 0:
        return f(x)

    return (get_slope(f, x + a, derivative_order - 1) - get_slope(f, x, derivative_order - 1)) / a  

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

def get_sqrt_taylor(dimension_sz: int) -> list[float]:

    func = lambda x: math.sqrt(x)
    return [get_slope(func, 0.000001, i) for i in range(dimension_sz)]

def get_log_taylor(dimension_sz: int) -> list[float]:

    func = lambda x: math.log(x)
    return [get_slope(func, 0.000001, i) for i in range(dimension_sz)]

def taylor_projection(f: list[float], x: float) -> float:

    # print(f)
    try:

        return sum([1 / math.factorial(i) * f[i] * (x ** i) for i in range(len(f))]) 
    except:
        return sys.float_info.max
        print(f)
        raise Exception()

def taylor_fog(f: list[float], g: list[float]) -> list[float]:

    func = lambda x: taylor_projection(f, taylor_projection(g, x))
    return [get_slope(func, 0, i) for i in range(max(len(f), len(g)))] 

#we are not calibrated in the right space
#newton approx direction is still in the original space
#so what do we want? we want to have a calibration function
#remember that calibration can be of form x1 * x2
#calibration in this case is just a simple dx1/dx, x2
#alright let's see what we could do - we want to approx every basic functions - and move to complex functions tmr - and multi-variables taylor the day after tomorrow - and centrality the day after that 
#x1*sin(x2*x)

#this is the list of the basic calibrated functions
#we'll try to come up with more optimization strategies before moving on to the next topic of flux == 0 (what goes around comes around by MaxLake) - this involes multi-variables taylor projection + advanced calibration in such environment

#there exists - random taylor optimization (with taylor's multipliers as stablizers) + gravitational waves (ripple effects) + damped gravitational waves (exponential)
#             - damped gravitational wave is lossless compression - there always exists a solution to compress x if the differential order + data_type resolution are high enough
#             - random step + exponential step newton optimization
#             - random_direction + exponential 2-step newton optimization (beneficial if the two variables are entirely not related - in the sense of gravitational waves x, y displacement)

#we'll move on to the difference of f(x) -> y or f(x, x1) -> y or f(x) -> <y, y1> or f(x, x1) -> <y, y1>
#what's the difference?
#the only difference is the locality of those projections or x
#we want to reorder those guys to have a smoother projection (by using centrality - context diffractor + multi_dimensional projection)
#remember what MaxLake got to tell

def random_taylor_optimization(approximator: TaylorApprox, instrument: Callable[[float], float], x_range: int, discretization_sz: int):

    dimension_sz                    = get_taylor_series_size(approximator.taylor_series)
    directional_vec: list[float]    = get_random_vector(dimension_sz)
    newton_exp_base                 = random.random() * 10
    newton_discretization_sz        = 10
    newton_iteration_sz             = 3
    multiplier                      = None
    deviation                       = None

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
    newton_iteration_sz         = 2
    multiplier                  = None
    deviation                   = None

    def newton_approx_func(multiplier: float):
        previous_value: list[float]     = taylor_series_to_value_arr(approximator.taylor_series)
        copied_value: list[float]       = copy.deepcopy(previous_value)
        scaled_x1_vec                   = scalar_multiply_vector(multiplier, x1_directional_vec)
        scaled_x2_vec                   = scalar_multiply_vector(multiplier, x2_directional_vec)
        directional_vec                 = taylor_convolution(scaled_x1_vec, taylor_fog(sin_cos_directional_vec, scaled_x2_vec)) #sin-cos calibration
        adjusted_value: list[float]     = add_vector(copied_value, directional_vec)

        write_taylor_series_value(approximator.taylor_series, adjusted_value)
        rs: float = calc_deviation(approximator.operation, instrument, x_range, discretization_sz)
        write_taylor_series_value(approximator.taylor_series, previous_value)

        return rs

    for j in range(newton_discretization_sz):
        exp_offset              = (newton_exp_base ** j) - 1
        (est_new_multiplier, y) = newton_approx(newton_approx_func, newton_iteration_sz, exp_offset)

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

        write_taylor_series_value(approximator.taylor_series, adjusted_value)
        rs: float = calc_deviation(approximator.operation, instrument, x_range, discretization_sz)
        write_taylor_series_value(approximator.taylor_series, previous_value)

        return rs

    for j in range(newton_discretization_sz):
        exp_offset              = (newton_exp_base ** j) - 1
        (est_new_multiplier, y) = newton_approx(newton_approx_func, newton_iteration_sz, exp_offset)

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
    newton_iteration_sz         = 2
    multiplier                  = None
    deviation                   = None

    def newton_approx_func(multiplier: float):
        previous_value: list[float]     = taylor_series_to_value_arr(approximator.taylor_series)
        copied_value: list[float]       = copy.deepcopy(previous_value)
        scaled_x1_vec                   = scalar_multiply_vector(multiplier, x1_directional_vec)
        scaled_x2_vec                   = scalar_multiply_vector(multiplier, x2_directional_vec)
        directional_vec                 = taylor_plus(scaled_x1_vec, taylor_fog(gravity_directional_vec, scaled_x2_vec)) #gravity calibration
        adjusted_value: list[float]     = add_vector(copied_value, directional_vec)

        write_taylor_series_value(approximator.taylor_series, adjusted_value)
        rs: float = calc_deviation(approximator.operation, instrument, x_range, discretization_sz)
        write_taylor_series_value(approximator.taylor_series, previous_value)

        return rs

    for j in range(newton_discretization_sz):
        exp_offset              = (newton_exp_base ** j) - 1
        (est_new_multiplier, y) = newton_approx(newton_approx_func, newton_iteration_sz, exp_offset)

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

        write_taylor_series_value(approximator.taylor_series, adjusted_value)
        rs: float = calc_deviation(approximator.operation, instrument, x_range, discretization_sz)
        write_taylor_series_value(approximator.taylor_series, previous_value)

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
    newton_iteration_sz         = 4
    multiplier                  = None
    deviation                   = None

    def newton_approx_func(multiplier: float):
        previous_value: list[float]     = taylor_series_to_value_arr(approximator.taylor_series)
        copied_value: list[float]       = copy.deepcopy(previous_value)
        scaled_x1_vec                   = scalar_multiply_vector(multiplier, x1_directional_vec)
        scaled_x2_vec                   = scalar_multiply_vector(multiplier, x2_directional_vec)
        directional_vec                 = taylor_plus(scaled_x1_vec, taylor_fog(gravity_directional_vec, scaled_x2_vec)) #gravity calibration
        adjusted_value: list[float]     = add_vector(copied_value, directional_vec)

        write_taylor_series_value(approximator.taylor_series, adjusted_value)
        rs: float = calc_deviation(approximator.operation, instrument, x_range, discretization_sz)
        write_taylor_series_value(approximator.taylor_series, previous_value)

        return rs

    for j in range(newton_discretization_sz):
        exp_offset              = (newton_exp_base ** j) - 1
        (est_new_multiplier, y) = newton_approx(newton_approx_func, newton_iteration_sz, exp_offset)

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

        write_taylor_series_value(approximator.taylor_series, adjusted_value)
        rs: float = calc_deviation(approximator.operation, instrument, x_range, discretization_sz)
        write_taylor_series_value(approximator.taylor_series, previous_value)

        return rs

    for j in range(newton_discretization_sz):
        exp_offset              = (newton_exp_base ** j) - 1
        (est_new_multiplier, y) = newton_approx(newton_approx_func, newton_iteration_sz, exp_offset)

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
    

    for _ in range(training_epoch_sz):
        inching_direction: list[float]  = [float(0)] * grad_dimension_sz
        inching_deviation: float        = None

        for __ in range(directional_optimization_sz):
            random_value        = random.randrange(0, 7)
            new_directional_vec = None
            deviation           = None 

            (new_directional_vec, deviation)    = calibrated_sin2_gravity_optimization(approximator, instrument, x_range, discretization_sz)

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

    #alright - let's implement this - we have directions, we have magnitude, => we have vector => we have pointer
    #                               - we have exponential => we have focus

    #what are we focusing on? we are focusing on the "gucci" - the spots (regions) that we statistically know that are better
    #this hints us that the operation we are looking for is a reordering operation (bijective space map)
    #what is wrong with this? we are using a "smarter" operation to improvise a dumb + naive operation - so who's gay?
    #we don't know who's gay - we just know that it must be a dumber operation to improvise a smarter operation - this proves that coach must be dumber than player in real life - otherwise he's not a socially productive coach
    #this sounds like the human-kind - because it really is so - we have the formula of taylor-fast - yet we don't have a recursive coach
    #how about coach coaching another person to be a coach? This is what we want

    #what if your coach is wrong? Do we listen to our coach or coach of the coach or coach of the coach of the coach or all of them?
    #alright - we have a window of things - and centrality to keep our resource finite
    #let's aim for 2 coaches - the random coach and the statistically best coach

    #alright, so to offset this problem, we use another radix of approximation called centrality - which we will implement 
    #centrality will boost our dumbness into another level - which we will leverage to coach another centrality

    #the best thing about centrality is the ability to create another word - lightning fast context projection from n dimension -> 1 dimension (linear) - and context diffraction
    #we will change the 1 dimension -> multi-dimensions to improve the context accuracy by using taylor approximation

    #alright - we want a bijective map of <directional_vector> or directional space -> sin | cos space
    #        - damped oscillation differential equation is a subset of the optimizables
    #        - this means we want to increase the possibility space of local randomness (we want to move more possibilities into our exponential "focus") to allow random value approximation - or turning points approximation
    #        - this does not substitute the above heuristic function but is used in conjunction

    #alright - there is x1(sin(x))
    #        - and there is sin(x*x1)
    #        - the general form is x1*sin(x*x2)

    #I was thinking what if we do taylor approximation differently - such is the combinatorial (or permutation) operation differs
    #then I realized this is not a necessity - because centrality built on top of taylor approximation would reorder things in the best possible way (we dont know this - we are hoping this)
    #so the taylor projection for centrality we wrote in the centrality_approx is the correct algorithm in the sense of completeness of approximation (we want multi-dimensional projection to not have projection collisions)
    #the only thing that we care about is the number of projecting dimensions and the number of projected dimensions
    #we want to do binary input and binary output - or we want hex input and hex output - we don't know

    #because ballistic missiles can only be operated on one-dimensional time coordinate
    #we must do dimensional reduction or dimensional expansion to dynamically adjust the number of taylor-series in our electrical circuit  
    #f(x1,x2,x3) -> f(x) or f(x) -> f(x1, x2, x3) for example 

    #yet we want to offset the numerical stability + training uncertainty by having ONE instrument function and newton approx in a random direction 
    #we'll show people the real way of approxing (there is no cosine similarity or database) - hint that it is not fast - we want to make it fast - we'll try - let's see

    #so we are approxing the <string> from the <all the data> at once
    #because this is the stable way of training
    #we'll write the centrality algorithm tmr

    #if we can approx sin(x) without cheating - I think we can move forward to implement centrality algorithm - this is a hard task
    #usually we want to increase the dimension size because we want to smoothen the curves - which helps with the training

    # print(get_sin_taylor(10))

    approxer: TaylorApprox  = get_taylor_series(4, 1)
    mapper: list[float]     = [random.random() for _ in range(32)] 
    def sqrt_func(x: float):

        return 1024 * mapper[int(x) % 32] + 3 * x**3 + 2 * x**2 + x + 1 #if this reaches < 10 deviation today - it is a success - if not then we must make sure that'd happen tmr - this is a hard task - we of course can increase the differential order -> 128 or 256 and do all sin waves - yet it is not optimization
                                                                        #what is this called? this is taylor lossless compression - by using sin waves 
                                                                        #this is a very hard task - because the <exponential focus> is the correct approach yet the calibrated environment is not - we need advanced calibration (a coach)

    train(approxer, sqrt_func, 1 << 13, 128, 32, 64)
    print(approxer.operation(2))
    print(calc_deviation(approxer.operation, sqrt_func, 2, 32))

    for i in range(len(approxer.taylor_series.series)):
        print(approxer.taylor_series.series[i].value)

    print()
    print(approxer.operation(1))

main()