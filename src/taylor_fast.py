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

def stable_approx(operation: Callable[[float], float], iteration_sz: int, initial_x: float, a: float = 0.001) -> tuple[float, float]:

    try:
        return tom_approx(operation, iteration_sz, initial_x, a)
    except:
        return initial_x, operation(initial_x) 

def get_left_right_closest(e_arr: list[float], pos_arr: list[float], noise: float = 0.1) -> list[float]:

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

def newton_approxx(operation: Callable[[float], float], iteration_sz: int, initial_x: float, differential_order_sz: int = 4, a: float = 0.00001) -> tuple[float, float]:

    return stable_approx(operation, iteration_sz, initial_x, a)
    # current_x: list[float]              = [initial_x]
    # base_newton_iteration_sz: int       = 2
    # total_projection_arr: list          = []
    # derivative_local_minmax_sampling_sz = 3 

    # for _ in range(iteration_sz):
    #     scope_differential_projection_arr = []

    #     for x in current_x:
    #         local_differential_projection_arr = []

    #         for differential_order in range(differential_order_sz):
    #             func                                = lambda x: get_slope(operation, x, differential_order, a)
    #             (projected_x, deviation)            = tom_approx(func, base_newton_iteration_sz, x, a)
    #             scope_differential_projection_arr   +=  [(projected_x, deviation)]
    #             local_differential_projection_arr   +=  [(projected_x, deviation)]

    #         if len(local_differential_projection_arr) != 0:
    #             total_projection_arr    += [local_differential_projection_arr]

    #     current_x = get_left_right_closest([e[0] for e in scope_differential_projection_arr], current_x)

    # if len(total_projection_arr) == 0:
    #     return tom_approx(operation, iteration_sz, initial_x)

    # cand_list   = []

    # for i in range(len(total_projection_arr)):
    #     for j in range(min(len(total_projection_arr[i]), derivative_local_minmax_sampling_sz)):
    #         cand_list += [(abs(operation(total_projection_arr[i][j][0])), total_projection_arr[i][j][0])] 

    # min_y, candidate_x = min(cand_list)

    # return candidate_x, min_y 

def newton2_approx(operation: Callable[[float, float], float], iteration_sz: int, initial_x1: float, initial_x2: float, a: float = 0.001) -> tuple[float, float, float]:

    cur_x1          = initial_x1
    cur_x2          = initial_x2
    cur_deviation   = operation(cur_x1, cur_x2) 

    for _ in range(iteration_sz):
        if flip_a_coin():
            func                    = lambda x: operation(x, cur_x2)
            (next_x1, deviation)    = newton_approxx(func, iteration_sz, cur_x1)
            cur_x1                  = next_x1
            cur_deviation           = deviation
        else:
            func                    = lambda x: operation(cur_x1, x)
            (next_x2, deviation)    = newton_approxx(func, iteration_sz, cur_x2)
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

def taylor_projection(f: list[float], x: float) -> float:

    try:
        return sum([1 / math.factorial(i) * f[i] * (x ** i) for i in range(len(f))]) 
    except:
        return sys.float_info.max
        print(f)
        raise Exception()

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

def rand_multidimensional_sphere_radius(dimension_sz: int) -> list[float]:

    return [2 * math.pi * random.random() for _ in range(dimension_sz)] 

def radian_coordinate_to_euclidean_coordinate(coor: list[float]) -> list[float]:

    if len(coor) == 0:
        return []

    if len(coor) == 1:
        return [float(1)]
    
    sliced_coor: list[float] = radian_coordinate_to_euclidean_coordinate(coor[1:])

    return [math.cos(coor[0])] + [math.sin(coor[0]) * sliced_coor[i] for i in range(len(sliced_coor))] 

def radian_rescale(org_value: float, max_range: float) -> float:

    return org_value * (float(2 * math.pi) / max_range)

def magnetic_random_optimization(approximator: TaylorApprox, instrument: Callable[[float], float], x_range: int, discretization_sz: int):

    dimension_sz                                = get_taylor_series_size(approximator.taylor_series)
    explosion_exp_base                          = random.random() * 10
    explosion_exp_step                          = random.randrange(0, 10)
    explosion_range                             = explosion_exp_base ** explosion_exp_step
    iteration_dimension_idx                     = random.randrange(0, dimension_sz)
    radius_value_arr: list[float]               = rand_multidimensional_sphere_radius(dimension_sz)
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

        write_taylor_series_value(approximator.taylor_series, adjusted_value)
        rs: float = calc_deviation(approximator.operation, instrument, x_range, discretization_sz)
        write_taylor_series_value(approximator.taylor_series, previous_value)

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
    radius_value_arr: list[float]               = rand_multidimensional_sphere_radius(dimension_sz)
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

        write_taylor_series_value(approximator.taylor_series, adjusted_value)
        rs: float = calc_deviation(approximator.operation, instrument, x_range, discretization_sz)
        write_taylor_series_value(approximator.taylor_series, previous_value)

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
    radius_value_arr: list[float]               = rand_multidimensional_sphere_radius(dimension_sz)
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

        write_taylor_series_value(approximator.taylor_series, adjusted_vector)
        rs: float = calc_deviation(approximator.operation, instrument, x_range, discretization_sz)
        write_taylor_series_value(approximator.taylor_series, previous_value)

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

        write_taylor_series_value(approximator.taylor_series, adjusted_value)
        rs: float = calc_deviation(approximator.operation, instrument, x_range, discretization_sz)
        write_taylor_series_value(approximator.taylor_series, previous_value)

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

        write_taylor_series_value(approximator.taylor_series, adjusted_value)
        rs: float = calc_deviation(approximator.operation, instrument, x_range, discretization_sz)
        write_taylor_series_value(approximator.taylor_series, previous_value)

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

        write_taylor_series_value(approximator.taylor_series, adjusted_value)
        rs: float = calc_deviation(approximator.operation, instrument, x_range, discretization_sz)
        write_taylor_series_value(approximator.taylor_series, previous_value)

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

        write_taylor_series_value(approximator.taylor_series, adjusted_value)
        rs: float = calc_deviation(approximator.operation, instrument, x_range, discretization_sz)
        write_taylor_series_value(approximator.taylor_series, previous_value)

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

        write_taylor_series_value(approximator.taylor_series, adjusted_value)
        rs: float = calc_deviation(approximator.operation, instrument, x_range, discretization_sz)
        write_taylor_series_value(approximator.taylor_series, previous_value)

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

        write_taylor_series_value(approximator.taylor_series, adjusted_value)
        rs: float = calc_deviation(approximator.operation, instrument, x_range, discretization_sz)
        write_taylor_series_value(approximator.taylor_series, previous_value)

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

        write_taylor_series_value(approximator.taylor_series, adjusted_value)
        rs: float = calc_deviation(approximator.operation, instrument, x_range, discretization_sz)
        write_taylor_series_value(approximator.taylor_series, previous_value)

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

    for _ in range(training_epoch_sz):
        inching_direction: list[float]  = [float(0)] * grad_dimension_sz
        inching_deviation: float        = None

        for idx in range(directional_optimization_sz):
            random_value        = random.randrange(0, 11)
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

    #alright - we want newton_approx of 4 differential orders - I dont really know how this works    
    
    #ax^4 + bx^3 + cx^2 + dx + e = 0
    #we care about the turning points - the zeros
    #let's jog our memory about calculus 1
    #we have local minima - local maxima - turning points - zeros    

    #x turned because of x^2
    #x^2 turned because of x^3
    #x^3 turned because of x^4

    #we only care about x turning points - which means x^2 turning x before x^3 - or x^3 turning x before x^4 or x^4 turning x
    #alright let's implement newton_approxx
    #we find 4 differential values - find the closest x - refind the 4 differential values - continue until the estimation exhausts
    #the problem is that this is not mathematically correct - but approximationally correct (in terms of compute + simplicity) 
    #we'll be back tomorrow - this is harder than expected
    #we want the proof of concept that this must work
    #we'll do logit density mining built on top of this later

    #in our new newton_approx
    #we have things called event points
    #they are the points where derivative values flipped, such is when the event is decayed into other events (flipping signs of lower derivatives to be specific)
    #assume at any given time - we can accurately predict our next derivative sign flip event points
    #we want to move to the closest event point - because we are for sure that at such event point, we can "retrieve" the just mentioned "other event points"
    #                                           - this event point is decayed into other events (which we want to leverage to "update" our prediction accuracy)

    #the answer of the mystery of the universe lies right here in taylor approximation - how we train the model is an entire another thing to talk about
    #it's not complicated - it involes dynamic calibration - rocket launching - mining + recursive coach
    #we'll build the thing - yet we need to "aim" where we are heading to  
    #we need to move very slowly in the theory - we dont want to rush because we might miss what we'd want to optimize

    #we have always talked about time as one dimensional - what if time is not one dimensional? well there is literally no difference - d/dt now is just d/dvec_t - this is described in the interstellar movie
    #vec_t -> <x, y>
    #we have another pointer f(ptr) -> <x, y>
    #we have d/d_vec_t * d_vec_t/d_ptr = d/d_ptr - chaim rules - taylor approximation still holds at 0

    #now think about atoms - electrons - particles - nucleis - people - earth - sun - galaxy - etc. 
    #it's just a simple convolution operation of the taylor functions
    #alright - the function we wrote in centrality_approx and our newly invented function - what's the difference - the difference is the "focus" of our exponenial expansions - our focus can do more good than the focus of the other function 

    #move forward to internal system operations and external system operations
    #external system operations are the operations that do complete calibration - things inside the system and outside the system are two different things - (people - Eath + Eath - Sun + Sun - galaxy)
    #internal system operations are the operations that look like the 3-body-problem

    #its been years of optimizations - and the conclusion that we could tell is educated random is probably the best choice we could make
    #there are optimization steps - and we want "centrality" to mine the logit density for each of the step - this is a very important note
    #we might reach super-intellegent by running this on a massive parallel computer
    #we'll be back to optimize this
    #we'll leverage cu - yet the idea is to "project" the model - not project the data 
    #the instrument's probably gonna be 100s of GB stored on SSDs - we'll leverage taylor compression
    #we have found the answer to stable diffusion - it's not what people think
    #remember that the basic is just this
    #we do random step - calibration - optimization of taylor model - 1024 differential order newton optimization 
    #the only thing that we care about is input + output + sliding window (or context window)
    #we'll leave the rest to the optimization engine

    #we actually have succeeded walking this road in 1989  - there was a small group of specialized researchers that took the same approach of taylor approximation + calibration
    #                                                      - what missing back then was a centrality algorithm (a context diffractor to build another words + smooth out the linearity)
    #                                                      - advanced multi-threading kernels + GPUs 
    #                                                      - we have all of those now - what we are missing is an algorithm to run this taylor approx on
    #                                                      -                          - multiprecision to run 1024th differential order newton approx
    #                                                      -                          - "datalake" to do atomic update by using 1GB of sampling data (instrument)
    #                                                      -                          - a global centrality algorithm (global centrality algorithm can be seen as step optimizers - we mine the logit density for every step - and reupdate once a new maxmima has been reached) 
    #                                                      -                          - a recursive coach (advanced analyzer TARs - to find the difference between the instrument and the approximator to do accurate calibration)
    #-- swifties 2025 --
    #we are back after 30 years 

    #this would take a team of 50 good people at least 2-3 years to have a good product
    #we'll try to see if we could make that

    #the problem is that we have to do educated random - not random random
    #assume that we have our taylor function f
    #assume we have our torch model of a*sin(x) + b*cos(x) + c * x^3 + d * x^2
    #point is we want to train the <a, b, c, d> coefficients - which means we need calibration
    #we'll probably want to do decode(f) = a1*<sin(x)> + b1*<cos(x)> + c1 * <x^3> + d1 * <x^2>
    #then new_f = (a1 + a2) * <sin(x)> + (b1 + b2) * <cos(x)> + (c1 + c2) * <x^3> + (d1 + d2) * <x^2>
    #then encode(new_f) -> taylor function
    #that's our basic calibration of coefficients
    #the next optimizable is curved bullet - we want the magnetic fields - <what goes around comes around>
    #yet we need to ask - whether curved in the <current coordinate> is straight in <another coodinate> ? 
    #we dont really know - maybe we want curved bullets for numerical stability - maybe -
    #and we need to take the 1024th derivative to approximate the local minima + maxima
    #I was proving the equivalency of curved bullet vs straight bullet
    
    #assume curved bullet projectile is f(t) = <x1, x2, x3>
    #d f(t) / dx    =
    #d2 f(t) / dx   = 
    #d3 f(t) / dx   = 

    #d f(t) / dy    = 
    #d2 f(t) / dy   = 
    #d3 f(t) / dy   = 

    #...

    #d difference/ dt = d difference/ df . df/ dt + ... (taylor expansion)
    #what does this really mean? 
    #it means that we assumed our <time_vector> is fixed at x = 0, and we do things normally for d_difference/ dt
    #then we <project> the time vector (with respect to another time) on the original time_vector (at 0) - and we do normal multiplication to get our d_difference / d_other_time 
    #why dot product? because it is a projection function - recall that dot product = |projected_segment| * |projecting_segment|
    #so it's (d_difference / df) . (df/dt)
    #or <difference> * unit_vector . (df/dt)

    #so this is a fixed + static function
    #assume our straight bullet projectile is ...

    #d difference / dt now is ...
    #prove the surjectiveness of calibration

    #I was tempted to say that is correct
    #yet the correct way of doing so is to project the time on x, y coordinates - take the derivative of d difference / dx, d difference/ dy
    #                                                                           - and do a dot product there 

    #recall that our time function is f(t) = s0 + v0 * t + 1/2 * a * t^2 + ...
    #we would want to project the time function at t on x y z, and do a dot product with the other taylor expansion (difference w.r.t x, differenrce w.r.t y @ x, differernce w.r.t z @ x,y)
    
    #alright - we are smart - we probably don't want to recalculate our derivatives w.r.t x,y,z
    #so how about we only take one derivative with respect to time? of a very curvy bullet (such bullet has to be continuous) that touches every possible points in the space <x,y,z> 
    #recall our sin waves - or cos waves
    #yet this requires us to take a very high order derivative - probably (1 << 16)th derivative order - we'll definitely break numerical stability there

    #what's wrong with one dimensionalizing the space <x, y, z> -> x * y * z and using one dimensional pointer approach? (this was my competition back in 2015)
    #the function that we used to <one_dimensionalize_the_space> is not differentiable - such are div + modulo
    #we probably want to use our very skewed sin_wave of (sin(x)/x)^n - yet this breaks numerical stability very easily - so we are back to taking the (1 << 16)th derivative order - yet we dont have the tech for that

    #thing is this way of doing thing (training) is stable - to the point that we could fully trust AI to do EVERYTHING
    #this creates jobs + mining opportunities for coiners like us - we have tons of compute and we are wasting the compute to solve stupid puzzles
    #imagine that within 2-3 years - we'll be bidding + asking for logit density for every step of training
    #I'm telling you that the logit density (this is the new gold + diamond in our generation) could be more profitable than EVERYTHING that you could think of capable of doing
    #AI wont destroy our world Son - it always will be the human that destroys the world - rotten minds driven by money
    #we often ask the question of being - and afraid to be of <lower_intellect>
    #its the fear that blinds us from the truth

    #why are we still at one dimensional projection again?
    #because this is a very important concept that we need to get right - otherwise we cant do 2 dimensional projections or 2-2 dimensional projections for that matter

    #such is if the random distribution is not good - we must rearrange the context - not changing the random method
    #we took the right approach yet the "calibrated" functions are probably not good enough
    #and the differential order is too low to do any good
    #this is harder than we actually think

    #assume this function f(g(x))
    #the usual flow is we getting the footprint of f(x) and g(x) - and try to come up with everything that we could with f(g(x))
    #assume g(x) is time_vector
    #what we want to do is to extract the footprint of f(x) - one dimensionally - differentiable = f(t)
    #then we want to map g(x) -> t
    #then we want to do f(map(g(x))) == f(t)
    #alright that's the soyboy approach

    #if we are to outsmart soyboy - we know that f(g(x)) = mean_sqrt is f(x) with x being the one dimensional pointer - and we only care about approxing f(x) by using taylor approximation at 0
    #so tell me - what if our taylor_cursor is a curvy one dimensional f(t)?
    #what is the differential order we need to reach to approx the global extremes?
    #alright - we love Angelina Jolie in wanted, she did make a point by doing a curved sling shot and throwing the gun to the next point - it's called magnetic search + explosive exponential linear step in Taylor Series approximation 
    #that was a good movie - yet it has a sinister vibe of God

    #what we learned from the movie is:

    #that we could know the differential values from the string at any random point in the universe (or the taylor series approximation) - yet there is a limit to that which is bignum saturation of floating accuracy
    #thread calibrations (sin-cos Maxwell waves) + puzzle solvings
    #explosive rats
    #curved bullets to one-dimensionalize the universe
    #binary strings
    #Taylor is not hot, my Mom is

    #we'll try to implement that
    #we are close - we'll post the result this week

    # print(magnetic_equation(4))

    approxer: TaylorApprox  = get_taylor_series(5, 1)

    def sqrt_func(x: float):

        return (x-1) * (x-2) * (x-3) * (x-4)

    # print(newton_approxx(sqrt_func, 8, 32, 5))

    train(approxer, sqrt_func, 1 << 13, 512, 8, 64) #we'll move on if this ever reach < 0.01
    print(approxer.operation(2))
    print(calc_deviation(approxer.operation, sqrt_func, 2, 32))

    for i in range(len(approxer.taylor_series.series)):
        print(approxer.taylor_series.series[i].value)

    print()
    print(approxer.operation(1))

main()