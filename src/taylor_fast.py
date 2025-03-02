from typing import Callable
import math 
import random

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

def newton_approx(operation: Callable[[float], float], iteration_sz: int, initial_x: float, a: float = 0.001) -> float:

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

    return cand_x

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

def get_distribution_vector(sz: int, decay: float, resolution: int) -> list[int]:
    
    pre_normalized_perc_list: list[float]   = []
    cursor_perc: float                      = float(1)

    for i in range(sz):
        pre_normalized_perc_list += [cursor_perc]
        cursor_perc *= decay

    total_perc: float                   = sum(pre_normalized_perc_list)
    normalized_perc_list: list[float]   = [e / total_perc for e in pre_normalized_perc_list]  
    rs: list[int]                       = []

    for i in range(sz):
        arr_sz: int = int(normalized_perc_list[i] * resolution)
        rs          += [i] * arr_sz

    return rs[:resolution]

def reverse_distribution_vector(inp: list[int], sz: int) -> list[int]:

    rs = []

    for i in range(len(inp)):
        rs += [sz - inp[i] - 1]

    return rs

def train(approximator: TaylorApprox, instrument: Callable[[float], float], training_epoch_sz: int, x_range: int, discretization_sz: int):

    newton_iteration_sz             = 16
    training_decay: float           = 0.5
    distribution_vector: list[int]  = reverse_distribution_vector(get_distribution_vector(len(approximator.taylor_series.series), training_decay, 100), len(approximator.taylor_series.series))

    for _ in range(training_epoch_sz):
        idx     = random.randrange(0, len(distribution_vector))
        cursor  = distribution_vector[idx]

        def newton_approx_func(x: float):
            previous_value: float                               = approximator.taylor_series.series[cursor].value
            approximator.taylor_series.series[cursor].value     = x
            rs:float                                            = calc_deviation(approximator.operation, instrument, x_range, discretization_sz) 
            approximator.taylor_series.series[cursor].value     = previous_value

            return rs

        est_new_x: float = newton_approx(newton_approx_func, newton_iteration_sz, approximator.taylor_series.series[cursor].value)
        approximator.taylor_series.series[cursor].value = est_new_x

def main():

    #something went wrong
    #let me show them mfs the real power of taylor fast + electrical engineering designs

    approxer: TaylorApprox  = get_taylor_series(5, 1)
    sqrt_func               = lambda x: math.sqrt(x)

    # print(approxer.operation(1))

    train(approxer, sqrt_func, 1024, 64, 64)
    # # approxer.taylor_series.series[0].value = 01
    # # approxer.taylor_series.series[1].value = 0.5

    print(approxer.operation(16))
    print(calc_deviation(approxer.operation, sqrt_func, 64, 64))

    # for i in range(len(approxer.taylor_series.series)):
    #     print(approxer.taylor_series.series[i].value)

    # print()
    # print(approxer.operation(1))

main()