import types 
from collections.abc import Callable
import math 
import json

#let's  actually work on this today and tmr - this is important - to approx sin | cos | sqrt | invsqrt by lightweight operation
#goal is to be able to approximate non-differentiable - and differentiable by using search approach
#this is important - because without these optimizations - concurrency is meaningless - because concurrency is actually premature optimizations - yeah
#minimum viable product - able to approx inv_sqrt like the wiki version without actually steering in the direction
#able to approx basic differentiable functions and return the correct functions 
#able to do heuristic search + call C++ function to bench
#able to do accurate binary search of constants - instead of discrete - by using newton approximation
#able to take in deviation, range and optimization_cost as inputs and returns a fastest tentative approximation
#this is going to be used for host_static_fast - which is not a replacement for the accurate version - but a dispatch option for users
#the flops in python is worse than I expect yet I think this is a decent place to start

def operation_add(lhs: object, rhs: object) -> object:
    return lhs + rhs

def operation_sub(lhs: object, rhs: object) -> object:
    return lhs - rhs

def operation_mul(lhs: object, rhs: object) -> object:
    return lhs * rhs

def operation_div(lhs: object, rhs: object) -> object:
    return lhs / max(rhs, 0.001)

def operation_and(lhs: object, rhs: object) -> object:
    return int(lhs) & int(rhs)

def operation_or(lhs: object, rhs: object) -> object:
    return int(lhs) | int(rhs)

def operation_xor(lhs: object, rhs: object) -> object:
    return int(lhs) ^ int(rhs)

def operation_exp(val: object) -> object:
    return math.exp(val)

def operation_log(val: object) -> object:
    return math.log(val)

def operation_sin(val: object) -> object:
    return math.sin(val)

def operation_cos(val: object) -> object:
    return math.cos(val)

def operation_inv(val: object) -> object:
    return 1 / val

def operation_neg(val: object) -> object:
    return -val

def self_to_pair_operation(operation: Callable[[object], object]) -> Callable[[object, object], object]:    
    return lambda a, b: operation(a)

def zero_operation() -> Callable[[object, object], object]:
    return lambda a, b: float(0)

def left_extract_operation() -> Callable[[object, object], object]:
    return lambda a, b: a

def right_extract_operation() -> Callable[[object, object], object]:
    return lambda a, b: b

def const_operation(c: float) -> Callable[[object, object], object]:
    return lambda a, b: c 

def bind_operation(operatable: Callable[[object, object], object], lhs: Callable[[object, object], object], rhs: Callable[[object, object], object]) -> Callable[[object, object], object]:
    return lambda a, b: operatable(lhs(a, b), rhs(a, b))

operation_arr       = ["add", "sub", "mul", "div"]
operation_dict      = {
    "add": operation_add,
    "sub": operation_sub,
    "mul": operation_mul,
    "div": operation_div,
    "and": operation_and,
    "or":  operation_or,
    "xor": operation_xor,
    "exp": operation_exp,
    "log": operation_log,
    "sin": operation_sin,
    "cos": operation_cos,
    "inv": operation_inv,
    "neg": operation_neg
}

self_transform_set  = {"exp", "log", "sin", "cos", "inv", "neg"}
pair_transform_set  = {"add", "sub", "mul", "div", "and", "or", "xor"}

def discretize(first: float, last: float, discretization_sz: int) -> list[float]:

    width: float    = (last - first) / discretization_sz
    rs: list[float] = []
    
    for i in range(discretization_sz):
        rs += [first + (i * width)]
    
    return rs

def combinatorial_zip(arr: list[list[object]]) -> list[tuple]:
    
    if (len(arr) == 0):
        return []
    
    if (len(arr) == 1):
        return [tuple([obj]) for obj in arr[0]]
    
    successor_arr: list[tuple] = combinatorial_zip(arr[1:])
    rs: list[tuple] = []

    for arr_e in arr[0]:
        for successor_e in successor_arr:
            rs += [tuple([arr_e] + list(successor_e))]
    
    return rs

def calc_deviation(expected_operation: Callable[[float, float], float], comparing_operation: Callable[[float, float], float], first: float, last: float, discretization_sz: float) -> float:

    discrete_value_arr: list[float]             = discretize(first, last, discretization_sz)
    pair_point_arr: list[tuple[float, float]]   = combinatorial_zip([discrete_value_arr, discrete_value_arr])
    sqr_sum: float                              = sum([(expected_operation(a, b) - comparing_operation(a, b)) ** 2 for (a, b) in pair_point_arr])
    denorm: float                               = float(len(pair_point_arr))

    return math.sqrt(sqr_sum / denorm)

def n_pow_x_set(arr: list[object], x: int) -> list[list[object]]:
    
    if (x == 0):
        return [[]]

    successor: list[list[object]]   = n_pow_x_set(arr, x - 1)
    rs: list[list[object]]          = []

    for i in range(len(arr)):
        for e in successor:
            rs += [[arr[i]] + e]

    return rs

def make_plan(transformer_list: list[str], const_list: list[float], operation_num: int) -> list[list[tuple[str, float, str]]]:
    
    print(transformer_list, const_list, operation_num)

    transform_sequence_list: list[list[str]]                            = n_pow_x_set(transformer_list, operation_num)
    const_sequence_list: list[list[float]]                              = n_pow_x_set(const_list, operation_num) 
    operation_kind_list: list[str]                                      = n_pow_x_set(["self_transform", "pair_lhs_transform", "pair_rhs_transform", "pair_clhs_transform", "pair_crhs_transform"], operation_num)
    combinatorial_list: list[tuple[list[str], list[float], list[str]]]  = combinatorial_zip([transform_sequence_list, const_sequence_list, operation_kind_list])
    rs: list[list[tuple[str, float, str]]]                              = []

    for (transform_sequence, const_sequence, transform_kind_sequence) in combinatorial_list:
        tmp_sequence    = []
        bad             = False

        for (transform_identifier, const_value, transform_kind) in zip(transform_sequence, const_sequence, transform_kind_sequence):
            if transform_identifier in self_transform_set and transform_kind in ["pair_lhs_transform", "pair_rhs_transform", "pair_clhs_transform", "pair_crhs_transform"]:
                bad = True
                break
            if transform_identifier in pair_transform_set and transform_kind in ["self_transform"]:
                bad = True
                break
            
            tmp_sequence += [(transform_identifier, const_value, transform_kind)]

        if not bad:
            rs += [tmp_sequence]

    return rs

def approx(operation: Callable[[float, float], float], first: float, last: float, transformer_list: list[str], operation_num: int, min_const_value: float, max_const_value: float, discretization_sz: float) -> str:

    discrete_const_arr: list[float]                     = discretize(min_const_value, max_const_value, discretization_sz)
    zipped_plan_arr: list[list[tuple[str, float, str]]] = make_plan(transformer_list, discrete_const_arr, operation_num)
    sorting_arr: list[tuple[float, str]]                = []

    for plan in zipped_plan_arr:
        cur_operation = zero_operation()
        operation_backtrack = []

        for (operation_name, operation_const, operation_kind) in plan:
            if operation_kind == "self_transform":
                cur_operation = bind_operation(self_to_pair_operation(operation_dict[operation_name]), cur_operation, zero_operation())
            elif operation_kind == "pair_lhs_transform":
                cur_operation = bind_operation(operation_dict[operation_name], left_extract_operation(), cur_operation)
            elif operation_kind == "pair_rhs_transform":
                cur_operation = bind_operation(operation_dict[operation_name], cur_operation, right_extract_operation())
            elif operation_kind == "pair_clhs_transform":
                cur_operation = bind_operation(operation_dict[operation_name], const_operation(operation_const), cur_operation)
            elif operation_kind == "pair_crhs_transform":
                cur_operation = bind_operation(operation_dict[operation_name], cur_operation, const_operation(operation_const))
            else:
                raise Exception()

        score: float    = calc_deviation(operation, cur_operation, first, last, discretization_sz)
        sorting_arr     += [(score, plan)]

    return min(sorting_arr)

def bitwise_or_df_da(a: int, b: int, r: int = 5) -> float:
    
    a       = int(a)
    b       = int(b)
    total   = float(0)

    for i in range(r):
        cur_f   = a | b
        prime_f = (a + i + 1) | b
        cur     = float(prime_f - cur_f) / (i + 1)
        total   += cur
    
    return total / r

def bitwise_and_df_da(a: int, b: int, r: int = 4) -> float:
    
    a       = int(a)
    b       = int(b)
    total   = float(0)

    for i in range(r):
        cur_f   = a & b
        prime_f = (a + i + 1) & b
        cur     = float(prime_f - cur_f) / (i + 1)
        total   += cur 
    
    return total / r

def bitwise_xor_df_da(a: int, b: int, r: int = 4) -> float:

    a       = int(a)
    b       = int(b)
    total   = float(0)

    for i in range(r):
        cur_f   = a ^ b 
        prime_f = (a + i + 1) ^ b
        cur     = float(prime_f - cur_f) / (i + 1)
        total   += cur 
    
    return total / r
 
def approx_bitwise_and_df_da(first: int, last: int) -> str:
    
    operation               = lambda a, b: bitwise_and_df_da(a, b)
    min_const_value         = 0
    max_const_value         = 256
    max_operation_num       = 2
    discretization_sz       = 10

    return approx(operation, first, last, operation_arr, max_operation_num, min_const_value, max_const_value, discretization_sz)

def approx_bitwise_and_df_db(first: int, last :int) -> str:
    
    operation               = lambda a, b: bitwise_and_df_da(b, a)
    min_const_value         = 0
    max_const_value         = 256
    max_operation_num       = 2
    discretization_sz       = 10

    return approx(operation, first, last, operation_arr, max_operation_num, min_const_value, max_const_value, discretization_sz)

def approx_bitwise_or_df_da(first: int, last: int) -> str:
    
    operation               = lambda a, b: bitwise_or_df_da(a, b)
    min_const_value         = 0
    max_const_value         = 256
    max_operation_num       = 2
    discretization_sz       = 10

    return approx(operation, first, last, operation_arr, max_operation_num, min_const_value, max_const_value, discretization_sz)

def approx_bitwise_or_df_db(first: int, last: int) -> str:
    
    operation               = lambda a, b: bitwise_or_df_da(b, a)
    min_const_value         = 0
    max_const_value         = 256
    max_operation_num       = 2
    discretization_sz       = 10

    return approx(operation, first, last, operation_arr, max_operation_num, min_const_value, max_const_value, discretization_sz)
    
def approx_bitwise_xor_df_da(first: int, last: int) -> str:
    
    operation               = lambda a, b: bitwise_xor_df_da(a, b)
    min_const_value         = 0
    max_const_value         = 256
    max_operation_num       = 2
    discretization_sz       = 10

    return approx(operation, first, last, operation_arr, max_operation_num, min_const_value, max_const_value, discretization_sz)

def approx_bitwise_xor_df_db(first: int, last: int) -> str:
    
    operation               = lambda a, b: bitwise_xor_df_da(b, a)
    min_const_value         = 0
    max_const_value         = 256
    max_operation_num       = 2
    discretization_sz       = 10

    return approx(operation, first, last, operation_arr, max_operation_num, min_const_value, max_const_value, discretization_sz)

def main():

    print(json.dumps(approx_bitwise_or_df_da(0, 256)))

main()