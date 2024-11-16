import types 
from collections.abc import Callable
import math 
import json
import random 

#alright guys - things weren't perfect - let's try later 
#let's introduce the concept of discrete function
#after passing the unit tests for basic differentiables and invsqrt test

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

self_transform_set          = {"exp", "log", "sin", "cos", "inv", "neg"}
pair_transform_set          = {"add", "sub", "mul", "div", "and", "or", "xor"}

OPS_TREE_KIND_LEFT_EXTRACT  = 0
OPS_TREE_KIND_RIGHT_EXTRACT = 1
OPS_TREE_KIND_CONST_EXTRACT = 2
OPS_TREE_KIND_PAIR_OPS      = 3 

class OperationTree:
    def __init__(self, left =  None, right = None, kind = None, pair_operation_name = None, const_value = None):
        self.left                   = left
        self.right                  = right
        self.kind                   = kind
        self.pair_operation_name    = pair_operation_name
        self.const_value            = const_value

def operation_tree_to_callable(root: OperationTree) -> Callable[[float, float], float]:

    if root == None:
        return zero_operation()
    
    if root.kind == None:
        return zero_operation()
    
    if root.kind == OPS_TREE_KIND_CONST_EXTRACT:
        return const_operation(root.const_value)
    
    if root.kind == OPS_TREE_KIND_LEFT_EXTRACT:
        return left_extract_operation()
    
    if root.kind == OPS_TREE_KIND_RIGHT_EXTRACT:
        return right_extract_operation()
    
    if root.kind == OPS_TREE_KIND_PAIR_OPS:
        return bind_operation(operation_dict[root.pair_operation_name], operation_tree_to_callable(root.left), operation_tree_to_callable(root.right))

    return zero_operation()

def pretty_print(root: OperationTree) -> str:

    if (root == None):
        return "0"
    
    if root.kind == None:
        return "0"

    if root.kind == OPS_TREE_KIND_CONST_EXTRACT:
        return str(root.const_value)
    
    if root.kind == OPS_TREE_KIND_LEFT_EXTRACT:
        return "a"
    
    if root.kind == OPS_TREE_KIND_RIGHT_EXTRACT:
        return "b"

    if root.kind == OPS_TREE_KIND_PAIR_OPS:
        return "%s(%s, %s)" % (root.pair_operation_name, pretty_print(root.left), pretty_print(root.right))

    return "0" 
    
def clone_tree(root: OperationTree) -> OperationTree:

    if (root == None):
        return None 
    
    rs: OperationTree       = OperationTree()
    rs.left                 = clone_tree(root.left)
    rs.right                = clone_tree(root.right)
    rs.kind                 = root.kind
    rs.pair_operation_name  = root.pair_operation_name
    rs.const_value          = root.const_value

    return rs

def operation_tree_count_ops(root: OperationTree) -> OperationTree:

    if (root == None):
        return 0

    return operation_tree_count_ops(root.left) + operation_tree_count_ops(root.right) + 1 

def operation_tree_node_at(root: OperationTree, idx: int) -> OperationTree:
    
    if root == None:
        return None 

    rs = operation_tree_node_at(root.left, idx)

    if rs != None:
        return rs

    idx     -= operation_tree_count_ops(root.left)
    new_rs  =  operation_tree_node_at(root.right, idx)

    if (new_rs != None):
        return new_rs

    idx     -= operation_tree_count_ops(root.right)

    if (idx == 0):
        return root
    
    return None

def make_operation_tree(transformer_list: list[str], const_value_arr: list[float], operation_num: int) -> list[OperationTree]:
    
    if operation_num == 0:
        return []

    possible_root_arr: list[OperationTree]  = []
    
    for transformer in transformer_list:
        possible_root_arr += [OperationTree(None, None, OPS_TREE_KIND_PAIR_OPS, transformer, None)] 

    operation_num                           -= 1
    left_root_arr: list[OperationTree]      = make_operation_tree(transformer_list, const_value_arr, operation_num)
    rs: list[OperationTree]                 = []

    for i in range(len(left_root_arr)):
        left_operation_sz                   = operation_tree_count_ops(left_root_arr[i])
        right_operation_sz                  = operation_num - left_operation_sz
        right_root_arr: list[OperationTree] = make_operation_tree(transformer_list, const_value_arr, right_operation_sz)

        for j in range(len(right_root_arr)):
            for z in range(len(possible_root_arr)):
                new_tree: OperationTree = clone_tree(possible_root_arr[z])
                new_tree.left           = clone_tree(left_root_arr[i])
                new_tree.right          = clone_tree(right_root_arr[j])
                rs                      += [new_tree]

    for const_value in const_value_arr:
        rs += [OperationTree(None, None, OPS_TREE_KIND_CONST_EXTRACT, None, const_value)]

    rs  += [OperationTree(None, None, OPS_TREE_KIND_LEFT_EXTRACT, None, None)]
    rs  += [OperationTree(None, None, OPS_TREE_KIND_RIGHT_EXTRACT, None, None)]

    return rs

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

def calc_deviation(instrument: Callable[[float, float], float], operation: Callable[[float, float], float], first: float, last: float, discretization_sz: int) -> float:

    discrete_value_arr: list[float]             = discretize(first, last, discretization_sz)
    pair_point_arr: list[tuple[float, float]]   = combinatorial_zip([discrete_value_arr, discrete_value_arr])
    sqr_sum: float                              = sum([(instrument(a, b) - operation(a, b)) ** 2 for (a, b) in pair_point_arr])
    denorm: float                               = float(len(pair_point_arr))

    return math.sqrt(sqr_sum / denorm)

def newton_approx(operation: Callable[[float], float], iteration_sz: int, initial_x: float, a: float = 0.1) -> float:

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
        
        if (abs(slope) < epsilon):
            break 

        cur_x   -= cur_y / slope

    return cand_x

def immutable_mutate_tree_const(root: OperationTree, tree_idx: int, x: float) -> OperationTree:

    new_root            = clone_tree(root)
    node                = operation_tree_node_at(new_root, tree_idx)
    node.const_value    = x

    return new_root

def approx(instrument: Callable[[float, float], float], first: float, last: float, transformer_list: list[str], operation_num: int, min_const_value: float, max_const_value: float, discretization_sz: int) -> str:

    discrete_const_arr: list[float]             = discretize(min_const_value, max_const_value, discretization_sz)
    tree_list: list[OperationTree]              = make_operation_tree(transformer_list, discrete_const_arr, operation_num)
    data_pts: list[tuple[float, OperationTree]] = []
    first_cut_tree_list: list[OperationTree]    = []
    newton_optimization_step                    = 1024
    newton_iteration_sz                         = 8

    for tree in tree_list:
        callable    = operation_tree_to_callable(tree)
        deviation   = calc_deviation(instrument, callable, first, last, discretization_sz)
        data_pts    += [(deviation, tree)]

    data_pts.sort(key = lambda x: x[0])
    first_cut_tree_list = [tree for (_, tree) in data_pts[:1234]]
    data_pts = []
    
    if len(first_cut_tree_list) == 0:
        raise Exception()

    for _ in range(newton_optimization_step):
        tree_idx: int               = random.randrange(0, len(first_cut_tree_list)) 
        cur_tree: OperationTree     = first_cut_tree_list[tree_idx]
        tree_sz: int                = operation_tree_count_ops(cur_tree)
        tree_idx: int               = random.randrange(0, tree_sz)
        tree_node: OperationTree    = operation_tree_node_at(cur_tree, tree_idx)
        
        if (tree_node.const_value == None):
            continue 
        
        func                        = lambda x: calc_deviation(instrument, operation_tree_to_callable(immutable_mutate_tree_const(cur_tree, tree_idx, x)), first, last, discretization_sz)
        approx_x                    = newton_approx(func, newton_iteration_sz, tree_node.const_value)
        tree_node.const_value       = approx_x

    for tree in first_cut_tree_list:
        callable    = operation_tree_to_callable(tree)
        deviation   = calc_deviation(instrument, callable, first, last, discretization_sz)
        data_pts    += [(deviation, tree)]

    score, tree = min(data_pts, key = lambda x: x[0])
    return score, pretty_print(tree)

def bitwise_or_df_da(a: int, b: int, r: int = 5) -> float:
        
    a       = int(a)
    b       = int(b)
    total   = float(0)
    
    for i in range(r):
        cur_f   = a * b
        prime_f = (a + i + 1) * b
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
    max_const_value         = 250
    max_operation_num       = 3
    discretization_sz       = 10

    return approx(operation, first, last, operation_arr, max_operation_num, min_const_value, max_const_value, discretization_sz)

def approx_bitwise_and_df_db(first: int, last :int) -> str:
    
    operation               = lambda a, b: bitwise_and_df_da(b, a)
    min_const_value         = 0
    max_const_value         = 250
    max_operation_num       = 3
    discretization_sz       = 10

    return approx(operation, first, last, operation_arr, max_operation_num, min_const_value, max_const_value, discretization_sz)

def approx_bitwise_or_df_da(first: int, last: int) -> str:
    
    operation               = lambda a, b: bitwise_or_df_da(a, b)
    min_const_value         = 0
    max_const_value         = 250
    max_operation_num       = 3
    discretization_sz       = 10

    return approx(operation, first, last, operation_arr, max_operation_num, min_const_value, max_const_value, discretization_sz)

def approx_bitwise_or_df_db(first: int, last: int) -> str:
    
    operation               = lambda a, b: bitwise_or_df_da(b, a)
    min_const_value         = 0
    max_const_value         = 250
    max_operation_num       = 3
    discretization_sz       = 10

    return approx(operation, first, last, operation_arr, max_operation_num, min_const_value, max_const_value, discretization_sz)
    
def approx_bitwise_xor_df_da(first: int, last: int) -> str:
    
    operation               = lambda a, b: bitwise_xor_df_da(a, b)
    min_const_value         = 0
    max_const_value         = 250
    max_operation_num       = 3
    discretization_sz       = 10

    return approx(operation, first, last, operation_arr, max_operation_num, min_const_value, max_const_value, discretization_sz)

def approx_bitwise_xor_df_db(first: int, last: int) -> str:
    
    operation               = lambda a, b: bitwise_xor_df_da(b, a)
    min_const_value         = 0
    max_const_value         = 250
    max_operation_num       = 3
    discretization_sz       = 10

    return approx(operation, first, last, operation_arr, max_operation_num, min_const_value, max_const_value, discretization_sz)

def approx_sqrt(first: int, last: int) -> str:

    operation               = lambda a, b: 1 / math.sqrt(a)
    min_const_value         = 0
    max_const_value         = 128
    max_operation_num       = 5
    discretization_sz       = 10

    return approx(operation, first, last, operation_arr, max_operation_num, min_const_value, max_const_value, discretization_sz)

def main():
    
    #alright - now is the time for discrete functions - discrete function is a RELU activator function

    # print(approx_bitwise_and_df_da(0, 256))
    # print(approx_bitwise_or_df_da(0, 256))
    # print(approx_bitwise_xor_df_da(0, 256))
    print(approx_sqrt(0, 256))

main()