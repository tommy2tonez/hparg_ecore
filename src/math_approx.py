import types 
from collections.abc import Callable
import math 
import json
import random 

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

def operation_discrete(lhs: object, rhs: object) -> object:
    if lhs < rhs:
        return lhs 
    
    return 0

def operation_exp(val) -> object:
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

operation_arr       = ["add", "sub", "mul", "div", "dsc", "and", "or", "xor"]
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
    "neg": operation_neg,
    "dsc": operation_discrete
}

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
    newton_iteration_sz                         = 16

    print(len(tree_list))

    for tree in tree_list:
        callable    = operation_tree_to_callable(tree)
        deviation   = calc_deviation(instrument, callable, first, last, discretization_sz)
        data_pts    += [(deviation, tree)]

    data_pts.sort(key = lambda x: x[0])
    first_cut_tree_list = [tree for (_, tree) in data_pts[:128]]
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
    max_const_value         = 256
    max_operation_num       = 8
    discretization_sz       = 5

    return approx(operation, first, last, operation_arr, max_operation_num, min_const_value, max_const_value, discretization_sz)

def approx_bitwise_and_df_db(first: int, last :int) -> str:
    
    operation               = lambda a, b: bitwise_and_df_da(b, a)
    min_const_value         = 0
    max_const_value         = 256
    max_operation_num       = 3
    discretization_sz       = 10

    return approx(operation, first, last, operation_arr, max_operation_num, min_const_value, max_const_value, discretization_sz)

def approx_bitwise_or_df_da(first: int, last: int) -> str:
    
    operation               = lambda a, b: bitwise_or_df_da(a, b)
    min_const_value         = 0
    max_const_value         = 256
    max_operation_num       = 3
    discretization_sz       = 10

    return approx(operation, first, last, operation_arr, max_operation_num, min_const_value, max_const_value, discretization_sz)

def approx_bitwise_or_df_db(first: int, last: int) -> str:
    
    operation               = lambda a, b: bitwise_or_df_da(b, a)
    min_const_value         = 0
    max_const_value         = 256
    max_operation_num       = 3
    discretization_sz       = 10

    return approx(operation, first, last, operation_arr, max_operation_num, min_const_value, max_const_value, discretization_sz)
    
def approx_bitwise_xor_df_da(first: int, last: int) -> str:
    
    operation               = lambda a, b: bitwise_xor_df_da(a, b)
    min_const_value         = 0
    max_const_value         = 256
    max_operation_num       = 3
    discretization_sz       = 10

    return approx(operation, first, last, operation_arr, max_operation_num, min_const_value, max_const_value, discretization_sz)

def approx_bitwise_xor_df_db(first: int, last: int) -> str:
    
    operation               = lambda a, b: bitwise_xor_df_da(b, a)
    min_const_value         = 0
    max_const_value         = 256
    max_operation_num       = 3
    discretization_sz       = 10

    return approx(operation, first, last, operation_arr, max_operation_num, min_const_value, max_const_value, discretization_sz)

def approx_sqrt(first: int, last: int) -> str:

    operation               = lambda a, b: math.sqrt(a)
    min_const_value         = 0
    max_const_value         = 256
    max_operation_num       = 5
    discretization_sz       = 10

    return approx(operation, first, last, operation_arr, max_operation_num, min_const_value, max_const_value, discretization_sz)

def main():
    
    #alright guys - let's take a break on data science today - I'll be back for A star + heuristic pruning tomorrow - goal is to able to approx heavy uint8_t and uint16_t operations (cos, sin, sqrt, etc.) + absolute accuracy for bitwise operators - within 15 operations - I hope - this is very very important
    #because our network's gonna be bitwise for numerical stability
    #we want to flops hard on the CPU - even though that's like the 1980 tech

    #alright - let's talk about asymmetric encryption | symmetric cracking methods
    #we know for sure that, the output of the encryption key is at best uniform, at worst skewed
    #uniform in the sense, or in the coordinate, that the output uniformly reduce the possibility space of the signing key 
    #assume our symmetric encryption function is f(x) -> y - think JWT signing algorithms
    #we are trying to model f(x) by approximating it - for every char created by the token - we are 1 char closer to estimating the token or reduce the possibility space by a factor of 256
    #for every two char created by the encryption key - we reduce the possibility space by a factor of 256 ** 2, and so on.
    #fortunately, JWT is not uniform distribution, and JWT signing secret is usually < 256 char - so the cracking time is actually much faster (I'm taking about instant fast) - if appropriate methods are being applied
    #this is a brute force version of that - I don't recommend anyone to actually crack JWT with this algorithm lol - the right tool for the right job is the network
    
    #what brother desparately needs was the network_kernelmap_x and my flat_datastructure
    #yeah brother - even on CPU - you need to contain the RAM usage - you don't let it CAP the RAM and die - that's the worst approach and irreversable approach EVER
    #this is the reason you want to RAID your storage - not for ingestion speed - god damn it
    #you want to allocate everything on the memregion - make sure that it's reachability is inside the memregion
    #and locality of the immutable flat_datastructure
    #best thing - you can dispatch to CUDA which runs the jobs for you
    #that's the definition of MPP - not multithreading and concurrency
    #multithreading and concurrency are premature optimizations - NEVER to be applicable in real life - only to be used for affined tasks (like draining kernel_network_buffer) and high latency IO tasks - other than that - NEVER use concurrency to boost your flops - that's what GPU is best at - and not CPU
    #I don't have bad intentions or whatever - I tell you the optimizables that can 10x your sales - yeah - that Neo4j after you implemented this can NEVER EVER beat the benchmark
    #truth is I dont know I spent 1 year to think about the optimizables that I could have for TigerGraph
    #the moment you followed Spark and friends was a bad moment - MPP is always about GPUs
    #I think about what you thought too - it's memregion locality - node collapses - this is actually a hard task that I haven't been able to solve yet
    #the only thing that I thought of was lambda as a service - circle infected region (by running BFS algorithms and friends) - dispatch it to distributed lambdas 
    #other non-heavy tasks like simple queries can be dispatched to the normal engine

    #I, however, pursue an entire different radix of Graph. It's dg - derivative of gradients (acceleration, jerk, snap, crackle, pop, whatever) - this is some new stuff that I will spend the next 2-3 years to work on
    #you might not see what I see yet it's always about time in this tensor transformation field
    #you want to time the backprop
    #you want to time the msgrbwd

    #I thought what you thought too - why don't I just fucking use a counter on the tile and backprop it? It's actually going to bottleneck the future architecture of dynamic pathing - and affect locality of dispatching - if you backprop it immediately after counter reaches 0 - you risk bad locality
    #the only way to solve the locality problem is to fatten the tile - which is what PyTorch has been doing - and yeah - I just reinvented PyTorch - yay
    #so it's actually about timing and reducing LOGIT_COUNT_PER_TILE yet maintaining the GPU flops

    print(approx_bitwise_and_df_da(0, 256))

main()