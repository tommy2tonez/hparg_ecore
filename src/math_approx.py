import types 
from collections.abc import Callable
import math 
import json
import random 
from queue import PriorityQueue

#this is a trillion dollar answer that I want yall to reflect on
#I want yall to prove that this method can actually approximate any function (continuous and non-continuous) to the absolute accuracy with sufficient number of hops
#I want yall to find the difference between THIS and the current state of the art transformer - which I call gradient retarded - because it's impossible to estimate the function accurately
#I want yall to figure out the way to throw every electrical signals at the network and make this predict everything to the absolute accuracy - yes - it's possible - but it requires crazy number of flops

#current transformer
#proof by induction:
#assume that our network is perfect, f(x) -> y, y is 100% accurate
#assume our incoming training data is z, f(z) -> p is not accurate
#in order for p to be accurate - p needs to reflect on all the tensors that it passes through, if the hops is 32 - then there are 32 layers of tensors that need to be mutated
#the possibility that the f(x) -> y ziggled just for the input z is near 0 - if it ziggles, then it has to ziggle for more than just z - which contradicts with the f(x) -> y is 100% in the beginning

#new transformer:
#proof by induction
#assume that our network is perfect, f(x) -> y is 100% accurate
#assume our incoming training data is z, f(z) -> p is not accurate
#then there exists a way for the function to continue to be 100% accurate such is modeled by the if x == z then f(x) else p is applied - and the function continues to be 100% accurate

#I want yall to prove that if the smaller the possibility space of a f(x) -> y - the higher the intelligence of the f(x) -> y
#possibility space can be briefly described as: 
#given N logits, possibility(f(x, N) -> y) = K
#hint, it's not attention. Attention is a part of dimensional reduction which limits the possibility space
#I want yall to prove that if the new transformer with a small possibility space can achieve super intelligence 

#next thing to think about is Green's theorem
#we have a circular ring of leaf logits - which is called an artifical brain
#when we talk about brain - we talk about forward and backward - not just forward - the brain that only does forward is retarded - such that it does not do real-time update of events
#a brain can become retarded after a certain period of real-time training (when it becomes saturated - we want to prevent this by using discrete switches - this is called transistor in electrical engineering)

#this circular ring of leaf logits guarantees that user A, B, C, D ... information is pushed the brain after maximum 1 brain frequency - brain frequency == adjecent_synchronous_time * len(ring)
#each leaf logit (tile) has it's own ring - where the each element of the ring is a different version of the same tile
#right - this brain of 3 billion devices - cannot have the worst case of take adjecent_synchronous_time * len(ring) to be updated
#what we want to do is to split the big ring into smaller rings 
#and do coin flip for the ring_master_node to either continue in the ring - or pulling from another random ring

def operation_nil(lhs: object, rhs: object) -> object:
    return type(lhs)()

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
    "dsc": operation_discrete,
    "nil": operation_nil
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


    def __lt__(self, other):
        return False 
    
    def __eq__(self, other):
        return False 
    
    def __gt__(self, other):
        return False 
    
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

def make_operation_tree(operation_num: int) -> list[OperationTree]:
    
    if operation_num == 0:
        return []

    possible_root_arr: list[OperationTree]  = []
    possible_root_arr += [OperationTree(None, None, OPS_TREE_KIND_PAIR_OPS, "nil", None)] 
    possible_root_arr += [OperationTree(None, None, OPS_TREE_KIND_PAIR_OPS, "dsc", None)]

    operation_num                           -= 1
    left_root_arr: list[OperationTree]      = make_operation_tree(operation_num)
    rs: list[OperationTree]                 = []

    for i in range(len(left_root_arr)):
        left_operation_sz                   = operation_tree_count_ops(left_root_arr[i])
        right_operation_sz                  = operation_num - left_operation_sz
        right_root_arr: list[OperationTree] = make_operation_tree(right_operation_sz)

        for j in range(len(right_root_arr)):
            for z in range(len(possible_root_arr)):
                new_tree: OperationTree = clone_tree(possible_root_arr[z])
                new_tree.left           = clone_tree(left_root_arr[i])
                new_tree.right          = clone_tree(right_root_arr[j])
                rs                      += [new_tree]

    rs += [OperationTree(None, None, OPS_TREE_KIND_CONST_EXTRACT, None, float(0.0))]
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

def immutable_mutate_tree_kind(root: OperationTree, tree_idx: int, kind: object) -> OperationTree:

    new_root            = clone_tree(root)
    node                = operation_tree_node_at(new_root, tree_idx)
    node.kind           = kind 

    return new_root 

def immutable_mutate_tree_operation(root: OperationTree, tree_idx: int, operation_name: object) -> OperationTree:

    new_root                    = clone_tree(root)
    node                        = operation_tree_node_at(new_root, tree_idx)
    node.pair_operation_name    = operation_name

    return new_root 

def randomize(arr: list[object]) -> object:

    if len(arr) == 0:
        raise Exception()
    
    idx = random.randrange(0, len(arr))
    return arr[idx]

def approx(instrument: Callable[[float, float], float], first: float, last: float, transformer_list: list[str], operation_num: int, optimization_step: int, min_const_value: float, max_const_value: float, discretization_sz: int) -> str:
    
    discrete_const_arr: list[float]             = discretize(min_const_value, max_const_value, discretization_sz)
    tree_list: list[OperationTree]              = make_operation_tree(operation_num)
    newton_iteration_sz                         = 8
    newton_sampling_sz                          = 32
    randomization_factor                        = 32
    max_search_size                             = 1 << 15
    pq: PriorityQueue                           = PriorityQueue()
    min_deviation: float                        = 1 << 50
    rs_tree: OperationTree                      = None  
    visited_map: dict[str, int]                 = dict()

    for tree in tree_list:
        callable: Callable[[float, float], float] = operation_tree_to_callable(tree)
        deviation: float = calc_deviation(instrument, callable, first, last, discretization_sz) 
        pq.put((2.0, tree))

    for iter in range(optimization_step):        
        print(iter)

        if (pq.empty()):
            break

        (deviation, tree) = pq.get()
        representable = json.dumps(pretty_print(tree))

        if (representable not in visited_map):
            visited_map[representable] = 0

        if (visited_map[representable] == 10):
            continue 

        visited_map[representable] += 1
        tree_sz = operation_tree_count_ops(tree)

        if (min_deviation > deviation):
            min_deviation = deviation
            rs_tree = tree

        for __ in range(newton_sampling_sz):
            tree_idx = random.randrange(0, tree_sz)
            tree_node: OperationTree = operation_tree_node_at(tree, tree_idx)

            if (tree_node.kind == OPS_TREE_KIND_LEFT_EXTRACT):
                pass
            elif (tree_node.kind == OPS_TREE_KIND_RIGHT_EXTRACT):
                pass
            elif (tree_node.kind == OPS_TREE_KIND_CONST_EXTRACT):
                f: Callable[[float], float] = lambda x: calc_deviation(instrument, operation_tree_to_callable(immutable_mutate_tree_const(tree, tree_idx, x)), first, last, discretization_sz)
                approx_x        = newton_approx(f, newton_iteration_sz, tree_node.const_value)
                new_tree        = immutable_mutate_tree_const(tree, tree_idx, approx_x)
                new_deviation   = calc_deviation(instrument, operation_tree_to_callable(new_tree), first, last, discretization_sz)
                pq.put((new_deviation, new_tree))
            elif (tree_node.kind == OPS_TREE_KIND_PAIR_OPS):
                pass
            else:
                raise Exception()
        
        for __ in range(randomization_factor):
            tree_idx = random.randrange(0, tree_sz)
            tree_node: OperationTree = operation_tree_node_at(tree, tree_idx)

            if (tree_node.kind == OPS_TREE_KIND_LEFT_EXTRACT or tree_node.kind == OPS_TREE_KIND_RIGHT_EXTRACT or tree_node.kind == OPS_TREE_KIND_CONST_EXTRACT):
                new_tree        = immutable_mutate_tree_kind(tree, tree_idx, randomize([OPS_TREE_KIND_LEFT_EXTRACT, OPS_TREE_KIND_RIGHT_EXTRACT, OPS_TREE_KIND_CONST_EXTRACT]))
                new_tree        = immutable_mutate_tree_const(new_tree, tree_idx, randomize(discrete_const_arr))
                new_deviation   = calc_deviation(instrument, operation_tree_to_callable(new_tree), first, last, discretization_sz)
                pq.put((new_deviation, new_tree))
            elif (tree_node.kind == OPS_TREE_KIND_PAIR_OPS):
                if (tree_node.pair_operation_name == "dsc"):
                    pass 
                else:
                    new_tree        = immutable_mutate_tree_operation(tree, tree_idx, randomize(transformer_list))
                    new_deviation   = calc_deviation(instrument, operation_tree_to_callable(new_tree), first, last, discretization_sz)
                    pq.put((new_deviation, new_tree))
            else:
                raise Exception()

    return min_deviation, pretty_print(rs_tree)  

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
 
def approx_bitwise_and_df_da(first: int, last: int, optimization_step: int) -> str:
    
    operation               = lambda a, b: bitwise_and_df_da(a, b)
    min_const_value         = 0
    max_const_value         = 256
    max_operation_num       = 15
    discretization_sz       = 5

    return approx(operation, first, last, operation_arr, max_operation_num, optimization_step, min_const_value, max_const_value, discretization_sz)

def approx_bitwise_and_df_db(first: int, last :int) -> str:
    
    operation               = lambda a, b: bitwise_and_df_da(b, a)
    min_const_value         = 0
    max_const_value         = 256
    max_operation_num       = 10
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

def approx_sqrt(first: int, last: int, optimization_step: int) -> str:

    operation               = lambda a, b: math.sqrt(a)
    min_const_value         = 0
    max_const_value         = 256
    max_operation_num       = 10
    discretization_sz       = 10

    return approx(operation, first, last, operation_arr, max_operation_num, optimization_step, min_const_value, max_const_value, discretization_sz)

def main():
    
    print(approx_sqrt(0, 256, 4096))

main()