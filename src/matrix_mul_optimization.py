
#

import math
import copy
from typing import Optional
import functools

def get_projection_storage_size(in_feature_sz: int, derivative_order_sz: int) -> int:

    if in_feature_sz < 0:
        raise Exception()

    if derivative_order_sz < 0:
        raise Exception()

    if in_feature_sz == 0:
        return 0

    if derivative_order_sz == 0:
        return 0

    if in_feature_sz == 1:
        return derivative_order_sz 

    return get_projection_storage_size(in_feature_sz - 1, derivative_order_sz) * derivative_order_sz 

def make_taylor_projection_storage(in_feature_sz: int, derivative_order_sz: int) -> list[float]:

    return [float(0) for _ in range(get_projection_storage_size(in_feature_sz, derivative_order_sz))]

def taylor_project(x: float, coeff_arr: list[float]) -> float:

    rs: float = 0.0

    for i in range(len(coeff_arr)):
        factorial_multiplier: float = float(1) / math.factorial(i)
        x_multiplier: float         = math.pow(x, i)
        coeff_multiplier: float     = coeff_arr[i]
        rs                          += factorial_multiplier * x_multiplier * coeff_multiplier

    return rs

def multidimensional_taylor_project(x: list[float], derivative_order_sz: int, storage_vec: list[float]) -> float:

    if derivative_order_sz <= 0:
        raise Exception()

    if len(x) == 0:
        raise Exception()

    required_storage_sz: int = get_projection_storage_size(len(x), derivative_order_sz)

    if required_storage_sz != len(storage_vec):
        raise Exception()

    if len(x) == 1:
        return taylor_project(x[0], storage_vec) 

    width: int                          = get_projection_storage_size(len(x) - 1, derivative_order_sz)
    projection_storage_vec: list[float] = []

    #f(x) = f(0) + f'(0) * t + f''(0) * 1/2 * t^2 + ...
    #we are removing the x[0], x[1] ... up to x[n - 1] by recursing the projection of derivatives

    for i in range(derivative_order_sz):
        first: int                      = width * i
        last: int                       = width * (i + 1)
        sub_storage_vec: list[float]    = storage_vec[first: last]
        projected_derivative: float     = multidimensional_taylor_project(x[:-1], derivative_order_sz, sub_storage_vec)
        projection_storage_vec          += [projected_derivative]

    return taylor_project(x[-1], projection_storage_vec)

def get_optimal_derivative_order_size(in_feature_sz: int, max_storage_sz: int) -> int:

    MAX_DERIVATIVE_ORDER_SZ: int = 10

    for i in range(MAX_DERIVATIVE_ORDER_SZ):
        order_sz: int               = MAX_DERIVATIVE_ORDER_SZ - i
        projection_storage_sz: int  = get_projection_storage_size(in_feature_sz, order_sz)

        if (projection_storage_sz <= max_storage_sz):
            return order_sz

    raise Exception()

def get_projection_storage_sz(in_feature_sz: int, max_storage_sz: int) -> int:

    return get_projection_storage_size(in_feature_sz, get_optimal_derivative_order_size(in_feature_sz, max_storage_sz))

def make_projection_storage(sz: int) -> list[float]:

    return [float(0) for _ in range(sz)]

def multidimensional_projection(x: list[float], projection_storage: list[float]) -> float:

    return multidimensional_taylor_project(x, get_optimal_derivative_order_size(len(x), len(projection_storage)), projection_storage) 

class Logit:

    def __init__(self,
                 projection_storage_vec: Optional[list[float]],
                 descendant_vec: Optional[list[object]],
                 leaf_value: Optional[float]):

        self.projection_storage_vec = copy.deepcopy(projection_storage_vec)
        self.descendant_vec         = copy.deepcopy(descendant_vec)
        self.leaf_value             = copy.deepcopy(leaf_value)

    def get_value(self) -> float:

        if self.leaf_value != None:
            return self.leaf_value
        elif self.descendant_vec != None:
            return multidimensional_projection([descendant.get_value() for descendant in self.descendant_vec], self.projection_storage_vec)
        else:
            raise Exception()

    def get_projection_storage_vec(self) -> list[float]:

        return copy.deepcopy(self.projection_storage_vec)

    def set_projection_storage_vec(self, projection_storage_vec: list[float]):

        if (projection_storage_vec != None and len(projection_storage_vec) != len(self.projection_storage_vec)):
            raise Exception()

        self.projection_storage_vec = copy.deepcopy(projection_storage_vec)


class SumLogit:

    def __init__(self,
                 lhs: Logit,
                 rhs: Logit):

        self.lhs = lhs
        self.rhs = rhs

    def get_value(self) -> float:

        return self.lhs.get_value() + self.rhs.get_value()

    def get_projection_storage_vec(self) -> list[float]:

        return []

    def set_projection_storage_vec(self, projection_storage_vec: list[float]):

        pass

class LogitPack:

    def __init__(self,
                 logit_vec: list[Logit]):
        
        self.logit_vec  = logit_vec
    
    def size(self) -> int:

        return len(self.logit_vec)

    def get(self, idx: int) -> Logit:

        return self.logit_vec[idx]

    def raw(self) -> list[Logit]:

        return list(self.logit_vec)
    
    def as_list(self) -> list[Logit]:

        return list(self.logit_vec)

def get_leaf(val: float) -> Logit:

    return Logit(None, None, val) 

def sum_logit(lhs: Logit, rhs: Logit) -> Logit:

    #this is complicated
    #a sum operation is essentially a x + y => initial displacement = 0, velocity = 1 

    return SumLogit(lhs, rhs)  

def one_sum(lhs: Logit, projection_storage_sz: int) -> Logit:

    actual_projection_storage_sz: int   = get_projection_storage_sz(1, projection_storage_sz)

    if actual_projection_storage_sz == 0:
        raise Exception()

    feature_vec: list[float]            = make_projection_storage(actual_projection_storage_sz)

    return Logit(feature_vec, [lhs], None)

def two_sum(lhs: Logit, rhs: Logit, projection_storage_sz: int) -> Logit:

    actual_projection_storage_sz: int   = get_projection_storage_sz(2, projection_storage_sz)

    if actual_projection_storage_sz == 0:
        raise Exception()

    feature_vec: list[float]            = make_projection_storage(actual_projection_storage_sz)

    return Logit(feature_vec, [lhs, rhs], None) 

def three_sum(x: Logit, x1: Logit, x2: Logit, projection_storage_sz: int) -> Logit:

    actual_projection_storage_sz: int   = get_projection_storage_sz(3, projection_storage_sz)

    if actual_projection_storage_sz == 0:
        raise Exception()

    feature_vec: list[float]            = make_projection_storage(actual_projection_storage_sz)

    return Logit(feature_vec, [x, x1, x2], None)

def four_sum(x: Logit, x1: Logit, x2: Logit, x3: Logit, projection_storage_sz: int) -> Logit:

    actual_projection_storage_sz: int   = get_projection_storage_sz(4, projection_storage_sz)

    if actual_projection_storage_sz == 0:
        raise Exception()

    feature_vec: list[float]            = make_projection_storage(actual_projection_storage_sz)

    return Logit(feature_vec, [x, x1, x2, x3], None)

def five_sum(x: Logit, x1: Logit, x2: Logit, x3: Logit, x4: Logit, projection_storage_sz: int) -> Logit:

    actual_projection_storage_sz: int   = get_projection_storage_sz(5, projection_storage_sz)

    if actual_projection_storage_sz == 0:
        raise Exception()

    feature_vec: list[float]            = make_projection_storage(actual_projection_storage_sz)

    return Logit(feature_vec, [x, x1, x2, x3, x4], None)

def six_sum(x: Logit, x1: Logit, x2: Logit, x3: Logit, x4: Logit, x5: Logit, projection_storage_sz: int) -> Logit:

    actual_projection_storage_sz: int   = get_projection_storage_sz(6, projection_storage_sz)

    if actual_projection_storage_sz == 0:
        raise Exception()

    feature_vec: list[float]            = make_projection_storage(actual_projection_storage_sz)

    return Logit(feature_vec, [x, x1, x2, x3, x4, x5], None)

def generic_sum(x_list: list[Logit], projection_storage_sz: int) -> Logit:

    if len(x_list) == 1:
        return one_sum(x_list[0], projection_storage_sz)
    elif len(x_list) == 2:
        return two_sum(x_list[0], x_list[1], projection_storage_sz)
    elif len(x_list) == 3:
        return three_sum(x_list[0], x_list[1], x_list[2], projection_storage_sz)
    elif len(x_list) == 4:
        return four_sum(x_list[0], x_list[1], x_list[2], x_list[3], projection_storage_sz)
    elif len(x_list) == 5:
        return five_sum(x_list[0], x_list[1], x_list[2], x_list[3], x_list[4], projection_storage_sz)
    elif len(x_list) == 6:
        return six_sum(x_list[0], x_list[1], x_list[2], x_list[3], x_list[4], x_list[5], projection_storage_sz)  
    else:
        raise Exception()

def flatten(tensor: object) -> list[object]:

    if type(tensor) == type(list()):
        return functools.reduce(lambda a, b: a + b, [flatten(sub_tensor) for sub_tensor in tensor], [])

    return [tensor]

def flatten_space(space: list[int]) -> list[int]:

    if len(space) == 0:
        raise Exception()

    sz: int = 1

    for i in range(len(space)):
        sz *= space[i]

    return [sz]; 

def shape_as(tensor: object, space: list[int]) -> object:

    if len(space) == 0:
        raise Exception()

    logit_vec: list[object]     = flatten(tensor)

    if flatten_space(space)[0] != len(logit_vec):
        raise Exception()

    if len(space) == 1:        
        return logit_vec

    sub_space: list[int]        = space[1:]
    flattened_sub_space_sz: int = flatten_space(sub_space)[0]
    rs: list[object]            = []

    for i in range(space[0]):
        first: int          = flattened_sub_space_sz * i
        last: int           = flattened_sub_space_sz * (i + 1)
        sub_tensor: object  = logit_vec[first: last]
        rs                  += [shape_as(sub_tensor, sub_space)]

    return rs

def make_empty(space: list[int]) -> object:

    if len(space) == 0:
        raise Exception()

    if len(space) == 1:
        return [object() for _ in range(space[0])]

    return [make_empty(space[1:]) for i in range(space[0])] 

def squeeze_space(space: list[int]) -> list[int]:

    if len(space) <= 1:
        return list(space)

    if space[-1] == 1:
        return squeeze_space(space[:-1]) 

    return list(space) 

def get_tensor_space_base(tensor: object) -> list[int]:

    if type(tensor) == type(list()):
        return [len(tensor)] + get_tensor_space_base(tensor[0])

    return [1]

def get_tensor_space(tensor: object) -> list[int]:

    rs: list[int] = get_tensor_space_base(tensor)

    if len(rs) > 1:
        return rs[:-1]
    
    return rs

def make_empty_as(tensor: object) -> object:

    return make_empty(get_tensor_space(tensor)) 

def rotate(tensor: object) -> object:

    space: list[int]    = get_tensor_space(tensor)

    if len(space) != 2:
        raise Exception()

    new_tensor: list    = make_empty(space[::-1])

    for i in range(space[0]):
        for j in range(space[1]):
            new_tensor[j][i] = tensor[i][j]

    return new_tensor

def sqrt(val: int) -> int:

    return int(math.sqrt(val))

def threepack_twosum(lhs: LogitPack, rhs: LogitPack, projection_storage_sz: int) -> LogitPack:

    if lhs.size() != 3:
        raise Exception()

    if rhs.size() != 3:
        raise Exception()

    rs_0: Logit     = six_sum(lhs.get(0), lhs.get(1), lhs.get(2), rhs.get(0), rhs.get(1), rhs.get(2), projection_storage_sz)
    rs_1: Logit     = six_sum(lhs.get(0), lhs.get(1), lhs.get(2), rhs.get(0), rhs.get(1), rhs.get(2), projection_storage_sz)
    rs_2: Logit     = six_sum(lhs.get(0), lhs.get(1), lhs.get(2), rhs.get(0), rhs.get(1), rhs.get(2), projection_storage_sz)

    return LogitPack([rs_0, rs_1, rs_2]) 

def pack_twosum(lhs: LogitPack, rhs: LogitPack, projection_storage_sz: int) -> LogitPack:

    if lhs.size() == 3 and rhs.size() == 3:
        return threepack_twosum(lhs, rhs, projection_storage_sz)

    raise Exception()

def sum_accum_2(lhs: LogitPack, rhs: LogitPack) -> LogitPack:

    if lhs.size() != rhs.size():
        raise Exception()

    return LogitPack([sum_logit(*pair) for pair in zip(lhs.as_list(), rhs.as_list())])

def sum_accum(*args) -> LogitPack:

    return functools.reduce(sum_accum_2, args[1:], args[0]) 

def forward_map_suffix_array(arr: list, suffix_arr: list[int]) -> list:
    
    for suffix in suffix_arr:
        rs_arr += [arr[suffix]]
    
    return rs_arr

def backward_map_suffix_array(arr: list, suffix_arr: list[int]) -> list:

    rs: list = [object() for _ in range(arr)]

    for i in range(len(suffix_arr)):
        rs[suffix_arr[i]] = arr[i] 

    return rs

def shake_x(logit_list: list[Brain],
            virtual_suffix_array: list[list[int]],
            projection_storage_sz: int,
            initial_iteration_sz: int,
            iteration_sz: int,
            storage_decay_rate: float,
            partitioning_resolution: float) -> list[Brain]:

    #alright, so what are we doing? why is this better?
    #we replaced normal matrix multiplication with 3 dimensional multiplication, every cell is a 3 dimensional feature vector x another 3 dimensional feature vector (this is the important clue)
    #we turned pigeonhole sort -> radixsort (x = x + f(x) -> x = x + f(x) and arg = x + f(x))
    #we used virtual matrix to add more crosses, normally we only do one cross, row+col
    #we recursively call the function to add projection accuration  
    #most importantly, we used 6 dimensional projection + Brain to increase numerical stability of semantics
    #last but not least, this is hard to implement, the calibration of the brain by using unit vector (cosine similarity of shapes) in an arbitrary Taylor Projection space that skews in the direction of trig (cos-sin waves) projection

    if iteration_sz == 0:
        return logit_list

    list_sz: int = len(logit_list)

    if list_sz not in [3, 9, 27, 81]:
        raise Exception()

    if list_sz == 3:
        first: Brain            = brain_accum(brain_twosum(logit_list[0], logit_list[1]), brain_twosum(logit_list[0], logit_list[2]), logit_list[0]) 
        second: Brain           = brain_accum(brain_twosum(logit_list[1], logit_list[0]), brain_twosum(logit_list[1], logit_list[2]), logit_list[1])
        third: Brain            = brain_accum(brain_twosum(logit_list[2], logit_list[0]), brain_twosum(logit_list[2], logit_list[1]), logit_list[2]) 

        _first: Brain           = brain_accum(brain_twosum(logit_list[0], logit_list[1]), brain_twosum(logit_list[0], logit_list[2]), logit_list[0]) 
        _second: Brain          = brain_accum(brain_twosum(logit_list[1], logit_list[0]), brain_twosum(logit_list[1], logit_list[2]), logit_list[1])
        _third: Brain           = brain_accum(brain_twosum(logit_list[2], logit_list[0]), brain_twosum(logit_list[2], logit_list[1]), logit_list[2]) 

        next_arg: list[Brain]   = [calibrate_brain(_first, projection_storage_sz, partitioning_resolution),
                                   calibrate_brain(_second, projection_storage_sz, partitioning_resolution),
                                   calibrate_brain(_third, projection_storage_sz, partitioning_resolution)]

        rs_arg_1: list[Brain]   = [first, second, third]
        rs_arg_2: list[Brain]   = shake_x(next_arg,
                                          virtual_suffix_array
                                          projection_storage_sz * storage_decay_rate,
                                          initial_iteration_sz,
                                          iteration_sz - 1,
                                          storage_decay_rate,
                                          partitioning_resolution)

        #this is the most important line in the history of projection radix sort, thank you John

        return list(map(brain_accum, zip(rs_arg_1, rs_arg_2)))

    dim_sz: int                                             = cube_root(list_sz)    
    space_sz: list[int]                                     = [dim_sz, dim_sz]
    virtual_transformed_logit_list: list[list[Brain]]       = []
    virtual_transformed_logit_list_2: list[list[Brain]]     = []

    for suffix_array in virtual_suffix_array:
        virtual_logit_list: list[Brain]                 = forward_map_suffix_array(logit_list, suffix_array)
        shaped_virtual_logit_list: list[list[Brain]]    = shape_as(virtual_logit_list, space_sz)
        transformed_logit_list: list[list[Brain]]       = []
        transformed_logit_list_2: list[list[Brain]]     = []

        for i in range(dim_sz):
            shaked_row: list[Brain]     = shake_x(shaped_virtual_logit_list_1[i],
                                                  get_initial_virtual_suffix_array(len(virtual_suffix_array), [cube_root(dim_sz), cube_root(dim_sz)])
                                                  projection_storage_sz,
                                                  initial_iteration_sz,
                                                  initial_iteration_sz,
                                                  storage_decay_rate,
                                                  partitioning_resolution)

            shaked_row_2: list[Brain]   = shake_x(shaped_virtual_logit_list_1[i],
                                                  get_initial_virtual_suffix_array(len(virtual_suffix_array), [cube_root(dim_sz), cube_root(dim_sz)]),
                                                  projection_storage_sz,
                                                  initial_iteration_sz,
                                                  initial_iteration_sz,
                                                  storage_decay_rate,
                                                  partitioning_resolution)

            transformed_logit_list      += [shaked_row]
            transformed_logit_list_2    += [shaked_row_2]

        virtual_transformed_logit_list      += [backward_map_suffix_array(shape_as(transformed_logit_list, list_sz), suffix_array)]
        virtual_transformed_logit_list_2    += [backward_map_suffix_array(shape_as(transformed_logit_list_2, list_sz), suffix_array)]

    virtual_transformed_logit_list      += [logit_list]
    rs_1: list[Brain]                   = pairwise_sum_accum(virtual_transformed_logit_list)
    virtual_transformed_logit_list_2    += [logit_list]
    rs_2: list[Brain]                   = list(map(calibrate_brain, pairwise_sum_accum(virtual_transformed_logit_list_2)))

    nxt_ctx_list: list[Brain]           = shake_x(rs_2,
                                                  list(map(flatten, map(rotate, map(shape_as_func(space_sz), virtual_suffix_array)))),
                                                  projection_storage_sz * storage_decay_rate,
                                                  initial_iteration_sz,
                                                  iteration_sz - 1,
                                                  storage_decay_rate,
                                                  partitioning_resolution)

    return list(map(brain_accum, zip(rs_1, nxt_ctx_list)))

def main():

    pass

main()

