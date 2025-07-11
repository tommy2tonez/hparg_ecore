
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

def shake_x(logit_list: list[LogitPack],
            projection_storage_sz: int,
            initial_iteration_sz: int,
            iteration_sz: int,
            decay_rate: int) -> list[LogitPack]:

    if iteration_sz == 0:
        return logit_list 

    list_sz: int = len(logit_list)

    if list_sz not in [2, 4, 16, 256, 65536]:
        raise Exception()

    if list_sz == 2:
        delta_pack_0: LogitPack = pack_twosum(logit_list[0], logit_list[1], projection_storage_sz)
        delta_pack_1: LogitPack = pack_twosum(logit_list[1], logit_list[0], projection_storage_sz)
        new_logit_0: LogitPack  = sum_accum(logit_list[0], delta_pack_0)
        new_logit_1: LogitPack  = sum_accum(logit_list[1], delta_pack_1)
        rs: list[LogitPack]     = [new_logit_0, new_logit_1]

        #the problem of Taylor Projection as pointed out by my colleagues is the resolution of the projection
        #note that we are still using the Taylor Projection, but the resolution of the numerical value is different, for example a binary Taylor Projection would look like -1 1 -1 1, to project sin + cos waves with optimal projection storage size

        #for the former layers, we'd want to decrease the resolution to increase derivative order, because we'd want to "group" the semantically equivalent context together (things that are fired to the same grid would be fired together in the next shake_x iterations), arguably, that we could achieve the same by allocating more storage for the traditional methods
        #essentially, the former layers would be having a lot of ups and downs to "radix" the context into their appropriate buckets instead of bending the input space in the direction of the output space  
        #in other words, we are changing our projection space to utilize more shapes over the linearity of those, this is where we got the decay rate wrong, decay rate is supposed to be the inverse decay of linearity, not the decay of projection_storage_sz ..., scientifically speaking
        #in that sense, the decay rate is the decay of discretization value 

        #the sole problem with those approaches is that we'd be allocating more resources for our search mission because the differential would be way way off

        #so its kind of difficult, because a derivative order of 8 would take probably terabytes of storage to project
        #however, if we are using binary coefficient, we'd be looking at /32 of the cost

        #if we are going under binary, we'd be looking at a 0.01 bit/ coefficient, for a 6 dimensional 10 derivative order space
        #for the lower layers, it'd be extremely hard to train, because like I said, we dont have the differential friction, we'd come up with a patch

        #this is where I'm wrong, again
        #essentially, those two values are distinct
        #the linearity index and the projection_storage_sz
        #we should still allocate more projection_storage_sz for the lower layers for they being bases
        #and we should decrease linearity index of the lower layer, the linearity index of 1 is essentially a line derivative order of 2 (supposedly 1)
        #linearity index of 0 is essentially a derivative order of 20 or 30, with high_discretization_sz, we'll syntheticalize the formula
        #so there is a linear correlation between the linearity index and derivative order and discretization_sz

        #for every iteration, we'd increase the linearity index and decrease the projection_storage_sz
        #for the upper layer is less important in the hierarchical tree and more linearly relevant (we just sorted those)

        return shake_x(rs, projection_storage_sz * decay_rate, initial_iteration_sz, iteration_sz - 1, decay_rate) #I could not come up with projection resolution and differential searchability yet. If this does not project correctly in the first layers, we'll be messed up very very badly

    dim_sz: int                                                 = sqrt(list_sz)    
    two_dimensional_logit_list: list[list[LogitPack]]           = shape_as(logit_list, [dim_sz, dim_sz])

    rotated_two_dimensional_logit_list: list[list[LogitPack]]   = rotate(two_dimensional_logit_list)
    transformed_logit_list: list[list[LogitPack]]               = []
    transformed_logit_list_2: list[list[LogitPack]]             = []
    transformed_logit_list_3: list[list[LogitPack]]             = []

    for i in range(dim_sz):
        shaked_row: list[LogitPack]     = shake_x(two_dimensional_logit_list[i], projection_storage_sz, initial_iteration_sz, initial_iteration_sz, decay_rate)
        transformed_logit_list          += [shaked_row]
        shaked_row_2: list[LogitPack]   = shake_x(two_dimensional_logit_list[i], projection_storage_sz, initial_iteration_sz, initial_iteration_sz, decay_rate)
        transformed_logit_list_2        += [shaked_row_2]
        shaked_row_3: list[LogitPack]   = shake_x(rotated_two_dimensional_logit_list[i], projection_storage_sz, initial_iteration_sz, initial_iteration_sz, decay_rate)
        transformed_logit_list_3        += [shaked_row_3]

    transformed_logit_list          = rotate(transformed_logit_list)
    transformed_logit_list_3        = rotate(transformed_logit_list_3)
    rs_list: list[list[LogitPack]]  = []

    #we are missing a suffix arrays implementation to offload the row-col pressures

    for i in range(dim_sz):
        org_list: list[LogitPack]           = two_dimensional_logit_list[i]
        ctx_list: list[LogitPack]           = transformed_logit_list[i]
        shaked_ctx_list: list[LogitPack]    = shake_x(ctx_list, projection_storage_sz, initial_iteration_sz, initial_iteration_sz, decay_rate)
        other_ctx_list: list[LogitPack]     = transformed_logit_list_2[i]
        other_ctx_list_2: list[LogitPack]   = transformed_logit_list_3[i] 
        new_row: list[LogitPack]            = []

        for j in range(dim_sz):
            new_logit: LogitPack    = sum_accum(org_list[j],
                                                other_ctx_list[j],
                                                shaked_ctx_list[j],
                                                other_ctx_list_2[j])

            new_row                 += [new_logit]

        rs_list += [new_row]

    return shake_x(flatten(rotate(rs_list)), projection_storage_sz * decay_rate, initial_iteration_sz, iteration_sz - 1, decay_rate)

def main():

    inp: list[LogitPack]    = [LogitPack([get_leaf(1), get_leaf(2), get_leaf(3)]) for _ in range(16)]
    output: list[LogitPack] = shake_x(inp, 8, 1, 1, 1)

    print(len(output))

main()
