
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
            storage_decay_rate: float,
            partitioning_resolution: float) -> list[LogitPack]:

    if iteration_sz == 0:
        return logit_list 

    list_sz: int = len(logit_list)

    if list_sz not in [2, 4, 16, 256, 65536]:
        raise Exception()

    if list_sz == 2:
        delta_pack_0_0: LogitPack   = pack_twosum(logit_list[0], logit_list[1], projection_storage_sz)
        delta_pack_0_1: LogitPack   = lowres_pack_twosum(logit_list[0], logit_list[1], projection_storage_sz, partitioning_resolution) #low resolution is for repartitioning the projection space because the "patterns" are of lower resolution than the actual number projections (which are taken cared of by the previous operation)
                                                                                                                                       #https://leetcode.com/problems/word-pattern/description/
                                                                                                                                       #I have considered very thoroughly about how these could be written, from col row calibration operation to base case projection to etc.
                                                                                                                                       #it all comes down to the base case projection, if you are more comfortable with 32 dimensional projections, then it should be the base case's problem
                                                                                                                                       #if we look at the multidimensional projections, we'd see a one -> many kind of projection, for each x[-1], there are derivative order base projections, which would help with the projection diffraction

                                                                                                                                       #that's precisely what we'd want to do in this case, because 3 dimensional + 3 dimensional = 6 dimensional projection is already a lot, we'd want to attempt to solve the problem of diffracting context by 
                                                                                                                                       #improve base case
                                                                                                                                            #improve the number of features in the base case, like in how we built the Taylor Projection multidimensional complete radix tree, we'd want to do one -> many, one x[-1] for many x[:-1] projections
                                                                                                                                            #essentially its a permutation operation, <pack_0_0, pack_0_1, pack_0_2>, <pack_1_0, pack_1_1, pack_1_2> -> <pack_1_0 * pack_0_1 + pack_1_1 * pack_0_1 + pack_1_2 * pack_0_1, etc.>
                                                                                                                                            #we'd want to do that to offset the diffracting responsibility, we'd want to have a lot of pointers because 6 dimensional projection is probably as far as we could do

                                                                                                                                            #improve the low resolution of the base case, essentially, we'd want to get the shapes right (our coefficient pointer is in a different space that bijectively relates to that of a higher discretization size of Taylor's coefficients, 0.1 bit per coefficient for example) to radix the projections into appropriate buckets, not the numerical values, so the low resolution would be a lossy compressor in the sense

                                                                                                                                       #improve suffix array diffraction of not base case
                                                                                                                                       #we are missing one thing
                                                                                                                                       #its a radix tree of LogitPack, I was trying to distinct the dimensional differences, what's the difference between 4 dimensional projection vs 2logit + 2logit dimensional projection ?
                                                                                                                                       
                                                                                                                                       #remember that we'd want to do projection within the [LogitPack] territory, as for all other operation, it is a calibration operation, logically speaking
                                                                                                                                       #as if [LogitPack] is a person, carrying information from A -> B, we'd want to talk about duplication of context later, essentially, we'd map a char -> an embedding of [[LogitPack, ...], ...]
                                                                                                                                       #so those two are two very different techniques of approximation tuning, or offseting the cost of inaccurate projection of the base cases

                                                                                                                                       #its hard, but the music is kind of starting from these lines 
                                                                                                                                       #I've seen videos about inaccurate movie projection or photo frame projections, essentially, we'd want to project a probably sliding window of what's gonna happen next, rather than an immediate next word, because it'd help with the stability

                                                                                                                                       #so if we are looking at the hierarchy, we are seeing the base of everything is a multidimensional Taylor Series projection (low + high resolution), low resolution to sort, high resolution to move the numerical values
                                                                                                                                       #because a projection of 16 dimensions is expensive, how about we pack the 16 dimensions into 4 packs of 4, and we'd keep the projection rule, our new rule is built on top of the shake_x logic

                                                                                                                                       #I dont know what the other approaches gonna look like, but the parallel dispatch version that leverages cache line + locality of accesses gonna look like this

        delta_pack_1_0: LogitPack   = pack_twosum(logit_list[1], logit_list[0], projection_storage_sz)
        delta_pack_1_1: LogitPack   = lowres_pack_twosum(logit_list[1], logit_list[0], projection_storage_sz, partitioning_resolution)

        new_logit_0: LogitPack      = sum_accum(logit_list[0], delta_pack_0_0, delta_pack_0_1)
        new_logit_1: LogitPack      = sum_accum(logit_list[1], delta_pack_1_0, delta_pack_1_1)
        rs: list[LogitPack]         = [new_logit_0, new_logit_1]

        return shake_x(rs,
                       projection_storage_sz * storage_decay_rate,
                       initial_iteration_sz,
                       iteration_sz - 1,
                       storage_decay_rate,
                       partitioning_resolution)

    #assume base case is accurate, assume not base case is not accurate
    #so the base case is not accurate, because we assumed that base case is sufficient to approx everything
    #problem is the base case is accurate if the matrix size is under a certain size
    #this is where we need an arbitrary bijective map, a shadow matrix to rotate and operate on to have two + or more + (two row-column intersection) to offset the logit density of the entire matrix being on the cross
    #so we have 1 real matrix and multiple shadow matrices to build flight paths between the logits

    #we were working on the maxflow + betweenness centrality problem
    #imagine that for a normal matrix, we do mlp + rotate + mlp, we offload the responsibility of the entire matrix information transferring to the cross, essentially, the data of the entire matrix has to flow to the cross to flow to a random cell, in other words, every cell is linked with | to a cross

    #how about we build a virtual matrix that is bijectively mapped to the original matrix (by using suffix array), and we rinse and repeat the process, so essentially, we do mlp + mlp + rotate + mlp + mlp for the first mlp is on the original matrix and the second mlp is on the virtual matrix 

    #now the random cell has two cross paths, so the burden of bearing the matrix is on the two cross paths not one, we have successfully decrease the "requirement" for accurate base case projection, I dont have the actual number for this

    #I guess the real problem statement is:
    #if the logit flow is uniformly distributed, we'd just fatten the leaf nodes, because it'd be equivalent to having multiple crosses in the sense of offloading the logit transfer responsibility

    #but the problem we are facing is the skin-tone problem, where a specific node needs more information from a certain set of nodes more than another
    #our thesis would be if the flow is not uniformly distributed, far from the deviation, it must be the partitioning of the matrix is not accurate at the point (too many important information is at the same location, can't summarize), so a virtual reorder of the matrix would bring that back to uniform distribution where we'd want to fatten the leafs to further solve the problem  
    #note that this is already the responsibility of the matrix when we do mlp, we just dont know how bad the original matrix really is in the sense of skewness, or what storage it takes for the matrix to do random matrix redistribution, we dont know 

    #it's insanely hard to solve the problem

    dim_sz: int                                                 = sqrt(list_sz)    
    two_dimensional_logit_list: list[list[LogitPack]]           = shape_as(logit_list, [dim_sz, dim_sz])

    rotated_two_dimensional_logit_list: list[list[LogitPack]]   = rotate(two_dimensional_logit_list)
    transformed_logit_list: list[list[LogitPack]]               = []
    transformed_logit_list_2: list[list[LogitPack]]             = []
    transformed_logit_list_3: list[list[LogitPack]]             = []

    for i in range(dim_sz):
        shaked_row: list[LogitPack]     = shake_x(two_dimensional_logit_list[i],
                                                  projection_storage_sz,
                                                  initial_iteration_sz,
                                                  initial_iteration_sz,
                                                  storage_decay_rate,
                                                  partitioning_resolution)

        transformed_logit_list          += [shaked_row]

        shaked_row_2: list[LogitPack]   = shake_x(two_dimensional_logit_list[i],
                                                  projection_storage_sz,
                                                  initial_iteration_sz,
                                                  initial_iteration_sz,
                                                  storage_decay_rate,
                                                  partitioning_resolution)

        transformed_logit_list_2        += [shaked_row_2]

        shaked_row_3: list[LogitPack]   = shake_x(rotated_two_dimensional_logit_list[i],
                                                  projection_storage_sz,
                                                  initial_iteration_sz,
                                                  initial_iteration_sz,
                                                  storage_decay_rate,
                                                  partitioning_resolution)

        transformed_logit_list_3        += [shaked_row_3]

    transformed_logit_list          = rotate(transformed_logit_list)
    transformed_logit_list_3        = rotate(transformed_logit_list_3)
    rs_list: list[list[LogitPack]]  = []

    for i in range(dim_sz):
        org_list: list[LogitPack]           = two_dimensional_logit_list[i]
        ctx_list: list[LogitPack]           = transformed_logit_list[i]
        shaked_ctx_list: list[LogitPack]    = shake_x(ctx_list,
                                                      projection_storage_sz,
                                                      initial_iteration_sz,
                                                      initial_iteration_sz,
                                                      storage_decay_rate,
                                                      partitioning_resolution)

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

    return shake_x(flatten(rotate(rs_list)),
                   projection_storage_sz * storage_decay_rate,
                   initial_iteration_sz,
                   iteration_sz - 1,
                   storage_decay_rate,
                   partitioning_resolution)

def main():

    inp: list[LogitPack]    = [LogitPack([get_leaf(1), get_leaf(2), get_leaf(3)]) for _ in range(16)]
    output: list[LogitPack] = shake_x(inp, 8, 1, 1, 1)
    print(len(output))

main()
