
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

def shake_x(logit_list: list[Brain],
            projection_storage_sz: int,
            initial_iteration_sz: int,
            iteration_sz: int,
            storage_decay_rate: float,
            partitioning_resolution: float) -> list[Brain]:

    if iteration_sz == 0:
        return logit_list # 

    list_sz: int = len(logit_list)

    if list_sz not in [2, 4, 16, 256, 65536]:
        raise Exception()

    #we are still having the Brain + Brain -> another Brain problem, we'll solve this by adding more feature, see that we are doing matrix multiplication on a 3 dimensional input x 3 dimensional input, this is very important
    #yet we have all of the problems solved: 

    #(1) the problem of reprojection x = x + f(x), arg = x + f(x), these are two different values
    #(2) the problem of virtual matrix
    #(3) the problem of 3 dimensional matrix multiplication
    #(4) the problem of calibration (cosine unit vector in the Taylor Projection space that focuses more on the sin cos projection space, kind of curvy instead of uniform distribution of Taylor's coefficients)
    #(5) the problem of storage_sz + partition resolution

    #we still can have 2D matrix of brain as base cases, we'd proceed to do normal matrix multiplication, this is getting very super confusing, essentially a Brain is a 2d array of LogitPack, we'd do row x col normal matrix multiplication, except for we are doing a Taylor Series projection, rotate rinse and repeat
    #I guess the question is how to make this runs as fast as possible, because we are getting very deep inside the logic and actually spins too many times

    #I've run the numbers, the problem is still at the base case, because there is literally nothing we could do for the other cases

    if list_sz == 2:
        lhs: list[LogitPack]    = []
        rhs: list[LogitPack]    = []

        for i in range(logit_list[0].size()):
            new_logit_pack: LogitPack = make_zero_logit_pack() 

            for j in range(logit_list[1].size()):
                new_logit_pack  = sum_accum(new_logit_pack, pack_twosum(logit_list[0].get(i), logit_list[1].get(j), projection_storage_sz))

            new_logit_pack  = sum_accum(logit_list[0].get(i), new_logit_pack)
            lhs             += [new_logit_pack]

        for i in range(logit_list[1].size()):
            new_logit_pack: LogitPack = make_zero_logit_pack()

            for j in range(len(logit_list[0].size())):
                new_logit_pack  = sum_accum(new_logit_pack, pack_twosum(logit_list[1].get(i), logit_list[0].get(j), projection_storage_sz))

            new_logit_pack  = sum_accum(logit_list[1].get(i), new_logit_pack)
            rhs             += [new_logit_pack]

        rs_arg_1: list[Brain]       = [Brain(lhs), Brain(rhs)]
        arg_lhs: list[LogitPack]    = []
        arg_rhs: list[LogitPack]    = []

        for i in range(logit_list[0].size()):
            new_logit_pack: LogitPack = make_zero_logit_pack() 

            for j in range(logit_list[1].size()):
                new_logit_pack  = sum_accum(new_logit_pack, pack_twosum(logit_list[0].get(i), logit_list[1].get(j), projection_storage_sz))

            new_logit_pack  = sum_accum(logit_list[0].get(i), new_logit_pack)
            arg_lhs         += [new_logit_pack]

        for i in range(logit_list[1].size()):
            new_logit_pack: LogitPack = make_zero_logit_pack()

            for j in range(len(logit_list[0].size())):
                new_logit_pack  = sum_accum(new_logit_pack, pack_twosum(logit_list[1].get(i), logit_list[0].get(j), projection_storage_sz))

            new_logit_pack  = sum_accum(logit_list[1].get(i), new_logit_pack)
            arg_rhs         += [new_logit_pack]

        next_arg: list[Brain]   = [calibrate_brain(Brain(lhs), partitioning_resolution), calibrate_brain(Brain(rhs), partitioning_resolution)]
        #to ease some of the problems, we'd have to assume that lhs rhs is already sorted in the sense of radix | other_semantic_space, we'd have to do another arg = etc.

        rs_arg_2: list[Brain]   = shake_x(next_arg,
                                          projection_storage_sz * storage_decay_rate,
                                          initial_iteration_sz,
                                          iteration_sz - 1,
                                          storage_decay_rate,
                                          partitioning_resolution)

        #this is the most important line in the history of projection radix sort, thank you John

        return list(map(brain_accum, zip(rs_arg_1, rs_arg_2)))

    #today we'll be focusing on the three main improvements:
    #one -> many projection, projection permutation and sum_accum for base case
    #virtual matrix paths, and simutanous sum accumulation, essentially we'd be working on multiple crosses to offload the responsibility of all matrix knowledge being on the cross
    #f(x) unscaling, essentially f(x) / n is the shape of the projection, if we look closely at the coefficients, their cosine vector is important in the case of shape projection, so that is what we'd do, we would be in the cosine space to map to the taylor projection space of derivative order of 100
    #alright, so essentially we'd be working on the coefficient vector of unit size of 1 instead of all the unit sizes 

    #assume that we have a pointer of 256 or one char, we'd want to discretize the unit space to 256 cubes in a multidimensional sphere
    #assume that we have a pointer of 65536 or two char, we'd want to discretize the unit space to 65536 cubes in a multidimensional sphere
    #the resolution of the shape would be propotional to the storage pointer (which is uint8_t, uint16_t, uint32_t etc.)
    #the only advantage edge we have when projecting shape is the sphere space of 1 around the origin instead of all over the place

    #it's insanely complicated how we got to this matrix code
    #I'm not saying that this is the best code but this is the logically correct code in terms of what we could do to stablize the matrix

    #despite my 2 days of non-stop thinking, I couldn't formulate an equation for exact # of rotations to keep the matrix from falling apart
    #I have used middle theorem to squeeze the possibility, assume this algorithmic approach of transferring all matrix information on the row to project exactly one cell in the matrix
    #this is precisely the problem (the base has to be as most strong as the matrix to hold the matrix (this is the skyscraper on stilt problem), this equals to the pair having to hold the matrix weight), we can't prove that by using other approaches
    #we'd allow leeways for the matrix having found a better way to lessen the weights, this is where I can't use middle theorem, what, where precisely is the equation ??? I have no idea

    #we'd want to do 3, 9, 27, 81 as clued by Mom

    #what I know for sure is that a 3 virtual matrices is better than a row + col + row_col transformation
    #so essentially, our original matrix is unchanged, we just rotate the arbitrary matrices that contain the indices of the original matrix, if that makes sense
    #what I also know is that we need 2x information (> 1x information) to transform the matrix (information-technology-wise speaking) in the row-col fashion without being afraid of misprojections, 1 is the original, 1 is the tmp, that's on the brain to carry the responsibility (what's why we have multiple brain cells, which is a set of LogitPacks)
    #the problem could not be better described https://leetcode.com/problems/game-of-life/description/

    #the only problem we have not been able to solve is the problem of radix sort
    #as we could see, we are trying to move the semantics into their appropriate buckets to do neuron firings, yet we could not recursively sort them into appropriate buckets but rather a pigeonhole sort of O(n) size, note that we are learning the map, which is not a good approach
    #if we could somehow learn it, the sorting way (or the sorting algorithm), and move our focal onto the sorted buckets to recursively project, that'd be ideal
    #believe it or not, we are missing a modulo operation

    #if you look at the radix sort, we are just radixing them into appropriate space, unscale the projection to see what's going on with the other radices
    #if we are radixing from right to left, we radix it into 256 buckets, then we'd want to unscale the projection, by essentially zooming into the remaining space
    #essentially a division followed by a multiplication to get the remaining projection, this division is an integer division
    #this is the sole implementation that is very important, otherwise we are just doing pigeonhole sort ...

    #imagine this sequence of 1|2|3|4

    #those that are in the 4th bucket are radixed into the same bucket of continuity compression
    #those that are in the 3rd bucket of 1|2|3 are radixed in the same bucket of continuity compression

    #problem is that in the sense of continuity compression, we'd want to do prefix followed by a modulo to get the lower bits, partition + 1|2|3|4 + projection + modulo -> partition + 2|3|4|0 + projection + modulo -> partition + 3|4|0|0 + projection + modulo -> partition 4|0|0|0 + projection
    #what I haven't been able to prove is this hasn't already done that, I guess my doubt really is the ability of the network to do bitshift and reprojection
    #until we have found a differential way to do this, I dont think modulo and bitshift is the way to do

    #the sorting mechanism of the network is very vague and hard to actually grasp, but keep the definition of this
    #essentially we have a semantic value, we would want to do: radix| other_semantic_value -> linear projection -> erase radix -> other_semantic_value (this is where we call recursion) -> ... -> rinse and repeat
    #we'd have to do the exact steps to get the engine to roll
    #what we are missing is the distinction of x = x + f(x) and the arg = x + f(x) or the recursive argument if that makes sense
    #because one is to do context projection (based on the radix), the other is to erase the radix into another semantic value, to continue the projection

    #so we'd turn a pigeonhole sort into a radix sort, whose policy is to sort the dimensions in the sense of relevancy and we'd want to do a modulo projection, this is my single biggest regret because I'd not be able to see continuity in my network
    #the other hard part is to reorder the original projection space based on the relevancy rule, we'll see about the approach
    #problem is that we'd do partition wrong (in the sense of keeping the trailing semantic) so we'd want to re radix the fellows, this requires padding bits to further continue the radix sort, we'd talk about this later, essentially, we'd want to dup or triple the byte size of the sorting data to further the radix sort 

    #because the flow path of the matrix is better that way

    dim_sz: int                                     = sqrt(list_sz)    
    space_sz: list[int]                             = [dim_sz, dim_sz]
    suffix_arr_1: list[int]                         = get_suffix_array(iteration_sz, space_sz, 0)
    suffix_arr_2: list[int]                         = get_suffix_array(iteration_sz, space_sz, 1)
    suffix_arr_3: list[int]                         = get_suffix_array(iteration_sz, space_sz, 2)

    virtual_logit_list_1: list[Brain]               = forward_map_suffix_array(logit_list, suffix_arr_1)
    virtual_logit_list_2: list[Brain]               = forward_map_suffix_array(logit_list, suffix_arr_2)
    virtual_logit_list_3: list[Brain]               = forward_map_suffix_array(logit_list, suffix_arr_3)

    shaped_virtual_logit_list_1: list[list[Brain]]  = shape_as(virtual_logit_list_1, space_sz)
    shaped_virtual_logit_list_2: list[list[Brain]]  = shape_as(virtual_logit_list_2, space_sz)
    shaped_virtual_logit_list_3: list[list[Brain]]  = shape_as(virtual_logit_list_3, space_sz)

    transformed_logit_list_1: list[list[Brain]]     = []
    transformed_logit_list_2: list[list[Brain]]     = []
    transformed_logit_list_3: list[list[Brain]]     = []

    #we'd want to do the virtual paths, later

    for i in range(dim_sz):
        shaked_row: list[Brain]     = shake_x(shaped_virtual_logit_list_1[i],
                                              projection_storage_sz,
                                              initial_iteration_sz,
                                              initial_iteration_sz,
                                              storage_decay_rate,
                                              partitioning_resolution)

        transformed_logit_list_1    += [shaked_row]

        shaked_row_2: list[Brain]   = shake_x(shaped_virtual_logit_list_2[i],
                                              projection_storage_sz,
                                              initial_iteration_sz,
                                              initial_iteration_sz,
                                              storage_decay_rate,
                                              partitioning_resolution)

        transformed_logit_list_2    += [shaked_row_2]

        shaked_row_3: list[Brain]   = shake_x(shaped_virtual_logit_list_3[i],
                                              projection_storage_sz,
                                              initial_iteration_sz,
                                              initial_iteration_sz,
                                              storage_decay_rate,
                                              partitioning_resolution)

        transformed_logit_list_3    += [shaked_row_3]

    post_shaped_virtual_logit_list_1: list[Brain]   = backward_map_suffix_array(flatten(transformed_logit_list_1), suffix_arr_1)
    post_shaped_virtual_logit_list_2: list[Brain]   = backward_map_suffix_array(flatten(transformed_logit_list_2), suffix_arr_2)
    post_shaped_virtual_logit_list_3: list[Brain]   = backward_map_suffix_array(flatten(transformed_logit_list_3), suffix_arr_3)
    rs_list: list[Brain]                            = []

    for i in range(list_sz):
        new_brain: Brain    = brain_accum(logit_list[i], post_shaped_virtual_logit_list_1[i], post_shaped_virtual_logit_list_2[i], post_shaped_virtual_logit_list_3[i])
        rs_list             += [new_brain]

    _virtual_logit_list_1: list[Brain]               = forward_map_suffix_array(logit_list, suffix_arr_1)
    _virtual_logit_list_2: list[Brain]               = forward_map_suffix_array(logit_list, suffix_arr_2)
    _virtual_logit_list_3: list[Brain]               = forward_map_suffix_array(logit_list, suffix_arr_3)

    _shaped_virtual_logit_list_1: list[list[Brain]]  = shape_as(_virtual_logit_list_1, space_sz)
    _shaped_virtual_logit_list_2: list[list[Brain]]  = shape_as(_virtual_logit_list_2, space_sz)
    _shaped_virtual_logit_list_3: list[list[Brain]]  = shape_as(_virtual_logit_list_3, space_sz)

    _transformed_logit_list_1: list[list[Brain]]     = []
    _transformed_logit_list_2: list[list[Brain]]     = []
    _transformed_logit_list_3: list[list[Brain]]     = []

    #we'd want to do the virtual paths, later

    for i in range(dim_sz):
        shaked_row: list[Brain]     = shake_x(_shaped_virtual_logit_list_1[i],
                                              projection_storage_sz,
                                              initial_iteration_sz,
                                              initial_iteration_sz,
                                              storage_decay_rate,
                                              partitioning_resolution)

        _transformed_logit_list_1   += [shaked_row]

        shaked_row_2: list[Brain]   = shake_x(_shaped_virtual_logit_list_2[i],
                                              projection_storage_sz,
                                              initial_iteration_sz,
                                              initial_iteration_sz,
                                              storage_decay_rate,
                                              partitioning_resolution)

        _transformed_logit_list_2   += [shaked_row_2]

        shaked_row_3: list[Brain]   = shake_x(_shaped_virtual_logit_list_3[i],
                                              projection_storage_sz,
                                              initial_iteration_sz,
                                              initial_iteration_sz,
                                              storage_decay_rate,
                                              partitioning_resolution)

        _transformed_logit_list_3   += [shaked_row_3]

    _post_shaped_virtual_logit_list_1: list[Brain]   = backward_map_suffix_array(flatten(_transformed_logit_list_1), suffix_arr_1)
    _post_shaped_virtual_logit_list_2: list[Brain]   = backward_map_suffix_array(flatten(_transformed_logit_list_2), suffix_arr_2)
    _post_shaped_virtual_logit_list_3: list[Brain]   = backward_map_suffix_array(flatten(_transformed_logit_list_3), suffix_arr_3)
    _rs_list: list[Brain]                            = []

    for i in range(list_sz):
        new_brain: Brain    = brain_accum(logit_list[i], _post_shaped_virtual_logit_list_1[i], _post_shaped_virtual_logit_list_2[i], _post_shaped_virtual_logit_list_3[i])
        _rs_list            += [new_brain]

    nxt_ctx_list: list[Brain] = shake_x(_rs_list,
                                        projection_storage_sz * storage_decay_rate,
                                        initial_iteration_sz,
                                        iteration_sz - 1,
                                        storage_decay_rate,
                                        partitioning_resolution)

    return list(map(brain_accum, zip(rs_list, nxt_ctx_list)))

def main():

    pass

main()

