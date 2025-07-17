
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

def brain_twosum(lhs: Brain, rhs: Brain) -> Brain:

    lhs_base: list[LogitPack]   = lhs.as_list()
    rhs_base: list[LogitPack]   = rhs.as_list()
    rs_base:  list[LogitPack]   = []

    for i in range(len(lhs_base)):
        brain_cell: LogitPack = make_empty_logit_pack()

        for j in range(len(rhs_base)):
            brain_cell = sum_accum(brain_cell, pack_twosum(lhs_base[i], rhs_base[j]))

        rs_base += [brain_cell]

    return Brain(rs_base)

def brain_accum_2(lhs: Brain, rhs: Brain) -> Brain:

    if lhs.size() != rhs.size():
        raise Exception()

    return Brain(list(map(sum_accum, zip(lhs.as_list(), rhs.as_list())))) 

def brain_accum(*args) -> Brain:

    return functools.reduce(brain_accum_2, args[1:], args[0]) 

def radian_to_euclidean_coordinate(feature_vec: list[float], r: float) -> list[float]:

    if len(feature_vec) == 0:
        raise Exception()

    if len(feature_vec) == 1:
        return [r * math.cos(feature_vec[0]), r * math.sin(feature_vec[1])] 

    x_projection_value: float   = math.cos(feature_vec[0]) * r 
    y_projection_value: float   = math.sin(feature_vec[0]) * r

    return [x_projection_value] + radian_to_euclidean_coordinate(feature_vec[1:], y_projection_value)

class CalibrationLogit:

    def __init__(self,
                 logit_list: list[Logit],
                 projection_storage_sz: int,
                 calibration_radius: float):

        if projection_storage_sz < 2:
            raise Exception() 

        self.logit_list             = list(logit_list)
        self.calibration_logit_list = [float(0) for i in range(projection_storage_sz - 1)]
        self.projection_storage_sz  = projection_storage_sz
        self.calibration_radius     = calibration_radius

    def get_value(self) -> float:

        projection_storage: list[float] = radian_to_euclidean_coordinate(self.calibration_logit_list, self.calibration_radius)
        projection_value: float         = multidimensional_projection(self.logit_list, projection_storage)

        return projection_value

    def get_projection_storage_vec(self) -> list[float]:

        return copy.deepcopy(self.calibration_logit_list)
    
    def set_projection_storage_vec(self, calibration_logit_list: list[float]):

        if (calibration_logit_list != None and len(calibration_logit_list) != len(self.calibration_logit_list)):
            raise Exception()

        self.calibration_logit_list = copy.deepcopy(calibration_logit_list)

def calibrate_brain_cell(brain_cell: LogitPack,
                         calibration_projection_storage_per_logit_sz: int,
                         calibration_radius: float) -> LogitPack:

    logit_list: list[Logit]     = brain_cell.as_list() 

    if len(logit_list) == 0:
        raise Exception()

    rs_logit_list: list[Logit]  = []

    for i in range(logit_list):
        calibration_logit: Logit    = CalibrationLogit(logit_list, calibration_projection_storage_per_logit_sz, calibration_radius)
        new_logit: Logit            = sum_logit(logit_list[i], calibration_logit)
        rs_logit_list               += [new_logit]

    new_brain_cell: LogitPack   = LogitPack(rs_logit_list)

    return new_brain_cell

def calibrate_brain(arg: Brain,
                    projection_storage_sz: int,
                    calibration_radius: float) -> Brain:

    #even though the storage pointer for the sphere is large, essentially taking as much space as the other projections
    #yet the differential space is small, so we'd want to store the euclidean semantic on the logit, and convert them into euclidean and then Taylor Coefficients and we'd roll
    # return arg 
    #the problem is that we couple the radian responsibility into the projection, so each Logit cannot carry their responsibility ...

    brain_cell_list: list[LogitPack]        = arg.as_list()
    new_brain_cell_list: list[LogitPack]    = []

    for brain_cell in brain_cell_list:
        new_brain_cell_list += [calibrate_brain_cell(brain_cell, projection_storage_sz, calibration_radius)]

    return Brain(new_brain_cell_list) 

def pairwise_brain_accum_2(lhs: list[Brain], rhs: list[Brain]) -> list[Brain]:

    return list(map(brain_accum, zip(lhs, rhs))) 

def pairwise_brain_accum(*args) -> list[Brain]:

    return functools.reduce(pairwise_brain_accum_2, args[1:], args[0])

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

    #I've spent a lot | too many hours to think about the equation to the point that I dont think that this could be further improved!
    #there are still the problems at the base case, but as for the general picture, we have done it correctly, this is sufficient to approx a really wide range of semantic, we'd have to implement a very efficient search + mining operation for that

    #the only major improvement we've made to the transformer is to not carry the output semantic around, essentially that's what happen in the transformer case when you want to do pigeonhole sort
    #we have to push the bars into their appropriate levels in order to do concurrent firing, from x -> y, this is the reason we need softmax, sometimes the y semantic is just not right enough to continue the firing + sort

    #we just changed the idea slightly from transforming from x -> y to radix accumulation from x -> y, for every layer of transformation, we are at the next radix sort to do linear projection
    #says that we have 1|2|3|4, 2|1|3|4

    #1 goes to the offset 10
    #2 goes to the offset 20

    #so we'd take care of that, continue to the 2|3|4 and the 1|3|4
    #2 goes to the offset 2, 1 goes to the offset 1
    #alright, what's the problem?
    #the problem is that this probably only works for uniform distribution of rules, as we could tell, the sorting space for 2 is different than that of 1, how about we alter the semantic beforehand to denote that right at the previous layer? this is where our transformation kicks in

    #as I become increasingly paranoid about what I could do for projection, I have come to a conclusion that a matrix multiplication for base case is the general idea (essentially we are doing dot products for row x column pair, except we are doing advanced dot product by using new operation of 6 dimensional projection -> 3 dimensional projection)
    #let's think of the matrix this way

    #our only clue to machine learning is projection

    #too much projection like 1024 dimensions -> 1 dimension or 3 dimensions would be bad, because it would break the projection space (in the sense of semantic), plus, the projecting context does not require that knowledge of being in the 1024th dimensions
    #patch: we are doing gossip + brain architect to do knowledge transfer to the brains and let the brains talk to each other, so we could improve the numerical + semantic stability of projection
    #immediate patch to naive projection: 6 dimensional projections -> 3 dimensional vector. Problem: how many accumulating features, essentially a multidimensional projection does not prevent us from doing sum accumulation (this is still counting as projection, yet this is still in the matrix multiplication territory)
    #problem, the matrix is only stable (good to do advanced matrix multiplication) if it is under a certain size, says 3x3 or 9x9
    #patch: we only do matrix multiplication for the base case
    #so the immediate doable for our current solution would be transforming a simple combinatorial approach to a 2d matrix multiplication row x col approach (all rows combines all cols dot product for dot product being the pack_twosum + sum_accum)

    #tomorrow we'd be talking about the radian Taylor Coefficents' coordinate to do shape projections, we'd want to be in radian coordinate because, our unit vector is of size 1, so essentially we'd kind of iterating the surface of the sphere to clue our shapes 
    #we can't skew the shape in the cos-sin projection space because it's not differentiable, whereas it is still differentiable in the space (even though it is kind of "spherely twisted," we dont really care, we just want to project the shape and the shape is differentiable, period) 

    #today we are going to cover the topic of calibration
    #what is calibration?
    #calibration is a reprojection of a context unit
    #whether that is 1 Logit, 1 LogitPack, 1 Brain, or 1 Matrix

    #point is calibration only works as an absolute unit of context, whose to be "intact" and to be "transferred"
    #I think about how calibration could work for a brain calibration, not a brain cell calibration

    #essentially it is a matrix "multiplication" operation for self x self, where the rule of the "multiplication" is now on the twisted sphere Taylor Projection space
    #note that the calibration process also needs to be shaked, in the sense of x = x + f(x) and arg = x + f(x), continue ...

    #recall that we are trying to do stable projection, stable projection only happens for 6 dimensional projection or 4 dimensional projection or (3 + 3 dimension space or 2 + 2 dimension space), with appropriate number of elements in the left matrix and the right matrix, I dont exactly know the number of brain cells
    #as long as we are hitting the number, we'll be fine

    #calibration is very very important, because it would move the domain space into appropriate buckets to do concurrent firing via multidimensional Taylor Projection
    #without calibration, we are probably clueless, we are too straight in a twisted space

    if iteration_sz == 0:
        return logit_list

    list_sz: int = len(logit_list)

    if list_sz not in [3, 9, 81]:
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

        return pairwise_brain_accum(rs_arg_1, rs_arg_2)

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
            shaked_row: list[Brain]     = shake_x(shaped_virtual_logit_list[i],
                                                  get_initial_virtual_suffix_array(len(virtual_suffix_array), [cube_root(dim_sz), cube_root(dim_sz)])
                                                  projection_storage_sz,
                                                  initial_iteration_sz,
                                                  initial_iteration_sz,
                                                  storage_decay_rate,
                                                  partitioning_resolution)

            shaked_row_2: list[Brain]   = shake_x(shaped_virtual_logit_list[i],
                                                  get_initial_virtual_suffix_array(len(virtual_suffix_array), [cube_root(dim_sz), cube_root(dim_sz)]),
                                                  projection_storage_sz,
                                                  initial_iteration_sz,
                                                  initial_iteration_sz,
                                                  storage_decay_rate,
                                                  partitioning_resolution)

            transformed_logit_list      += [shaked_row]
            transformed_logit_list_2    += [shaked_row_2]

        virtual_transformed_logit_list      += [backward_map_suffix_array(shape_as(transformed_logit_list, [list_sz]), suffix_array)]
        virtual_transformed_logit_list_2    += [backward_map_suffix_array(shape_as(transformed_logit_list_2, [list_sz]), suffix_array)]

    virtual_transformed_logit_list      += [logit_list]
    rs_1: list[Brain]                   = pairwise_brain_accum(*virtual_transformed_logit_list)
    virtual_transformed_logit_list_2    += [logit_list]
    rs_2: list[Brain]                   = list(map(calibrate_brain, pairwise_brain_accum(*virtual_transformed_logit_list_2)))

    nxt_ctx_list: list[Brain]           = shake_x(rs_2,
                                                  list(map(flatten, map(rotate, map(shape_as_func(space_sz), virtual_suffix_array)))),
                                                  projection_storage_sz * storage_decay_rate,
                                                  initial_iteration_sz,
                                                  iteration_sz - 1,
                                                  storage_decay_rate,
                                                  partitioning_resolution)

    return pairwise_brain_accum(rs_1, nxt_ctx_list)

def main():

    pass

main()

