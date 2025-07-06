
#

def get_projection_storage_size(in_feature_sz: int, derivative_order_sz: int) -> int:

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

    rs: float = 0

    for i in range(len(coeff_arr)):
        factorial_multiplier: float = float(1) / math.factorial(i)
        x_multiplier: float         = math.pow(x, i)
        coeff_multiplier: float     = coeff_arr[i]
        rs                          += factorial_multiplier * x_multiplier * coeff_multiplier

    return rs

def multidimensional_taylor_project(x: list[float], derivative_order_sz: int, storage_vec: list[float]) -> float:

    if derivative_order_sz == 0:
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

    def get_value() -> float:

        if self.leaf_value != None:
            return self.leaf_value
        elif self.descendant_vec != None:
            return multidimensional_projection([descendant.get_value() for descendant in self.descendant_vec], self.projection_storage_vec)
        else:
            raise RuntimeException()

    def get_projection_storage_vec() -> list[float]:

        return copy.deepcopy(self.projection_storage_vec)

    def set_projection_storage_vec(projection_storage_vec: list[float]):

        if (projection_storage_vec != None and len(projection_storage_vec) != len(self.projection_storage_vec)):
            raise RuntimeException()

        self.projection_storage_vec = copy.deepcopy(projection_storage_vec)

def get_leaf(val: float) -> Logit:

    return Logit(None, None, val) 

def one_sum(lhs: Logit, projection_storage_sz: int) -> Logit:

    actual_projection_storage_sz: int   = get_projection_storage_sz(1, projection_storage_sz)

    if actual_projection_storage_sz == 0:
        raise RuntimeException()

    feature_vec: list[float]            = make_projection_storage(actual_projection_storage_sz)

    return Logit(feature_vec, [lhs], None)

def two_sum(lhs: Logit, rhs: Logit, projection_storage_sz: int) -> Logit:

    actual_projection_storage_sz: int   = get_projection_storage_sz(2, projection_storage_sz)

    if actual_projection_storage_sz == 0:
        raise RuntimeException()

    feature_vec: list[float]            = make_projection_storage(actual_projection_storage_sz)

    return Logit(feature_vec, [lhs, rhs], None) 

def three_sum(x: Logit, x1: Logit, x2: Logit, projection_storage_sz: int) -> Logit:

    actual_projection_storage_sz: int   = get_projection_storage_sz(3, projection_storage_sz)

    if actual_projection_storage_sz == 0:
        raise RuntimeException()

    feature_vec: list[float]            = make_projection_storage(actual_projection_storage_sz)

    return Logit(feature_vec, [x, x1, x2], None)

def four_sum(x: Logit, x1: Logit, x2: Logit, x3: Logit, projection_storage_sz: int) -> Logit:

    actual_projection_storage_sz: int   = floor_projection_storage_sz(4, projection_storage_sz)

    if actual_projection_storage_sz == 0:
        raise RuntimeException()

    feature_vec: list[float]            = make_projection_storage(actual_projection_storage_sz)

    return Logit(feature_vec, [x, x1, x2, x3], None)

def generic_sum(x_list: list[Logit], projection_storage_sz: int) -> Logit:

    if len(x_list) == 1:
        return one_sum(x_list[0], projection_storage_sz)
    elif len(x_list) == 2:
        return two_sum(x_list[0], x_list[1], projection_storage_sz)
    elif len(x_list) == 3:
        return three_sum(x_list[0], x_list[1], x_list[2], projection_storage_sz)
    elif len(x_list) == 4:
        return four_sum(x_list[0], x_list[1], x_list[2], x_list[3], projection_storage_sz)
    else:
        raise RuntimeException()

def twosum_tree_accumulate(x_list: list[Logit], projection_storage_sz: int) -> Logit:

    if len(x_list) == 0:
        raise Exception()

    if len(x_list) == 1:
        return one_sum(x_list[0], projection_storage_sz) 

    if len(x_list) % 2 != 0:
        raise Exception()

    lhs_list: list[Logit]   = x_list[:len(x_list) / 2] 
    rhs_list: list[Logit]   = x_list[len(x_list) / 2:]

    return two_sum(twosum_tree_accumulate(lhs_list, projection_storage_sz),
                   twosum_tree_accumulate(rhs_list, projection_storage_sz),
                   projection_storage_sz)

def threesum_tree_accumulate(x_list: list[Logit], projection_storage_sz: int) -> Logit:

    if len(x_list) == 0:
        raise Exception()

    if len(x_list) == 1:
        return one_sum(x_list[0], projection_storage_sz)

    if len(x_list) % 3 != 0:
        raise Exception()

    width: int                  = len(x_list) / 3
    first_list: list[Logit]     = x_list[:width]
    second_list: list[Logit]    = x_list[width: width * 2]
    third_list: list[Logit]     = x_list[width * 2:]

    return three_sum(threesum_tree_accumulate(first_list, projection_storage_sz),
                     threesum_tree_accumulate(second_list, projection_storage_sz),
                     threesum_tree_accumulate(third_list, projection_storage_sz),
                     projection_storage_sz) 

#everything before this line is accurate, this is the base of everything, we'd want to further improve the taylor projection by actually changing the projecting range, but we'll be talking about that later
#essentially, we'd want to make things sharp instead of curvy like the Taylor Projection

#because the projection_storage_sz would actually be too unrealistic for a word vec of 1024 words
#the only way we could patch this is by running centrality
#centrality, on one hand, actually builds a very high level vocabulary from low level words 
#like images inputted into our brain from pixel muscle -> brain connection -> word detection -> semanticalization -> etc.
#centrality, on the other hand, solves the problem of projection storage sz

#the problem of centrality is precisely that, we can't assume that the words at the beginning are the absolute words, every transforming phase, the words make sense, so it's important to keep the semantic of the row + col at every given step of the iteration
#otherwise, we'd be forced to alter the semantic of the next matrix in a destructive interference way

#we'd attempt to solve the problem by making sure that every intermediate centrality node is not saturated in the sense of progressively transforming the aggregating cell
#this is the mentioned problem, when I was talking about we were putting too much pressure on the joints 
#yet we have found a solution of perfect square recursion

def two_fold(logit_list: list[Logit], projection_storage_sz: int) -> list[Logit]:

    sz: int             = len(logit_list)

    if sz % 2 != 0:
        raise Exception()

    left: list[Logit]   = logit_list[:(sz / 2)]
    right: list[Logit]  = logit_list[(sz / 2):]

    return [two_sum(lhs, rhs, projection_storage_sz) for (lhs, rhs) in zip(left, right)]

def quad_fold(logit_list: list[Logit], projection_storage_sz: int) -> list[Logit]:

    return two_fold(two_fold(logit_list, projection_storage_sz), projection_storage_sz)

def to_logit_vec(tensor: object) -> list[Logit]:

    if type(tensor) == type([]):
        return functools.reduce(lambda a, b: a + b, [to_logit_vec(sub_tensor) for sub_tensor in tensor], [])

    if type(tensor) == type(Logit()):
        return tensor

    raise Exception()

def flatten_space(space: list[int]) -> list[int]:

    if len(space) == 0:
        raise Exception()

    sz: int = 1

    for i in range(len(space)):
        rs *= space[i]

    return rs; 

def shape_as(tensor: object, space: list[int]) -> object:

    if len(space) == 0:
        raise Exception()

    logit_vec: list[Logit]      = to_logit_vec(tensor)

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

def flatten(tensor: object) -> list[Logit]:

    return to_logit_vec(tensor)

def sqrt(val: int) -> int:
    
    return int(math.sqrt(val))

def cube_root(val: int) -> int:

    return int(val ** (1.f / 3))

def two_shake(logit_list: list[Logit], projection_storage_sz: int, iteration_sz: int) -> list[Logit]:

    list_sz: int = len(logit_list)

    if list_sz not in [1, 2, 4, 16, 256, 65536]:
        raise Exception()

    if list_sz == 1:
        return [one_sum(logit_list[0], projection_storage_sz)]

    if list_sz == 2:
        return [two_sum(logit_list[0], logit_list[1], projection_storage_sz),
                two_sum(logit_list[0], logit_list[1], projection_storage_sz)]

    dim_sz: int                                     = sqrt(list_sz)
    two_dimensional_logit_list: list[list[Logit]]   = shape_as(logit_list, [dim_sz, dim_sz])
    transformed_logit_list_1: list[list[Logit]]     = []
    transformed_logit_list_2: list[list[Logit]]     = []

    for i in range(dim_sz):
        feature_vec_1: list[Logit]      = list()
        feature_vec_2: list[Logit]      = list()
        shaked_row: list[Logit]         = two_shake(two_dimensional_logit_list[i], projection_storage_sz, iteration_sz)

        for j in range(dim_sz):
            first_dimension_representation: Logit   = twosum_tree_accumulate(shaked_row, projection_storage_sz)
            second_dimension_representation: Logit  = twosum_tree_accumulate(shaked_row, projection_storage_sz)

            feature_vec_1.append(first_dimension_representation)
            feature_vec_2.append(second_dimension_representation)

        transformed_logit_list_1.append(feature_vec_1)
        transformed_logit_list_2.append(feature_vec_2)

    rotated_transformed_logit_list_1: list[list[Logit]] = rotate(transformed_logit_list_1)
    rotated_transformed_logit_list_2: list[list[Logit]] = rotate(transformed_logit_list_2)
    rs_list: list[list[Logit]]                          = []

     for i in range(dim_sz):
        lhs: list[Logit]        = two_dimensional_logit_list[i]
        rhs_1: list[Logit]      = two_shake(rotated_transformed_logit_list_1[i], projection_storage_sz, iteration_sz)
        rhs_2: list[Logit]      = two_shake(rotated_transformed_logit_list_2[i], projection_storage_sz, iteration_sz)
        new_row: list[Logit]    = []

        for j in range(dim_sz):
            new_row += [three_sum(lhs[j], rhs_1[j], rhs_2[j], projection_storage_sz)] #we are probably wrong here, I dont know yet, yet the joints for the rotated_transformed_logit_1 and rotated_transformed_logit_2 might loose the global context in the way

        rs_list                 += [new_row]

    return two_shake(flatten(rotate(rs_list)), projection_storage_sz, iteration_sz - 1)

def quad_shake(logit_list: list[Logit], projection_storage_sz: int, iteration_sz: int) -> list[Logit]:

    list_sz: int = len(logit_list)

    if list_sz not in [1, 2, 4, 16, 256, 65536]:
        raise Exception()

    if list_sz == 1:
        return [one_sum(logit_list[0], projection_storage_sz)] 

    if list_sz == 2:
        return [two_sum(logit_list[0], logit_list[1], projection_storage_sz),
                two_sum(logit_list[0], logit_list[1], projection_storage_sz)]

    if list_sz == 4:
        return two_shake(logit_list, projection_storage_sz, iteration_sz)

    if list_sz == 16:
        return two_shake(logit_list, projection_storage_sz, iteration_sz)

    dim_sz: int                                     = sqrt(list_sz)
    two_dimensional_logit_list: list[list[Logit]]   = shape_as(logit_list, [dim_sz, dim_sz])

    transformed_logit_list_1: list[list[Logit]]     = []
    transformed_logit_list_2: list[list[Logit]]     = []
    transformed_logit_list_3: list[list[Logit]]     = []
    transformed_logit_list_4: list[list[Logit]]     = []

    for i in range(dim_sz):
        feature_vec_1: list[Logit]      = list()
        feature_vec_2: list[Logit]      = list()
        feature_vec_3: list[Logit]      = list()
        feature_vec_4: list[Logit]      = list()
        shaked_row: list[Logit]         = quad_shake(two_dimensional_logit_list[i], projection_storage_sz, iteration_sz)

        for j in range(dim_sz):
            first_dimension_representation: Logit   = twosum_tree_accumulate(shaked_row, projection_storage_sz)
            second_dimension_representation: Logit  = twosum_tree_accumulate(shaked_row, projection_storage_sz)
            third_dimension_representation: Logit   = twosum_tree_accumulate(shaked_row, projection_storage_sz)
            fourth_dimension_representation: Logit  = twosum_tree_accumulate(shaked_row, projection_storage_sz)

            feature_vec_1.append(first_dimension_representation)
            feature_vec_2.append(second_dimension_representation)
            feature_vec_3.append(third_dimension_representation)
            feature_vec_4.append(fourth_dimension_representation)

        transformed_logit_list_1.append(feature_vec_1)
        transformed_logit_list_2.append(feature_vec_2)
        transformed_logit_list_3.append(feature_vec_3)
        transformed_logit_list_4.append(feature_vec_4)

    rotated_transformed_logit_list_1: list[list[Logit]]     = rotate(transformed_logit_list_1)
    rotated_transformed_logit_list_2: list[list[Logit]]     = rotate(transformed_logit_list_2)
    rotated_transformed_logit_list_3: list[list[Logit]]     = rotate(transformed_logit_list_3)
    rotated_transformed_logit_list_4: list[list[Logit]]     = rotate(transformed_logit_list_4)
    rs_list: list[list[Logit]]                              = []

    for i in range(dim_sz):
        lhs: list[Logit]        = two_dimensional_logit_list[i]
        rhs_1: list[Logit]      = quad_shake(rotated_transformed_logit_list_1[i], projection_storage_sz, iteration_sz)
        rhs_2: list[Logit]      = quad_shake(rotated_transformed_logit_list_2[i], projection_storage_sz, iteration_sz)
        rhs_3: list[Logit]      = quad_shake(rotated_transformed_logit_list_3[i], projection_storage_sz, iteration_sz)
        rhs_4: list[Logit]      = quad_shake(rotated_transformed_logit_list_4[i], projection_storage_sz, iteration_sz)
        new_row: list[Logit]    = []

        for j in range(dim_sz):
            new_logit: Logit    = two_sum(lhs, 
                                          two_sum(two_sum(rhs_1, rhs_2, projection_storage_sz),
                                                  two_sum(rhs_3, rhs_4, projection_storage_sz),
                                                  projection_storage_sz),
                                          projection_storage_sz)

            new_row += [new_logit]

        rs_list                 += [new_row]

    return quad_shake(flatten(rotate(rs_list)), projection_storage_sz, iteration_sz - 1)

#this is probably the most important implementation in the 21st century
#point is that we are using three dimensional projection to increase the projection logics + decreasing the tree height, this is super very important to increase intellect
#we are projecting the row context -> a tessaract => aggregating all tessaracts to project a summary of the matrix for every cell, this is used as a centrality node, which is aggregated back to the old node by using a two_sum operation
#we implemented the context_logit wrong, we are putting too much responsibility on the two_sum hinge which would break (imagine maxflow) if the context cannot be distributed to the base, we can't say that the next iteration's gonna fix the problem, that's not how logic works, everything in the iteration is accounted for
#we'd want to actually cat the tessaracts on one row and do another shake, yet this would break the property of perfect squares (we have yet to find the way)

def tri_shake(logit_list: list[Logit], projection_storage_sz: int, iteration_sz: int) -> list[Logit]:

    list_sz: int = len(logit_list)

    if list_sz not in [1, 3, 9, 81, 6531]:
        raise Exception()

    if list_sz == 1:
        return [one_sum(logit_list[0], projection_storage_sz)]

    if list_sz == 3:
        return [three_sum(logit_list[0], logit_list[1], logit_list[2], projection_storage_sz), 
                three_sum(logit_list[0], logit_list[1], logit_list[2], projection_storage_sz),
                three_sum(logit_list[0], logit_list[1], logit_list[3], projection_storage_sz)]

    dim_sz: int                                     = cube_root(list_sz)
    two_dimensional_logit_list: list[list[Logit]]   = shape_as(logit_list, [dim_sz, dim_sz])

    transformed_logit_list_1: list[list[Logit]]     = []
    transformed_logit_list_2: list[list[Logit]]     = []
    transformed_logit_list_3: list[list[Logit]]     = []

    for i in range(dim_sz):
        feature_vec_1: list[Logit]      = list()
        feature_vec_2: list[Logit]      = list()
        feature_vec_3: list[Logit]      = list()
        shaked_row: list[Logit]         = tri_shake(two_dimensional_logit_list[i], projection_storage_sz, iteration_sz) #we are OK here, we are still in the assume phase of the recursive definition

        for j in range(dim_sz):
            first_dimension_representation: Logit   = threesum_tree_accumulate(shaked_row, projection_storage_sz)
            second_dimension_representation: Logit  = threesum_tree_accumulate(shaked_row, projection_storage_sz)
            third_dimension_representation: Logit   = threesum_tree_accumulate(shaked_row, projection_storage_sz) 

            feature_vec_1.append(first_dimension_representation)
            feature_vec_2.append(second_dimension_representation)
            feature_vec_3.append(third_dimension_representation)

        transformed_logit_list_1.append(feature_vec_1)
        transformed_logit_list_2.append(feature_vec_2)
        transformed_logit_list_3.append(feature_vec_3)

    #everything before this line is not changable

    #in order for the induction to work, we have to assume that every cell in the matrix is self-representable, we'd have to assume that the matrix would learn the way
    #this means that we'd have to increase the cell information, essentially increasing the dimensions of the cells not their storage in order for this to work
    #imagine that this is a real graph, every cell in the matrix is a node, and we are doing centrality on the graph

    #this morning I was doing an analysis of joint pressure
    #it seems like the problem only arises (specifically the maltransform) if the context logit (twosum(twosum(rhs_1, rhs_2), twosum(rh3, rh4))) is saturated
    #                                                                         the context logit is not logically representable (everything in two dimensions isn't sensible, it's very super hard to describe things in two dimensions without context collisions or decreasing the euclidean relevancy)

    #we'd attempt to fix the problem by actually doing a four sum projection, because it would fix the joint pressure by splitting the intermediate representation (the centrality representation) into a three dimensional context (not only that this increases the representable dimensions, but also increase the amount of minimum storable, so this is a win-win) 

    #the second problem is the shake of the individual summary frames, we are losing the global context from the tessaracts

    rotated_transformed_logit_list_1: list[list[Logit]]     = rotate(transformed_logit_list_1)
    rotated_transformed_logit_list_2: list[list[Logit]]     = rotate(transformed_logit_list_2)
    rotated_transformed_logit_list_3: list[list[Logit]]     = rotate(transformed_logit_list_3)
    rs_list: list[list[Logit]]                              = []

    for i in range(dim_sz):
        lhs: list[Logit]                = two_dimensional_logit_list[i]
        ctx_list: list[Logit]           = rotated_transformed_logit_list_1[i] + rotated_transformed_logit_list_2[i] + rotated_transformed_logit_list_3[i]
        shaked_ctx_list: list[Logit]    = tri_shake(ctx_list, projection_storage_sz, iteration_sz)
        new_row: list[Logit]            = []

        for j in range(dim_sz):
            idx_1: int          = j
            idx_2: int          = idx_1 + dim_sz
            idx_3: int          = idx_2 + dim_sz

            new_logit: Logit    = four_sum(lhs[j],
                                           shaked_ctx_list[idx_1],
                                           shaked_ctx_list[idx_2],
                                           shaked_ctx_list[idx_3],
                                           projection_storage_sz)

            new_row             += [new_logit]

        rs_list                 += [new_row]

    return tri_shake(flatten(rotate(rs_list)), projection_storage_sz , iteration_sz - 1)

#what amazed me is these lines of code being actually sufficient to approximate a very intellectual being
#we are missing a map + reduce function to evaluate the directions of these guys (we'd want to train on a complete dataset at once)
#yet we are still using the search implementation to train neural networks
#the algorithm is not hard to be understood, it's essentially a very dense network running on centrality, each node contains a multidimensional word to communicate with their neighbors efficiently
#we are training to extract the centrality algorithm, so there are centrality rules that could be reused, like in linear one row == multiple cols

#when I was trying to prove if the global_ctx_list > row_ctx_list, it's a hard proof to work on
#assume that shake_x is sufficient to approx everything
#we are stuck at the pack_threesum, we can prove that the result of the pack_threesum is better than that of the row_ctx version right after the evaluation
#yet we can't prove that another shake (or pack_threesum) would reduce the quality of the logits which would hinder the induction proof
#alright, so the problem is that the pack_threesum being not sufficient to continue the "sufficient to approx everything" (this is the logic loophole)

#in other words, the returning result is an intermediate result, not an argument to the next which can be directly compared
#so we actually went on with both, the row_ctx and the global_ctx version, which can only be better, not worse

def fourpack_onesum(lhs: LogitPack) -> LogitPack:

    if lhs.size() != 4:
        raise Exception() 

    lhs_0_0: Logit  = four_sum(lhs[0], lhs[1], lhs[2], lhs[3], projection_storage_sz)
    lhs_0_1: Logit  = four_sum(lhs[0], lhs[1], lhs[2], lhs[3], projection_storage_sz)
    lhs_0_2: Logit  = four_sum(lhs[0], lhs[1], lhs[2], lhs[3], projection_storage_sz)
    lhs_0_3: Logit  = four_sum(lhs[0], lhs[1], lhs[2], lhs[3], projection_storage_sz)

    return LogitPack([lhs_0_0, lhs_0_1, lhs_0_2, lhs_0_3])

def fourpack_twosum(lhs: LogitPack, rhs: LogitPack, projection_storage_sz: int) -> LogitPack:

    if lhs.size() != 4:
        raise Exception()

    if rhs.size() != 4:
        raise Exception()

    lhs_0_0: Logit  = four_sum(lhs[0], lhs[1], lhs[2], lhs[3], projection_storage_sz)
    lhs_0_1: Logit  = four_sum(lhs[0], lhs[1], lhs[2], lhs[3], projection_storage_sz)
    lhs_0_2: Logit  = four_sum(lhs[0], lhs[1], lhs[2], lhs[3], projection_storage_sz)
    rhs_0_0: Logit  = four_sum(rhs[0], rhs[1], rhs[2], rhs[3], projection_storage_sz)
    rhs_0_1: Logit  = four_sum(rhs[0], rhs[1], rhs[2], rhs[3], projection_storage_sz)
    rhs_0_2: Logit  = four_sum(rhs[0], rhs[1], rhs[2], rhs[3], projection_storage_sz)

    rs_0: Logit     = six_sum(lhs_0_0, lhs_0_1, lhs_0_2, rhs_0_0, rhs_0_1, rhs_0_2, projection_storage_sz)

    lhs_1_0: Logit  = four_sum(lhs[0], lhs[1], lhs[2], lhs[3], projection_storage_sz)
    lhs_1_1: Logit  = four_sum(lhs[0], lhs[1], lhs[2], lhs[3], projection_storage_sz)
    lhs_1_2: Logit  = four_sum(lhs[0], lhs[1], lhs[2], lhs[3], projection_storage_sz)
    rhs_1_0: Logit  = four_sum(rhs[0], rhs[1], rhs[2], rhs[3], projection_storage_sz)
    rhs_1_1: Logit  = four_sum(rhs[0], rhs[1], rhs[2], rhs[3], projection_storage_sz)
    rhs_1_2: Logit  = four_sum(rhs[0], rhs[1], rhs[2], rhs[3], projection_storage_sz)

    rs_1: Logit     = six_sum(lhs_1_0, lhs_1_1, lhs_1_2, rhs_1_0, rhs_1_1, rhs_1_2, projection_storage_sz)   

    lhs_2_0: Logit  = four_sum(lhs[0], lhs[1], lhs[2], lhs[3], projection_storage_sz)
    lhs_2_1: Logit  = four_sum(lhs[0], lhs[1], lhs[2], lhs[3], projection_storage_sz)
    lhs_2_2: Logit  = four_sum(lhs[0], lhs[1], lhs[2], lhs[3], projection_storage_sz)
    rhs_2_0: Logit  = four_sum(rhs[0], rhs[1], rhs[2], rhs[3], projection_storage_sz)
    rhs_2_1: Logit  = four_sum(rhs[0], rhs[1], rhs[2], rhs[3], projection_storage_sz)
    rhs_2_2: Logit  = four_sum(rhs[0], rhs[1], rhs[2], rhs[3], projection_storage_sz)

    rs_2: Logit     = six_sum(lhs_2_0, lhs_2_1, lhs_2_2, rhs_2_0, rhs_2_1, rhs_2_2, projection_storage_sz)

    lhs_3_0: Logit  = four_sum(lhs[0], lhs[1], lhs[2], lhs[3], projection_storage_sz)
    lhs_3_1: Logit  = four_sum(lhs[0], lhs[1], lhs[2], lhs[3], projection_storage_sz)
    lhs_3_2: Logit  = four_sum(lhs[0], lhs[1], lhs[2], lhs[3], projection_storage_sz)
    rhs_3_0: Logit  = four_sum(rhs[0], rhs[1], rhs[2], rhs[3], projection_storage_sz)
    rhs_3_1: Logit  = four_sum(rhs[0], rhs[1], rhs[2], rhs[3], projection_storage_sz)
    rhs_3_2: Logit  = four_sum(rhs[0], rhs[1], rhs[2], rhs[3], projection_storage_sz)

    rs_3: Logit     = six_sum(lhs_3_0, lhs_3_1, lhs_3_2, rhs_3_0, rhs_3_1, rhs_3_2, projection_storage_sz)

    return LogitPack([rs_0, rs_1, rs_2, rs_3])

#alright, it's a coincidence but it seems like 420.69 is unevitable

def fourpack_threesum(first: LogitPack, second: LogitPack, third: LogitPack, projection_storage_sz: int) -> LogitPack:

    return fourpack_twosum(fourpack_twosum(first, second, projection_storage_sz), third, projection_storage_sz)

def pack_twosum(lhs: LogitPack, rhs: LogitPack, projection_storage_sz: int) -> LogitPack:

    if lhs.size() == 4 and rhs.size() == 4:
        return fourpack_twosum(lhs, rhs, projection_storage_sz)

    raise Exception()

def pack_threesum(first: LogitPack, second: LogitPack, third: LogitPack, projection_storage_sz: int) -> LogitPack:

    if first.size() == 4 and second.size() == 4 and third.size() == 4:
        return fourpack_threesum(first, second, third, projection_storage_sz)

    raise Exception()

#alright, probably there are a lot of different ways to approximate a projection space, but I'd stick with the continuous differential approach
#the sole improvement that we made is making every cell a tessaract, this helps with the centrality communication (if two random nodes are one-dimensionals, they'd be talking in a very bad language, ambiguous, not clear), we made the two nodes 4 dimensionals

#we faced the storage_sz constraints, ideally, when we join two tessaracts, we'd want to keep the sum <= 4
#yet 4 sums would be two two-dimensional-nodes
#so we'd want to make it 6-sum, because it would be 2 three-dimensional-nodes, which is OK
#we further improved it by making it a tessaract and project the tessaract to 3 dimensions, so it could only be better not worse 

#the remaining problem is the map-reduce problem, we'd want to "stitch" the projection space by projecting all output for a set of inputs, and accumulate the deviation and move in the direction
#and the Wanted problem, the mis-stitch would require a reverse operation + etc., a train that keeps moving forward and removing the commits as fast as possible

#this is a very important lecture
#the increasing the dimension from 1 dimension -> 3 dimensions would still have the same <storage> or pointer numerical range, but the euclidean relevancy are hugely changed in the senses of sorting relevant things together or allowing that to happen, this is very important for the succeeding projections
#the optimal solution would probably somewhere around 12 dimensions hyperspace in string theory

#the question that I've always wanted to answer is "given a context space (domain space) and their optimal all pairs relevancy scores," could we somehow arrange them in an arbitrary euclidean space that their euclidean distances == relevancy scores

#alright, so the solution would be making a graph to connect all points, draw a circle or a sphere to pick the next position for the neighbors  
#if the neighbor has not been previously mentioned, we are OK, we are fine, mission accomplished
#but if the neighbor has been previously mentioned, we'd have to find the best position in the stack to make sure we could minimize the distance
#we are back to our beloved problem, the rotating two arms problem, which is used for one-dimensionalizing the projection space, so essentially, the second point could be anywhere in the space 
#a greedy approach would be to bind the neighbors that have not been previously mentioned, and run the multiarms algorithm to find the coordinate where the cosine distance is minimal compared to the original all-pair scores space
#the exact answer to this question is rather very hard to mathematically describe, we just dont have the words for that yet, so our best bet is increasing the dimensions and hope that it actually works (theoretically, increasing the dimensions can only be better, not worse, for the lower dimension answer is a subset of that)

#the two arms is actually the answer to everything, we can prove that the tip the two arms chain could reach every point in the graph, thus further the induction of placing a certain point on the graph 
#we'd need to chain the arms in order to prove that arm_1, arm_2, arm_3, etc.
#                                                   [   two_arm ]
#                                                         [   two_arm  ]
#yet this is only the greedy approach, because we can't really change the previous coordinates, but rather placing the next coordinates on the graph where it would be locally best

#alright, so what if we could somehow featurize an input as binaries, 0 means doesn't have the feature, 1 means has the feature
#we could prove the completeness of relevancy projection, but the question of whether this is equivalent to the relevancy score is an entire another topic to talk about

#how about we approach this at a different perspective, angle?
#how about we assume that we have found the euclidean projection space for the previous n - 1 pairs, we are to further the induction for the current coordinate
#okay, we'd have to keep a dictionary of all possible previouses, and we'd want to maintain that for the current coordinate
#it is an intersection of the previous possible set and the current set of restrictions, we'd have to write the formula for this
#theoretically we could solve this by using discretization of projection space + recursion (dynamic programming), we just dont have the exact number for this
#this is actually an optimizable, we'd want to demote the projection space to speed up the search time

#why does this sound like nucleus and electron?

#we need to think of the LogitPack as an operating brain, we are gossiping the information to stitch the complete picture
#so what's the problem? our brain cannot be changed from first -> last like the transformer, because it would loose the context in the way, the brain at the latter phases is not as general in the sense of entropy as the brain at the former phases

#we'd need to map each initial cell -> an arbitrary brain (which we'd call LogitPack), this is the embedding layer in torch
#we'd do gossiping, or centrality on the set of brains, and project the brain back to the original cell form 

#the function is correctly implemented in the sense of brain gossiping
#we'd want to implement another function on top of this function to do f(x) -> y, f'(y) -> x

#the joining of the two centrality nodes is probably the hinge that is difficult to solve
#theoretically, the algorithm works as projection_storage_sz approaches -> oo
#we'd definitely need to offset the theoretical

#because we are very precise people, we'd want to use finite decimal place implementation from BigNum, a missing 1000th decimal place would have a devastating effect on the network
#and we'd want to use map-reduce to train the network, essentially we are projecting all the outputs for the current network given the inputs, comparing to that of the instrument, and inch forward in the direction
#in order for that to happen, we'd need to strictly stay on the differential territory, there is no softmax, no square_root, no etc., just Taylor Series and friends 

#the problem of these networks is that we can't be greedy, we'd want to stay on the 1KB -> 64KB network to get the best of the network, and kind of scale from the smaller networks
#an attempt to train a very large network would result in a very very undesirable result
#this is precisely why compression rate of 0.1% is important, we dont know how to reach the level of intellectual otherwise

#problems:
#the joints from the lower layers would collapse due to context saturation
#we'd have to keep the original context throughout the centrality gossip

#essentially, we'd want to have a channel for gossiping and a channel to store the org_x
#alright, another upgrade from keeping the original would be having a sliding decay window of context trails, just to make sure that we aren't stressing the lower layers, it'd be very bad in terms of logit density distribution
#                                                                                                             because the upper layers having the same storage sz are not doing as much job as the lower layers (or it is bottlenecked by the lower layers) 

#we'd be working on this brain for a month or two to get some financial aids first
#it's not that complicated
#the formula is simple: centrality, 3 dimensional array, tessaracts, a centrality algorithm, Multidimensional Taylor projections + trailing memorizations + map-reduce search + multiprecision lib + tons of compute power to compute | project at unprecended speed + erase commits at that speed also
#recall the train in Wanted, no trains get stopped on the bridge
#we wont be getting lots of money, a couple hundred mils is good enough for us to keep the project going + hire some people

#we are missing a plus operation, a diagonal operation, and a channel operation

#at the high level, we are having our original context floating around and a gossip context
#a gossip context is a product of another gossip context and another original context
#this is getting confusing

#a window of original context interaction to make sure that we are "reminding" the matrix of their initial missions, not to have destructive interferences because we can't transcend from the original context
#this requires a carefully engineered reminding window to work

#a centrality node join == two gossip values interacting, by a plus operation (alright, they got this right) 
#the plus acts as a calibration layer for communication, the original context is there to cancel the ripple effect of mis-calibrated

#the solely two sum operation only works as the storage sz approaches infinity, we are forcing a recontext at this point to make sure that our base joints are intact
#we'd want to offset that by forcing the two sum to approximate a calibration layer, which we'd want to peel the calibration layer one by one
#so instead of having the twosum to be accurate 100% about the projecting context, we just need it to approx a delta to move the current context into a valid context  

def shake_x(logit_list: list[LogitPack], projection_storage_sz: int, iteration_sz: int) -> list[LogitPack]:

    list_sz: int = len(logit_list)

    if list_sz not in [1, 2, 4, 16, 256, 65536]:
        raise Exception()

    if list_sz == 1:
        return [pack_one_sum(logit_list[0], projection_storage_sz)]

    if list_sz == 2:
        return [pack_two_sum(logit_list[0], logit_list[1], projection_storage_sz),
                pack_two_sum(logit_list[0], logit_list[1], projection_storage_sz)]

    dim_sz: int                                         = sqrt(list_sz)
    two_dimensional_logit_list: list[list[LogitPack]]   = shape_as(logit_list, [dim_sz, dim_sz])
    transformed_logit_list: list[list[LogitPack]]       = []
    transformed_logit_list_2: list[list[LogitPack]]     = []

    for i in range(dim_sz):
        shaked_row: list[LogitPack]     = shake_x(two_dimensional_logit_list[i], projection_storage_sz, iteration_sz)
        transformed_logit_list.append(shaked_row)
        shaked_row_2: list[LogitPack]   = shake_x(two_dimensional_logit_list[i], projection_storage_sz, iteration_sz)
        transformed_logit_list_2.append(shaked_row_2)

    transformed_logit_list          = rotate(transformed_logit_list)
    rs_list: list[list[LogitPack]]  = []

    for i in range(dim_sz):
        org_list: list[LogitPack]           = two_dimensional_logit_list[i]
        ctx_list: list[LogitPack]           = transformed_logit_list[i]
        shaked_ctx_list: list[LogitPack]    = shake_x(ctx_list, projection_storage_sz, iteration_sz)
        other_ctx_list: list[LogitPack]     = transformed_logit_list_2[i]
        new_row: list[LogitPack]            = []

        for j in range(dim_sz):
            new_logit: LogitPack    = pack_threesum(org_list[j], other_ctx_list[j], shaked_ctx_list[j], projection_storage_sz, iteration_sz) #this is where I can't prove, the recursive property is detached at this point, because I can't prove if it is more efficient than a pure row transformation, I'd have to add that as an optional information to make sure that we are covering the corner cases 
            new_row                 += [new_logit]

        rs_list                             += [new_row]

    return shake_x(flatten(rotate(rs_list)), projection_storage_sz, iteration_sz - 1) 

class MatMulPolicy:

    def get_order() -> list[list[int]]:
        pass 

def optimize_matmul(dot_product_sz: int,
                    domain_points: object,
                    projection_points: object,
                    max_storage_sz: int) -> MatMulPolicy:

    pass
