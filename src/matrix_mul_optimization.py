
#

def floor_projection_storage_sz(in_feature_sz: int, max_storage_sz: int) -> int:
    pass

def make_projection_storage(sz: int) -> list[float]:
    pass

def multi_dimensional_projection(in_features: list[float], projection_storage: list[float]) -> float:
    pass 

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
            return multi_dimensional_projection([descendant.get_value() for descendant in self.descendant_vec], self.projection_storage_vec)
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

    actual_projection_storage_sz: int   = floor_projection_storage_sz(1, projection_storage_sz)

    if actual_projection_storage_sz == 0:
        raise RuntimeException()

    feature_vec: list[float]            = make_projection_storage(actual_projection_storage_sz)

    return Logit(feature_vec, [lhs], None)

def two_sum(lhs: Logit, rhs: Logit, projection_storage_sz: int) -> Logit:

    actual_projection_storage_sz: int   = floor_projection_storage_sz(2, projection_storage_sz)

    if actual_projection_storage_sz == 0:
        raise RuntimeException()

    feature_vec: list[float]            = make_projection_storage(actual_projection_storage_sz)

    return Logit(feature_vec, [lhs, rhs], None) 

def three_sum(x: Logit, x1: Logit, x2: Logit, projection_storage_sz: int) -> Logit:

    actual_projection_storage_sz: int   = floor_projection_storage_sz(3, projection_storage_sz)

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

#without loss of generality
#16 -> 4 x 4

#attempt to accumulate col by using sparse graph (binary tree of twosum) -> [0, 1, 2, 3] -> [0], [0, 1, 2, 3] -> [1], [0, 1, 2, 3] -> [2], [0, 1, 2, 3] -> 3
#rotate -> shake again -> pairwise two sum
#-> rotate -> rinse and repeat

## of rotations (or loop), iteration_sz must be of pow2 size (I just have that hinge)
#two dimensions     == two binary tree of two sums
#three dimensions   == ...

def two_shake(logit_list: list[Logit], projection_storage_sz: int, iteration_sz: int) -> list[Logit]:

    list_sz: int = len(logit_list)

    if not is_pow2(list_sz):
        raise Exception()

    # if list_sz == 4:
    #     return logit_list

    if iteration_sz == 0:
        return logit_list

    dim_sz: int                                                 = sqrt(list_sz)
    two_dimensional_logit_list: list[list[Logit]]               = shape_as(logit_list, [dim_sz, dim_sz])
    transformed_two_dimensional_logit_list: list[list[Logit]]   = []

    for i in range(dim_sz):
        feature_vec: list[list[Logit]]  = list()
        shaked_row: list[Logit]         = two_shake(two_dimensional_logit_list[i], , iteration_sz)

        for j in range(dim_sz):
            accumulated_logit: Logit            = twosum_binary_tree_accumulate(shaked_row) #uniform rule for accumulation, in other words, we can reuse the rules for other rows (or cols), we can use threesums or foursums to reduce the tree height which would help with the context loss (we can argue otherwise if we can prove that if the two_shake rs is uniformly distributed in an arbitrary space, we can continue such properties)
            another_accumulated_logit: Logit    = twosum_binary_tree_accumulate(shaked_row) #uniform rule for accumulation

            feature_vec.append([accumulated_logit, another_accumulated_logit])

        transformed_two_dimensional_logit_list.append(feature_vec)

    transformed_two_dimensional_logit_list  = rotate(shape_as(transformed_two_dimensional_logit_list, [dim_sz, dim_sz]))
    transformed_two_dimensional_logit_list  = [two_shake(flatten(feature_vec), , iteration_sz) for feature_vec in transformed_two_dimensional_logit_list] #

    flattened_transformed_list: list[Logit] = flatten(transformed_two_dimensional_logit_list)
    rs_list: list[Logit]                    = list()

    for i in range(list_sz):
        new_logit: Logit = three_sum(logit_list[i], flattened_transformed_list[i * 2], flattened_transformed_list[i * 2 + 1]) #alright, we are doing threesum because dimensional reduction of row, col -> 1 dimension is unstable, so we'd have to project that -> 2 dimensions, each row now contains a fuzzy representation of the entire matrix
        rs_list.append(new_logit)

    return two_shake(rotate(rs_list), , iteration_sz - 1)

class MatMulPolicy:

    def get_order() -> list[list[int]]:
        pass 

#we are trying to do 2 sum, a sparse binary tree to do a dimensional reduction, a.k.a. a dot product of a two vectors
#problems, the context of the base is not uniformly distributed, this is the only problem
#one could argue that a sparse binary tree should suffice if the context of the base is uniformly distributed in an arbitrary space

#so I guess our problem is to "rediffract" the base context, by using a milkshake operation

#a milkshake operation is essentially x + f(x) (in the transformer)
#thing is that this milk shake operation does not force the f(x) to pollute the original context if f(x) does not improve the succeeding context

#we'd want to replace the + operation by a two-sum operation (which was described in the multivariate_taylor_series)
#as for the milk shake operation, essentially we'd want to do two_sum three sum + rotate + rinse and repeat, this is another transformer

#we need to understand that the . operation is to do fast projection of col states, an intermediate representation in one dimensional space to loosely describe the col states + heading the col states in a transforming direction
#we'd want to improve the . operation and the + operation solely   

#after we have found our new matrix multiplication technique, we'd want to optimize that by using an optimizer to transform the operations -> SIMD + cuda friendly operations, by reordering + reapprox the operation order to tune with the current hardware instruction set 
#we'd want to find the sweet spot between the storage size and projection accuracy, we'd want to have a function to give an optimal instruction of transforming given a specific storage size
#we'd probably want to do search, we can't be too greedy about the number of keyvalues included in the instrument
#so we'd want to either use multi-precision to offset the cost or we'd want to use multiple instruments to sing to the projection space
#we'd make a purchase before August 22nd, mark my words !!!

#I have been thinking about the centrality algorithms (betweenness centrality + pagerank + differential + etc.), their effects on ML for too long
#it seems like we are doing matrix multiplication, dot product very wrong

#recall that in a normal centrality algorithm, we have fixed nodes, we are propagating the values from neighbor nodes -> the current node
#the problem is that our dot product is in one dimension, and we rotate the matrix to do centrality

#so there are two problems:
#the problem of context collision of dot product, and the problem of rotation (this is not exactly a problem)

#so an attempt to solve this would be

#we have a matrix
#2 dimensional projection dot product (combinatorial <rox> . <col> -> <a, b> instead of <a>) -> tranformed matrix
#-> rotate transformed matrix, do another good shake on the transformed matrix rows and do a centrality threesum operation on the original matrix
#then we'd want to rotate the original matrix
#-> we have a matrix

#we are working on square units

#the hinge of this assignment is keeping the "semantic" of the inputs, we have to be able to prove that our inputs are progressively moving in the direction of the output without losing their context in the way
#so a mid-way representation of input-output is not forced, but a product of choices


def optimize_matmul(dot_product_sz: int,
                    domain_points: object,
                    projection_points: object,
                    max_storage_sz: int) -> MatMulPolicy:

    pass