
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

#this two_shake operation is very important, because every entry in the logit_list is actually a unit, we have yet to tell if this is 1 logit or 1 tile

#the problem about the machine learning recursive definition is that we have to be increasingly paranoid about keeping the definition

#what is the two_shake operation actually does:
#input:     a 1 dimensional matrix
#output:    a matched matrix whose every Entry is "well mixed" + "polymorphic" in the sense of sufficient to approx every possible output given the input matrix 

#we are not OK here
def two_fold(logit_list: list[Logit], projection_storage_sz: int) -> list[Logit]:

    sz: int             = len(logit_list)

    if sz % 2 != 0:
        raise Exception()

    left: list[Logit]   = logit_list[:(sz / 2)]
    right: list[Logit]  = logit_list[(sz / 2):]

    return [two_sum(lhs, rhs) for (lhs, rhs) in zip(left, right)]

def quad_fold(logit_list: list[Logit], projection_storage_sz: int) -> list[Logit]:

    return two_fold(two_fold(logit_list, ), )

def quad_partition(logit_list: list[Logit]) -> list[Logit]:

    rs = [list(), list(), list(), list()]

    for i in range(len(logit_list)):
        idx     = i % 4
        rs[idx] += [logit_list[i]]

    return rs[0] + rs[1] + rs[2] + rs[3] 

def get_numerical_representation(logit_list: list[Logit], projection_storage_sz: int) -> Logit:

    return twosum_tree_accumulate(logit_list, projection_storage_sz) #essentially a binary tree to do two sum, logit_list is the base of the binary tree

def shuffle(logit_list: list[Logit]) -> list[Logit]:

    pass

#1 1 
#2 2 
#4 4
#16 16
#256 256 (we need this) 
#65536 65536

#this probably still has room for improvement, because we are assuming that the logit_list responsibility is uniform, lhs + rhs would not quad_shake in the sense of uniform distribution as much as we wanted   

#the moral of this function is to shake a square

#by projecting the row context -> 4 dimensions (to avoid context collision), rotate so that each row contains a fuzzy representation of the matrix
#we'd want to do another quad_shake (recall that because the shake actually keeps the original semantic of the input at the cell, so we'd want to do a quad_partition followed by a quad fold so we can make sure that the context is not worse than it was before the function invoke) 
#we'd then want to mix the lhs + rhs to do another shake and another fold
#we'd want to rotate and do another shake, the number of such rotation has to be of even numbers

#this is probably the most important transformation, we'd want to improve the get_numerical_representation (I dont know howtos yet)

def quad_shake(logit_list: list[Logit], projection_storage_sz: int, iteration_sz: int) -> list[Logit]:

    list_sz: int = len(logit_list)

    if list_sz not in [1, 2, 4, 16, 256, 65536]:
        raise Exception()

    if list_sz == 1:
        return [one_sum(logit_list[0], projection_storage_sz)] 

    if list_sz == 2:
        return [two_sum(logit_list[0], logit_list[1], projection_storage_sz / 2), two_sum(logit_list[1], logit_list[0], projection_storage_sz / 2)]

    dim_sz: int                                     = sqrt(list_sz)
    two_dimensional_logit_list: list[list[Logit]]   = shape_as(logit_list, [dim_sz, dim_sz])

    transformed_logit_list_1: list[list[Logit]]     = []
    transformed_logit_list_2: list[list[Logit]]     = []
    transformed_logit_list_3: list[list[Logit]]     = []
    transformed_logit_list_4: list[list[Logit]]     = []

    #our generic formula's gonna look like this
    #alright people, because we are extremely frustrated if the square is not either 1x1 2x1 2x2 4x4 16x16 or 256 x 256
    #we'd want to change the implementation just for the shake of fitting the squares

    for i in range(dim_sz):
        feature_vec_1: list[Logit]      = list()
        feature_vec_2: list[Logit]      = list()
        feature_vec_3: list[Logit]      = list()
        feature_vec_4: list[Logit]      = list()

        shaked_row: list[Logit]         = quad_shake(two_dimensional_logit_list[i], , iteration_sz) #we are OK here, we are still in the assume phase of the recursive definition

        for j in range(dim_sz):
            #the question is probably how difficult that is the project a row -> 4 dimensional vector in a continuous, sensible space, we would want to minimize that as an engineer 
            #if we shuffle the row, we'd put more burden on the shoulder of the shaked_row responsibility to make the representation "sensible"
            #because a quad_shake already sets up for the get_numerical_representation to succeed, I dont think this is a necessity

            first_dimension_representation: Logit   = get_numerical_representation(shaked_row)
            second_dimension_representation: Logit  = get_numerical_representation(shaked_row) #this is easier to get the numerical representation of the row, because we'd want to avoid the extremely skewed cases of 0 1 -> 0 == context lost, assume that the quad_shake is not there, a shuffle() is a necessity, quad_shake() is to reduce the burden of distribution of context (so if this is easier then the converged curves of having the quad_shake() should be more optimal in the case)
            third_dimension_representation: Logit   = get_numerical_representation(shaked_row)
            fourth_dimension_representation: Logit  = get_numerical_representation(shaked_row)

            feature_vec_1.append(first_dimension_representation)
            feature_vec_2.append(second_dimension_representation)
            feature_vec_3.append(third_dimension_representation)
            feature_vec_4.append(fourth_dimension_representation)

            #we are also OK here, we assume that the quad_shake is sufficient in the sense of uniformly distribution for twosum_binary_tree_accumulate to project a fuzzy representation of the row

        transformed_logit_list_1.append(feature_vec_1)
        transformed_logit_list_2.append(feature_vec_2)
        transformed_logit_list_3.append(feature_vec_3)
        transformed_logit_list_4.append(feature_vec_4)

    #we are OK here

    rotated_transformed_logit_list_1: list[list[Logit]]     = rotate(transformed_logit_list_1)
    rotated_transformed_logit_list_2: list[list[Logit]]     = rotate(transformed_logit_list_2)
    rotated_transformed_logit_list_3: list[list[Logit]]     = rotate(transformed_logit_list_3)
    rotated_transformed_logit_list_4: list[list[Logit]]     = rotate(transformed_logit_list_4)

    shaked_1: list[list[Logit]]                             = shape_as(quad_shake(flatten(rotated_transformed_logit_list_1), , iteration_sz), [dim_sz, dim_sz])
    shaked_2: list[list[Logit]]                             = shape_as(quad_shake(flatten(rotated_transformed_logit_list_2), , iteration_sz), [dim_sz, dim_sz])
    shaked_3: list[list[Logit]]                             = shape_as(quad_shake(flatten(rotated_transformed_logit_list_3), , iteration_sz), [dim_sz, dim_sz])
    shaked_4: list[list[Logit]]                             = shape_as(quad_shake(flatten(rotated_transformed_logit_list_4), , iteration_sz), [dim_sz, dim_sz])

    context_logit: list[list[Logit]]                        = shape_as(quad_fold(flatten(shaked_1) + flatten(shaked_2) + flatten(shaked_3) + flatten(shaked_4), ), [dim_sz, dim_sz])

    #we are OK here

    # transformed_four_dimensional_logit_list = [quad_fold(quad_shake(quad_partition(flatten(feature_vec)), , iteration_sz), ) for feature_vec in transformed_four_dimensional_logit_list] #this is not a smaller square, which can either do not have a recursive base or not a square

    #we are still OK here

    rs_list: list[list[Logit]]                              = []

    for i in range(dim_sz):
        lhs: list[Logit]    = two_dimensional_logit_list[i]
        rhs: list[Logit]    = context_logit[i]
        rs_list             += [two_fold(lhs + rhs)]

    return quad_shake(flatten(rotate(rs_list)), , iteration_sz - 1)

#
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
