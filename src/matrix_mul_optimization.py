
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

def optimize_matmul(dot_product_sz: int,
                    domain_points: object,
                    projection_points: object,
                    max_storage_sz: int) -> MatMulPolicy:

    pass