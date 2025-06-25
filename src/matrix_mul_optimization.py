
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

#how precisely could we do this?
#by using dynamic programming
#for every newly generated projection space (idx 0 (two sums) idx 1), we'd add that -> to the back of the list

#we'd want to maintain a definition of straightness of projection space / storage_sz for every combination, kind of a logit density mining operation
#because we dont really know the patterns, we'd want to learn this pattern and extends that to another large size patterns 
#this is a very important project that we'd want to get done within this week

#alright, do we see the pattern yet???
#we have 1 sum 2 sum 3 sum 4 sum
#now we are trying to make 5 sum 6 sum 7 sum 8 sum 9 sum 10 sum
#how about 11 sum 12 sum 13 sum 14 sum 15 sum
#the cycle of sum actually repeats
#we kind of building sum on top of sum, by running statistics, and climb the ladder of sum length
#alright, this is actually another learning problem, the problem of network bindings because we cant really bind the network reasonably

#and we'd want to clue from there

#the problem is actually centrality, we have advanced vocab thanks to centrality, we are running "pagerank" on the network
#we need to find the policy for efficient matrix multiplication to aid the centrality in approxing the vocab
#we need to run the search algorithm to find deviation space, we'd probably want to run the search on different datasets to avoid numerical problems (or using a multiprecision lib)
#if we have completed these, we'd probably have a conscious AI
#AI in my view is radixed as, finite buffer AI + thru everything context AI
#finite buffer AI is human
#thru_everything_context AI is doing projection of a bunch, there is no temporary buffer
#in order to get there, we'd need the finite buffer AI on every training node to listen to the strings (this is hard)

#100% stock projection is possible, we'd be multi-millionaire to run this on cloud soon, stay tuned !!!

def optimize_matmul(dot_product_sz: int,
                    domain_points: object,
                    projection_points: object,
                    max_storage_sz: int) -> MatMulPolicy:

    pass