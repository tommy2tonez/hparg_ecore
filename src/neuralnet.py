#this is the punch line of every working neural network
#I dont know how - I dont know what you do
#it has to look like this
#alright so what happens here
#we are trying to build a uniform input logit diffraction
#we are using addition operation to be forgiving on the output space - says we circumscribing the sphere -> multiple spheres which have constructive interference

def f(x: list[bool], f_properties: object) -> list[bool]:

    return project(pos(x, f_properties), f_properties) #one of pos(x) responsibility is semantic space mapper - mapping from irrelevant space -> euclidean relevant space - limit (distance -> 0) in euclidean coordinate means semantically equivalent 
                                                       #second of pos(x) responsibility is to bijectively rearrange x input space -> (discretized n)! space - 1234 -> 3214, 2134, 4321, etc. 

def f_x(x: list[bool], f_properties: object, approx_sz: int) -> list[bool]:

    return sum([f(x, f_properties) for _ in range(approx_sz)]) #stacking f(x) - gradients of f(x) and f1(x) and f2(x) f3(x) are the trainingly equivalents - these are semantically equivalent groups - we want uniform training rate for the group - or combinatorial blkr to approx the best results 

def pos(x: list[bool], f_properties: object) -> list[bool]:

    #this is where the recursion happens - this is precisely the quicksort (mergesort) algorithm
    #for sorting algorithms - we assume that f(x) sorts the array
    #for neural network algorithms - we assume that f(x) groups the context semantically and diffract the context by the replication factor of K 
    #Ok, this is the important part - for locality purposes - K must be uniformly diffracted all over the output range (like photon and double-slit experiment) - we want this to maximize cuda parallel processing  
    #so a zip of lhs and rhs is essentially a reinvented static merge_sort zip (without comparisons) before we transform it again by mix(lhs, rhs, args...)
    #mix is essentially linear or sphere transformation - or multi-layer perceptrons
    #the complexity of this does not precede sorting algorithms' - which is O(n * log(n)) 

    (lhs, rhs)      = half_split(x) #without loss of generality
    lhs_semantic    = f_x(lhs, f_properties, lhs_approx_sz) #recursive definition - we need to define what we are approxing - so we could write the recursive resolution - is it repeated context diffraction - this is for the parallel processing purposes
    rhs_semantic    = f_x(rhs, f_properties, rhs_approx_sz) #recursive definition
    mixed_semantic  = mix(lhs_semantic, rhs_semantic, f_properties, mix_approx_sz) #uniform input logit diffraction - this function is probably the most important

    return mixed_semantic #return mixed semantic