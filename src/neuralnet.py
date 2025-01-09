#this is the punch line of every working neural network
#I dont know how - I dont know what you do
#it has to look like this
#alright so what happens here
#we are trying to build a uniform input logit diffraction
#we are using addition operation to be forgiving on the output space - says we circumscribing the sphere -> multiple spheres which have constructive interference | or f(x) static activations (we can say that hash(logit_vec) mod MODULO) to avoid saturated training rate
#alright - to avoid training destructive interference - we must use hash(unique_representation(x)) - x is the unique representation of the base input logit 
#this is a very very important note 
#alrights - our model can scale - such that there is no such a thing as saturated loss_rate
#yet the logit density does not scale - this is the most important in intelligent system not intelligence system
#logit density can be briefly described as training_data_sz/ neural_network_size for the same discretized loss_rate 
#we'll spend 10 years of doctoral to study logit density
#these are tough materials - the study of the shapes
#alrights - so what do we learn? we learn that intelligence system has low logit density - but good at retaining informations
#                                we learn that intelligent system has high logit density and good at making bullshits - you tell them Aurora 6pm Seattle netflix and chill
#we have 3 weeks to build this guys
#we'll be rich guys - stay tuned
#we'll probably be the firsts to build a distributed network for 1M personal devices (with this rate - people will include top-of-the-line GPUs in iphones soon enough) - which compete for logit density solutions
#we'll build a bid/ask system that is powered by the people, run by the people - just to move logits around

#alrights - Dad got a point - the initial input is actually skewed - in the semantic space
#so it has a synthetic shape - because human is not good at diffracting semantic languagely
#we can either - enhance the semantic - or we can best fit a synthetic shape (that is not sphere or linear for the initial layers)
#context diffractor or mix(x) is actually the transformer layers - x + f(x) + g(x + f(x)) - in a wave-like property (the recursion candy) - thus add
#we want to train the layers by blocking other layers - to approx the semantic space for a certain layer then we train others to continue to approx semantic space, etc.
#what we are missing is actually static projection (or blkr + initial values) - if we aren't forcing it to be of a certain shape - then we aren't hitting the inital requirement of f(x) being a semantic mapper

#alrights - let's start from a clean slate
#assume we have x and f(x)
#x semantic space is uniformly distributed (has a sphere shape)
#f(x) semantic space is uniformly distributed (has a sphere shape)
#sizeof(f(x)) < sizeof(x) 
#yeah - we are doing semantic compression

#we would want to do f(mix(x)) = y - mix is the transformer layer and f is the output
#the training question is whether we could appropriately train given the semantic layer x and g(x), g(x) as in f(g(x)), as we could train x and f(x) 
#or the semantic layer x and k(x), k(x) as in f(g(k(x))) for that matter
#alrights - consider this - we have an appropriate f(x) layer - we block the f(x) layer to train the g(x) layer - g(x) is now the direct f(x) proxy - and we train x g(x) as we train x f(x)
#g(x) now produce random semantic space - which is uniformly distributed - according to our f(x) shape - and now we block g(x) layer to train f(x) layer
#so g(x) is our new x input - and we are again doing f(x)

#so what's the catch - the catch is our training loss must be 0% in order for this to work
#the only reason that it is not 0% is because we have completely densified the logit - and we solve that by adding another semantic mapper layer 
#okay - so this is the solution we mentioned in app_compass
#we have completed one revolution of training - yay
#we'll be back with path + results - this is just theoretically speakings - the path prunables

#Mom - mathematical operations arent sufficient as base operations - and I hope that people would come up with a synthetic solution for the base operations (we want to use math_approx for that)
#we know that our coordinate is probably not correct - a coordinate that could describe things as waves-like and Hz as x as in f(x) is preferred in the cuda-environment
#we know that the ideal semantic space is continuous and has good distribution: projection_counter(x)/dx = avg - somewhat looks like a sphere
#we know that the input semantic space is skewed and does not have good distribution: projection_counter(x)/dx != avg
#we know that locality trades for logit density
#we know that log discretizations (uint3_t base 5) are important for neural network training
#we know that group logit training is important for neural network training
#we know that neural network is only good for compression and training f(x) -> x
#we know that all the above things are logit mining - there is not a real solution than to just spin the core and mine the logit density

def f(x: list[bool], f_properties: object) -> list[bool]:

    return project(pos(x, f_properties), f_properties) #one of pos(x) responsibility is semantic space mapper - mapping from irrelevant space -> euclidean relevant space - limit (distance -> 0) in euclidean coordinate means semantically equivalent 
                                                       #second of pos(x) responsibility is to bijectively rearrange x input space -> (discretized n)! space - 1234 -> 3214, 2134, 4321, etc. 

def f_x(x: list[bool], f_properties: object, approx_sz: int) -> list[bool]:

    func_vec        = [f(x, f_properties, cursor) for _ in range(approx_sz)]
    inclusive_vec   = static_exclusion(func_vec, initial_input_unique_representation, f_properties) #leaf logits diffraction

    return sum(inclusive_vec)  #stacking f(x) - gradients of f(x) and f1(x) and f2(x) f3(x) are the trainingly equivalents - these are semantically equivalent groups - we want uniform training rate for the group - or combinatorial blkr to approx the best results 

def pos(x: list[bool], f_properties: object) -> list[bool]:

    #this is where the recursion happens - this is precisely the quicksort (mergesort) algorithm
    #for sorting algorithms - we assume that f(x) sorts the array
    #for neural network algorithms - we assume that f(x) groups the context semantically and diffract the context by the replication factor of K (okay - when we diffract the context - we are wasting leaf logits to move things around - this is the argument between locality and logit_density)
    #Ok, this is the important part - for locality purposes - K must be uniformly diffracted all over the output range (like photon and double-slit experiment) - we want this to maximize cuda parallel processing  
    #so a zip of lhs and rhs is essentially a reinvented static merge_sort zip (without comparisons) before we transform it again by mix(lhs, rhs, args...)
    #mix is essentially linear or sphere transformation - or multi-layer perceptrons
    #the complexity of this does not precede sorting algorithms' - which is O(n * log(n)) 

    #okay - people ask where the activator function is
    #activator function is, actually, not what people think - hint it is not if else activation per se in the sense of training
    #it is the bad leaf logit density diffraction - in another words, it has the equivalent terrible job of our circle implementation (left | right) 
    #the true discrete operation is hard to implement - I dont know if there is a reasonable implementation than just random seeds

    #because all our functions are - recursively defined and symmetric - it is equivalently described as above 
    #the only purpose of activators is to resource utilize the leaf logits efficiently
    #such is there is no all in one basket (says I want to utilize 3 out of 15 logits for a certain set of input, another 3 out of 15 logits for another set of inputs etc.) 
    #not for the cause that we stated in math_approx.py

    (lhs, rhs)      = half_split(x) #without loss of generality - the state of the art transformer actually split this row_sz - and do concurrent parallel semantic mixing then do transpose(0, 1) which is a mixed_semantic - then another MLP 
    lhs_semantic    = f_x(lhs, f_properties, lhs_approx_sz) #recursive definition - we need to define what we are approxing - so we could write the recursive resolution - is it repeated context diffraction - this is for the parallel processing purposes
    rhs_semantic    = f_x(rhs, f_properties, rhs_approx_sz) #recursive definition
    mixed_semantic  = mix(lhs_semantic, rhs_semantic, f_properties, mix_approx_sz) #uniform input logit diffraction - this function is probably the most important

    return mixed_semantic #return mixed semantic - which is now wave-liked instead of euclidean-liked shape - this is a hint that our euclidian semantic coordinate is probably not correct - unless our operation window is on euclidean coordinate - which is an output assumption