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

#it's not underestimated when people said FFT (fast fourier transform) is probably the most important convention
#because it allows the addition operation to have constructive interference without breaking numerical stability
#alright consider this dummy solution
#says tommy is on sin(x), thomas is on sin(2x)
#we discretize the frequency - add them together - and now we have perfect encoding without losing numerical stability
#alright - let's talk about frequency discretization
#we have <x, x1, x2> without loss of generality - x represents unit frequency of range 0-10Hz, x1 represents unit frequency of range 0-100Hz, x2 represent frequency of range 0-1000Hz
#ok - now we have frequency vector - we want to map this dude to the semantic euclidean u(x) - where closer in x means closer integral((f(x) - f'(x)).dx) - how tf do we map this dude - alright - its the math_approx.py guys - good luck
#or we have a pretrained static continuous transformation p(x) -> euclidean
#we dont train the data on those leaf logits - those are immutables 
#this is literally BFS - you start at a point - you ziggle - and you make up the output - until all the data is reasonably mapped  
#in other words, this is the circle problem - we get the intergral value and we draw the circle (sphere) around the random point - recursively doing that until the condition of intergral_value == euclid_distance(x, y) is met - this is a hard task - i'm not saying this is easy - this requires correct discretization + everything to make this right

#the only reason we are doing frequencies' coordinate is because we are using constructive interference operation addition without breaking numerical stability (lossless encode)
#alrights - I've been talking bullshits - actually 1Hz 2Hz 1.5Hz interference is lossless and it's called FFT - converting from the discretized (x -> f(x)) domain -> the frequency domain 
#what we actually want to store is the fuzzy chad, x = chad, f(x) = chad - and doing addition operation to have constructive interference on the output
#so we actually map the chad -> frequency -> the euclidean coordinate then do projection
#the mapping is continuous and pretrained - without further training (immu logits)

#we assume literally everything is ordered avg discretized chad - from x to f(x) - we leverage the continuity of euclidean (or radian) coordinate by mapping f(euclid(x)) - note that the function is euclid_continuous but the output is chad (this is an assumption)
#other than that I don't really see the whys

#a good coder is a good assumer - they break the function responsibiliies correctly 
#we assume this does that that does other things - then we use recursion resolution to solve the problem

#alrights - we assume everything as unordered_bag of context - it's kinda lossless in the sense
#if the context already encodes the orderness - then lossless unordered_bag is essentially lossless ordered_bag
#if the context does not already encode the orderness - then we have to invent a way to encode the order of x and y - which is preprocessing the data and make it lossless unordered bag compatible
#we do this by using one sign bit 
#says 10 10 -> 010 110 
#this is not good for constructive interference - but we are talking about lossless compression 

#let's elaborate on this further - there are two ways to do this: one is (x1 * 2 + 1) + (x2 * 2 + 0) to preserve interference (full constructive interference) - we now assume that x1 and x2 are frequency domain - not chad domain (alright maybe chad is not that good - we could stay on frequency domain to do things) - so to do lossless compression - we must convert chad back to frequency domain and train another neural network on this
#                                - alright - sin(x) + sin(1.5x) and sin(2x) + sin(3x) what's the mathemetical difference? this is the quedstion that only neural network could answer appropriately
#                                - second way is x1 * 2 + x2 to preserve interference - and reduce the semantic space by 2 ** tree_height - this requires precond of len(x1) == len(x2) - in terms of semantic_space_bit_length
#                                - the way we do lossless and lossy compression will determine how good our model is
#                                - we want to discretize this space - the chad space - and map it to the euclidean - such that sqrt((chad1(x) - chad2(x)) * (chad1(x) - chad2(x))) ~= integral(chad1(x)) - integral(chad2(x)) - we want propotional relations - we dont care about actual values - because it's semantic mapping
#                                - after we have the euclidean output - we want to train this (chad -> euclid) on another continuous neural network - that has the training loss rate of 0% - we make this immutable - and now we have chad -> euclid or radian coordinate to do continous projection 
#                                - we want to map this guy for EVERY f(x) projection - f(x) -> f(chad_to_euclid(x))

#alright - so we just learned that converting to the chad coordinate to do add operation (constructive interference) has better numerical stability - and convert it back to frequency domain then fixed euclid semantic coordinate and do continuous mapping is a good way to do things
#this is easily described not easily written

#* chad coordinate is essentially x1, x2, x3, x4
#x1's range is [0, 5)
#y1 is area/deltax = 2
#x2's range is [5, 10)
#y2 avg is 4
#etc.
#chad is good but we'll probably have other coordinates to do this better

#Mom talked about context diffractor and divider earlier - we have to assume that the x and f(x) density space are uniformly distributed - projection_counter(x) / dx = area/range for all x c range 
#alright - we'll build a regex engine + regex path optimization + convergence analysis to mine logit density  

#mapping x - f(x)
def f(x: list[bool], f_properties: object) -> list[bool]:

    return project(pos(x, f_properties), f_properties) #one of pos(x) responsibility is semantic space mapper - mapping from irrelevant space -> euclidean relevant space - limit (distance -> 0) in euclidean coordinate means semantically equivalent 
                                                       #second of pos(x) responsibility is to bijectively rearrange x input space -> (discretized n)! space - 1234 -> 3214, 2134, 4321, etc. 

#trade density for intelligence
def f_x(x: list[bool], f_properties: object, approx_sz: int) -> list[bool]:

    func_vec        = [f(x, f_properties, cursor) for _ in range(approx_sz)]
    inclusive_vec   = static_exclusion(func_vec, initial_input_unique_representation, f_properties) #leaf logits diffraction

    return sum(inclusive_vec)  #stacking f(x) - gradients of f(x) and f1(x) and f2(x) f3(x) are the trainingly equivalents - these are semantically equivalent groups - we want uniform training rate for the group - or combinatorial blkr to approx the best results 

#diffract + mix context
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

    #^^^
    #these are the most important lines
    # (lhs, rhs)      = half_split(x) #without loss of generality - the state of the art transformer actually split this row_sz - and do concurrent parallel semantic mixing then do transpose(0, 1) which is a mixed_semantic - then another MLP 
    # lhs_semantic    = f_x(lhs, f_properties, lhs_approx_sz) #recursive definition - we need to define what we are approxing - so we could write the recursive resolution - is it repeated context diffraction - this is for the parallel processing purposes
    # rhs_semantic    = f_x(rhs, f_properties, rhs_approx_sz) #recursive definition
    # mixed_semantic  = mix(lhs_semantic, rhs_semantic, f_properties, mix_approx_sz) #uniform input logit diffraction - this function is probably the most important

    #vvv

    #alrights - let's redefine this - 
    #f_x is context mixer + diffractor
    #merge is transpose(0, 1) equivalent of square matrix

    for _ in range(mixed_sz):
        (lhs, rhs)      = half_split(x)
        lhs_semantic    = f_x(lhs, f_properties, lhs_approx_sz, _) #...
        rhs_semantic    = f_x(rhs, f_properties, rhs_approx_sz, _) #these are same function f_x - without loss of generality - this is dimensional reduction + cuda parallel processing - we are treating f(x) like a real function, says sorting function for instance
        new_semantic    = merge(lhs_semantic, rhs_semantic) #mix is supposed to be a recursive function - but we make things simple here
        x               = x + new_semantic  #we are doing constructive interference - Chad's rules

    return x #return mixed semantic - which is now wave-liked instead of euclidean-liked shape - this is a hint that our euclidian semantic coordinate is probably not correct - unless our operation window is on euclidean coordinate - which is an output assumption