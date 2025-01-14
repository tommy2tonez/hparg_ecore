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
#let's do the reverse of soy boy - and now we are mapping euclidean -> chad (whose euclidean we build from chad coordinate)
#so a constructive interference operation would look like chad_to_euclid(euclid_to_chad(x) + euclid_to_chad(x1)) - euclid_to_chad and chad_to_euclid are static functions - yet able to propagate gradients
#or frequency_to_euclid(euclid_to_frequency(x) * 2 + euclid_to_frequency(x1) * 2 + 1) - without loss of generality
#or etc.

#alright - Dad, Mom got a point - the semantic space of chad_to_euclid(euclid_to_chad(x) + euclid_to_chad(x1)) is minimal - the semantic space of frequency * 2 + frequency * 2 + 1 is maximal 
#we need to find the middle points - where is the middle point - it is the separate interference of two or more semantic spaces - point is we want constructive interference - we aren't trading destructive interference for lossy compression (though we can) by doing frequency(x) * 1.1 + other_frequency 
#if the semantic space is too large and our logit density stays the same - the required logits scale exponentially - which is a bad thing
#sandy beach is actually a good application name - we have waves, leafs are sands - we are "collecting" the semantic of the pacific ocean  

#our only use case for neural net is compression because compression is nice in terms of inputs and outputs - and the only thing we should concern about is logit density (compression_rate/ leaf_logit_byte)
#our PHD focus is building a regex engine - says the users want to mine logit density of a certain model using regex syntax
#we are the ones filling the specifics - by running convergence analysis and paths + top of the lines optimization techniques

#thank you Dad, we are adding real-time density analysis of leaf logits and using blkr to densify the logits by using heuristics 
#I dont know if that's necessary if the model has uniform influence of leaf logits and our semantic space is uniform distributed - either direction is fine
#the truth is our brain has always been doing compressions
#that's why 1 + 1 for the first grade is the equivalence of doing an integral for a high schooler - the math semantic space of the high schooler is well shaped - but the learning rate - the d logit_density/dt (derivative of gradient - we can say that) is the same or reduced or even saturated
#we want to have multi layers of compression for this precise reason - yet - logit density of the upper networks plays a VERY VERY crucial role for the compression rate of the lower networks - bad logit density of upper network means lower network cannot further compress the semantic space  
#there is also the concept of semantic space shaping - which is an advanced concept - we aren't heading there for now - we use discretized logarit uint3_t immu + multiplier for linear - says (10**3 + a) * x
#we'll be there guys - the day of 0.1% compression rate - I do think it's possible and it's simple - it's not 1024 pages of machine learning - its more like 1024 lines of heuristic and pruning

#alrights - let's talk about this scientific terms
#first, Mom is right - about having pretrained semantic mapping
#there are bad pretrained mapping and good pretrained mapping - bad pretrained mapping use tons of logits but achieve the result - good pretrained has better logit density
#Mom was talking about good pretrained and retrain
#I was talking about the bad pretrained, continuous and fixed - only used for backpropping gradients (alright - bad pretrain does not backprop gradient as well as good pretrain - we might want to have custom backward - which is weird)

#the constructive interference is just one of the very specific example that we could use
#now - let's discuss about this in-depth
#we know that if our semantic space is too large - the model is unable to learn
#or if our semantic space is not dense enough - it's hard for the model to learn continuous projection

#let's say we have uint8_t - the semantic space is 256
#let's say we have uint16_t - the semantic space is 65536
#let's say each of the context vector in the semantic space has bijective relation to the chad's coordinate
#alright - this is the hard part - we need to buff the chad's coordinate to be able to maintain the semantic relation of euclidean coordinate  by the formula: distance() == integral_diff()
#constructive interference in this case is just + operation - moving towards to (a + b) * 2 + 1
#then we want to fit this into the original space - without breaking numerical stability (extremely hard) - this is the float responsibility - they use correct exponent (or do they?) to represent this
#machine learning is, at heart, data manipulation
#we probably want to translate our language to a nicer language (by using good compression) - before training our AI

#alright - I feel lazy today - so let's talk about this for an hour - about semantic space - float, exponent and sin cos interference - and normalization of semantic space
#what the hell is attention? Attention is probably a way to have unique representation of context without doing the Chad's rules plus dimensional reduction of possibility to increase intelligent
#without attention, we have bad numerical stability of context representation - thus the model is unable to learn

#we know that float32_t is to fool people into believing that the number could represent every possible number in the universe - well it's partially true
#but float32_t semantic space is not better than uint32_t semantic space - in terms of information storage - they both maximum represent 1 << 32 context points

#yet float32_t with correct exponent (base 2 exponent) - would be able to do constructive interference without breaking numerical stability of the a and b in y = (a + b) * 2 + 1
#because float32_t use logarit discretization of euclidean space to represent semantic space
#wheras uint32_t use continuous of euclidean space (we are talking N set) to represent semantic space 
#pay very close attention to the problems I have stated and I have solved - and human kind probably stands a chance in the future
#we actually don't need 80 million training or human data - we entirely train our data on logics + synthetic data 
#we've been wasting crazy semantic space (trillion logits network) - actual AI's only 1.5GB if the datatypes are used correctly - and logit density + saturation are monitored correctly  
#the only reason we are scaling petabytes or even exabytes of network is because we are not training AI - we are training future transaction infrastructure - this level requires extreme optimizations and density mining

#alright let's talk more about increasing logit density of base operations
#we know of oval, circle and line segments
#let's talk about continuity - what the hell is continuity? continuity is the ability to draw a graph without lifting the pen
#what if we want to draw every possibilities of continuity by evenly discretizing the space into grids 

#let's say I have a grid of size 8x8 = 64
#I want to do dfs - and get every possibility of continuity then I would either do projection of space to store information or magnifization of the compact space 
#then I encode the information and make it a as in f(x) = a * x of linear
#and I make the a differentiable - by using neighbor rules

#alright - let's limit the domain and range of a function by using circle rules
#we must assume that the output space is enclosed by a circle - such is there exists a circle that contains the domain and the range
#then our grid projection must be of enclosed shape

#this is called fluid projection - fluid projection if done right can be very very impactful

#what do we learn? 
#we learn that uniform distribution of leaf logit influence is very important (transformer succeeded solely at this via destructive interference training - this is why swapping transformer layer does not change the result, all leaf logits are mostly equal)
#we learn that the attention layer enhances semantic to allow unique representation of context space to allow better y as in f(x) -> y - and decrease possibility space (which increases intellect)
#we learn that loud interference is important for destructive interferece (how we achieve this is probably via regex training)
#we learn that fluid projection increase logit density if done correctly - people wasting 8 bit space for linear semantic - we waste 8 bit space to represent every possibility of continuity - we are not the same
#we'll probably ditch linear for fluid projection - this is probably costly

#mapping x - f(x)
#this is a regex - we'll build a regex optimizer to optimize for logit density

#alright - let's look at the core of machine learning
#it's polynomial function - if you do linear 20 times - you approx accurately 20 results  - we are talking as if x is the only mutating variable in the <x, x1, x2, ...> input space
#if you do linear 50 times - you approx accurately 50 results - we are encoding the semantic space by using linear 
#then you plus the result for every layer 
#alright - that's machine learning

#how about continuity and radian coordinate? 
#first why continuity? because if things arent continuous - we aren't mapping relevant semantics - we are doing random projection of not intelligent things
#let's make things simple by thinking of every 2D projection space could be within at least one circle
#and the point we start drawing is the last point that the pen touches without lifting

#alright - there is the problem of gradient approx of the left operand
#ok - without loss of generality, we have f(x) = a * x
#we are dealing with discrete logic here - let's say that we have the gradient or the delta value being theta for f(x)
#we want to do best distance projection for a - and move a in the direction that best approxes the new value which is f(x) + theta
#and we want to have an unordered_map or some random magic function for the job
#point is instead of storing uint8_t for linear semantic - we are doing every continuous projection possible within the uint8_t space by enumerating the continuity possibilities
#we know better that those continuous functions could be described by using mathematical continuous methods - yet it does not have good numerical stability 
#why radian coordinate? well - because it best describes continuity without infinity and friends

#so we have learnt the basics of data information - it's entropy, continuity
#in the high entropy semantic space - every continuous function trades 1 for 1 - 1 leaf logit for 1 output logit
#in the low entropy semantic space (which is continuous) - the logit density is infinity

#machine learning is deentropizer - we are moving from high entropy semantic space -> low entropy semantic space
#we adhere to the rule of minimum wage via mathematical formula, whether it is Chad's or polynomial
#1 leaf logit for 1 output logit
#the extra is fries

def f(x: list[bool], f_properties: object) -> list[bool]:

    return project(pos(x, f_properties), f_properties) #one of pos(x) responsibility is semantic space mapper - mapping from irrelevant space -> euclidean relevant space - limit (distance -> 0) in euclidean coordinate means semantically equivalent 
                                                       #second of pos(x) responsibility is to bijectively rearrange x input space -> (discretized n)! space - 1234 -> 3214, 2134, 4321, etc. 

#trade density for intelligence
def f_x(x: list[bool], f_properties: object, approx_sz: int) -> list[bool]:

    func_vec        = [f(x, f_properties, cursor) for _ in range(approx_sz)]
    inclusive_vec   = static_exclusion(func_vec, initial_input_unique_representation, f_properties) #leaf logits diffraction

    return sum(inclusive_vec) #this is definitely constructive interference  #stacking f(x) - gradients of f(x) and f1(x) and f2(x) f3(x) are the trainingly equivalents - these are semantically equivalent groups - we want uniform training rate for the group - or combinatorial blkr to approx the best results 

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
        x               = x + new_semantic  #we are doing constructive interference - Chad's rules - now I want you to think whether destructive interference is required or only loud constructive interference is required (if loud constructive interference is allowed - think about how we could achieve that)

    return x #return mixed semantic - which is now wave-liked instead of euclidean-liked shape - this is a hint that our euclidian semantic coordinate is probably not correct - unless our operation window is on euclidean coordinate - which is an output assumption