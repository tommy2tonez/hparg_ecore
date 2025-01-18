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
#in a high entropy semantic space - every continuous function trades 1 for 1 - 1 leaf logit for 1 output logit
#in a low entropy semantic space (which is continuous) - the logit density is infinity
#alright - we are talking in terms of enumerated continuous space (uint8_t) - we can actually have high entropy continuous space - which is machine learning (or what we are trying to learn)
#what we choose to be enumerated continuous functions would determine the success of our base operations - this is a very important note - if we aren't getting the enumerated continuous functions right - we are wasting fries

#machine learning is deentropizer - we are moving from high entropy semantic space -> low entropy semantic space
#we adhere to the rule of minimum wage via mathematical formula, whether it is Chad's or polynomial
#1 leaf logit for 1 output logit
#the extra is fries

#alright - so we have learnt the basics of semantic mapper - crypto or machine learning or huffman encoding or gzip
#the theory is simple - assume that we have the semantic mapper of size N (leaf logit size)
#the space complexity of deentropizer of linear size O(N)
#if the secret key is 256 bytes - then we need 256 + C output token bytes to accurately estimate the secret key (C is the extra semantic - including the programming language semantic space + source code + overheads + etc.)
#this is precisely the reason for our invention of uniform distribution encoding method
#we also have learnt that the only thing that's been wasting money is destructive interference of learning (we ziggle the polymorphic eqn by 0.00001 and all the semantic changes) - we want to offset this cost accurately by picking pivots (not fries) and right heuristics (this is another research topic) 
#the destructive interference of learning is so bad that scaling the transformer - polynomial order - would result in worser model (either the model regex is wrong or the training is wrong or model and training are wrong)
#we call this guided model training (this is an entire another research topic)
#we'll probably have clients to do bid/ask for logit density mining - for many purposes - crypto crack/ intellect/ semantic compression/ etc.
#this is logarit regex mining - we'll be there
#apache is nice - we'll be talking about session encoding + decoding directly via network traffic later - constant time

#alright - let's admit one thing - we are monkeys - we speak in simple terms so let's make this simple
#assume we are in the coordinate of x = <input, output>, y = <deviation>
#polynomial order of 10 = 10 correctly predicted deviation

#alright - let's talk in terms of uncertainty - let's say that the accepted window of output deviation is 0.1
#and our flex grid is 0.1 (or resoluton) - and the x axis that is within the circle has the length of 10
#then our maximum correctly predicted output is 100
#polynomial order of 10 flex = 1000 correctly predicted deviation
#why? because it is (x-1)(x-2) * (x-3)(x-4)
#and multiplication acts as a lossless compressor in this case
#alright - so what have we been talking about? we are talking about linear wasting resolution (alright - this is informationally speaking debatable - because there are methods to offset this cost)
#and we talking about linear can't get us fries - because linear only stores linear semantics 
#and we are talking about locality compression - linear is sufficient to compress EVERY semantic possible that are not semantically relevant (not local) but not sufficient to compress local context
#locality compression is a very major concept in information technology - because every possible problem in Computer Science can be quantified as locality and affinity problems
#we have just talked about the very minute subset of locality compression which is knot theory  

#what is a good example of locality compression? it's permutational reflection
#assume an array (image) of arr = [1, 2, 3, 4, 5, 6]
#we want to store a = [2, 1, 3]
#so we store the suffix and remove the suffix out of the array iteratively, which algorithm has the result space complexity of n! / (n - size(a))! 
#so instead of storing uint8_t * size(a) - we are storing at max n! decimal space for sizeof(a) 
#let's talk in terms of logit density - we have reduced the logit density by a factor of uint8_t * size(a)/ byte_size(n! decimal space) 
#so what are we doing? we map it to another semantic space - we leverage the fact that <continuity and our defined enclosed space> means that everything can only be passed once (it is <x, y> pair not y or not x) - and we enumerate every permutational possibilities  

#let's reuse the locality algorithm that we mentioned in the stock trading ballinger
#we change the semantic coordinate of the continuous projection -> chart compression
#the lossy compression ratio is ln(x!) / (ln(256) * x) for x is the number of projection and we are storing x on uint8_t     
#we have suffix compression of continuous chart
#we are aiming for 90% compression rate compared to linear - so we are looking at 3 projection points - because too few projection points would not correctly describe the chart - and too many projection points would stretch our semantic space -> the original space

#let's assume our continuous locality compression is dfs - of 3 bits - up, down or stays - this is dfs
#there are 1024 different ways to do locality compression
#consider this case - 1 - 3, 2 - 4, 3 - 3, 4 - 2
#then instead of doing 8 bits for random range, etc.
#pay very close attention to the things I mentioned about locality and continuity and enumeration

#is there a better locality algorithm? - this question is up for debate
#let's assume the set of our defined continuous function is U
#let's evenly discretize 256 different continuous shapes - and we want to move from one shape to another (shape shifter) to fit the projection pattern
#what the hell is evenly discretize? It is euclid distance = integral_difference * C for all pairs + each point has a bijective relation with a matrix's cell that spans the range + domain of the output space 
#let's logaritally discretize 256 different continuous shapes

#what is the other mathematically lossless compression? it's Chad 
#why should we use Chad compression (full constructive interference) over other compression (linear)? (1): because Chad has loud interference (which replaces destructive interference), (2): we can described Chad's in euclidean coordinate by using integral_difference() == euclidean_difference()
#in other words, Chad compression has a more stable approach than linear compression
#we assume

#what really is the struggle between choosing the right lossless compression method and still getting the fries? 
#imagine this eqn (x-1)(x-2)(x-3)(x-4)(x-5) - this is polynomial
#you ziggle 1 -> 0.9 - and literally everything changes semantically
#is there a way that we could circumscribe the affected area to limit destructive interference of training?

#imagine this eqn sin(x) + sin(2x) + sin(3x) + sin(4x) + sin(5x)
#you ziggle 1 -> 0.9.
#it still changes everything semantically - but ... there is the importance of semantic (by magnitude of 1, 2, 3, 4, 5 or loudness (amplitude)) - and the new result is not very deviated from the old result
#alright - we have just spoken of two lossless compression methods that we know of - there are probably other better ways to do this - which is a research topic of regex mining problems

#so what the hell do we want? we want integral difference of delta(I) = I1 - I2 
#and the distribution of integral difference - it must be either uniform distribution on the domain or local distribution - we need to come up with a synthetic way to do this
#so machine learning is essentially lossless compression + minimum wages + fries + Chad + semantic reduction (we want to reduce semantic window of the, WLOG, linear operation into operatable window to leverage locality compression of projections) + training ziggle integral difference distribution (uniform or locally skewed) + the projectable space (flash exp + linear would increase the range dramatically)
#after we got the regex for all of that right - we'll begin to implement other advanced algorithms to do logit density mining

#alright - we want to talk about polynomial without exp() and polynomial with exp()
#the difference is the polynomial with exp() can correctly predict skewed semantic context space where polynomial without exp() can not

#let's open desmos and see the difference between (x-1) * (x-2) * (x-3) * (x-4) and (x-1) * (x-2 + e^x) * (x-3) * (x-4)
#the problem with polynomial prediction is it takes incredibly long training time to train the polynomial to look reasonable
#til the point one could argue that if the integral difference distribution is local uniform or local skewed - then it would offset the cost tremendously - we probably want destructive interference for this 
#because most of the training has destructive interference and it moves very slowly in the right direction

#thing is we want to flex the graph to touch the points without exponent interference and we probably want to preprocess our semantic space to look round + continuous + reasonable
#let's go back to our original formula of unordered set of zeros - we are looking at the semantic space of decimal size n!/((n - x)! * x!) - compared to the space of byte size (uint8_t) * n in the linear case
#let's say we want to encode the shapes - we would want to leverage our recursive ballinger compression - suffix array of f -> suffix array of f' -> suffix array of f'', etc.

#we want to include continuity - which is back to problem of enclosed same area - different shape - euclid_distance() == integral_difference() - this is another research topic
#we want to keep the enumeration of shapes small - within 256 or 1024 - because otherwise - we are not making semantic connections (we are not getting fries - we are storing irrelevant data)
#Chad's posulate: in the most compressed semantic coordinate, there is at least one possibility that things could be continuous. Chad says he does not which set of shapes, it is just continuous  

#alright - we'll implement this

#we'll work more on the theory today
#there's a curse between lossless compression and fries
#the more fries you get, the lesser the lossless compression gonna be
#because you store the information as shapes as trade off for lossless compression
#but the information theory stays the same
#1 leaf logit for 1 output logit (on average)

#consider this polynomial function (x-1)(x-2)(x-3)(x-4)
#4 outputs for 4 linears
#now we want to flex the shape at x = 1.8 to get y = -2 to get the fries - we can't because it violates information technology - you have reached the maximum storage for irrelevant data
#not the same in the sense of flex shape - flex shape gets nothing right individually - but together - they get things right - this is uniform distribution of responsibility
#so flex shape is better on average (in the sense of best fit line - because it stores information as continuous shape or semantics - not absolute lossless information like linear - well we can say that flex shape stores absolute information like linear in the sense of shape - but we aren't getting into the recursion of it all)

#alright - so we talk about locality compression - which is continuity in general, not linear, not sin, not cos, not exp, not ln, not mathematical eqns
#consider we want to store polynomial set of 2 zeros on the domain [0, 256] and we want to store those on the domain [0, 8]
#which can be expressed as (x - a) * (x - b) or n!/((n - x)! * x!)
#the difference is from the "set" - not "linear" problem
#alright things get confused for locality and set theory here
#but wait, if we are storing data as zeros and the function is continuous, we probably dont want to repeat the zeros (though we can - two zeros mean something semantically different)

#goal of machine learning is to maximize locality compression (we want to be able to store two zeros on the [0, 8] rather than two zeros on the domain [0, 256])
#the ultimate goal is to get the set of flex shapes correctly - and shift from one shape to another by using heuristic - not mathematically eqns (left operand)
#flex shape also allows smoother training - in the sense that a local changes wont affect the function globally like linear which looks like a variant of plus 1/x * g(x)

#alright - what if we go full regard mode and flex everything?
#this is where the trinity is from - flex is a trinity operation, flex(lhs, rhs, flex_shape) = y - we are on radian so lhs and rhs are theta and alpha - being the xy radian and xz radian - and the flex shape is somewhat a spheroid
#the propagation of gradient -> lhs, rhs, and flex_shape are tedious - we want fast approximator for these
#the flex heuristic function is also tedious - assume f(x, y, z) -> f'(x, y, z), we want heuristic(z) = z' and f(x, y, z') = f'(x, y, z) 
#theoretically - this works - but the training process usually says otherwise - if the shape is too flex - we aren't getting the minimum wages 
#says that the shapes are all plus operations - then we aren't actually learning anything 
#so getting the set of flex shapes right is actually a topic worth 4 years of PHD research
#I just make things simple and do fixed volume - and kinda twist the shapes in different directions - we talked about integral difference and integral difference distributions - we kinda want to move in the direction
#let's actually dwell on the ideas for 2-3 days before we start this project (we need to get the ideas right - and prove that this can approx every possible continuous function - or there exists an optimal solution by using flex projections)

#like flex is the only trinity operation we have beside constructive interference pair operation which is add - what is the secret of the add operation is mathematic? it seems like add is the only operation that we can't project
#it seems like add is the coordinate calibration function - such that it changes the semantic coordinate, adder with respect to the addee, or addee with respect to the adder, not necessarily projecting anything
#alright so what's the difference between f(x, y) = x + y and f(x, y) - x = y
#this is numerical stability issue that we would want to talk about - or is it?
#let's throw away all mathematical projection operations like exp, log, sin, cos, inverse, sqrt, etc.
#we must keep lossless compression operation mul and coordinate calibration operation add  
#because without add - we stuck in one context space - we aren't resoluting the context space (calibrating the space layers by layers)
#without mul - we are not getting the lossless compression - which is bad numerically speaking - this is up for debate - things can be done without mul (this is back to the Chad's shape + add - sin(x) + sin(2x) + sin(3x), etc.)
#so the three operations are mul, add, and flex - which are the tri-force

#we only care about uniform distribution of leaf logit on the result
#such is a random leaf logit influence on a random result logit is the same for all possible pair (before training)
#to be able to understand - we must unlearn the mathematics - for all that math expresses is continuity and locality compression
#so our new ultimate goal is actually good locality compression (another word for math operations) by using right set of flex, regex of initial seeds, uniform distribution of influence, and radian coordinate 
#and good heuristic approximation + path optimizations
#if you got these rights in a 1990 computer - you'd unlock crypto secret keys in constant time - that's literally the danger we are facing right now

#alright
#consider input logits, leaf logits and output logits
#each input logit must have uniform influence on output logit
#each leaf logit must have uniform influence on output logit
#we must be able to prove that if a function is continuous - then it most compact form can be expressed as continuous functions - we'll talk about this later

#let's get back to the string theory - what does string theory state? that the universe is made of invisible semantic strings
#string is another word for continuity and semantic space, we add those strings together - which is semantic space calibration
#so our trinity flex operation is f(x, y, flex_shape) = <x, y, z> in euclidean coordinate
#we approx the string's shape - we stretch the string in x, y, z, and we do semantic calibration 
#this is the clearest proof that string theory could be one of the possibility of what the universe is made of

#let's work on the theory of shape estimation today
#assume we have two operations: add and flex
#let's assume we only have the operation flex (or we are looking at the flex only of the flex and add)
#without loss of generality, we have f(x) = x, and we kinda want to twist the f(x), to do bucket propagation of unordered range 
#because flex1(flex2(flex3(x))) would turn the range into unordered eventually
#so we want to look at this as unordered range propagation and n! permutation sort
#alright this is not good - because flex1(flex2(flex3(x))) slowly turns our "semantic information" into unordered range of information - or destructive interference with respect to initial purposes

#this is precisely why we need semantic calibration which is the add operation between flex operations
#so instead of doing flex1(flex2(flex3(x))) - we are doing flex1(flex(x) + x)) + ...
#so let's say we have 30 strings that made up the universe
#we guess the first string out of 30 strings, we guess the second string out of 29 strings, we guess the third string out of 28 strings, etc.
#we are collapsing the semantic spaces(strings) iteratively, what could possibly go wrong with this?
#the first string and the last string has a very unbalanced workload - we aren't doing uniform distribution of leaf logits - this is a very important optimizable
#alright what does this mean? do we do plus 30 different layers? No - because we aren't collapsing the spaces by doing that
#we want to reduce the shape resolution exponentially from the first layer -> the last layer
#let's say that our flex shape can represent 256 shapes for the last layer
#our flex shape can represent 512 shapes for the next to last layer
#we don't want to compromise same resolution for the first -> last layer and risk saturated leaf logits
#the first string contains 2x more information than the 2nd string
#second string contains 2x more information than the 3rd string
#etc.
#if we set this up right - we can do this

#alright - back to the question of flat earthers and round eathers - we have combinatorial operation, whose row is render rule and column is state
#linear render rule is simple - so we can describe it separately (in the sense of pair operation and path)
#"spheroid" render rule is not (we cant describe it as pair operation or trinity operation or etc.)

#alright - back to the original mathematical rules
#if we aren't doing self operation (x linear x' - attention paper), we aren't getting the lossless compression - polynomial order is 1
#let's move the focal to x1 (x1 is the only variable, all others are constants) as for x = <x1, x2, x3, x4, x5, ...>
#then y = a * x1 + b for y is a random cell in the output matrix - without self operation
#so it's important to do self operation - to increase polynomial order which helps with lossless compression 
#alright - we've been talking about rope - rotational positional embedding - it's essentially Chad (or flex Joe) that we described - we are turning linear relation to sin cos constructive interference which represents the context better 
#so instead of y = a * x1 + b, we are looking at y = sin(2x1) + sin(3x1) + ax1 + b, etc.
#so that's why rope is in the attention layer
#with our new invention, we are looking at y = flex2(flex1(x1) + x1 + C ...
#we are collapsing the 2-dimensional string (continuous function) layer by layer - by using multi dimensional flex function (which is on radian coordinate without loss of generality)
#alright - continuity can't be easily described - Mom talked about knot theories - this is advanced topic - if there are knots - we aren't on radian coordinate
#there are problems with numerical stabilities - described above with flex(flex(flex)) would turn things into unordered range compression - we need numerical stability from first layer -> output layer (because that's what we can really count on) - not in-between layers
#things like y = ax + b or y = ax + sin(x) + sin(2x) + b
#but the ideas remain the same
#we want to collapse the semantic spaces iteratively - by using flex shapes of continuity (which is locality compression) - and offset the cost of destructive interference

#it's best for flex1 to have better resolution than flex2, which has better resolution than flex3
#because the first string is going to create ripple effect for the following strings - and if we aren't getting things right (or stuck between the right things) - drawing a best-fit linear line between irrelevant points - we cause a learning ripple effect - which has destructive interference (and move towards equilibrium of uniform responsibility to avoid ripple effects which is part of the learning - thus swapping layers of transformers won't affect results)
#alright so we are not only learning the input -> output, we are also learning to avoid ripple effects in transformers - this is usually the problems - there aren't enough logits to learn to avoid that - we need to offset this cost
#we are now back to the 2x denser continuity - what does that even mean? it probably means that we use 2x as much shapes (or 2x resolution) to "not stuck between the right things" or "less stuck between the right things" - alright (what are usually the right things and how we enumerate those are very hard to get right)

#so our only concerns in the two above cases are enumerated flex shapes 
#and our f(x, y) = x + y vs f(x, y) - x = y. What's the difference? 
#one is fake semantic calibration, other is true semantic calibration - we want true calibration for flex shapes - because we already talked about flex(flex(flex(x))) would destroy our flexibility - in the sense of turning the result in to unordered range compression
#is f(x, y) = x - (-y), possibly with correct shapes? or is there a way to force calibrations? limit the transformation? this question is up for debate - imma just put the question mark here - for now - the add is the coordinate calibration operator

#enough theory - we have to see if this actually works

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

    for _ in range(mixed_sz):
        (lhs, rhs)      = half_split(x)
        lhs_semantic    = f_x(lhs, f_properties, lhs_approx_sz, _) #...
        rhs_semantic    = f_x(rhs, f_properties, rhs_approx_sz, _) #these are same function f_x - without loss of generality - this is dimensional reduction + cuda parallel processing - we are treating f(x) like a real function, says sorting function for instance
        new_semantic    = merge(lhs_semantic, rhs_semantic) #mix is supposed to be a recursive function - but we make things simple here
        x               = x + new_semantic  #we are doing constructive interference - Chad's rules - now I want you to think whether destructive interference is required or only loud constructive interference is required (if loud constructive interference is allowed - think about how we could achieve that)

    return x #return mixed semantic - which is now wave-liked instead of euclidean-liked shape - this is a hint that our euclidian semantic coordinate is probably not correct - unless our operation window is on euclidean coordinate - which is an output assumption