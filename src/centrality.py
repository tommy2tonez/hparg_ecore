#
#inspired by Taylor MultiDimensional projection, we'd attempt to write a centrality approach using solely the three operations, the two sum operation, the three sum operation and the four sum operation
#we'd replace the plus operation with the two sum operation
#we'd replace the matrix multiplication operation with an operation consists of the two sum, three sum and four sum (essentially, this is also a centrality operation)

#essentially, matrix multiplication is a dimensional reduction operation with row x column -> 1 with row being the render rules, and column being the state
#we'd want to slightly change the dimensional reduction with, without loss of generality, a1 * b1 + a2 * b2 + a3 * b3 + a4 * b4 + a5 * b5 = two_sum(two_sum(two_sum(a1, b1), two_sum(a2, b2)), two_sum(a3, b3))...
#we've yet to know the exact order of those operations, yet we are 100% certain that it looks like a centrality operation, consists of a dense network to do dimensional reduction, not a sparse binary tree
#the accuracy upper bound of the dense network is essentially a n-sum approximation

#I guess the real question is a concrete mathematical formula to establish the connection between two approaches
#we never really get things for free, we are trading projection accuracy for projection storage, we need to find the exact formula for that, hint, it would look like a recursive combinatorial reduction 
#we dont have time to do 10 years of doctoral to find the exact eqn, we are lowlife folks, we run statistics, run instruments, and kind of pick the best possible combination (given the storage vs projection sufficiency curve), we'd devote 3 days on this problem

#we'd want to devote roughly 1 week to work on this problem

#we have already found our input and output semantic space for stocks, it's, essentially, a sorted order to maximize learning penalties, we are very progressive people
#the only persisting problem is the problem of logit density, we'd want to mine the logit density and kind of being as providing as possible in terms of input sufficiency  
