
#let's start another proof of concept
#in this proof of concept
#we want to approx centrality models by using state expansion approach
#and our moral compass is the time-series prediction of velocity + acceleration + jerk + crack + snap + etc.
#our A * heuristic is the best possible next move in the time-series prediction result
#our input consists of the previous moves + the f'(x) f''(x) f'''(x) f''''(x) at x = 0 - without loss of generality
#we want to explore what happened if we use centrality to improve centrality to improve centrality
#we have a week to work on this

#let's see
#let's start very simple
#we have regex rules of centrality + regex rules of training
#we have a binary input data

#our state is the model
#our adjecent states (the actionables) are the expansions of the regex rules

#we want to do random regex at first - to collect data in terms of state + f'(0), f''(0), f'''(0), f''''(0)

#the "path" algorithm that we want to use is exponential backstep + backtrack algorithm 
#we want to unhinge the "best" possible path - exponentially rewind - and remember to not step in the already explored direction
#the moral compass of which direction to explore is the "time-series" prediction - our time step is state1-state2-state3-state4 etc.

import math 

#alright let's have a scratch of what centrality means - centrality means uniform distribution of rules + propagation
#                                                      - centrality means propagate the value to the neighbor nodes
#                                                      - centrality means the algorithm must converge

#how about taylor series: taylor series can approx every differential function by knowing the differential values at 0 including f(0), f'(0), f''(0), f'''(0), f''''(0), ... - up to 256 differential orders
#our time-series prediction is actually from f'(0) to another f'(0) to another f'(0) - which is the time step
#the way we train our time-series has <support_line> + punishment based on how far down through the support lines the next move is gonna be

#in the midst of all centrality algorithms, we want to know what the next inching direction is going to be (which is the time-series forcast)
#let's work on this

#let's start very simple - without loss of generality

#we want to have a regex rule for the lhs column (for speed) and the rhs row  - and we multiply the rules for all the rows 
#then we move to something complicated - the idea of an arbitrage row by having a static bijective translation rule
#then we regex the lhs to be a random leaf or self operation (leaf|self)

#wait a second
#everything could be seen as a pair operation - an intercourse of two variables
#how about a three-variables - or n variables intercourse - well that's where taylor-series projection + calibration kicked in (EVERYTHING can be quantified as projection + calibration - which is operator add)
#we want to build our centrality on top of this concept

#-----

#why is taylor series the most important concept - because it is not only suitable for doing one dimensional projection but also n-dimensional projection approx f(x) -> y 
#without loss of generality

#assume we are doing one dimensional projection
#we need to find f(0), f'(0), f''(0), f'''(0) ... up to 256 differential orders

#assume we are doing two dimensional projection f(x, y)
#we need to find f(0, 0), f'_wrtx(0, 0), f''_wrtx(0, 0) ... up to 256 differential orders
#we also need to find f(x1, 0), f'_wrty(x1, 0), f''_wrty(x1, 0) ... up to 256 differential orders
#the f(x1, 0), f'_wrty(x1, 0) look familiar - yes - they are one-dimensional projection functions - which we've already solved 

#assume we are doing three dimensional projection f(x, y, z)
#by using induction - taylor-series is probably the most important concept in machine learning - taylor swift is, therefore, very important to approx modern infrastructure

#taylor swift + centrality is probably the proof of concept that we are going to write this week
#yall better hurry with the ballistic missiles before taylor swift smacking your asses 

def e(x: int) -> float:

    rs = float()

    for i in range(128):
        rs += 1 / math.factorial(i) * (x**i)

    return rs

print(e(12))