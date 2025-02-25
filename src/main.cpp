//
#include <random>
#include <iostream>
#include <algorithm>
#include <stdint.h>
#include <stdlib.h>
#include <vector>
#include <functional>
#include <chrono>
#include <utility>
#include <array>
#include <cstring>
#include <mutex>
#include <thread>
#include <atomic>

//alright I was trying to explain memory orderings because this is a difficult topic

//assume that the mutual exclusion is responsible for a finite memory pool - such is the memory reachable by the mutex is self-induced
//then std::memory_order_acquire and std::memory_order_release should suffice

//assume that the mutual exclusion is responsible for the unique_ptr<> (without loss of generality) argument
//then std::memory_order_acquire and std::memory_order_release should transfer the responsiblity of memory synchronization in the sense of the caller and the callee pre and post the transaction should "see" the same data

//assume that the mutual exclusion is responsible for the shared_ptr<> (without loss of generality) argument
//then std::memory_order_acquire and std::memory_order_release release should transfer the responsibility of memory synchronization in the sense of pointer value and the constness of the pointing values (including virtualization header, and the const arguments of the pointing values)

//as a contrary to common beliefs, std::shared_ptr<> operator = or initialization without a lock is malicious for the reason being - it only works IF the caller has the correct pointer value (induction), 
//                                                                                                                                 - there is no virtualization header - and there is no constness of fields in the pointing structure
//                                                                                                                                 - mutual exclusion is at a fixed address whose responsibility is to serialize memory access of all the members of the structure

//people asked me why I wrote code that is the opposite with what people would do
//because the formula is to do the exact reverse of what a soy boy would do and you would be successful
//this applies to life, stock market and everything
//event-driven everything is hard to write yet it's the true formula for a massive training - we'll see someday
//point is to ingest all sliding snapshots of virtual machine buffers concurrently (L1 L2 L3 and RAM) and try to train the trees
//we'll be surprised by the browsers' data

//collecting the core data is a very hard problem - we want to implement a virtual machine - and implement our own "cache" mechanisms - and train the data based on such
//we want to run EVERYTHING on the tree - including focus + domain reduction + etc - we'll find a way

//we don't even care if the backprop is thru - we just keep ingesting realtime data
//we'd break numerical stability of gradient update yet that's another topic to explore 
//we'd hope to train these on synthetic data

//we aint selling guys - diamond hands
//let's see how this scam works
//let's show the naysayers the true neural network - not their 3rd grade kid stuff
//we've never been so close to socialism than now - in this very decade
//socialism is 10^24 bytes/ second
//we'll be there

int main(){

}