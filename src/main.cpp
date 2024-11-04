#define DEBUG_MODE_FLAG true

// #include <stdio.h>
// #include <stdint.h>
// #include "network_fileio_unified_x.h"
// #include <iostream>
#include "network_kernel_mailbox_impl1.h"
// #include <iostream>
// #include "stdx.h"
// #include <functional>
// #include <random> 
// #include <algorithm>

int main(){

    //there's no such thing as too hard to implement in comp-sci - if it's not easy to implement - then you implemented it wrong
    //find the joins | the splits, and dig from there - use dry approach - always - compromise your compoennt - give user a way to restart + respawn the component 
    //be very specific - don't try to solve everything - you won't solve anything if you are not specific
    //from the specific implementations build a generic solution - it's always that way - don't inverse the natural flow
    //there's no such thing as un-spaghetize a large code base - best you could do is building a dependency tree and their height (called HEADER_CONTROL) - this is to allow future engineers to replace the components by trying to decreasing the HEADER_CONTROL - the codebase that has the max(HEADER_CONTROL) < 10 is a good code_base - or as we call it no spaghetti  
    //and most importantly - get the damn thing working then try to fit the aesthetic of other people
}