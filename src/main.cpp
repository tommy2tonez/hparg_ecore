#define DEBUG_MODE_FLAG true

// #include "stdx.h"
// #include <type_traits>
// #include "network_tileops_host_static.h"
// #include <iostream>
// #include <math.h>
// #include <utility>
// #include <functional>
// #include <algorithm> 
// #include "network_memlock_proxyspin.h"

#include <chrono>
#include <memory>
#include <stdint.h>
#include <stdlib.h>
#include <vector>
#include <random>
#include <functional>
#include <utility>
#include <algorithm>
#include <bit>
#include <iostream>
#include "network_uma_tlb_impl1.h"
#include "network_uma.h"


int main(){

    dg::network_uma::lockmap_safewait_many<1>({});
}