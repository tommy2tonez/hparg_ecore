#define DEBUG_MODE_FLAG true

#include "network_tileops_host_poly.h"
#include <vector>
#include <iostream>

int main(){

    std::vector<uint8_t> lhs(256, 1);
    std::vector<uint8_t> rhs(256, 2);
    std::vector<uint8_t> dst(256, 0);
    
    dg::network_tileops_host_poly::fwd_pair(dst.data(), lhs.data(), rhs.data(), dg::network_tileops_host_poly::make_dispatch_code(dg::network_tileops_host_poly::uuu_8_8_8, dg::network_tileops_host_poly::ops_linear));

    for (size_t i = 0u; i < 256; ++i){
        std::cout << static_cast<size_t>(dst[i]) << std::endl;
    }
}