#define DEBUG_MODE_FLAG true
#define STRONG_MEMORY_ORDERING_FLAG true

#include <iostream>
#include "network_datastructure.h"

int main()
{
    dg::network_datastructure::unordered_map_variants::cyclic_unordered_node_set<std::string> set(32);

    intmax_t offset = 0; 
    intmax_t lookback_sz = 32;
    intmax_t incremental_sz = 16;
    bool break_value;

    while (true)
    {
        for (size_t i = 0u; i < lookback_sz; ++i)
        {
            intmax_t value = (offset - 1) - i;
            std::cout << "finding value " << value << "<>" << set.contains(std::to_string(value)) << std::endl; 
        }

        std::cin >> break_value;
        
        for (size_t i = 0u; i < incremental_sz; ++i)
        {
            size_t nxt_value = offset + i;
            std::cout << "inserting value "<< nxt_value << std::endl;
            set.insert(std::to_string(nxt_value));
        }

        std::cout << "-------------------------------------------" << std::endl;
        offset += incremental_sz;
    }
}