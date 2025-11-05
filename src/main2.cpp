#define DEBUG_MODE_FLAG true
#define STRONG_MEMORY_ORDERING_FLAG true

#include <iostream>
// #include "network_datastructure.h"
#include "network_kernel_allocator.h"
#include "network_kernel_buffer.h"
#include "network_kernel_mailbox_impl1.h"

int main()
{
    using namespace dg::network_kernel_mailbox_impl1::allocation;

    dg::network_kernel_mailbox_impl1::allocation::init(dg::network_kernel_mailbox_impl1::allocation::Config{.total_mempiece_count = 1 << 10, .mempiece_sz = 1 << 10, .affined_refill_sz = 1 << 8, .affined_mem_vec_capacity = 1 << 8, .affined_free_vec_capacity = 1 << 8});

    internal_kernel_buffer buf{};
    buf = std::string_view("Hello World");

    std::cout << std::string_view(buf) << std::endl;
    // auto allocator = dg::network_kernel_allocator::ComponentFactory::make_affined_map_allocator({}, {}, {}, {});
    // auto buf = dg::network_kernel_buffer::kernel_string();
    // buf.size();
    // auto buf2 = buf;
    // buf.at(0);
    // buf[0];
    // buf.front();
    // buf.back();
    // buf.begin();
    // buf.end();
    // buf.cbegin();
    // buf.cend();
    // buf.data();
    // buf.reserve({});
    // buf.resize({});
    // buf.capacity();
    // buf.clear();
    // buf.push_back({});
    // buf.pop_back();
    // buf.swap(buf2);
    
    // dg::network_datastructure::unordered_map_variants::cyclic_unordered_node_set<std::string> set(32);

    // intmax_t offset = 0; 
    // intmax_t lookback_sz = 32;
    // intmax_t incremental_sz = 16;
    // bool break_value;

    // while (true)
    // {
    //     for (size_t i = 0u; i < lookback_sz; ++i)
    //     {
    //         intmax_t value = (offset - 1) - i;
    //         std::cout << "finding value " << value << "<>" << set.contains(std::to_string(value)) << std::endl; 
    //     }

    //     std::cin >> break_value;
        
    //     for (size_t i = 0u; i < incremental_sz; ++i)
    //     {
    //         size_t nxt_value = offset + i;
    //         std::cout << "inserting value "<< nxt_value << std::endl;
    //         set.insert(std::to_string(nxt_value));
    //     }

    //     std::cout << "-------------------------------------------" << std::endl;
    //     offset += incremental_sz;
    // }
}