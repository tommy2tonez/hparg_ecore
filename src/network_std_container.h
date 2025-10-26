#ifndef __DG_NETWORK_STD_CONTAINER_H__
#define __DG_NETWORK_STD_CONTAINER_H__

//define HEADER_CONTROL 5

#include <type_traits> 
#include "network_type_traits_x.h"
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <string>
#include "network_allocation.h"
#include "network_datastructure.h"
#include "network_hash_factory.h"

namespace dg::network_std_container{

    template <class T>
    using container_hasher          = dg::network_hash_factory::default_hasher<T>;

    template <class T>
    using container_equal_to        = dg::network_hash_factory::default_equal_to<T>;

    template <class T>
    using unordered_set             = std::unordered_set<T, container_hasher<T>, container_equal_to<T>, dg::network_allocation::NoExceptAllocator<T>>;

    template <class T>
    using unordered_unstable_set    = dg::network_datastructure::unordered_map_variants::unordered_node_set<T, std::size_t, container_hasher<T>, container_equal_to<T>, dg::network_allocation::NoExceptAllocator<T>>;

    template <class T>
    using cyclic_unordered_node_set = dg::network_datastructure::unordered_map_variants::cyclic_unordered_node_set<T, std::size_t, container_hasher<T>, container_equal_to<T>, dg::network_allocation::NoExceptAllocator<T>>;

    template <class Key, class Value>
    using unordered_map             = std::unordered_map<Key, Value, container_hasher<Key>, container_equal_to<Key>, dg::network_allocation::NoExceptAllocator<std::pair<const Key, Value>>>;

    // template <class Key, class Value>
    // using unordered_unstable_map    = dg::network_datastructure::unordered_map_variants::unordered_node_map<Key, Value, std::size_t, std::integral_constant<bool, true>, container_hasher<Key>, container_equal_to<Key>, dg::network_allocation::NoExceptAllocator<std::pair<const Key, Value>>>;

    template <class Key, class Value>
    using unordered_unstable_map    = std::unordered_map<Key, Value, container_hasher<Key>, container_equal_to<Key>, dg::network_allocation::NoExceptAllocator<std::pair<const Key, Value>>>;

    template <class Key, class Value>
    using unordered_unstable_map2   = dg::network_datastructure::unordered_map_variants::unordered_node_map<Key, Value, std::size_t, std::integral_constant<bool, true>, container_hasher<Key>, container_equal_to<Key>, dg::network_allocation::NoExceptAllocator<std::pair<const Key, Value>>>;

    
    template <class Key, class Value>
    using cyclic_unordered_node_map = dg::network_datastructure::unordered_map_variants::cyclic_unordered_node_map<Key, Value, container_hasher<Key>, std::size_t, std::integral_constant<bool, true>, container_equal_to<Key>, dg::network_allocation::NoExceptAllocator<std::pair<const Key, Value>>>;

    template <class T>
    using vector                    = std::vector<T, dg::network_allocation::NoExceptAllocator<T>>;

    template <class T>
    using deque                     = std::deque<T, dg::network_allocation::NoExceptAllocator<T>>;

    template <class T>
    using pow2_cyclic_queue         = dg::network_datastructure::cyclic_queue::pow2_cyclic_queue<T, dg::network_allocation::NoExceptAllocator<T>>;

    using string                    = std::basic_string<char, std::char_traits<char>, dg::network_allocation::NoExceptAllocator<char>>;
}

namespace dg{

    template <class Key>
    using unordered_set             = dg::network_std_container::unordered_set<Key>;

    template <class Key>
    using unordered_unstable_set    = dg::network_std_container::unordered_unstable_set<Key>;

    template <class Key>
    using cyclic_unordered_node_set = dg::network_std_container::cyclic_unordered_node_set<Key>;

    template <class Key, class Value>
    using unordered_map             = dg::network_std_container::unordered_map<Key, Value>;

    template <class Key, class Value>
    using unordered_unstable_map    = dg::network_std_container::unordered_map<Key, Value>;

    template <class Key, class Value>
    using unordered_unstable_map2    = dg::network_std_container::unordered_unstable_map2<Key, Value>;

    template <class Key, class Value>
    using cyclic_unordered_node_map = dg::network_std_container::cyclic_unordered_node_map<Key, Value>;

    template <class T>
    using vector                    = dg::network_std_container::vector<T>;

    template <class T>
    using deque                     = dg::network_std_container::deque<T>;

    using string                    = dg::network_std_container::string;

    template <class T>
    using pow2_cyclic_queue         = dg::network_std_container::pow2_cyclic_queue<T>;
}

#endif