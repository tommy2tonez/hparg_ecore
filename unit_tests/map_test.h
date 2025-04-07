#include <iostream>
#include "../src/dg_map_variants.h"
#include "../src/dense_hash_map/dense_hash_map.hpp"
#include "../src/network_datastructure.h"
#include <random>
#include <utility>
#include <algorithm>
#include <chrono>
#include <vector>
#include <functional>
#include <unordered_map>

namespace map_test{

    struct PairNullValueGen{

        constexpr auto operator()() -> std::pair<uint32_t, size_t>{
            return std::make_pair(std::numeric_limits<uint32_t>::max(), size_t{});
        }
    };

    template <class T>
    auto to_const_reference(T& value) -> const T&{

        return value;
    }

    template <template <class...> class Map, class Key, class Value>
    void test_map(){

        const size_t TEST_SZ = size_t{1} << 15;

        std::unordered_map<Key, Value> std_map{};
        Map<Key, Value> cmp_map{};

        auto seed                   = std::chrono::high_resolution_clock::now().time_since_epoch().count();
        auto key_rand_gen           = std::bind(std::uniform_int_distribution<Key>{}, std::mt19937_64(seed));
        auto val_rand_gen           = std::bind(std::uniform_int_distribution<Value>{}, std::mt19937_64(seed * 2));
        auto ops_gen                = std::bind(std::uniform_int_distribution<size_t>(0u, 11u), std::mt19937_64(seed * 4));
        auto random_clear_gen       = std::bind(std::uniform_int_distribution<size_t>(0u, size_t{1} << 20), std::mt19937_64(seed * 8)); 
        auto random_clear_sz_gen    = std::bind(std::uniform_int_distribution<size_t>(0u, size_t{1} << 20), std::mt19937_64(seed * 16));
        auto clear_gen              = std::bind(std::uniform_int_distribution<size_t>(0u, size_t{1u} << 24), std::mt19937_64(seed * 32));

        // while (true){
        for (size_t i = 0u; i < TEST_SZ; ++i){
            Key key         = key_rand_gen();
            Value val       = val_rand_gen();
            size_t ops      = ops_gen();
            size_t clear    = random_clear_gen();
            size_t do_clear = clear_gen(); 

            if (ops == 0u){
                std_map.emplace(key, val);
                cmp_map.emplace(key, val);
            } else if (ops == 1u){
                std_map.try_emplace(key, val);
                cmp_map.try_emplace(key, val);   
            } else if (ops == 2u){
                std_map.insert({key, val});
                cmp_map.insert({key, val});
            } else if (ops == 3u){
                std_map.insert_or_assign(key, val);
                cmp_map.insert_or_assign(key, val);
            } else if (ops == 4u){
                std_map[key] = val;
                cmp_map[key] = val;
            } else if (ops == 5u){
                auto std_map_ptr            = std_map.find(key);
                auto cmp_map_ptr            = cmp_map.find(key);
                bool is_valid_std_map_ptr   = std_map_ptr != std_map.end();
                bool is_valid_cmp_map_ptr   = cmp_map_ptr != cmp_map.end();

                if (is_valid_cmp_map_ptr != is_valid_std_map_ptr){
                    std::cout << "mayday" << std::endl;
                    std::abort();
                }

                if (is_valid_std_map_ptr){
                    if (std_map_ptr->second != cmp_map_ptr->second){
                        std::cout << "mayday" << std::endl;
                        std::abort();
                    }
                }
            } else if (ops == 6u){
                auto std_map_ptr            = to_const_reference(std_map).find(key);
                auto cmp_map_ptr            = to_const_reference(cmp_map).find(key);
                bool is_valid_std_map_ptr   = std_map_ptr != std_map.end();
                bool is_valid_cmp_map_ptr   = cmp_map_ptr != cmp_map.end();

                if (is_valid_cmp_map_ptr != is_valid_std_map_ptr){
                    std::cout << "mayday" << std::endl;
                    std::abort();
                }

                if (is_valid_std_map_ptr){
                    if (std_map_ptr->second != cmp_map_ptr->second){
                        std::cout << "mayday" << std::endl;
                        std::abort();
                    }
                }
            } else if (ops == 7u){
                bool std_map_contain_rs     = std_map.contains(key);
                bool cmp_map_contain_rs     = cmp_map.contains(key);

                if (std_map_contain_rs != cmp_map_contain_rs){
                    std::cout << "mayday" << std::endl;
                    std::abort();
                }
            } else if (ops == 8u){
                bool std_map_contain_rs     = std_map.contains(key);
                bool cmp_map_contain_rs     = cmp_map.contains(key);

                if (std_map_contain_rs != cmp_map_contain_rs){
                    std::cout << "mayday" << std::endl;
                    std::abort();
                }

                if (std_map_contain_rs){
                    if (std_map.at(key) != cmp_map.at(key)){
                        std::cout << "mayday" << std::endl;
                        std::abort();
                    }
                }
            } else if (ops == 9u){
                bool std_map_contain_rs     = std_map.contains(key);
                bool cmp_map_contain_rs     = cmp_map.contains(key);

                if (std_map_contain_rs != cmp_map_contain_rs){
                    std::cout << "mayday" << std::endl;
                    std::abort();
                }

                if (std_map_contain_rs){
                    if (to_const_reference(std_map).at(key) != to_const_reference(cmp_map).at(key)){
                        std::cout << "mayday" << std::endl;
                        std::abort();
                    }
                }  
            } else if (ops == 10u){
                if (std_map.size() != cmp_map.size()){
                    std::cout << "mayday" << std::endl;
                    std::abort();
                }
            } else if (ops == 11u){
                auto map_pair = std::make_pair(key, val);
                std_map.insert(&map_pair, &map_pair + 1);
                cmp_map.insert(&map_pair, &map_pair + 1);
            }

            if (clear == 0u){
                size_t dispatching_clear_sz = std::min(std::min(static_cast<size_t>(random_clear_sz_gen()), std_map.size()), cmp_map.size());
                auto it = std_map.begin();

                for (size_t i = 0u; i < dispatching_clear_sz; ++i){
                    cmp_map.erase(it->first);
                    it = std_map.erase(it);
                }
            }

            if (clear == 1u){
                size_t dispatching_clear_sz = std::min(std::min(static_cast<size_t>(random_clear_sz_gen()), std_map.size()), cmp_map.size());
                auto it = cmp_map.begin();

                for (size_t i = 0u; i < dispatching_clear_sz; ++i){
                    std_map.erase(it->first);
                    it = cmp_map.erase(it);
                }
            }

            if (do_clear == 0u){
                std_map.clear();
                cmp_map.clear();
            }
        }

        for (auto it = cmp_map.begin(); it != cmp_map.end(); ++it){
            if (it->second != std_map.at(it->first)){
                std::cout << "mayday" << std::endl;
                std::abort();
            }
        }

        if (std::distance(cmp_map.begin(), cmp_map.end()) != cmp_map.size()){
            std::cout << "mayday" << std::endl;
            std::abort();
        }

        for (auto it = cmp_map.cbegin(); it != cmp_map.cend(); ++it){
            if (it->second != std_map.at(it->first)){
                std::cout << "mayday" << std::endl;
                std::abort();
            }
        }

        if (std::distance(cmp_map.cbegin(), cmp_map.cend()) != cmp_map.size()){
            std::cout << "mayday" << std::endl;
            std::abort();
        }

        for (auto& map_pair: std_map){
            if (map_pair.second != cmp_map.at(map_pair.first)){
                std::cout << "mayday" << std::endl;
                std::abort();
            }
        }

        for (auto& map_pair: cmp_map){
            if (map_pair.second != std_map.at(map_pair.first)){
                std::cout << "mayday" << std::endl;
                std::abort();
            }
        }

        if (std_map.size() != cmp_map.size()){
            std::cout << "mayday" << std::endl;
            std::abort();
        }

        std::cout << "passed" << std::endl;
    }

    template <class Key, class Value>
    using tmp_fast_map = dg::map_variants::unordered_unstable_fast_map<uint32_t, Value, PairNullValueGen>;

    template <class Key, class Value>
    using tmp_fastinsert_map = dg::map_variants::unordered_unstable_fast_map<uint32_t, Value, PairNullValueGen>;

    void run(){

        std::cout << "__MAP_TEST_BEGIN__" << std::endl;
        std::cout << "testing dg::map_variants::unordered_unstable_map" << std::endl;
        test_map<dg::map_variants::unordered_unstable_map, uint16_t, uint16_t>();
        std::cout << "testing dg::map_variants::unordered_unstable_fast_map" << std::endl;
        test_map<tmp_fast_map, uint16_t, uint16_t>();
        std::cout << "testing dg::map_variants::unordered_unstable_fastinsert_map" << std::endl;
        test_map<tmp_fastinsert_map, uint16_t, uint16_t>();
        std::cout << "__MAP_TEST_END__" << std::endl;
        std::cout << "testing dg::map_variants::unordered_node_map" << std::endl;
        test_map<dg::network_datastructure::unordered_map_variants::unordered_node_map, uint16_t, uint16_t>();
        std::cout << "__MAP_TEST_END__" << std::endl;

    }
}

