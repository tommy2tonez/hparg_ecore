#ifndef __GRAPH_OPTIMIZER_H__
#define __GRAPH_OPTIMIZER_H__

#include <stdint.h>
#include <stdlib.h>
#include <algorithm>
#include <utility>
#include "network_datastructure.h"
#include "sort_variants.h"
#include "network_hash_factory.h"

namespace graph_optimizer
{
    struct BinaryFCDEdgeInformation
    {
        uint32_t src;
        uint32_t dst;
        double score;
    };

    struct BinaryFCDResult
    {
        std::vector<uint32_t> vertices;
    };

    template <class FloatType, class UnsignedType>
    class FloatNormalizer
    {
        private:

            FloatType first;
            FloatType last;
            FloatType discrete_sz;

            static_assert(std::is_floating_point_v<FloatType>);
            static_assert(std::is_unsigned_v<UnsignedType>);

            static inline constexpr FloatType MIN_DISCRETE_SZ   = std::numeric_limits<FloatType>::min();

        public:

            constexpr FloatNormalizer(FloatType first, FloatType last)
            {
                if (std::isnan(first))
                {
                    throw std::invalid_argument("bad float normalizer's first, not a number");
                }

                if (std::isinf(first))
                {
                    throw std::invalid_argument("bad float normalizer's first, inf");
                }

                if (std::isnan(last))
                {
                    throw std::invalid_argument("bad float normalizer's last, not a number");
                }

                if (std::isinf(last))
                {
                    throw std::invalid_argument("bad float normalizer's last, inf");
                }

                if (last < first)
                {
                    throw std::invalid_argument("bad float normalizer range, < 0");
                }

                this->first                     = first;
                this->last                      = last;
                FloatType tentative_discrete_sz = (last - first) / std::numeric_limits<UnsignedType>::max();
                this->discrete_sz               = std::max(tentative_discrete_sz, MIN_DISCRETE_SZ); 
            }

            constexpr auto operator()(FloatType value) noexcept -> UnsignedType 
            {
                return (value - this->first) * this->discrete_sz;
            }
    };

    class BinaryFCDOptimizer
    {
        private:

            template <class T>
            using default_hasher        = dg::network_hash_factory::default_hasher<T>;

            template <class Key>
            using local_unordered_set   = dg::network_datastructure::unordered_map_variants::unordered_node_set<Key, uint32_t, default_hasher<Key>>;  

            template <class Key, class Value>
            using local_unordered_map   = dg::network_datastructure::unordered_map_variants::unordered_node_map<Key, Value, uint32_t, std::integral_constant<bool, true>, default_hasher<Key>>;

            struct BinaryClassifyResult
            {
                std::vector<uint32_t> flat_community_vec;
                std::vector<std::pair<uint32_t, uint32_t>> flat_community_range_vec;
            };

        public:

            auto optimize(const std::vector<BinaryFCDEdgeInformation>& edge_vec, size_t optimization_step = 32u) -> BinaryFCDResult
            {
                std::vector<BinaryFCDEdgeInformation> filtered_edge_vec = this->to_edge_vec(this->to_edge_graph(edge_vec));
                BinaryClassifyResult community_result = this->to_initial_group(this->to_initial_community(filtered_edge_vec));

                for (size_t i = 0u; i < optimization_step; ++i)
                {
                    if (community_result.flat_community_range_vec.size() <= 1)
                    {
                        break;
                    }

                    size_t old_sz                   = community_result.flat_community_range_vec.size();
                    BinaryClassifyResult nxt_bcr    = this->binary_classify(filtered_edge_vec, community_result.flat_community_vec, community_result.flat_community_range_vec); 
                    size_t new_sz                   = nxt_bcr.flat_community_range_vec.size();
                    bool is_progressing             = old_sz != new_sz;

                    if (!is_progressing)
                    {
                        break;
                    }

                    community_result                = std::move(nxt_bcr);
                }

                return BinaryFCDResult
                {
                    .vertices = std::move(community_result.flat_community_vec)
                };
            }

        private:

            auto to_edge_graph(const std::vector<BinaryFCDEdgeInformation>& edge_vec) -> local_unordered_map<std::pair<uint32_t, uint32_t>, double>
            {
                local_unordered_map<std::pair<uint32_t, uint32_t>, double> rs{};

                for (const auto& edge: edge_vec)
                {
                    if (std::isnan(edge.score))
                    {
                        throw std::invalid_argument("bad edge score, not a number");
                    }

                    if (std::isinf(edge.score))
                    {
                        throw std::invalid_argument("bad edge score, inf");
                    }

                    uint32_t src_vtx    = std::min(edge.src, edge.dst);
                    uint32_t dst_vtx    = std::max(edge.src, edge.dst);
                    auto new_edge       = std::make_pair(src_vtx, dst_vtx); 

                    rs[new_edge]        = edge.score;
                }

                return rs;
            }

            auto to_edge_vec(const local_unordered_map<std::pair<uint32_t, uint32_t>, double>& graph) -> std::vector<BinaryFCDEdgeInformation>
            {
                std::vector<BinaryFCDEdgeInformation> rs(graph.size());
                size_t i = 0u;

                for (const auto& map_pair: graph)
                {
                    const auto& [src_vtx, dst_vtx] = map_pair.first;
                    rs[i++] = BinaryFCDEdgeInformation
                    {
                        .src    = src_vtx,
                        .dst    = dst_vtx,
                        .score  = map_pair.second
                    };
                }

                return rs;
            }

            auto to_initial_community(const std::vector<BinaryFCDEdgeInformation>& edge_vec) -> local_unordered_set<uint32_t>
            {
                local_unordered_set<uint32_t> rs{};

                for (const auto& edge: edge_vec)
                {
                    rs.insert(edge.src);
                    rs.insert(edge.dst);
                }

                return rs;
            }

            auto to_initial_group(const local_unordered_set<uint32_t>& vertex_set) -> BinaryClassifyResult 
            {
                std::vector<uint32_t> rs(vertex_set.size());
                std::vector<std::pair<uint32_t, uint32_t>> rs_range(vertex_set.size());

                size_t i = 0u;

                for (uint32_t vertex: vertex_set)
                {
                    rs[i]       = vertex;
                    rs_range[i] = {i, i + 1};
                    i           += 1;
                }

                return BinaryClassifyResult
                {
                    .flat_community_vec         = std::move(rs),
                    .flat_community_range_vec   = std::move(rs_range)
                };
            }

            auto group_size(const std::pair<uint32_t, uint32_t>& first_last_range) -> size_t
            {
                return first_last_range.second - first_last_range.first;
            }

            auto binary_classify(const std::vector<BinaryFCDEdgeInformation>& edge_vec,
                                 const std::vector<uint32_t>& flat_community_vec,
                                 const std::vector<std::pair<uint32_t, uint32_t>>& flat_community_range_vec) -> BinaryClassifyResult 
            {
                local_unordered_map<uint32_t, std::pair<uint32_t, uint32_t>> reverse_map{};
                reverse_map.reserve(flat_community_vec.size());

                for (size_t i = 0u; i < flat_community_range_vec.size(); ++i)
                {
                    const auto& [first, last] = flat_community_range_vec[i];

                    for (size_t j = first; j < last; ++j)
                    {
                        uint32_t node_idx       = flat_community_vec[j];
                        uint32_t community_idx  = i; 
                        reverse_map[node_idx]   = {community_idx, last - first};
                    }
                }

                local_unordered_map<std::pair<uint32_t, uint32_t>, double> community_score_map{};

                //we know of an optimzation, that is called the hierarchical cache access, to make sure that the base cache access is within the 64KB boundaries to avoid cache misses
                //this is particularly useful in this case of cache access, where we'd literally touch all the possible src_vtx and dst_vtx
                //but I guess the question is that we'd now trade one entropy for another, this is weird
                //I think this is OK enough, further optimizations should be cuda's, because we can't trade readability for micro optimzations

                for (const auto& edge: edge_vec)
                {
                    uint32_t src_vtx;
                    uint32_t dst_vtx;
                    double score;

                    std::tie(src_vtx, dst_vtx, score)   = std::make_tuple(edge.src, edge.dst, edge.score);

                    auto [src_community_idx, src_community_sz]  = reverse_map.at(src_vtx);
                    auto [dst_community_idx, dst_community_sz]  = reverse_map.at(dst_vtx);

                    if (dst_community_idx <= src_community_idx)
                    {
                        continue;
                    }

                    auto group_id                   = std::make_pair(src_community_idx, dst_community_idx);
                    size_t group_sz                 = src_community_sz + dst_community_sz;
                    size_t sqr_sz                   = group_sz * group_sz;
                    double norm_value               = score / sqr_sz; 

                    community_score_map[group_id]   += norm_value;
                }

                if (community_score_map.empty())
                {
                    return BinaryClassifyResult{
                        .flat_community_vec = flat_community_vec,
                        .flat_community_range_vec = flat_community_range_vec
                    };
                }

                auto [min_iter, max_iter]   = std::minmax_element(community_score_map.begin(), community_score_map.end(), [](const auto& lhs, const auto& rhs){return lhs.second < rhs.second;});
                double value_first          = min_iter->second;
                double value_last           = max_iter->second;

                FloatNormalizer<long double, uint32_t> normalizer(value_first, value_last);
                std::vector<uint64_t> encoded_vec{};

                for (size_t i = 0u; i < community_score_map.size(); ++i)
                {
                    if (i > std::numeric_limits<uint32_t>::max())
                    {
                        throw std::runtime_error("bad unsigned operation, max value reached");
                    }

                    uint32_t group_id       = i;
                    uint32_t score          = normalizer(std::next(community_score_map.begin(), group_id)->second);
                    uint64_t encoded_value  = (static_cast<uint64_t>(score) << 32) | group_id; 

                    encoded_vec.push_back(encoded_value);
                }

                dg::sort_variants::quicksort::quicksort(encoded_vec.data(), std::next(encoded_vec.data(), encoded_vec.size()));
                // std::sort(encoded_vec.data(), std::next(encoded_vec.data(), encoded_vec.size()));

                local_unordered_set<uint32_t> visited_set                               = {}; 
                std::vector<uint32_t> new_flat_community_vec                            = {};
                std::vector<std::pair<uint32_t, uint32_t>> new_flat_community_range_vec = {};

                for (size_t i = 0u; i < encoded_vec.size(); ++i)
                {
                    size_t back_idx         = encoded_vec.size() - (i + 1u); 
                    uint32_t group_id       = encoded_vec[back_idx] & ((uint64_t{1} << 32) - 1u);
                    uint32_t src_group_id   = std::next(community_score_map.begin(), group_id)->first.first;
                    uint32_t dst_group_id   = std::next(community_score_map.begin(), group_id)->first.second;

                    if (visited_set.contains(src_group_id))
                    {
                        continue;
                    }

                    if (visited_set.contains(dst_group_id))
                    {
                        continue;
                    }

                    visited_set.insert(src_group_id);
                    visited_set.insert(dst_group_id);

                    uint32_t new_first                  = new_flat_community_vec.size();
                    uint32_t new_last                   = new_first + this->group_size(flat_community_range_vec[src_group_id]) + this->group_size(flat_community_range_vec[dst_group_id]);

                    const auto& [src_first, src_last]   = flat_community_range_vec[src_group_id];
                    const auto& [dst_first, dst_last]   = flat_community_range_vec[dst_group_id]; 

                    std::copy(std::next(flat_community_vec.begin(), src_first), std::next(flat_community_vec.begin(), src_last), std::back_inserter(new_flat_community_vec));
                    std::copy(std::next(flat_community_vec.begin(), dst_first), std::next(flat_community_vec.begin(), dst_last), std::back_inserter(new_flat_community_vec));

                    new_flat_community_range_vec.push_back({new_first, new_last});
                }

                for (size_t i = 0u; i < flat_community_range_vec.size(); ++i)
                {
                    if (visited_set.contains(i))
                    {
                        continue;
                    }

                    const auto& [group_first, group_last]   = flat_community_range_vec[i];
                    uint32_t new_first                      = new_flat_community_vec.size();
                    uint32_t new_last                       = new_first + (group_last - group_first);

                    std::copy(std::next(flat_community_vec.begin(), group_first), std::next(flat_community_vec.begin(), group_last), std::back_inserter(new_flat_community_vec));
                    new_flat_community_range_vec.push_back({new_first, new_last});
                }

                return BinaryClassifyResult
                {
                    .flat_community_vec         = std::move(new_flat_community_vec),
                    .flat_community_range_vec   = std::move(new_flat_community_range_vec)
                };
            }
    };
}

#endif