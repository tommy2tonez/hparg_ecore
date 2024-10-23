#ifndef __CUBLAS_X_H__
#define __CUBLAS_X_H__

#include <stdint.h>
#include <stdlib.h>
#include <vector>
#include <memory>
#include <string>
#include <chrono>
#include "network_compact_serializer.h"
#include <vector>
#include <string>

namespace dg::cublas_x::syntax_tree{

    using transform_kind_t = uint8_t; 

    enum transform_option: transform_kind_t{
        transform_kind_saturate_01          = 0u,
        transform_kind_rounddown_optional   = 1u,
        transform_kind_rounddown            = 2u,
        transform_kind_roundup_optional     = 3u,
        transform_kind_roundup              = 4u,
        transform_kind_fastmath             = 5u,
        transform_kind_roundeven_optional   = 6u,
        transform_kind_roundeven            = 7u,
        transform_kind_roundzero_optional   = 8u,
        transform_kind_roundzero            = 9u,
        transform_kind_clone                = 10u,
        transform_kind_relu                 = 11u,
        transform_kind_cast_u8              = 12u,
        transform_kind_cast_u16             = 13u,
        transform_kind_cast_u32             = 14u,
        transform_kind_cast_f8_native       = 15u,
        transform_kind_cast_f16_brain       = 16u,
        transform_kind_cast_f16_iec559      = 17u,
        transform_kind_cast_f32_iec559      = 18u,
        transform_kind_sign                 = 19u,
        transform_kind_exp                  = 20u,
        transform_kind_exp2                 = 21u,
        transform_kind_exp10                = 22u,
        transform_kind_log                  = 23u,
        transform_kind_log2                 = 24u,
        transform_kind_log10                = 25u,
        transform_kind_abs                  = 26u,
        transform_kind_cos                  = 27u,
        transform_kind_acos                 = 28u,
        transform_kind_sin                  = 29u,
        transform_kind_asin                 = 30u,
        transform_kind_tan                  = 31u,
        transform_kind_atan                 = 32u,
        transform_kind_sqrt                 = 33u,
        transform_kind_invsqrt              = 34u,
        transform_kind_negative             = 35u,
        transform_kind_negate               = 36u,
        transform_kind_transpose            = 37u,
        transform_kind_linear               = 38u,
        transform_kind_dot                  = 39u,
        transform_kind_add                  = 40u,
        transform_kind_sub                  = 41u,
        transform_kind_mul                  = 42u,
        transform_kind_div                  = 43u,
        transform_kind_pow                  = 44u,
        transform_kind_min                  = 45u,
        transform_kind_max                  = 46u,
        transform_kind_none                 = 47u
    };

    using logit_kind_t = uint8_t; 

    enum logit_option: logit_kind_t{
        u8          = 0u,
        u16         = 1u,
        u32         = 2u,
        f8          = 3u,
        f16brain    = 4u,
        f16iec559   = 5u,
        f32iec559   = 6u
    };

    struct MatrixDimension{
        size_t row_sz;
        size_t column_sz;
    };

    struct AbstractNode{
        std::vector<std::unique_ptr<AbstractNode>> descendants;
        transform_kind_t transform_kind;
        MatrixDimension dim;
        std::string value_identifier;
        logit_kind_t logit_kind;
    };

    struct CollapsedAbstractNode{
        std::unique_ptr<AbstractNode> upper; //this is weird
        std::vector<std::unique_ptr<AbstractNode>> leaf_descendants; //this is weird
    };

    struct Node{
        std::vector<std::unique_ptr<Node>> descendants;
        transform_kind_t transform_kind;
        MatrixDimension dim;
        std::string value_identifier;
        logit_kind_t logit_kind;
        std::shared_ptr<cuda_ptr_t> data;
    };

    auto transform_kind_cstr(transform_kind_t transform_kind) -> const char *{

    }
}

namespace dg::cublas_x::exception{

    struct invalid_argument: std::exception{};
}

namespace dg::cublas_x::exec_engine{

    class ExecutorInterface{

        public:

            virtual ~ExecutorInterface() noexcept = default;
            virtual void exec(int device_id, const std::multimap<std::string, void *>& arguments, void * dst, size_t dst_cap) = 0;
    };
}

namespace dg::cublas_x::opti_engine{
 
    class OptimizerInterface{

        public:

            virtual ~OptimizerInterface() noexcept = default;
            virtual auto optimize(cublas_handle_t, const syntax_tree::AbstractNode&) -> syntax_tree::AbstractNode = 0;
    };
}

namespace dg::cublas_x::exhaustive_ss_opti_engine{

    class SpaceRandomizerInterface{

        public:

            virtual ~SpaceRandomizerInterface() noexcept = default;
            virtual auto randomize(const syntax_tree::MatrixDimension&) -> std::shared_ptr<cuda_ptr_t> = 0;
    };

    class AbstractNodeRandomizerInterface{

        public:

            virtual ~AbstractNodeRandomizerInterface() noexcept = default;
            virtual auto randomize(const std::unique_ptr<syntax_tree::AbstractNode>&) -> std::unique_ptr<syntax_tree::Node> = 0;
    };

    class StateSearchEngineInterface{

        public:

            virtual ~StateSearchEngineInterface() noexcept = default;
            virtual auto search(const std::unique_ptr<syntax_tree::AbstractNode>&) -> std::vector<std::unique_ptr<syntax_tree::AbstractNode>> = 0;
    };

    class BenchmarkEngineInterface{

        public:

            virtual ~BenchmarkEngineInterface() noexcept = default;
            virtual auto benchmark(int device_id, const std::unique_ptr<syntax_tree::Node>&) -> std::chrono::nanoseconds = 0;
    };

    class IdentifierGeneratorInterface{

        public:

            virtual ~IdentifierGeneratorInterface() noexcept = default;
            virtual auto id(const std::unique_ptr<syntax_tree::AbstractNode>&) -> std::string = 0;
    };

    class AbstractNodeCollapserInterface{

        public:

            virtual ~AbstractNodeCollapserInterface() noexcept = default;
            virtual auto collapse(const std::unique_ptr<syntax_tree::AbstractNode>&) -> std::vector<std::unique_ptr<CollapsedAbstractNode>> = 0; 
    };

    class AbstractNodeUniqueRepresentationGeneratorInterface{

        public:

            virtual ~AbstractNodeUniqueRepresentationGeneratorInterface() noexcept = default;
            virtual auto to_unique_representation(const std::unique_ptr<syntax_tree::AbstractNode>&) -> std::unique_ptr<syntax_tree::AbstractNode> = 0;
    };
} 


namespace dg::cublas_x::utility{

    auto deepcopy(const std::unique_ptr<syntax_tree::AbstractNode>& node) -> std::unique_ptr<syntax_tree::AbstractNode>{

        if (!node){
            return nullptr;
        }

        auto rs                 = std::make_unique<syntax_tree::AbstractNode>();
        rs->transform_kind      = node->transform_kind;
        rs->dim                 = node->dim;
        rs->value_identifier    = node->value_identifier;

        for (const auto& child: node->descendants){
            rs->descendants.push_back(deepcopy(child));
        }

        return rs;
    } 

    auto make_identifier(const std::string& identifier) -> std::string{

        auto new_identifier = std::vector<std::string> {identifier};
        auto rs             = std::string(dg::network_compact_serializer::size(new_identifier), ' ');
        dg::network_compact_serializer::serialize_into(rs.data(), new_identifier);

        return rs;
    } 

    auto combine_identifier(const std::string& lhs, const std::string& rhs) -> std::string{
        
        // return lhs + rhs;

    }
}

namespace dg::cublas_x::exhaustive_ss_opti_engine{

    using namespace syntax_tree; 

    template <class arithemtic_ops_t>
    struct coerced_x_math{

        static __device__ inline auto sign(arithemtic_ops_t value) -> arithemtic_ops_t{

        } 

        static __device__ inline auto exp(arithemtic_ops_t value) -> arithemtic_ops_t{

        }

        static __device__ inline auto ln(arithemtic_ops_t value) -> arithemtic_ops_t{

        }

        static __device__ inline auto abs(arithemtic_ops_t value) -> arithemtic_ops_t{

        }

        static __device__ inline auto cos(arithemtic_ops_t value) -> arithemtic_ops_t{

        }

        static __device__ inline auto acos(arithemtic_ops_t value) -> arithemtic_ops_t{

        }

        static __device__ inline auto sin(arithemtic_ops_t value) -> arithemtic_ops_t{

        }

        static __device__ inline auto asin(arithemtic_ops_t value) -> arithemtic_ops_t{

        }

        static __device__ inline auto tan(arithemtic_ops_t value) -> arithemtic_ops_t{

        }

        static __device__ inline auto atan(arithemtic_ops_t value) -> arithmetic_ops_t{

        }

        static __device__ inline auto sqrt(arithemtic_ops_t value) -> arithemtic_ops_t{

        }

        static __device__ inline auto invsqrt(arithemtic_ops_t value) -> arithmetic_ops_t{

        }

        static __device__ inline auto negative(arithemtic_ops_t value) -> arithmetic_ops_t{

        }

        static __device__ inline auto add(arithemtic_ops_t lhs, arithemtic_ops_t rhs) -> arithmetic_ops_t{

        }

        static __device__ inline auto sub(arithemtic_ops_t lhs, arithemtic_ops_t rhs) -> arithmetic_ops_t{

        }

        static __device__ inline auto mul(arithemtic_ops_t lhs, arithemtic_ops_t rhs) -> arithemtic_ops_t{

        }

        static __device__ inline auto div(arithemtic_ops_t lhs, arithemtic_ops_t rhs) -> arithmetic_ops_t{

        }

        static __device__ inline auto pow(arithemtic_ops_t lhs, arithemtic_ops_t rhs) -> arithmetic_ops_t{

        }

        template <size_t RHS_VALUE>
        static __device__ inline auto pow(arithemtic_ops_t lhs, const std::integral_constant<size_t, RHS_VALUE>) -> arithmetic_ops_t{

        }

        static __device__ inline auto fma(arithmetic_ops_t first, arithmetic_ops_t second, arithemtic_ops_t third) -> arithmetic_ops_t{

        }

        static __device__ inline auto min(arithemtic_ops_t lhs, arithemtic_ops_t rhs) -> arithmetic_ops_t{

        }

        static __device__ inline auto max(arithemtic_ops_t lhs, arithemtic_ops_t rhs) -> arithmetic_ops_t{

        }

        static __device__ inline auto eqcmp_mul(arithemtic_ops_t lcmp, arithemtic_ops_t rcmp, arithemtic_ops_t val) -> arithmetic_ops_t{

        }
    };

    //this is the very basic of optimization - heuristics could be applied to do pruning - eliminating state search
    //this component alone could be 20k-30k LOC
    //there are many objectives:
    //(1): heuristics pruning 
    //(2): base case greedy optimizations
    //(3): fast greedy optimization
    //(4): fast grouping - rotate + permute
    //(5): trade off between tree-height + tree nodes and runtime  

    class AbstractNodeCollapser: public virtual AbstractNodeCollapserInterface{

        private:

            static inline constexpr uint8_t PATHNODE_BITCONTROL_END_OF_PATH     = 0b11;
            static inline constexpr uint8_t PATHNODE_BITCONTROL_END_AT_ROOT     = 0b01;
            static inline constexpr uint8_t PATHNODE_BITCONTROL_CONTINUE        = 0b00;

            struct PathNode{
                uint8_t bit_control;
                size_t descendant_idx;
            };

            using path_t = std::vector<PathNode>; 

        public:

            auto collapse(const std::unique_ptr<syntax_tree::AbstractNode>& root) -> std::vector<std::unique_ptr<CollapsedAbstractNode>>{

                std::vector<std::vector<path_t>> block_instruction_set = this->get_block_possibilities(root);
                return this->make_collapsed_node_from_block_instruction_set(root, block_instruction_set);
            }
        
        private:

            auto get_space(const std::vector<std::vector<std::vector<path_t>>>& inp) -> std::vector<size_t>{

                auto rs = std::vector<size_t>();

                for (const auto& e: inp){
                    rs.push_back(e.size());
                }

                return rs;
            } 

            auto get_space_size(const std::vector<size_t>& space) -> size_t{

                if (space.empty()){
                    return 0u;
                }

                size_t space_sz = 1u;

                for (size_t i = 0; i < space.size(); ++i){
                    space_sz *= space[i];
                }
                
                return space_sz;
            }

            void iter_increment(std::vector<size_t>& ptr, const std::vector<size_t>& space){

                for (size_t i = 0; i < space.size(); ++i){
                    ptr[i] += 1;

                    if (ptr[i] != space[i]){
                        return;
                    }

                    ptr[i] = 0u;
                }
            }

            auto permute_join(const std::vector<std::vector<std::vector<path_t>>>& block_instruction) -> std::vector<std::vector<path_t>>{
                
                std::vector<std::vector<path_t>> rs{};
                std::vector<size_t> space = get_space(block_instruction);
                std::vector<size_t> ptr(0u, space.size());
                size_t idx = 0u;
                size_t space_size = this->get_space_size(space);

                while (idx != space_size){
                    std::vector<path_t> cur_block_instruction{};

                    for (size_t i = 0u; i < ptr.size(); ++i){
                        std::vector<path_t> descendant_block_instruction = block_instruction[i][ptr[i]];

                        for (auto& descendant_path: descendant_block_instruction){
                            descendant_path.insert(descendant_path.begin(), PathNode{PATHNODE_BITCONTROL_CONTINUE, i});
                        }

                        cur_block_instruction.insert(cur_block_instruction.end(), descendant_block_instruction.begin(), descendant_block_instruction.end());
                    }

                    rs.push_back(std::move(cur_block_instruction));
                    ++idx;
                    iter_increment(ptr, space);
                }

                return rs;
            }

            auto get_block_possibilities(const std::unique_ptr<syntax_tree::AbstractNode>& root) -> std::vector<std::vector<path_t>>{

                if (root == nullptr){
                    return {};
                }

                if (root->descendants.empty()){
                    return {{{PathNode{PATHNODE_BITCONTROL_END_OF_PATH, {}}}}, {{PathNode{PATHNODE_BITCONTROL_END_AT_ROOT, {}}}}};
                }

                std::vector<std::vector<std::vector<path_t>>> descendants_possibilities = {};

                for (const auto& descendant: root->descendants){
                    descendants_possibilities.push_back(this->get_block_possibilities(descendant));
                }

                std::vector<std::vector<path_t>> rs = this->permute_join(descendants_possibilities);
                rs.push_back(std::vector<path_t>{path_t{PathNode{PATHNODE_BITCONTROL_END_OF_PATH, {}}}});

                return rs;
            }

            auto traverse_and_prune(const std::unique_ptr<syntax_tree::AbstractNode>& root, std::vector<intmax_t>& backtrack, const std::unordered_set<std::string>& leaf_hashset) -> std::pair<std::unique_ptr<AbstractNode>, std::vector<std::unique_ptr<AbstractNode>>>{

                // if (root == nullptr){
                //     return {};
                // }

                // std::string backtrack_str_rep(dg::network_compact_serializer::size(backtrack), ' ');
                // dg::network_compact_serializer::serialize_into(backtrack_str_rep.data(), backtrack);
                
                // if (leaf_hashset.contains(backtrack_str_rep)){
                //     return {nullptr, {}};
                // }



            } 

            auto make_collapsed_node_from_block_instruction_set(const std::unique_ptr<syntax_tree::AbstractNode>& root, const std::vector<std::vector<path_t>>& instruction_set) -> std::vector<std::unique_ptr<CollapsedAbstractNode>>{
                    
                std::vector<std::unique_ptr<CollapsedAbstractNode>> rs{};

                for (const auto& instruction: instruction_set){
                    std::unordered_set<std::string> end_of_path_hashset = this->to_eop_hashset(instruction);
                    std::vector<intmax_t> backtrack = {-1};  
                    auto [abstract_root, descendants] = traverse_and_prune(root, end_of_path_hashset);
                    auto rs = std::make_unique<CollapsedAbstractNode>(CollapsedAbstractNode{std::move(abstract_root), std::move(descendants)});
                    rs.push_back(std::move(rs));
                }
            }
    };

    class BaseStateSearchEngine: public virtual StateSearchEngineInterface{

        private:

            std::unordered_map<std::string, transform_kind_t> transform_map;
            std::unique_ptr<IdentifierGeneratorInterface> id_gen;
            std::unique_ptr<AbstractNodeCollapserInterface> abstract_node_collapser;
            std::unique_ptr<AbstractNodeUniqueRepresentationGeneratorInterface> abstract_node_unique_rep_generator;

        public:

            BaseStateSearchEngine(std::unordered_map<std::string, transform_kind_t> transform_map,
                                  std::unique_ptr<IdentifierGeneratorInterface> id_gen,
                                  std::unique_ptr<AbstractNodeCollapserInterface> abstract_node_collapser,
                                  std::unique_ptr<AbstractNodeUniqueRepresentationGeneratorInterface> abstract_node_unique_rep_generator) noexcept: transform_map(std::move(transform_map)),
                                                                                                                                                    id_gen(std::move(id_gen)),
                                                                                                                                                    abstract_node_collapser(std::move(abstract_node_collapser)),
                                                                                                                                                    abstract_node_unique_rep_generator(std::move(abstract_node_unique_rep_generator)){}

            auto search(const std::unique_ptr<AbstractNode>& root) -> std::vector<std::unique_ptr<AbstractNode>>{

                if (!this->is_precond_met(root)){
                    return {utility::deepcopy(root)};
                }

                std::vector<std::unique_ptr<CollapsedAbstractNode>> collapsed_node_vec = this->abstract_node_collapser->collapse(root);
                std::vector<std::unique_ptr<AbstractNode>> rs{};

                for (auto& collapsed_node: collapsed_node_vec){
                    std::unique_ptr<AbstractNode> uniq_rep  = this->abstract_node_unique_rep_generator->to_unique_representation(collapsed_node->upper);
                    std::string id                          = this->id_gen->id(uniq_rep);
                    auto map_ptr                            = this->transform_map.find(id);

                    if (map_ptr == this->transform_map.end()){
                        continue;
                    }

                    auto appendee               = std::make_unique<AbstractNode>();
                    appendee->dim               = collapsed_node->upper->dim;
                    appendee->logit_kind        = collapsed_node->upper->logit_kind;
                    appendee->value_identifier  = collapsed_node->upper->value_identifier;
                    appendee->transform_kind    = map_ptr->second;
                    appendee->descendants       = this->map_descendants(collapsed_node->upper, uniq_rep, collapsed_node->leaf_descendants);

                    rs.push_back(std::move(appendee));
                }

                return rs;
            }
    };

    class PermuteStateSearchEngine: public virtual StateSearchEngineInterface{

        private:

            std::unique_ptr<StateSearchEngineInterface> base_search;

        public:

            PermuteStateSearchEngine(std::unique_ptr<StateSearchEngineInterface> base_search) noexcept: base_search(std::move(base_search)){}

            auto search(const std::unique_ptr<AbstractNode>& root) -> std::vector<std::unique_ptr<AbstractNode>>{

                return this->internal_search(root);
            }
        
        private:

            auto get_space(const std::vector<std::vector<std::unique_ptr<AbstractNode>>>& inp) -> std::vector<size_t>{

                auto rs = std::vector<size_t>();

                for (const auto& e: inp){
                    rs.push_back(e.size());
                }

                return rs;
            } 

            auto get_space_size(const std::vector<size_t>& space) -> size_t{

                if (space.empty()){
                    return 0u;
                }

                size_t space_sz = space[0u];

                for (size_t i = 1u; i < space.size(); ++i){
                    space_sz *= space[i];
                }
                
                return space_sz;
            }

            void iter_increment(std::vector<size_t>& ptr, const std::vector<size_t>& space){

                for (size_t i = 0; i < space.size(); ++i){
                    ptr[i] += 1;

                    if (ptr[i] != space[i]){
                        return;
                    }

                    ptr[i] = 0u;
                }
            }

            auto extract_abstract_node(const std::vector<std::vector<std::unique_ptr<AbstractNode>>>& inp, const std::vector<size_t>& ptr) -> std::vector<std::unique_ptr<AbstractNode>>{

                std::vector<std::unique_ptr<AbstractNode>> rs{};

                for (size_t i = 0; i < ptr.size(); ++i){
                    size_t idx = ptr[i]; 
                    rs.push_back(utility::deepcopy(inp[i][idx]));
                }

                return rs;
            }

            auto make_root_possibilities(const std::unique_ptr<AbstractNode>& root, 
                                         const std::vector<std::vector<std::unique_ptr<AbstractNode>>>& descendants) -> std::vector<std::unique_ptr<AbstractNode>>{
                
                auto rs                     = std::vector<std::unique_ptr<AbstractNode>>{};
                std::vector<size_t> space   = this->get_space(descendants); 
                std::vector<size_t> ptr     = std::vector<size_t>(space.size(), 0u);
                size_t space_size           = this->get_space_size(space);
                size_t idx                  = 0u; 

                while (idx != space_size){
                    auto appendee               = std::make_unique<AbstractNode>();
                    appendee->descendants       = this->extract_abstract_node(descendants, ptr);
                    appendee->dim               = root->dim;
                    appendee->logit_kind        = root->logit_kind;
                    appendee->transform_kind    = root->transform_kind;
                    appendee->value_identifier  = root->value_identifier;
                    rs.push_back(std::move(appendee));
                    this->iter_increment(ptr, space);
                    ++idx;
                }

                return rs;
            } 

            auto internal_search(const std::unique_ptr<AbstractNode>& root) -> std::vector<std::unique_ptr<AbstractNode>>{

                if (root == nullptr){
                    return {};
                }

                if (root->descendants.size() == 0u){
                    return {utility::deepcopy(root)};
                }

                std::vector<std::vector<std::unique_ptr<AbstractNode>>> descendant_permute_set{};

                for (size_t i = 0; i < root->descendants.size(); ++i){
                    descendant_permute_set.push_back(internal_search(root->descendants[i]));
                }

                std::vector<std::unique_ptr<AbstractNode>> root_possibilities = this->make_root_possibilities(root, descendant_permute_set);
                std::vector<std::unique_ptr<AbstractNode>> rs{};

                for (const std::unique_ptr<AbstractNode>& cur: root_possibilities){
                    auto cur_valid = this->base_search->search(cur);
                    std::copy(std::make_move_iterator(cur_valid.begin()), std::make_move_iterator(cur_valid.end()), std::back_inserter(rs));
                }

                return rs;
            }
    };

    class OptimizerEngine: public virtual opti_engine::OptimizerInterface{

        private:

            std::unique_ptr<AbstractNodeRandomizerInterface> abstract_node_randomizer;
            std::unique_ptr<StateSearchEngineInterface> state_search_engine;
            std::unique_ptr<BenchmarkEngineInterface> benchmark_engine;
        
        public:

            explicit OptimizerEngine(std::unique_ptr<AbstractNodeRandomizerInterface> abstract_node_randomizer,
                                     std::unique_ptr<StateSearchEngineInterface> state_search_engine,
                                     std::unique_ptr<BenchmarkEngineInterface> benchmark_engine) noexcept: abstract_node_randomizer(std::move(abstract_node_randomizer)),
                                                                                                           state_search_engine(std::move(state_search_engine)),
                                                                                                           benchmark_engine(std::move(benchmark_engine)){}
            
            auto optimize(cublas_handle_t cublas_handle_object, const syntax_tree::AbstractNode& node) -> syntax_tree::AbstractNode{

                std::vector<syntax_tree::AbstractNode> abstract_states = this->state_search_engine->search(node);
                std::chrono::nanoseconds max_ts = std::chrono::duration_values<std::chrono::nanoseconds>::max();
                syntax_tree::AbstractNode rs    = syntax_tree::deepcopy(node);

                for (const syntax_tree::AbstractNode& abstract_state: abstract_states){
                    syntax_tree::Node state         = this->abstract_node_randomizer->randomize(abstract_state);
                    std::chrono::nanoseconds cur_ts = this->benchmark_engine->benchmark(cublas_handle_object, state);

                    if (cur_ts < max_ts){
                        max_ts = cur_ts;
                        rs = syntax_tree::deepcopy(abstract_state);
                    }
                }

                return rs;
            } 
    };

} 

namespace dg::cublas_x{

    using AbstractNode  = syntax_tree::AbstractNode;
    using cublas_plan_t = std::unique_ptr<AbstractNode>;
    using logit_kind_t  = syntax_tree::logit_kind_t;

    auto cublas_make_matrix(size_t m, size_t n, std::string value_identifier, logit_kind_t logit_kind) -> std::unique_ptr<AbstractNode>{
        
        using namespace syntax_tree; 

        if (m == 0u){
            throw exception::invalid_argument();
        }

        if (n == 0u){
            throw exception::invalid_argument{};
        }

        return std::make_unique<AbstractNode>(AbstractNode{{}, transform_kind_none, MatrixDimension{m, n}, utility::make_identifier(value_identifier), logit_kind});
    }
    
    auto cublas_split(const std::unique_ptr<AbstractNode>& plan, size_t split_factor) -> std::vector<std::unique_ptr<AbstractNode>>{

    }

    auto cublas_join(const std::vector<std::unique_ptr<AbstractNode>>& plan) -> std::unique_ptr<AbstractNode>{
        
    }

    auto cublas_mono_saturate_01(const std::unique_ptr<AbstractNode>& plan) -> std::unique_ptr<AbstractNode>{

        using namespace syntax_tree;

        if (plan == nullptr){
            throw exception::invalid_argument();
        }

        auto descendants                = std::vector<std::unique_ptr<AbstractNode>>{utility::deepcopy(plan)};
        transform_kind_t transform_kind = transform_kind_saturate_01;
        MatrixDimension dim             = plan->dim;
        std::string identifier          = utility::combine_identifier(plan->value_identifier, transform_kind_cstr(transform_kind_saturate_01));
        logit_kind_t logit_kind         = plan->logit_kind; 

        return std::make_unique<AbstractNode>(AbstractNode{std::move(descendants), transform_kind, dim, std::move(identifier), logit_kind});
    }

    auto cublas_mono_rounddown_optional(const std::unique_ptr<AbstractNode>& plan) -> std::unique_ptr<AbstractNode>{

        using namespace syntax_tree;

        if (plan == nullptr){
            throw exception::invalid_argument();
        }

        auto descendants                = std::vector<std::unique_ptr<AbstractNode>>{utility::deepcopy(plan)};
        transform_kind_t transform_kind = transform_kind_rounddown_optional;
        MatrixDimension dim             = plan->dim;
        std::string identifier          = utility::combine_identifier(plan->value_identifier, transform_kind_cstr(transform_kind_rounddown_optional));
        logit_kind_t logit_kind         = plan->logit_kind; 

        return std::make_unique<AbstractNode>(AbstractNode{std::move(descendants), transform_kind, dim, std::move(identifier), logit_kind});
    } 

    auto cublas_mono_rounddown(const std::unique_ptr<AbstractNode>& plan) -> std::unique_ptr<AbstractNode>{

        using namespace syntax_tree;

        if (plan == nullptr){
            throw exception::invalid_argument();
        }

        auto descendants                = std::vector<std::unique_ptr<AbstractNode>>{utility::deepcopy(plan)};
        transform_kind_t transform_kind = transform_kind_rounddown;
        MatrixDimension dim             = plan->dim;
        std::string identifier          = utility::combine_identifier(plan->value_identifier, transform_kind_cstr(transform_kind_rounddown)); 
        logit_kind_t logit_kind         = plan->logit_kind; 

        return std::make_unique<AbstractNode>(AbstractNode{std::move(descendants), transform_kind, dim, std::move(identifier), logit_kind});
    }

    auto cublas_mono_roundup_optional(const std::unique_ptr<AbstractNode>& plan) -> std::unique_ptr<AbstractNode>{

        using namespace syntax_tree;

        if (plan == nullptr){
            throw exception::invalid_argument();
        }

        auto descendants                = std::vector<std::unique_ptr<AbstractNode>>{utility::deepcopy(plan)};
        transform_kind_t transform_kind = transform_kind_roundup_optional;
        MatrixDimension dim             = plan->dim;
        std::string identifier          = utility::combine_identifier(plan->value_identifier, transform_kind_cstr(transform_kind_roundup_optional));
        logit_kind_t logit_kind         = plan->logit_kind; 

        return std::make_unique<AbstractNode>(AbstractNode{std::move(descendants), transform_kind, dim, std::move(identifier), logit_kind});
    }

    auto cublas_mono_roundup(const std::unique_ptr<AbstractNode>& plan) -> std::unique_ptr<AbstractNode>{

        using namespace syntax_tree;

        if (plan == nullptr){
            throw exception::invalid_argument();
        }

        auto descendants                = std::vector<std::unique_ptr<AbstractNode>>{utility::deepcopy(plan)};
        transform_kind_t transform_kind = transform_kind_roundup;
        MatrixDimension dim             = plan->dim;
        std::string identifier          = utility::combine_identifier(plan->value_identifier, transform_kind_cstr(transform_kind_roundup));
        logit_kind_t logit_kind         = plan->logit_kind; 

        return std::make_unique<AbstractNode>(AbstractNode{std::move(descendants), transform_kind, dim, std::move(identifier), logit_kind});
    }

    auto cublas_mono_fastmath(const std::unique_ptr<AbstractNode>& plan) -> std::unique_ptr<AbstractNode>{

        using namespace syntax_tree;
        
        if (plan == nullptr){
            throw exception::invalid_argument();
        }

        auto descendants                = std::vector<std::unique_ptr<AbstractNode>>{utility::deepcopy(plan)};
        transform_kind_t transform_kind = transform_kind_fastmath;
        MatrixDimension dim             = plan->dim;
        std::string identifier          = utility::combine_identifier(plan->value_identifier, transform_kind_cstr(transform_kind_fastmath));
        logit_kind_t logit_kind         = plan->logit_kind; 

        return std::make_unique<AbstractNode>(AbstractNode{std::move(descendants), transform_kind, dim, std::move(identifier), logit_kind});
    }

    auto cublas_mono_roundeven_optional(const std::unique_ptr<AbstractNode>& plan) -> std::unique_ptr<AbstractNode>{

        using namespace syntax_tree;

        if (plan == nullptr){
            throw exception::invalid_argument();
        }

        auto descendants                = std::vector<std::unique_ptr<AbstractNode>>{utility::deepcopy(plan)};
        transform_kind_t transform_kind = transform_kind_roundeven_optional;
        MatrixDimension dim             = plan->dim;
        std::string identifier          = utility::combine_identifier(plan->value_identifier, transform_kind_cstr(transform_kind_roundeven_optional));
        logit_kind_t logit_kind         = plan->logit_kind; 

        return std::make_unique<AbstractNode>(AbstractNode{std::move(descendants), transform_kind, dim, std::move(identifier), logit_kind});
    }

    auto cublas_mono_roundeven(const std::unique_ptr<AbstractNode>& plan) -> std::unique_ptr<AbstractNode>{

        using namespace syntax_tree;

        if (plan == nullptr){
            throw exception::invalid_argument();
        }

        auto descendants                = std::vector<std::unique_ptr<AbstractNode>>{utility::deepcopy(plan)};
        transform_kind_t transform_kind = transform_kind_roundeven;
        MatrixDimension dim             = plan->dim;
        std::string identifier          = utility::combine_identifier(plan->value_identifier, transform_kind_cstr(transform_kind_roundeven));
        logit_kind_t logit_kind         = plan->logit_kind; 

        return std::make_unique<AbstractNode>(AbstractNode{std::move(descendants), transform_kind, dim, std::move(identifier), logit_kind});
    }

    auto cublas_mono_roundzero_optional(const std::unique_ptr<AbstractNode>& plan) -> std::unique_ptr<AbstractNode>{

        using namespace syntax_tree;

        if (plan == nullptr){
            throw exception::invalid_argument();
        }

        auto descendants                = std::vector<std::unique_ptr<AbstractNode>>{utility::deepcopy(plan)};
        transform_kind_t transform_kind = transform_kind_roundzero_optional;
        MatrixDimension dim             = plan->dim;
        std::string identifier          = utility::combine_identifier(plan->value_identifier, transform_kind_cstr(transform_kind_roundzero_optional));
        logit_kind_t logit_kind         = plan->logit_kind; 

        return std::make_unique<AbstractNode>(AbstractNode{std::move(descendants), transform_kind, dim, std::move(identifier), logit_kind});
    }

    auto cublas_mono_roundzero(const std::unique_ptr<AbstractNode>& plan) -> std::unique_ptr<AbstractNode>{

        using namespace syntax_tree;

        if (plan == nullptr){
            throw exception::invalid_argument();
        }

        auto descendants                = std::vector<std::unique_ptr<AbstractNode>>{utility::deepcopy(plan)};
        transform_kind_t transform_kind = transform_kind_roundzero;
        MatrixDimension dim             = plan->dim;
        std::string identifier          = utility::combine_identifier(plan->value_identifier, transform_kind_cstr(transform_kind_roundzero));
        logit_kind_t logit_kind         = plan->logit_kind; 

        return std::make_unique<AbstractNode>(AbstractNode{std::move(descendants), transform_kind, dim, std::move(identifier), logit_kind});
    }

    auto cublas_mono_clone(const std::unique_ptr<AbstractNode>& plan) -> std::unique_ptr<AbstractNode>{

        using namespace syntax_tree;

        if (plan == nullptr){
            throw exception::invalid_argument();
        }

        auto descendants                = std::vector<std::unique_ptr<AbstractNode>>{utility::deepcopy(plan)};
        transform_kind_t transform_kind = transform_kind_clone;
        MatrixDimension dim             = plan->dim;
        std::string identifier          = utility::combine_identifier(plan->value_identifier, transform_kind_cstr(transform_kind_clone));
        logit_kind_t logit_kind         = plan->logit_kind; 

        return std::make_unique<AbstractNode>(AbstractNode{std::move(descendants), transform_kind, dim, std::move(identifier), logit_kind});
    }

    auto cublas_mono_relu(const std::unique_ptr<AbstractNode>& plan) -> std::unique_ptr<AbstractNode>{

        using namespace syntax_tree;

        if (plan == nullptr){
            throw exception::invalid_argument();
        }

        auto descendants                = std::vector<std::unique_ptr<AbstractNode>>{utility::deepcopy(plan)};
        transform_kind_t transform_kind = transform_kind_relu;
        MatrixDimension dim             = plan->dim;
        std::string identifier          = utility::combine_identifier(plan->value_identifier, transform_kind_cstr(transform_kind_relu));
        logit_kind_t logit_kind         = plan->logit_kind;

        return std::make_unique<AbstractNode>(AbstractNode{std::move(descendants), transform_kind, dim, std::move(identifier), logit_kind});
    }

    auto cublas_mono_cast(const std::unique_ptr<AbstractNode>& plan, logit_kind_t logit_kind) -> std::unique_ptr<AbstractNode>{

        using namespace syntax_tree;

        if (plan == nullptr){
            throw exception::invalid_argument();
        }

        auto descendants = std::vector<std::unique_ptr<AbstractNode>>{utility::deepcopy(plan)};
        transform_kind_t transform_kind{};

        switch (logit_kind){
            case u8:
                transform_kind = transform_kind_cast_u8;
                break;
            case u16:
                transform_kind = transform_kind_cast_u16;
                break;
            case u32:
                transform_kind = transform_kind_cast_u32;
                break;
            case f8:
                transform_kind = transform_kind_cast_f8_native;
                break;
            case f16brain:
                transform_kind = transform_kind_cast_f16_brain;
                break;
            case f16iec559:
                transform_kind = transform_kind_cast_f16_iec559;
                break;
            case f32iec559:
                transform_kind = transform_kind_cast_f32_iec559;
                break;
            default:
                throw exception::invalid_argument();
                break;
        }

        MatrixDimension dim     = plan->dim;
        std::string identifier  = utility::combine_identifier(plan->value_identifier, transform_kind_cstr(transform_kind));

        return std::make_unique<AbstractNode>(AbstractNode{std::move(descendants), transform_kind, dim, std::move(identifier), logit_kind});
    }

    auto cublas_mono_sign(const std::unique_ptr<AbstractNode>& plan) -> std::unique_ptr<AbstractNode>{

        using namespace syntax_tree;

        if (plan == nullptr){
            throw exception::invalid_argument();
        }

        auto descendants                = std::vector<std::unique_ptr<AbstractNode>>{utility::deepcopy(plan)};
        transform_kind_t transform_kind = transform_kind_sign;
        MatrixDimension dim             = plan->dim;
        std::string identifier          = utility::combine_identifier(plan->value_identifier, transform_kind_cstr(transform_kind_sign));
        logit_kind_t logit_kind         = plan->logit_kind;

        return std::make_unique<AbstractNode>(AbstractNode{std::move(descendants), transform_kind, dim, std::move(identifier), logit_kind});
    }

    auto cublas_mono_exp(const std::unique_ptr<AbstractNode>& plan) -> std::unique_ptr<AbstractNode>{

        using namespace syntax_tree;

        if (plan == nullptr){
            throw exception::invalid_argument();
        }

        auto descendants                = std::vector<std::unique_ptr<AbstractNode>>{utility::deepcopy(plan)};
        transform_kind_t transform_kind = transform_kind_exp;
        MatrixDimension dim             = plan->dim;
        std::string identifier          = utility::combine_identifier(plan->value_identifier, transform_kind_cstr(transform_kind_exp));
        logit_kind_t logit_kind         = plan->logit_kind;

        return std::make_unique<AbstractNode>(AbstractNode{std::move(descendants), transform_kind, dim, std::move(identifier), logit_kind});
    }

    auto cublas_mono_exp2(const std::unique_ptr<AbstractNode>& plan) -> std::unique_ptr<AbstractNode>{

        using namespace syntax_tree;

        if (plan == nullptr){
            throw exception::invalid_argument();
        }

        auto descendants                = std::vector<std::unique_ptr<AbstractNode>>{utility::deepcopy(plan)};
        transform_kind_t transform_kind = transform_kind_exp2;
        MatrixDimension dim             = plan->dim;
        std::string identifier          = utility::combine_identifier(plan->value_identifier, transform_kind_cstr(transform_kind_exp2));
        logit_kind_t logit_kind         = plan->logit_kind;

        return std::make_unique<AbstractNode>(AbstractNode{std::move(descendants), transform_kind, dim, std::move(identifier), logit_kind});
    }

    auto cublas_mono_exp10(const std::unique_ptr<AbstractNode>& plan) -> std::unique_ptr<AbstractNode>{

        using namespace syntax_tree;

        if (plan == nullptr){
            throw exception::invalid_argument();
        }

        auto descendants                = std::vector<std::unique_ptr<AbstractNode>>{utility::deepcopy(plan)};
        transform_kind_t transform_kind = transform_kind_exp10;
        MatrixDimension dim             = plan->dim;
        std::string identifier          = utility::combine_identifier(plan->value_identifier, transform_kind_cstr(transform_kind_exp10));
        logit_kind_t logit_kind         = plan->logit_kind;

        return std::make_unique<AbstractNode>(AbstractNode{std::move(descendants), transform_kind, dim, std::move(identifier), logit_kind});
    }

    auto cublas_mono_log(const std::unique_ptr<AbstractNode>& plan) -> std::unique_ptr<AbstractNode>{

        using namespace syntax_tree;

        if (plan == nullptr){
            throw exception::invalid_argument();
        }

        auto descendants                = std::vector<std::unique_ptr<AbstractNode>>{utility::deepcopy(plan)};
        transform_kind_t transform_kind = transform_kind_log;
        MatrixDimension dim             = plan->dim;
        std::string identifier          = utility::combine_identifier(plan->value_identifier, transform_kind_cstr(transform_kind_log));
        logit_kind_t logit_kind         = plan->logit_kind;

        return std::make_unique<AbstractNode>(AbstractNode{std::move(descendants), transform_kind, dim, std::move(identifier), logit_kind});
    }

    auto cublas_mono_log2(const std::unique_ptr<AbstractNode>& plan) -> std::unique_ptr<AbstractNode>{

        using namespace syntax_tree;

        if (plan == nullptr){
            throw exception::invalid_argument();
        }

        auto descendants                = std::vector<std::unique_ptr<AbstractNode>>{utility::deepcopy(plan)};
        transform_kind_t transform_kind = transform_kind_log2;
        MatrixDimension dim             = plan->dim;
        std::string identifier          = utility::combine_identifier(plan->value_identifier, transform_kind_cstr(transform_kind_log2));
        logit_kind_t logit_kind         = plan->logit_kind;

        return std::make_unique<AbstractNode>(AbstractNode{std::move(descendants), transform_kind, dim, std::move(identifier), logit_kind});
    }

    auto cublas_mono_log10(const std::unique_ptr<AbstractNode>& plan) -> std::unique_ptr<AbstractNode>{

        using namespace syntax_tree;

        if (plan == nullptr){
            throw exception::invalid_argument();
        }

        auto descendants                = std::vector<std::unique_ptr<AbstractNode>>{utility::deepcopy(plan)};
        transform_kind_t transform_kind = transform_kind_log10;
        MatrixDimension dim             = plan->dim;
        std::string identifier          = utility::combine_identifier(plan->value_identifier, transform_kind_cstr(transform_kind_log10));
        logit_kind_t logit_kind         = plan->logit_kind;

        return std::make_unique<AbstractNode>(AbstractNode{std::move(descendants), transform_kind, dim, std::move(identifier), logit_kind});
    } 

    auto cublas_mono_abs(const std::unique_ptr<AbstractNode>& plan) -> std::unique_ptr<AbstractNode>{

        using namespace syntax_tree;

        if (plan == nullptr){
            throw exception::invalid_argument();
        }

        auto descendants                = std::vector<std::unique_ptr<AbstractNode>>{utility::deepcopy(plan)};
        transform_kind_t transform_kind = transform_kind_abs;
        MatrixDimension dim             = plan->dim;
        std::string identifier          = utility::combine_identifier(plan->value_identifier, transform_kind_cstr(transform_kind_abs));
        logit_kind_t logit_kind         = plan->logit_kind;

        return std::make_unique<AbstractNode>(AbstractNode{std::move(descendants), transform_kind, dim, std::move(identifier), logit_kind});
    }

    auto cublas_mono_cos(const std::unique_ptr<AbstractNode>& plan) -> std::unique_ptr<AbstractNode>{

        using namespace syntax_tree;

        if (plan == nullptr){
            throw exception::invalid_argument();
        }

        auto descendants                = std::vector<std::unique_ptr<AbstractNode>>{utility::deepcopy(plan)};
        transform_kind_t transform_kind = transform_kind_cos;
        MatrixDimension dim             = plan->dim;
        std::string identifier          = utility::combine_identifier(plan->value_identifier, transform_kind_cstr(transform_kind_cos));
        logit_kind_t logit_kind         = plan->logit_kind;

        return std::make_unique<AbstractNode>(AbstractNode{std::move(descendants), transform_kind, dim, std::move(identifier), logit_kind});
    }

    auto cublas_mono_acos(const std::unique_ptr<AbstractNode>& plan) -> std::unique_ptr<AbstractNode>{

        using namespace syntax_tree;

        if (plan == nullptr){
            throw exception::invalid_argument{};
        }

        auto descendants                = std::vector<std::unique_ptr<AbstractNode>>{utility::deepcopy(plan)};
        transform_kind_t transform_kind = transform_kind_acos;
        MatrixDimension dim             = plan->dim;
        std::string identifier          = utility::combine_identifier(plan->value_identifier, transform_kind_cstr(transform_kind_acos));
        logit_kind_t logit_kind         = plan->logit_kind;

        return std::make_unique<AbstractNode>(AbstractNode{std::move(descendants), transform_kind, dim, std::move(identifier), logit_kind});
    }

    auto cublas_mono_sin(const std::unique_ptr<AbstractNode>& plan) -> std::unique_ptr<AbstractNode>{

        using namespace syntax_tree;

        if (plan == nullptr){
            throw exception::invalid_argument();
        }

        auto descendants                = std::vector<std::unique_ptr<AbstractNode>>{utility::deepcopy(plan)};
        transform_kind_t transform_kind = transform_kind_sin;
        MatrixDimension dim             = plan->dim;
        std::string identifier          = utility::combine_identifier(plan->value_identifier, transform_kind_cstr(transform_kind_sin));
        logit_kind_t logit_kind         = plan->logit_kind;

        return std::make_unique<AbstractNode>(AbstractNode{std::move(descendants), transform_kind, dim, std::move(identifier), logit_kind});
    }

    auto cublas_mono_asin(const std::unique_ptr<AbstractNode>& plan) -> std::unique_ptr<AbstractNode>{

        using namespace syntax_tree;

        if (plan == nullptr){
            throw exception::invalid_argument();
        }

        auto descendants                = std::vector<std::unique_ptr<AbstractNode>>{utility::deepcopy(plan)};
        transform_kind_t transform_kind = transform_kind_asin;
        MatrixDimension dim             = plan->dim;
        std::string identifier          = utility::combine_identifier(plan->value_identifier, transform_kind_cstr(transform_kind_asin));
        logit_kind_t logit_kind         = plan->logit_kind;

        return std::make_unique<AbstractNode>(AbstractNode{std::move(descendants), transform_kind, dim, std::move(identifier), logit_kind});
    }
 
    auto cublas_mono_tan(const std::unique_ptr<AbstractNode>& plan) -> std::unique_ptr<AbstractNode>{

        using namespace syntax_tree;

        if (plan == nullptr){
            throw exception::invalid_argument();
        }

        auto descendants                = std::vector<std::unique_ptr<AbstractNode>>{utility::deepcopy(plan)};
        transform_kind_t transform_kind = transform_kind_tan;
        MatrixDimension dim             = plan->dim;
        std::string identifier          = utility::combine_identifier(plan->value_identifier, transform_kind_cstr(transform_kind_tan));
        logit_kind_t logit_kind         = plan->logit_kind;

        return std::make_unique<AbstractNode>(AbstractNode{std::move(descendants), transform_kind, dim, std::move(identifier), logit_kind});
    }

    auto cublas_mono_atan(const std::unique_ptr<AbstractNode>& plan) -> std::unique_ptr<AbstractNode>{

        using namespace syntax_tree;

        if (plan == nullptr){
            throw exception::invalid_argument();
        }

        auto descendants                = std::vector<std::unique_ptr<AbstractNode>>{utility::deepcopy(plan)};
        transform_kind_t transform_kind = transform_kind_atan;
        MatrixDimension dim             = plan->dim;
        std::string identifier          = utility::combine_identifier(plan->value_identifier, transform_kind_cstr(transform_kind_atan));
        logit_kind_t logit_kind         = plan->logit_kind;

        return std::make_unique<AbstractNode>(AbstractNode{std::move(descendants), transform_kind, dim, std::move(identifier), logit_kind});
    }

    auto cublas_mono_sqrt(const std::unique_ptr<AbstractNode>& plan) -> std::unique_ptr<AbstractNode>{

        using namespace syntax_tree;

        if (plan == nullptr){
            throw exception::invalid_argument();
        }

        auto descendants                = std::vector<std::unique_ptr<AbstractNode>>{utility::deepcopy(plan)};
        transform_kind_t transform_kind = transform_kind_sqrt;
        MatrixDimension dim             = plan->dim;
        std::string identifier          = utility::combine_identifier(plan->value_identifier, transform_kind_cstr(transform_kind_sqrt));
        logit_kind_t logit_kind         = plan->logit_kind;

        return std::make_unique<AbstractNode>(AbstractNode{std::move(descendants), transform_kind, dim, std::move(identifier), logit_kind});
    }

    auto cublas_mono_invsqrt(const std::unique_ptr<AbstractNode>& plan) -> std::unique_ptr<AbstractNode>{

        using namespace syntax_tree;

        if (plan == nullptr){
            throw exception::invalid_argument();
        }

        auto descendants                = std::vector<std::unique_ptr<AbstractNode>>{utility::deepcopy(plan)};
        transform_kind_t transform_kind = transform_kind_invsqrt;
        MatrixDimension dim             = plan->dim;
        std::string identifier          = utility::combine_identifier(plan->value_identifier, transform_kind_cstr(transform_kind_invsqrt));
        logit_kind_t logit_kind         = plan->logit_kind; 

        return std::make_unique<AbstractNode>(AbstractNode{std::move(descendants), transform_kind, dim, std::move(identifier), logit_kind});
    }

    auto cublas_mono_negative(const std::unique_ptr<AbstractNode>& plan) -> std::unique_ptr<AbstractNode>{

        using namespace syntax_tree;

        if (plan == nullptr){
            throw exception::invalid_argument();
        }

        auto descendants                = std::vector<std::unique_ptr<AbstractNode>>{utility::deepcopy(plan)};
        transform_kind_t transform_kind = transform_kind_negative;
        MatrixDimension dim             = plan->dim;
        std::string identifier          = utility::combine_identifier(plan->value_identifier, transform_kind_cstr(transform_kind_negative));
        logit_kind_t logit_kind         = plan->logit_kind;

        return std::make_unique<AbstractNode>(AbstractNode{std::move(descendants), transform_kind, dim, std::move(identifier), logit_kind});
    }

    auto cublas_mono_negate(const std::unique_ptr<AbstractNode>& plan) -> std::unique_ptr<AbstractNode>{

        using namespace syntax_tree;

        if (plan == nullptr){
            throw exception::invalid_argument();
        }

        auto descendants                = std::vector<std::unique_ptr<AbstractNode>>{utility::deepcopy(plan)};
        transform_kind_t transform_kind = transform_kind_negate;
        MatrixDimension dim             = plan->dim;
        std::string identifier          = utility::combine_identifier(plan->value_identifier, transform_kind_cstr(transform_kind_negate));
        logit_kind_t logit_kind         = plan->logit_kind;

        return std::make_unique<AbstractNode>(AbstractNode{std::move(descendants), transform_kind, dim, std::move(identifier), logit_kind});
    }

    auto cublas_mono_transpose(const std::unique_ptr<AbstractNode>& plan) -> std::unique_ptr<AbstractNode>{

        using namespace syntax_tree;

        if (plan == nullptr){
            throw exception::invalid_argument();
        }

        auto descendants                = std::vector<std::unique_ptr<AbstractNode>>{utility::deepcopy(plan)};
        transform_kind_t transform_kind = transform_kind_transpose;
        MatrixDimension dim             = plan->dim;
        std::string identifier          = utility::combine_identifier(plan->value_identifier, transform_kind_cstr(transform_kind_transpose));
        logit_kind_t logit_kind         = plan->logit_kind;

        return std::make_unique<AbstractNode>(AbstractNode{std::move(descendants), transform_kind, dim, std::move(identifier), logit_kind});
    }

    auto cublas_pair_linear(const std::unique_ptr<AbstractNode>& lhs, const std::unique_ptr<AbstractNode>& rhs) -> std::unique_ptr<AbstractNode>{

        using namespace syntax_tree;

        if (lhs == nullptr){
            throw exception::invalid_argument();
        }

        if (rhs == nullptr){
            throw exception::invalid_argument();
        }

        if (lhs->dim.column_sz != rhs->dim.row_sz){
            throw exception::invalid_argument();
        }

        if (lhs->logit_kind != rhs->logit_kind){
            throw exception::invalid_argument();
        }

        auto descendants                = std::vector<std::unique_ptr<AbstractNode>>{utility::deepcopy(lhs), utility::deepcopy(rhs)};
        transform_kind_t transform_kind = transform_kind_linear;
        MatrixDimension dim             = MatrixDimension{lhs->dim.row_sz, rhs->dim.column_sz};
        std::string identifier          = utility::combine_identifier(utility::combine_identifier(lhs->value_identifier, rhs->value_identifier), transform_kind_cstr(transform_kind_linear));
        logit_kind_t logit_kind         = lhs->logit_kind;

        return std::make_unique<AbstractNode>(AbstractNode{std::move(descendants), transform_kind, dim, std::move(identifier), logit_kind});
    }

    auto cublas_pair_dot(const std::unique_ptr<AbstractNode>& lhs, const std::unique_ptr<AbstractNode>& rhs) -> std::unique_ptr<AbstractNode>{

        using namespace syntax_tree;

        if (lhs == nullptr){
            throw exception::invalid_argument();
        } 

        if (rhs == nullptr){
            throw exception::invalid_argument();
        }

        if (lhs->dim.column_sz != rhs->dim.column_sz){
            throw exception::invalid_argument();
        }

        if (lhs->dim.row_sz != rhs->dim.row_sz){
            throw exception::invalid_argument();
        }

        auto descendants                = std::vector<std::unique_ptr<AbstractNode>>{utility::deepcopy(lhs), utility::deepcopy(rhs)};
        transform_kind_t transform_kind = transform_kind_dot;
        MatrixDimension dim             = lhs->dim;
        std::string identifier          = utility::combine_identifier(utility::combine_identifier(lhs->value_identifier, rhs->value_identifier), transform_kind_cstr(transform_kind_linear));
        logit_kind_t logit_kind         = lhs->logit_kind;

        return std::make_unique<AbstractNode>(AbstractNode{std::move(descendants), transform_kind, dim, std::move(identifier), logit_kind});
    }

    auto cublas_pair_add(const std::unique_ptr<AbstractNode>& lhs, const std::unique_ptr<AbstractNode>& rhs) -> std::unique_ptr<AbstractNode>{

        using namespace syntax_tree;

        if (lhs == nullptr){
            throw exception::invalid_argument();
        }

        if (rhs == nullptr){
            throw exception::invalid_argument();
        }

        if (lhs->dim.column_sz != rhs->dim.column_sz){
            throw exception::invalid_argument();
        }

        if (lhs->dim.row_sz != rhs->dim.row_sz){
            throw exception::invalid_argument();
        }

        if (lhs->logit_kind != rhs->logit_kind){
            throw exception::invalid_argument();
        }

        auto descendants                = std::vector<std::unique_ptr<AbstractNode>>{utility::deepcopy(lhs), utility::deepcopy(rhs)};
        transform_kind_t transform_kind = transform_kind_add;
        MatrixDimension dim             = lhs->dim;
        std::string identifier          = utility::combine_identifier(utility::combine_identifier(lhs->value_identifier, rhs->value_identifier), transform_kind_cstr(transform_kind_add));
        logit_kind_t logit_kind         = lhs->logit_kind;

        return std::make_unique<AbstractNode>(AbstractNode{std::move(descendants), transform_kind, dim, std::move(identifier), logit_kind});
    }

    auto cublas_pair_sub(const std::unique_ptr<AbstractNode>& lhs, const std::unique_ptr<AbstractNode>& rhs) -> std::unique_ptr<AbstractNode>{

        using namespace syntax_tree;

        if (lhs == nullptr){
            throw exception::invalid_argument();
        }

        if (rhs == nullptr){
            throw exception::invalid_argument();
        }

        if (lhs->dim.column_sz != rhs->dim.column_sz){
            throw exception::invalid_argument();
        }

        if (lhs->dim.row_sz != rhs->dim.row_sz){
            throw exception::invalid_argument();
        }

        if (lhs->logit_kind != rhs->logit_kind){
            throw exception::invalid_argument();
        }

        auto descendants                = std::vector<std::unique_ptr<AbstractNode>>{utility::deepcopy(lhs), utility::deepcopy(rhs)};
        transform_kind_t transform_kind = transform_kind_sub;
        MatrixDimension dim             = lhs->dim;
        std::string identifier          = utility::combine_identifier(utility::combine_identifier(lhs->value_identifier, rhs->value_identifier), transform_kind_cstr(transform_kind_sub));
        logit_kind_t logit_kind         = lhs->logit_kind;

        return std::make_unique<AbstractNode>(AbstractNode{std::move(descendants), transform_kind, dim, std::move(identifier), logit_kind});
    }

    auto cublas_pair_mul(const std::unique_ptr<AbstractNode>& lhs, const std::unique_ptr<AbstractNode>& rhs) -> std::unique_ptr<AbstractNode>{

        using namespace syntax_tree;

        if (lhs == nullptr){
            throw exception::invalid_argument();
        }

        if (rhs == nullptr){
            throw exception::invalid_argument();
        }

        if (lhs->dim.column_sz != rhs->dim.column_sz){
            throw exception::invalid_argument();
        }

        if (lhs->dim.row_sz != rhs->dim.row_sz){
            throw exception::invalid_argument();
        }

        if (lhs->logit_kind != rhs->logit_kind){
            throw exception::invalid_argument();
        }

        auto descendants                = std::vector<std::unique_ptr<AbstractNode>>{utility::deepcopy(lhs), utility::deepcopy(rhs)};
        transform_kind_t transform_kind = transform_kind_mul;
        MatrixDimension dim             = lhs->dim;
        std::string identifier          = utility::combine_identifier(utility::combine_identifier(lhs->value_identifier, rhs->value_identifier), transform_kind_cstr(transform_kind_mul));
        logit_kind_t logit_kind         = lhs->logit_kind;

        return std::make_unique<AbstractNode>(AbstractNode{std::move(descendants), transform_kind, dim, std::move(identifier), logit_kind});
    }

    auto cublas_pair_div(const std::unique_ptr<AbstractNode>& lhs, const std::unique_ptr<AbstractNode>& rhs) -> std::unique_ptr<AbstractNode>{

        using namespace syntax_tree;

        if (lhs == nullptr){
            throw exception::invalid_argument();
        }

        if (rhs == nullptr){
            throw exception::invalid_argument();
        }

        if (lhs->dim.column_sz != rhs->dim.column_sz){
            throw exception::invalid_argument();
        }

        if (lhs->dim.row_sz != rhs->dim.row_sz){
            throw exception::invalid_argument();
        }

        if (lhs->logit_kind != rhs->logit_kind){
            throw exception::invalid_argument();
        }

        auto descendants                = std::vector<std::unique_ptr<AbstractNode>>{utility::deepcopy(lhs), utility::deepcopy(rhs)};
        transform_kind_t transform_kind = transform_kind_div;
        MatrixDimension dim             = lhs->dim;
        std::string identifier          = utility::combine_identifier(utility::combine_identifier(lhs->value_identifier, rhs->value_identifier), transform_kind_cstr(transform_kind_div));
        logit_kind_t logit_kind         = lhs->logit_kind;

        return std::make_unique<AbstractNode>(AbstractNode{std::move(descendants), transform_kind, dim, std::move(identifier), logit_kind});
    }

    auto cublas_pair_pow(const std::unique_ptr<AbstractNode>& lhs, const std::unique_ptr<AbstractNode>& rhs) -> std::unique_ptr<AbstractNode>{

        using namespace syntax_tree;

        if (lhs == nullptr){
            throw exception::invalid_argument();
        }

        if (rhs == nullptr){
            throw exception::invalid_argument();
        }

        if (lhs->dim.column_sz != rhs->dim.column_sz){
            throw exception::invalid_argument();
        }

        if (lhs->dim.row_sz != rhs->dim.row_sz){
            throw exception::invalid_argument();
        }

        auto descendants                = std::vector<std::unique_ptr<AbstractNode>>{utility::deepcopy(lhs), utility::deepcopy(rhs)};
        transform_kind_t transform_kind = transform_kind_pow;
        MatrixDimension dim             = lhs->dim;
        std::string identifier          = utility::combine_identifier(utility::combine_identifier(lhs->value_identifier, rhs->value_identifier), transform_kind_cstr(transform_kind_div));
        logit_kind_t logit_kind         = lhs->logit_kind;

        return std::make_unique<AbstractNode>(AbstractNode{std::move(descendants), transform_kind, dim, std::move(identifier), logit_kind});
    }

    auto cublas_pair_min(const std::unique_ptr<AbstractNode>& lhs, const std::unique_ptr<AbstractNode>& rhs) -> std::unique_ptr<AbstractNode>{

        using namespace syntax_tree;

        if (lhs == nullptr){
            throw exception::invalid_argument();
        }

        if (rhs == nullptr){
            throw exception::invalid_argument();
        }

        if (lhs->dim.column_sz != rhs->dim.column_sz){
            throw exception::invalid_argument();
        }

        if (lhs->dim.row_sz != rhs->dim.row_sz){
            throw exception::invalid_argument();
        }

        auto descendants                = std::vector<std::unique_ptr<AbstractNode>>{utility::deepcopy(lhs), utility::deepcopy(rhs)};
        transform_kind_t transform_kind = transform_kind_min;
        MatrixDimension dim             = lhs->dim;
        std::string identifier          = utility::combine_identifier(utility::combine_identifier(lhs->value_identifier, rhs->value_identifier), transform_kind_cstr(transform_kind_min));
        logit_kind_t logit_kind         = lhs->logit_kind;
        
        return std::make_unique<AbstractNode>(AbstractNode{std::move(descendants), transform_kind, dim, std::move(identifier), logit_kind});
    }

    auto cublas_pair_max(const std::unique_ptr<AbstractNode>& lhs, const std::unique_ptr<AbstractNode>& rhs) -> std::unique_ptr<AbstractNode>{

        using namespace syntax_tree;

        if (lhs == nullptr){
            throw exception::invalid_argument();
        }

        if (rhs == nullptr){
            throw exception::invalid_argument();
        }

        if (lhs->dim.column_sz != rhs->dim.column_sz){
            throw exception::invalid_argument();
        }

        if (lhs->dim.row_sz != rhs->dim.row_sz){
            throw exception::invalid_argument();
        }

        auto descendants                = std::vector<std::unique_ptr<AbstractNode>>{utility::deepcopy(lhs), utility::deepcopy(rhs)};
        transform_kind_t transform_kind = transform_kind_max;
        MatrixDimension dim             = lhs->dim;
        std::string identifier          = utility::combine_identifier(utility::combine_identifier(lhs->value_identifier, rhs->value_identifier), transform_kind_cstr(transform_kind_max));
        logit_kind_t logit_kind         = lhs->logit_kind;

        return std::make_unique<AbstractNode>(AbstractNode{std::move(descendants), transform_kind, dim, std::move(identifier), logit_kind});
    }

    auto cublas_optimize_fast(int device_id, const std::unique_ptr<AbstractNode>& plan, size_t buf_cap) -> std::unique_ptr<AbstractNode>{

    }

    auto cublas_optimize_slow(int device_id, const std::unique_ptr<AbstractNode>& plan, size_t buf_cap) -> std::unique_ptr<AbstractNode>{

    }

    auto cublas_make_executable(const std::unique_ptr<AbstractNode>& plan) -> std::unique_ptr<exec_engine::ExecutorInterface>{

    }
} 

#endif