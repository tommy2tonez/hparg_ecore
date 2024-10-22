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

    using logit_kind_t  = uint8_t; 
    
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

    // class EngineLookupInterface{

    //     public:

    //         virtual ~EngineLookupInterface() noexcept = default;
    //         virtual auto lookup(syntax_tree::transform_ins_t) -> std::shared_ptr<exec_engine::ExecutorInterface> = 0;
    // };
 
    class SpaceRandomizerInterface{

        public:

            virtual ~SpaceRandomizerInterface() noexcept = default;
            virtual auto randomize(const syntax_tree::Space&) -> std::shared_ptr<cuda_ptr_t> = 0;
    };

    class AbstractNodeRandomizerInterface{

        public:

            virtual ~AbstractNodeRandomizerInterface() noexcept = default;
            virtual auto randomize(const syntax_tree::AbstractNode&) -> syntax_tree::Node = 0;
    };

    class StateSearchEngineInterface{

        public:

            virtual ~StateSearchEngineInterface() noexcept = default;
            virtual auto search(const syntax_tree::AbstractNode&) -> std::vector<syntax_tree::AbstractNode> = 0;
    };

    class BenchmarkEngineInterface{

        public:

            virtual ~BenchmarkEngineInterface() noexcept = default;
            virtual auto benchmark(cublas_handle_t, const syntax_tree::Node&) -> std::chrono::nanoseconds = 0;
    };
} 

namespace dg::cublas_x::exhaustive_ss_opti_engine{

    class BaseStateSearchEngine: public virtual StateSearchEngineInterface{

        private:

        
        public:

            auto search(const syntax_tree::AbstractNode& node) -> std::vector<syntax_tree::AbstractNode>{

            }
    };

    class PermuteStateSearchEngine: public virtual StateSearchEngineInterface{

        private:

            std::unique_ptr<StateSearchEngineInterface> base_search;

        public:

            explicit PermuteStateSearchEngine(std::unique_ptr<StateSearchEngineInterface> base_search) noexcept: base_search(std::move(base_search)){}

            auto search(const syntax_tree::AbstractNode& node) -> std::vector<syntax_tree::AbstractNode>{

                return this->internal_search(node);
            }
        
        private:

            auto get_space(const std::vector<std::vector<syntax_tree::AbstractNode>>& inp) -> std::vector<size_t>{ //should be vector_utility

                auto rs = std::vector<size_t>();

                for (const auto& e: inp){
                    rs.push_back(e.size());
                }

                return rs;
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

            auto extract_abstract_node(const std::vector<std::vector<syntax_tree::AbstractNode>>& inp, const std::vector<size_t>& ptr) -> std::vector<syntax_tree::AbstractNode>{

                std::vector<syntax_tree::AbstractNode> rs{};

                for (size_t i = 0; i < ptr.size(); ++i){
                    size_t idx = ptr[i]; 
                    rs.push_back(syntax_tree::deepcopy(inp[i][idx]));
                }

                return rs;
            }

            auto make_root_possibilities(const syntax_tree::AbstractNode& node, const std::vector<std::vector<syntax_tree::AbstractNode>>& descendants) -> std::vector<syntax_tree::AbstractNode>{
                
                std::vector<syntax_tree::AbstractNode> rs{};
                std::vector<size_t> space   = this->get_space(descendants); 
                std::vector<size_t> ptr     = std::vector<size_t>(space.size(), 0u);

                while (ptr != space){
                    std::vector<syntax_tree::AbstractNode> cur_descendants = this->extract_abstract_node(descendants, ptr);
                    rs.push_back(syntax_tree::make_abstract_node(cur_descendants, node.transform_type, node.space));
                    this->iter_increment(ptr, space);
                }

                return rs;
            } 

            auto internal_search(const syntax_tree::AbstractNode& node) -> std::vector<syntax_tree::AbstractNode>{

                std::vector<std::vector<syntax_tree::AbstractNode>> descendant_permute_set{};

                for (size_t i = 0; i < node.descendants.size(); ++i){
                    descendant_permute_set.push_back(internal_search(*node.descendants[i]));
                }

                if (descendant_permute_set.size() == 0u){
                    return {node};
                }

                std::vector<syntax_tree::AbstractNode> root_possibilities   = this->make_root_possibilities(node, descendant_permute_set);
                std::vector<syntax_tree::AbstractNode> root_valids          = {};
                
                for (const syntax_tree::AbstractNode& cur: root_possibilities){
                    auto cur_valid = this->base_search->search(cur); 
                    std::copy(std::make_move_iterator(cur_valid.begin()), std::make_move_iterator(cur_valid.end()), std::back_inserter(root_valids));
                }

                return root_valids;
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
        
        return lhs + rhs;
    }
}

namespace dg::cublas_x{

    //alright - let's get this correct today and tomorrow - this is not as complicated as many people think - given a list of base-optimizables - build a tree of operations - this is 101 

    using AbstractNode  = syntax_tree::AbstractNode;
    using cublas_plan_t = std::unique_ptr<AbstractNode>;
    using logit_kind_t  = syntax_tree::logit_kind_t;

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