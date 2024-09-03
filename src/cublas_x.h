#ifndef __CUBLAS_X_H__
#define __CUBLAS_X_H__

#include <stdint.h>
#include <stdlib.h>
#include <vector>
#include <memory>
#include <string>
#include <chrono>

namespace dg::cublas_x::syntax_tree{

    using transform_ins_t = std::string;  

    struct Space{
        size_t row_sz;
        size_t column_sz;
        size_t replication_factor;
    };

    struct AbstractNode{
        std::vector<std::unique_ptr<AbstractNode>> descendants;
        std::string transform_type;
        Space space;
    };

    struct Node{
        std::vector<std::unique_ptr<Node>> descendants;
        transform_ins_t transform_type;
        std::shared_ptr<cuda_ptr_t> data;
    };
}

namespace dg::cublas_x::exec_engine{

    class ExecutorInterface{

        public:

            virtual ~ExecutorInterface() noexcept = default;
            virtual void exec(cublas_handle_t, cuda_ptr_t *) = 0;
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

            //this is a hard-problem
            //WLOG, assume binary tree 
            //split the tree into two after segmentation blocks (reachable by root, not-reachable by root)
            //reachable_by_root -> = ordered_hash of the graph -> direct look of dictionary
            //not_reachable_by_root -> child of reachable_by_root  
            //segmentation blocks permutation == |combinatorial_space(1, N)| + |combinatorial_space(2, N)| + ... + |combinatorial_space(N, N)| 
            //do tmr

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

namespace dg::cublas_x{

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

    using matrix_option_t  = uint8_t;
    using logit_option_t   = uint8_t; 
    
    enum logit_option: logit_option_t{
        u8          = 0u,
        u16         = 1u,
        u32         = 2u,
        f8          = 3u, //specifier
        f16brain    = 4u,
        f16iec559   = 5u,
        f32iec559   = 6u
    };

    auto cublas_make_matrix(size_t m, size_t n, size_t ld, logit_option_t logit_option) -> cublas_plan_t{

    } 

    auto cublas_make_matrix(size_t m, size_t n, logit_option_t logit_option) -> cublas_plan_t{

    }
    
    auto cublas_mono_saturate_01(cublas_plan_t) -> cublas_plan_t{

    } 

    auto cublas_mono_rounddown_optional(cublas_plan_t) -> cublas_plan_t{

    } 

    auto cublas_mono_rounddown(cublas_plan_t) -> cublas_plan_t{

    }

    auto cublas_mono_roundup_optional(cublas_plan_t) -> cublas_plan_t{

    }

    auto cublas_mono_roundup(cublas_plan_t) -> cublas_plan_t{

    }

    auto cublas_mono_fastmath(cublas_plan_t) -> cublas_plan_t{

    }

    auto cublas_mono_roundeven_optional(cublas_plan_t) -> cublas_plan_t{

    }

    auto cublas_mono_roundeven(cublas_plan_t) -> cublas_plan_t{

    }

    auto cublas_mono_roundzero_optional(cublas_plan_t) -> cublas_plan_t{

    }

    auto cublas_mono_roundzero(cublas_plan_t) -> cublas_plan_t{

    }

    auto cublas_mono_clone(cublas_plan_t) -> cublas_plan_t{

    }

    auto cublas_mono_relu(cublas_plan_t) -> cublas_plan_t{

    }

    auto cublas_mono_cast(cublas_plan_t, logit_option_t) -> cublas_plan_t{

    }

    auto cublas_mono_sign(cublas_plan_t) -> cublas_plan_t{

    }

    auto cublas_mono_exp(cublas_plan_t) -> cublas_plan_t{

    }

    auto cublas_mono_exp2(cublas_plan_t) -> cublas_plan_t{

    }

    auto cublas_mono_exp10(cublas_plan_t) -> cublas_plan_t{

    }

    auto cublas_mono_log(cublas_plan_t) -> cublas_plan_t{

    }

    auto cublas_mono_log2(cublas_plan_t) -> cublas_plan_t{

    }

    auto cublas_mono_log10(cublas_plan_t) -> cublas_plan_t{

    } 

    auto cublas_mono_abs(cublas_plan_t) -> cublas_plan_t{

    }

    auto cublas_mono_cos(cublas_plan_t) -> cublas_plan_t{

    }

    auto cublas_mono_acos(cublas_plan_t) -> cublas_plan_t{

    }

    auto cublas_mono_sin(cublas_plan_t) -> cublas_plan_t{

    }

    auto cublas_mono_asin(cublas_plan_t) -> cublas_plan_t{

    }

    auto cublas_mono_tan(cublas_plan_t) -> cublas_plan_t{

    }

    auto cublas_mono_atan(cublas_plan_t) -> cublas_plan_t{

    }

    auto cublas_mono_sqrt(cublas_plan_t) -> cublas_plan_t{

    }

    auto cublas_mono_invsqrt(cublas_plan_t) -> cublas_plan_t{

    }

    auto cublas_mono_negative(cublas_plan_t) -> cublas_plan_t{

    }

    auto cublas_mono_negate(cublas_plan_t) -> cublas_plan_t{

    }

    auto cublas_pair_linear(cublas_plan_t, cublas_plan_t) -> cublas_plan_t{

    }

    auto cublas_pair_tlinear(cublas_plan_t, cublas_plan_t) -> cublas_plan_t{

    }

    auto cublas_pair_dot(cublas_plan_t, cublas_plan_t) -> cublas_plan_t{

    }

    auto cublas_pair_add(cublas_plan_t, cublas_plan_t) -> cublas_plan_t{

    }

    auto cublas_pair_sub(cublas_plan_t, cublas_plan_t) -> cublas_plan_t{

    }

    auto cublas_pair_mul(cublas_plan_t, cublas_plan_t) -> cublas_plan_t{

    }

    auto cublas_pair_div(cublas_plan_t, cublas_plan_t) -> cublas_plan_t{

    }

    auto cublas_pair_pow(cublas_plan_t, cublas_plan_t) -> cublas_plan_t{

    }

    auto cublas_pair_min(cublas_plan_t, cublas_plan_t) -> cublas_plan_t{

    }

    auto cublas_pair_max(cublas_plan_t, cublas_plan_t) -> cublas_plan_t{

    }

    auto cublas_optimize_fast(cublas_handle_t, cublas_plan_t) -> cublas_plan_t{

    }

    auto cublas_optimize_slow(cublas_handle_t, cublas_plan_t) -> cublas_plan_t{

    }

    auto cublas_make_executable(cublas_plan_t) -> std::unique_ptr<ExecutorInterface>{

    }
} 

#endif