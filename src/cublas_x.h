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
#include <utility>
#include <algorithm>
#include <optional>
#include <random>
#include <mutex>
#include <functional>
#include "stdx.h"

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

        template <class Reflector>
        void dg_reflect(const Reflector& reflector){
            reflector(row_sz, column_sz);
        }

        template <class Reflector>
        void dg_reflect(const Reflector& reflector) const{
            reflector(row_sz, column_sz);
        }
    };
    
    struct AbstractNode{
        std::vector<std::unique_ptr<AbstractNode>> descendants;
        std::vector<size_t> descendants_positional_vec;
        transform_kind_t transform_kind;
        MatrixDimension dim;
        std::string val_id;
        std::string id;
        logit_kind_t logit_kind;

        template <class Reflector>
        void dg_reflect(const Reflector& reflector) const{
            reflector(descendants, descendants_positional_vec, transform_kind, dim, val_id, id, logit_kind);
        }

        template <class Reflector>
        void dg_reflect(const Reflector& reflector){
            reflector(descendants, descendants_positional_vec, transform_kind, dim, val_id, id, logit_kind);
        }
    };

    struct CollapsedAbstractNode{
        std::unique_ptr<AbstractNode> upper;
        std::vector<std::unique_ptr<AbstractNode>> leaf_descendants;

        template <class Reflector>
        void dg_reflect(const Reflector& reflector) const{
            reflector(upper, leaf_descendants);
        }

        template <class Reflector>
        void dg_reflect(const Reflector& reflector){
            reflector(upper, leaf_descendants);
        }
    };

    struct CudaPtr{
        void * cuda_ptr;
        int cuda_device_id;
    };

    struct Node{
        std::vector<std::unique_ptr<Node>> descendants;
        std::vector<size_t> descendants_positional_vec;
        transform_kind_t transform_kind;
        MatrixDimension dim;
        std::string value_identifier;
        logit_kind_t logit_kind;
        std::shared_ptr<CudaPtr> cuda_ptr;
    };

    auto transform_kind_cstr(transform_kind_t transform_kind) -> const char *{

        switch (transform_kind){
            case transform_kind_saturate_01:
                return "transform_kind_saturate_01";
            case transform_kind_rounddown_optional:
                return "transform_kind_rounddown_optional";
            case transform_kind_rounddown:
                return "transform_kind_rounddown";
            case transform_kind_roundup_optional:
                return "transform_kind_roundup_optional";
            case transform_kind_roundup:
                return "transform_kind_roundup";
            case transform_kind_fastmath:
                return "transform_kind_fastmath";
            case transform_kind_roundeven_optional:
                return "transform_kind_roundeven_optional";
            case transform_kind_roundeven:
                return "transform_kind_roundeven";
            case transform_kind_roundzero_optional:
                return "transform_kind_roundzero_optional";
            case transform_kind_roundzero:
                return "transform_kind_roundzero";
            case transform_kind_clone:
                return "transform_kind_clone";
            case transform_kind_relu:
                return "transform_kind_relu";
            case transform_kind_cast_u8:
                return "transform_kind_cast_u8";
            case transform_kind_cast_u16:
                return "transform_kind_cast_u16";
            case transform_kind_cast_u32:
                return "transform_kind_cast_u32";
            case transform_kind_cast_f8_native:
                return "transform_kind_cast_f8_native";
            case transform_kind_cast_f16_brain:
                return "transform_kind_cast_f16_brain";
            case transform_kind_cast_f16_iec559:
                return "transform_kind_cast_f16_iec559";
            case transform_kind_cast_f32_iec559:
                return "transform_kind_cast_f32_iec559";
            case transform_kind_sign:
                return "transform_kind_sign";
            case transform_kind_exp:
                return "transform_kind_exp";
            case transform_kind_exp2:
                return "transform_kind_exp2";
            case transform_kind_exp10:
                return "transform_kind_exp10";
            case transform_kind_log:
                return "transform_kind_log";
            case transform_kind_log2:
                return "transform_kind_log2";
            case transform_kind_log10:
                return "transform_kind_log10";
            case transform_kind_abs:
                return "transform_kind_abs";
            case transform_kind_cos:
                return "transform_kind_cos";
            case transform_kind_acos:
                return "transform_kind_acos";
            case transform_kind_sin:
                return "transform_kind_sin";
            case transform_kind_asin:
                return "transform_kind_asin";
            case transform_kind_tan:
                return "transform_kind_tan";
            case transform_kind_atan:
                return "transform_kind_atan";
            case transform_kind_sqrt:
                return "transform_kind_sqrt";
            case transform_kind_invsqrt:
                return "transform_kind_invsqrt";
            case transform_kind_negative:
                return "transform_kind_negative";
            case transform_kind_negate:
                return "tranform_kind_negate";
            case transform_kind_transpose:
                return "transform_kind_transpose";
            case transform_kind_linear:
                return "transform_kind_linear";
            case transform_kind_dot:
                return "transform_kind_dot";
            case transform_kind_add:
                return "transform_kind_add";
            case transform_kind_sub:
                return "transform_kind_sub";
            case transform_kind_mul:
                return "transform_kind_mul";
            case transform_kind_div:
                return "transform_kind_div";
            case transform_kind_pow:
                return "transform_kind_pow";
            case transform_kind_min:
                return "transform_kind_min";
            case transform_kind_max:
                return "transform_kind_max";
            case transform_kind_none:
                return "transform_kind_none";
            default:
                std::abort();
                return ""; 
            }
    }
}

namespace dg::cublas_x::exception{

    struct invalid_argument: std::exception{};
}

namespace dg::cublas_x::engine{

    class ExecutorInterface{

        public:

            virtual ~ExecutorInterface() noexcept = default;
            virtual void exec(int device_id, const std::unordered_map<std::string, void *>& arguments, void * dst, size_t dst_cap) = 0;
    };

    class OptimizerInterface{

        public:

            virtual ~OptimizerInterface() noexcept = default;
            virtual auto set_cuda_device(int) -> OptimizerInterface& = 0;
            virtual auto set_memory_cap(size_t) -> OptimizerInterface& = 0;
            virtual auto optimize(const std::unique_ptr<syntax_tree::AbstractNode>&) -> std::unique_ptr<syntax_tree::AbstractNode> = 0;
    };
}

namespace dg::cublas_x::utility{

    template <class T>
    auto deepcopy(const T& inp) -> T{

        dg::network_compact_serializer::deserialize<T>(dg::network_compact_serializer::serialize<std::string>(inp));
    }

    auto make_identifier(const std::string& identifier) -> std::string{

    } 

    auto combine_identifier(const std::string& lhs, const std::string& rhs) -> std::string{
        
    }

    struct IdFactory{

        private:

            static inline constexpr size_t ID_SIZE = 32;
            static inline std::mutex mtx{};
            static inline auto random_device = std::bind(std::uniform_int_distribution<char>{}, std::mt19937{});

        public:

            static auto next_id() -> std::string{

                auto lck_grd = stdx::lock_guard(mtx);
                std::string rs(ID_SIZE, ' ');
                std::generate(rs.begin(), rs.end(), std::ref(random_device));

                return rs;
            }
    };

    auto next_id() -> std::string{

        return IdFactory::next_id();
    }
}

namespace dg::cublas_x::exhaustive_ss_opti_engine{

    using namespace syntax_tree; 

    class SpaceRandomizerInterface{

        public:

            virtual ~SpaceRandomizerInterface() noexcept = default;
            virtual auto randomize(const syntax_tree::MatrixDimension&) -> std::shared_ptr<syntax_tree::CudaPtr> = 0;
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

    class AbstractNodeIdentifierGeneratorInterface{

        public:

            virtual ~AbstractNodeIdentifierGeneratorInterface() noexcept = default;
            virtual auto id(const std::unique_ptr<syntax_tree::AbstractNode>&) -> std::string = 0;
    };

    class AbstractNodeCollapserInterface{

        public:

            virtual ~AbstractNodeCollapserInterface() noexcept = default;
            virtual auto collapse(const std::unique_ptr<syntax_tree::AbstractNode>&) -> std::vector<std::unique_ptr<CollapsedAbstractNode>> = 0; 
    };
    
    class AbstractNodeOverheadCalculatorInterface{

        public:

            virtual ~AbstractNodeOverheadCalculatorInterface() noexcept = default;
            virtual auto get_memory_overhead(const std::unique_ptr<syntax_tree::AbstractNode>&) -> size_t = 0;
    };

    class AbstractNodeUniqueRepresentationGeneratorInterface{

        public:

            virtual ~AbstractNodeUniqueRepresentationGeneratorInterface() noexcept = default;
            virtual auto to_unique_representation(const std::unique_ptr<syntax_tree::AbstractNode>&) -> std::unique_ptr<syntax_tree::AbstractNode> = 0;
    };

    class AbstractNodeCollapser: public virtual AbstractNodeCollapserInterface{

        public:

            auto collapse(const std::unique_ptr<syntax_tree::AbstractNode>& root) -> std::vector<std::unique_ptr<CollapsedAbstractNode>>{

                return this->internal_collapse(root);
            }
        
        private:

            auto get_space(const std::vector<std::vector<std::unique_ptr<CollapsedAbstractNode>>>& inp) -> std::vector<size_t>{

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

                return std::accumulate(space.begin(), space.end(), size_t{1u}, std::multiplies<>{});
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

            auto make_collapsed_node(const std::unique_ptr<syntax_tree::AbstractNode>& root, const std::vector<std::unique_ptr<CollapsedAbstractNode>>& descendants) -> std::unique_ptr<CollapsedAbstractNode>{

                if (root == nullptr){
                    std::abort();
                }

                auto rs     = std::make_unique<CollapsedAbstractNode>();
                rs->upper   = utility::deepcopy(root);
                
                for (size_t i = 0u; i < descendants.size(); ++i){
                    rs->upper->descendants[i] = utility::deepcopy(descendants[i]->upper);
                    std::vector<std::unique_ptr<AbstractNode>> cur_leaf_descendants = utility::deepcopy(descendants[i]->leaf_descendants);
                    std::copy(std::make_move_iterator(cur_leaf_descendants.begin()), std::make_move_iterator(cur_leaf_descendants.end()), std::back_inserter(rs->leaf_descendants));
                }

                return rs;
            } 

            auto permute_join(const std::unique_ptr<syntax_tree::AbstractNode>& root, const std::vector<std::vector<std::unique_ptr<CollapsedAbstractNode>>>& descendants) -> std::vector<std::unique_ptr<CollapsedAbstractNode>>{
                
                if (root == nullptr){
                    std::abort();
                }

                std::vector<size_t> space   = this->get_space(descendants);
                auto rs                     = std::vector<std::unique_ptr<CollapsedAbstractNode>>{};
                auto ptr                    = std::vector<size_t>(0u, space.size());
                size_t idx                  = 0u;
                size_t space_size           = this->get_space_size(space);

                while (idx != space_size){
                    std::vector<std::unique_ptr<CollapsedAbstractNode>> cand = {};

                    for (size_t i = 0u; i < ptr.size(); ++i){
                        cand.push_back(utility::deepcopy(descendants[i][ptr[i]]));
                    }

                    std::unique_ptr<CollapsedAbstractNode> collapsed_node = this->make_collapsed_node(root, cand); 
                    rs.push_back(std::move(collapsed_node));
                    ++idx;
                    iter_increment(ptr, space);
                }

                return rs;
            }

            auto internal_collapse(const std::unique_ptr<syntax_tree::AbstractNode>& root) -> std::vector<std::unique_ptr<CollapsedAbstractNode>>{
                
                if (root == nullptr){
                    return {};
                }

                auto descendants_collapsed_vec = std::vector<std::vector<std::unique_ptr<CollapsedAbstractNode>>>{};

                for (const auto& descendant: root->descendants){
                    descendants_collapsed_vec.push_back(this->internal_collapse(descendant));
                }

                std::vector<std::unique_ptr<CollapsedAbstractNode>> rs      = this->permute_join(root, descendants_collapsed_vec);
                std::unique_ptr<CollapsedAbstractNode> self_abstract_node   = std::make_unique<CollapsedAbstractNode>();
                self_abstract_node->upper                                   = utility::deepcopy(root);
                self_abstract_node->upper->descendants                      = {};
                self_abstract_node->upper->descendants_positional_vec       = {};
                self_abstract_node->upper->transform_kind                   = syntax_tree::transform_kind_none;
                self_abstract_node->leaf_descendants.push_back(utility::deepcopy(root));
                rs.push_back(std::move(self_abstract_node));

                return rs;
            }
    };

    class AbstractNodeTransformIdentifierGenerator: public virtual AbstractNodeIdentifierGeneratorInterface{

        public:

            auto id(const std::unique_ptr<AbstractNode>& root) -> std::string{

                std::vector<transform_kind_t> transform_kind_vec{};
                this->postorder_traversal(root, transform_kind_vec);

                return dg::network_compact_serializer::serialize<std::string>(transform_kind_vec);
            }
        
        private:

            void postorder_traversal(const std::unique_ptr<AbstractNode>& root, std::vector<transform_kind_t>& rs){

                if (root == nullptr){
                    return;
                }

                for (const auto& descendant: root->descendants){
                    this->postorder_traversal(descendant, rs);
                }
                
                rs.push_back(root->transform_kind);
            }
    };

    class AbstractNodeUniqueRepresentationGenerator: public virtual AbstractNodeUniqueRepresentationGeneratorInterface{

        private:

            std::unique_ptr<AbstractNodeIdentifierGeneratorInterface> id_gen;

        public:

            AbstractNodeUniqueRepresentationGenerator(std::unique_ptr<AbstractNodeIdentifierGeneratorInterface> id_gen) noexcept: id_gen(std::move(id_gen)){}

            auto to_unique_representation(const std::unique_ptr<AbstractNode>& root) -> std::unique_ptr<AbstractNode>{

                if (root == nullptr){
                    return nullptr;
                }

                std::vector<std::unique_ptr<AbstractNode>> descendant_vec{};
                std::unique_ptr<AbstractNode> rs = utility::deepcopy(root);
                rs->descendants = {};

                for (const auto& descendant: root->descendants){
                    descendant_vec.push_back(this->to_unique_representation(descendant));
                }

                std::vector<std::pair<std::string, size_t>> cmpable_representation_vec{};

                for (const auto& descendant: descendant_vec){
                    cmpable_representation_vec.push_back(std::make_pair(this->id_gen->id(descendant), cmpable_representation_vec.size()));
                }

                std::sort(cmpable_representation_vec.begin(), cmpable_representation_vec.end());

                for (size_t i = 0u; i < cmpable_representation_vec.size(); ++i){
                    size_t old_idx = std::get<1>(cmpable_representation_vec[i]);
                    rs->descendants_positional_vec[i] = root->descendants_positional_vec[old_idx];
                    rs->descendants.push_back(std::move(descendant_vec[old_idx]));
                }

                return rs;
            }
    };

    class AbstractNodeOverheadCalculator: public virtual AbstractNodeOverheadCalculatorInterface{

        public:

            auto get_memory_overhead(const std::unique_ptr<AbstractNode>& root) -> size_t{
                
                if (root == nullptr){
                    return 0u;
                }

                size_t rs = root->dim.column_sz * root->dim.row_sz;

                for (const auto& descendant: root->descendants){
                    rs += this->get_memory_overhead(descendant);
                }

                return rs;
            }
    };

    class BaseStateSearchEngine: public virtual StateSearchEngineInterface{

        private:

            std::unordered_map<std::string, transform_kind_t> transform_map;
            std::unique_ptr<AbstractNodeIdentifierGeneratorInterface> id_gen;
            std::unique_ptr<AbstractNodeCollapserInterface> abstract_node_collapser;
            std::unique_ptr<AbstractNodeUniqueRepresentationGeneratorInterface> abstract_node_unique_rep_generator;

        public:

            BaseStateSearchEngine(std::unordered_map<std::string, transform_kind_t> transform_map,
                                  std::unique_ptr<AbstractNodeIdentifierGeneratorInterface> id_gen,
                                  std::unique_ptr<AbstractNodeCollapserInterface> abstract_node_collapser,
                                  std::unique_ptr<AbstractNodeUniqueRepresentationGeneratorInterface> abstract_node_unique_rep_generator) noexcept: transform_map(std::move(transform_map)),
                                                                                                                                                    id_gen(std::move(id_gen)),
                                                                                                                                                    abstract_node_collapser(std::move(abstract_node_collapser)),
                                                                                                                                                    abstract_node_unique_rep_generator(std::move(abstract_node_unique_rep_generator)){}

            auto search(const std::unique_ptr<AbstractNode>& root) -> std::vector<std::unique_ptr<AbstractNode>>{

                std::vector<std::unique_ptr<CollapsedAbstractNode>> collapsed_node_vec = this->abstract_node_collapser->collapse(root);
                std::vector<std::unique_ptr<AbstractNode>> rs{};

                for (const auto& collapsed_node: collapsed_node_vec){
                    std::unique_ptr<AbstractNode> uniq_rep  = this->abstract_node_unique_rep_generator->to_unique_representation(collapsed_node->upper);
                    std::string id                          = this->id_gen->id(uniq_rep);
                    auto map_ptr                            = this->transform_map.find(id);

                    if (map_ptr == this->transform_map.end()){
                        continue;
                    }

                    auto appendee = this->map_descendants(collapsed_node->upper, uniq_rep, collapsed_node->leaf_descendants);
                    rs.push_back(std::move(appendee));
                }

                return rs;
            }

        private:

            void postorder_traversal_leaf_identifier(const std::unique_ptr<AbstractNode>& root, std::vector<std::string>& identifier_vec){

                if (root == nullptr){
                    return;
                }

                for (auto& descendant: root->descendants){
                    this->postorder_traversal_leaf_identifier(descendant, identifier_vec);
                }

                if (root->descendants.empty()){
                    identifier_vec.push_back(root->id);
                }
            }

            auto map_descendants(const std::unique_ptr<AbstractNode>& old_node, const std::unique_ptr<AbstractNode>& new_node, const std::vector<std::unique_ptr<AbstractNode>>& descendants) -> std::unique_ptr<AbstractNode>{

                if (new_node == nullptr){
                    return nullptr;
                }

                auto old_node_leaf_identifier_vec   = std::vector<std::string>{};
                auto new_node_leaf_identifier_vec   = std::vector<std::string>{};
                auto old_descendants                = utility::deepcopy(descendants);
                auto new_descendants                = std::vector<std::unique_ptr<AbstractNode>>{};
                auto rs                             = utility::deepcopy(new_node); 

                this->postorder_traversal_leaf_identifier(old_node, old_node_leaf_identifier_vec);
                this->postorder_traversal_leaf_identifier(new_node, new_node_leaf_identifier_vec);

                for (size_t i = 0u ; i < descendants.size(); ++i){
                    auto ptr    = std::find(old_node_leaf_identifier_vec.begin(), old_node_leaf_identifier_vec.end(), new_node_leaf_identifier_vec[i]);
                    size_t idx  = std::distance(old_node_leaf_identifier_vec.begin(), ptr);
                    auto dptr   = descendants.begin() + idx; 

                    new_descendants.push_back(utility::deepcopy(*dptr));
                    old_node_leaf_identifier_vec.erase(ptr);
                    old_descendants.erase(dptr);
                }

                rs->descendants = std::move(new_descendants);
                return rs;
            }
    };

    class CombinatorialStateSearchEngine: public virtual StateSearchEngineInterface{

        private:

            std::unique_ptr<StateSearchEngineInterface> base_search;

        public:

            CombinatorialStateSearchEngine(std::unique_ptr<StateSearchEngineInterface> base_search) noexcept: base_search(std::move(base_search)){}

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

                return std::accumulate(space.begin(), space.end(), size_t{1u}, std::multiplies<>{});
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

            auto make_root_possibilities(const std::unique_ptr<AbstractNode>& root, 
                                         const std::vector<std::vector<std::unique_ptr<AbstractNode>>>& descendants) -> std::vector<std::unique_ptr<AbstractNode>>{
                
                auto rs                     = std::vector<std::unique_ptr<AbstractNode>>{};
                std::vector<size_t> space   = this->get_space(descendants); 
                std::vector<size_t> ptr     = std::vector<size_t>(space.size(), 0u);
                size_t space_size           = this->get_space_size(space);
                size_t idx                  = 0u; 

                while (idx != space_size){
                    auto appendee = std::make_unique<AbstractNode>();
                    
                    for (size_t i = 0u; i < ptr.size(); ++i){
                        appendee->descendants.push_back(utility::deepcopy(descendants[i][ptr[i]]));
                    }
                    
                    appendee->descendants_positional_vec    = root->descendants_positional_vec;
                    appendee->transform_kind                = root->transform_kind;
                    appendee->dim                           = root->dim;
                    appendee->val_id                        = root->val_id;
                    appendee->id                            = root->id;
                    appendee->logit_kind                    = root->logit_kind;

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

    class OptimizerEngine: public virtual engine::OptimizerInterface{

        private:

            struct OptimizationConfig{
                int cuda_device_id;
                size_t optimization_memory_cap;
            };

            std::unique_ptr<AbstractNodeRandomizerInterface> abstract_node_randomizer;
            std::unique_ptr<StateSearchEngineInterface> state_search_engine;
            std::unique_ptr<BenchmarkEngineInterface> benchmark_engine;
            std::unique_ptr<AbstractNodeOverheadCalculatorInterface> overhead_calculator;
            OptimizationConfig config;

        public:

            OptimizerEngine(std::unique_ptr<AbstractNodeRandomizerInterface> abstract_node_randomizer,
                            std::unique_ptr<StateSearchEngineInterface> state_search_engine,
                            std::unique_ptr<BenchmarkEngineInterface> benchmark_engine,
                            std::unique_ptr<AbstractNodeOverheadCalculatorInterface> overhead_calculator) noexcept: abstract_node_randomizer(std::move(abstract_node_randomizer)),
                                                                                                                    state_search_engine(std::move(state_search_engine)),
                                                                                                                    benchmark_engine(std::move(benchmark_engine)),
                                                                                                                    overhead_calculator(std::move(overhead_calculator)),
                                                                                                                    config(OptimizationConfig{-1, std::numeric_limits<size_t>::max()}){}

            auto set_cuda_device(int cuda_device_id) -> OptimizerInterface&{

                this->config.cuda_device_id = cuda_device_id;
                return *this;
            }
            
            auto set_memory_cap(size_t memory_cap) -> OptimizerInterface&{

                this->config.optimization_memory_cap = memory_cap;
                return *this;
            }

            auto optimize(const std::unique_ptr<AbstractNode>& root) -> std::unique_ptr<AbstractNode>{

                if (!root){
                    return {};
                }

                std::chrono::nanoseconds max_ts     = std::chrono::duration_values<std::chrono::nanoseconds>::max();
                std::unique_ptr<AbstractNode> rs    = utility::deepcopy(root);

                for (const std::unique_ptr<AbstractNode>& abstract_state: this->state_search_engine->search(root)){
                    size_t memory_overhead = this->overhead_calculator->get_memory_overhead(abstract_state);

                    if (memory_overhead > config.optimization_memory_cap){
                        continue;
                    }

                    std::unique_ptr<Node> state     = this->abstract_node_randomizer->randomize(abstract_state);
                    std::chrono::nanoseconds cur_ts = this->benchmark_engine->benchmark(config.cuda_device_id, state);

                    if (cur_ts < max_ts){
                        max_ts = cur_ts;
                        rs = utility::deepcopy(abstract_state);
                    }
                }

                size_t memory_overhead = this->overhead_calculator->get_memory_overhead(rs);

                if (memory_overhead > config.optimization_memory_cap){
                    throw std::exception();
                }

                return rs;
            } 
    };
} 

namespace dg::cublas_x{

    //goals tmr:   - get basic optimizations done - skip, fuse, simd  
    //             - optimization tree height -> 1 << 6
    //I'll be right back
    
    using AbstractNode  = syntax_tree::AbstractNode;
    using cublas_plan_t = std::unique_ptr<AbstractNode>;
    using logit_kind_t  = syntax_tree::logit_kind_t;

    auto cublas_make_matrix(size_t m, size_t n, logit_kind_t logit_kind, std::optional<std::string> id = std::nullopt, std::optional<std::string> val_id = std::nullopt) -> std::unique_ptr<AbstractNode>{
        
        using namespace syntax_tree; 

        if (m == 0u){
            throw exception::invalid_argument();
        }

        if (n == 0u){
            throw exception::invalid_argument{};
        }

        if (!id.has_value()){
            id = utility::next_id(); 
        }

        if (!val_id.has_value()){
            val_id = utility::next_id();
        }

        return std::make_unique<AbstractNode>(AbstractNode{{}, {}, transform_kind_none, MatrixDimension{m, n}, *val_id, *id, logit_kind});
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
        auto positional_vec             = std::vector<size_t>{0u};
        transform_kind_t transform_kind = transform_kind_saturate_01;
        MatrixDimension dim             = plan->dim;
        std::string val_id              = utility::combine_identifier(plan->val_id, transform_kind_cstr(transform_kind_saturate_01));
        std::string id                  = utility::combine_identifier(plan->id, utility::next_id());
        logit_kind_t logit_kind         = plan->logit_kind; 

        return std::make_unique<AbstractNode>(AbstractNode{std::move(descendants), std::move(positional_vec), transform_kind, dim, std::move(val_id), std::move(id), logit_kind});
    }

    auto cublas_mono_rounddown_optional(const std::unique_ptr<AbstractNode>& plan) -> std::unique_ptr<AbstractNode>{

        using namespace syntax_tree;

        if (plan == nullptr){
            throw exception::invalid_argument();
        }

        auto descendants                = std::vector<std::unique_ptr<AbstractNode>>{utility::deepcopy(plan)};
        auto positional_vec             = std::vector<size_t>{0u};
        transform_kind_t transform_kind = transform_kind_rounddown_optional;
        MatrixDimension dim             = plan->dim;
        std::string val_id              = utility::combine_identifier(plan->val_id, transform_kind_cstr(transform_kind_rounddown_optional));
        std::string id                  = utility::combine_identifier(plan->id, utility::next_id());
        logit_kind_t logit_kind         = plan->logit_kind; 

        return std::make_unique<AbstractNode>(AbstractNode{std::move(descendants), std::move(positional_vec), transform_kind, dim, std::move(val_id), std::move(id), logit_kind});
    } 

    auto cublas_mono_rounddown(const std::unique_ptr<AbstractNode>& plan) -> std::unique_ptr<AbstractNode>{

        using namespace syntax_tree;

        if (plan == nullptr){
            throw exception::invalid_argument();
        }

        auto descendants                = std::vector<std::unique_ptr<AbstractNode>>{utility::deepcopy(plan)};
        auto positional_vec             = std::vector<size_t>{0u};
        transform_kind_t transform_kind = transform_kind_rounddown;
        MatrixDimension dim             = plan->dim;
        std::string val_id              = utility::combine_identifier(plan->val_id, transform_kind_cstr(transform_kind_rounddown)); 
        std::string id                  = utility::combine_identifier(plan->id, utility::next_id());
        logit_kind_t logit_kind         = plan->logit_kind; 

        return std::make_unique<AbstractNode>(AbstractNode{std::move(descendants), std::move(positional_vec), transform_kind, dim, std::move(val_id), std::move(id), logit_kind});
    }

    auto cublas_mono_roundup_optional(const std::unique_ptr<AbstractNode>& plan) -> std::unique_ptr<AbstractNode>{

        using namespace syntax_tree;

        if (plan == nullptr){
            throw exception::invalid_argument();
        }

        auto descendants                = std::vector<std::unique_ptr<AbstractNode>>{utility::deepcopy(plan)};
        auto positional_vec             = std::vector<size_t>{0u};
        transform_kind_t transform_kind = transform_kind_roundup_optional;
        MatrixDimension dim             = plan->dim;
        std::string val_id              = utility::combine_identifier(plan->val_id, transform_kind_cstr(transform_kind_roundup_optional));
        std::string id                  = utility::combine_identifier(plan->id, utility::next_id());
        logit_kind_t logit_kind         = plan->logit_kind; 

        return std::make_unique<AbstractNode>(AbstractNode{std::move(descendants), std::move(positional_vec), transform_kind, dim, std::move(val_id), std::move(id), logit_kind});
    }

    auto cublas_mono_roundup(const std::unique_ptr<AbstractNode>& plan) -> std::unique_ptr<AbstractNode>{

        using namespace syntax_tree;

        if (plan == nullptr){
            throw exception::invalid_argument();
        }

        auto descendants                = std::vector<std::unique_ptr<AbstractNode>>{utility::deepcopy(plan)};
        auto positional_vec             = std::vector<size_t>{0u};
        transform_kind_t transform_kind = transform_kind_roundup;
        MatrixDimension dim             = plan->dim;
        std::string val_id              = utility::combine_identifier(plan->val_id, transform_kind_cstr(transform_kind_roundup));
        std::string id                  = utility::combine_identifier(plan->id, utility::next_id());
        logit_kind_t logit_kind         = plan->logit_kind; 

        return std::make_unique<AbstractNode>(AbstractNode{std::move(descendants), std::move(positional_vec), transform_kind, dim, std::move(val_id), std::move(id), logit_kind});
    }

    auto cublas_mono_fastmath(const std::unique_ptr<AbstractNode>& plan) -> std::unique_ptr<AbstractNode>{

        using namespace syntax_tree;
        
        if (plan == nullptr){
            throw exception::invalid_argument();
        }

        auto descendants                = std::vector<std::unique_ptr<AbstractNode>>{utility::deepcopy(plan)};
        auto positional_vec             = std::vector<size_t>{0u};
        transform_kind_t transform_kind = transform_kind_fastmath;
        MatrixDimension dim             = plan->dim;
        std::string val_id              = utility::combine_identifier(plan->val_id, transform_kind_cstr(transform_kind_fastmath));
        std::string id                  = utility::combine_identifier(plan->id, utility::next_id());
        logit_kind_t logit_kind         = plan->logit_kind; 

        return std::make_unique<AbstractNode>(AbstractNode{std::move(descendants), std::move(positional_vec), transform_kind, dim, std::move(val_id), std::move(id), logit_kind});
    }

    auto cublas_mono_roundeven_optional(const std::unique_ptr<AbstractNode>& plan) -> std::unique_ptr<AbstractNode>{

        using namespace syntax_tree;

        if (plan == nullptr){
            throw exception::invalid_argument();
        }

        auto descendants                = std::vector<std::unique_ptr<AbstractNode>>{utility::deepcopy(plan)};
        auto positional_vec             = std::vector<size_t>{0u};
        transform_kind_t transform_kind = transform_kind_roundeven_optional;
        MatrixDimension dim             = plan->dim;
        std::string val_id              = utility::combine_identifier(plan->val_id, transform_kind_cstr(transform_kind_roundeven_optional));
        std::string id                  = utility::combine_identifier(plan->id, utility::next_id());
        logit_kind_t logit_kind         = plan->logit_kind; 

        return std::make_unique<AbstractNode>(AbstractNode{std::move(descendants), std::move(positional_vec), transform_kind, dim, std::move(val_id), std::move(id), logit_kind});
    }

    auto cublas_mono_roundeven(const std::unique_ptr<AbstractNode>& plan) -> std::unique_ptr<AbstractNode>{

        using namespace syntax_tree;

        if (plan == nullptr){
            throw exception::invalid_argument();
        }

        auto descendants                = std::vector<std::unique_ptr<AbstractNode>>{utility::deepcopy(plan)};
        auto positional_vec             = std::vector<size_t>{0u};
        transform_kind_t transform_kind = transform_kind_roundeven;
        MatrixDimension dim             = plan->dim;
        std::string val_id              = utility::combine_identifier(plan->val_id, transform_kind_cstr(transform_kind_roundeven));
        std::string id                  = utility::combine_identifier(plan->id, utility::next_id());
        logit_kind_t logit_kind         = plan->logit_kind; 

        return std::make_unique<AbstractNode>(AbstractNode{std::move(descendants), std::move(positional_vec), transform_kind, dim, std::move(val_id), std::move(id), logit_kind});
    }

    auto cublas_mono_roundzero_optional(const std::unique_ptr<AbstractNode>& plan) -> std::unique_ptr<AbstractNode>{

        using namespace syntax_tree;

        if (plan == nullptr){
            throw exception::invalid_argument();
        }

        auto descendants                = std::vector<std::unique_ptr<AbstractNode>>{utility::deepcopy(plan)};
        auto positional_vec             = std::vector<size_t>{0u};
        transform_kind_t transform_kind = transform_kind_roundzero_optional;
        MatrixDimension dim             = plan->dim;
        std::string val_id              = utility::combine_identifier(plan->val_id, transform_kind_cstr(transform_kind_roundzero_optional));
        std::string id                  = utility::combine_identifier(plan->id, utility::next_id());
        logit_kind_t logit_kind         = plan->logit_kind; 

        return std::make_unique<AbstractNode>(AbstractNode{std::move(descendants), std::move(positional_vec), transform_kind, dim, std::move(val_id), std::move(id), logit_kind});
    }

    auto cublas_mono_roundzero(const std::unique_ptr<AbstractNode>& plan) -> std::unique_ptr<AbstractNode>{

        using namespace syntax_tree;

        if (plan == nullptr){
            throw exception::invalid_argument();
        }

        auto descendants                = std::vector<std::unique_ptr<AbstractNode>>{utility::deepcopy(plan)};
        auto positional_vec             = std::vector<size_t>{0u};
        transform_kind_t transform_kind = transform_kind_roundzero;
        MatrixDimension dim             = plan->dim;
        std::string val_id              = utility::combine_identifier(plan->val_id, transform_kind_cstr(transform_kind_roundzero));
        std::string id                  = utility::combine_identifier(plan->id, utility::next_id());
        logit_kind_t logit_kind         = plan->logit_kind;

        return std::make_unique<AbstractNode>(AbstractNode{std::move(descendants), std::move(positional_vec), transform_kind, dim, std::move(val_id), std::move(id), logit_kind});
    }

    auto cublas_mono_clone(const std::unique_ptr<AbstractNode>& plan) -> std::unique_ptr<AbstractNode>{

        using namespace syntax_tree;

        if (plan == nullptr){
            throw exception::invalid_argument();
        }

        auto descendants                = std::vector<std::unique_ptr<AbstractNode>>{utility::deepcopy(plan)};
        auto positional_vec             = std::vector<size_t>{0u};
        transform_kind_t transform_kind = transform_kind_clone;
        MatrixDimension dim             = plan->dim;
        std::string val_id              = utility::combine_identifier(plan->val_id, transform_kind_cstr(transform_kind_clone));
        std::string id                  = utility::combine_identifier(plan->id, utility::next_id());
        logit_kind_t logit_kind         = plan->logit_kind; 

        return std::make_unique<AbstractNode>(AbstractNode{std::move(descendants), std::move(positional_vec), transform_kind, dim, std::move(val_id), std::move(id), logit_kind});
    }

    auto cublas_mono_relu(const std::unique_ptr<AbstractNode>& plan) -> std::unique_ptr<AbstractNode>{

        using namespace syntax_tree;

        if (plan == nullptr){
            throw exception::invalid_argument();
        }

        auto descendants                = std::vector<std::unique_ptr<AbstractNode>>{utility::deepcopy(plan)};
        auto positional_vec             = std::vector<size_t>{0u};
        transform_kind_t transform_kind = transform_kind_relu;
        MatrixDimension dim             = plan->dim;
        std::string val_id              = utility::combine_identifier(plan->val_id, transform_kind_cstr(transform_kind_relu));
        std::string id                  = utility::combine_identifier(plan->id, utility::next_id());
        logit_kind_t logit_kind         = plan->logit_kind;

        return std::make_unique<AbstractNode>(AbstractNode{std::move(descendants), std::move(positional_vec), transform_kind, dim, std::move(val_id), std::move(id), logit_kind});
    }

    auto cublas_mono_cast(const std::unique_ptr<AbstractNode>& plan, logit_kind_t logit_kind) -> std::unique_ptr<AbstractNode>{

        using namespace syntax_tree;

        if (plan == nullptr){
            throw exception::invalid_argument();
        }

        auto descendants    = std::vector<std::unique_ptr<AbstractNode>>{utility::deepcopy(plan)};
        auto positional_vec = std::vector<size_t>{0u};
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
        std::string val_id      = utility::combine_identifier(plan->val_id, transform_kind_cstr(transform_kind));
        std::string id          = utility::combine_identifier(plan->id, utility::next_id()); 

        return std::make_unique<AbstractNode>(AbstractNode{std::move(descendants), std::move(positional_vec), transform_kind, dim, std::move(val_id), std::move(id), logit_kind});
    }

    auto cublas_mono_sign(const std::unique_ptr<AbstractNode>& plan) -> std::unique_ptr<AbstractNode>{

        using namespace syntax_tree;

        if (plan == nullptr){
            throw exception::invalid_argument();
        }

        auto descendants                = std::vector<std::unique_ptr<AbstractNode>>{utility::deepcopy(plan)};
        auto positional_vec             = std::vector<size_t>{0u};
        transform_kind_t transform_kind = transform_kind_sign;
        MatrixDimension dim             = plan->dim;
        std::string val_id              = utility::combine_identifier(plan->val_id, transform_kind_cstr(transform_kind_sign));
        std::string id                  = utility::combine_identifier(plan->id, utility::next_id()); 
        logit_kind_t logit_kind         = plan->logit_kind;

        return std::make_unique<AbstractNode>(AbstractNode{std::move(descendants), std::move(positional_vec), transform_kind, dim, std::move(val_id), std::move(id), logit_kind});
    }

    auto cublas_mono_exp(const std::unique_ptr<AbstractNode>& plan) -> std::unique_ptr<AbstractNode>{

        using namespace syntax_tree;

        if (plan == nullptr){
            throw exception::invalid_argument();
        }

        auto descendants                = std::vector<std::unique_ptr<AbstractNode>>{utility::deepcopy(plan)};
        auto positional_vec             = std::vector<size_t>{0u};
        transform_kind_t transform_kind = transform_kind_exp;
        MatrixDimension dim             = plan->dim;
        std::string val_id              = utility::combine_identifier(plan->val_id, transform_kind_cstr(transform_kind_exp));
        std::string id                  = utility::combine_identifier(plan->id, utility::next_id());
        logit_kind_t logit_kind         = plan->logit_kind;

        return std::make_unique<AbstractNode>(AbstractNode{std::move(descendants), std::move(positional_vec), transform_kind, dim, std::move(val_id), std::move(id), logit_kind});
    }

    auto cublas_mono_exp2(const std::unique_ptr<AbstractNode>& plan) -> std::unique_ptr<AbstractNode>{

        using namespace syntax_tree;

        if (plan == nullptr){
            throw exception::invalid_argument();
        }

        auto descendants                = std::vector<std::unique_ptr<AbstractNode>>{utility::deepcopy(plan)};
        auto positional_vec             = std::vector<size_t>{0u};
        transform_kind_t transform_kind = transform_kind_exp2;
        MatrixDimension dim             = plan->dim;
        std::string val_id              = utility::combine_identifier(plan->val_id, transform_kind_cstr(transform_kind_exp2));
        std::string id                  = utility::combine_identifier(plan->id, utility::next_id());
        logit_kind_t logit_kind         = plan->logit_kind;

        return std::make_unique<AbstractNode>(AbstractNode{std::move(descendants), std::move(positional_vec), transform_kind, dim, std::move(val_id), std::move(id), logit_kind});
    }

    auto cublas_mono_exp10(const std::unique_ptr<AbstractNode>& plan) -> std::unique_ptr<AbstractNode>{

        using namespace syntax_tree;

        if (plan == nullptr){
            throw exception::invalid_argument();
        }

        auto descendants                = std::vector<std::unique_ptr<AbstractNode>>{utility::deepcopy(plan)};
        auto positional_vec             = std::vector<size_t>{0u};
        transform_kind_t transform_kind = transform_kind_exp10;
        MatrixDimension dim             = plan->dim;
        std::string val_id              = utility::combine_identifier(plan->val_id, transform_kind_cstr(transform_kind_exp10));
        std::string id                  = utility::combine_identifier(plan->id, utility::next_id());
        logit_kind_t logit_kind         = plan->logit_kind;

        return std::make_unique<AbstractNode>(AbstractNode{std::move(descendants), std::move(positional_vec), transform_kind, dim, std::move(val_id), std::move(id), logit_kind});
    }

    auto cublas_mono_log(const std::unique_ptr<AbstractNode>& plan) -> std::unique_ptr<AbstractNode>{

        using namespace syntax_tree;

        if (plan == nullptr){
            throw exception::invalid_argument();
        }

        auto descendants                = std::vector<std::unique_ptr<AbstractNode>>{utility::deepcopy(plan)};
        auto positional_vec             = std::vector<size_t>{0u};
        transform_kind_t transform_kind = transform_kind_log;
        MatrixDimension dim             = plan->dim;
        std::string val_id              = utility::combine_identifier(plan->val_id, transform_kind_cstr(transform_kind_log));
        std::string id                  = utility::combine_identifier(plan->id, utility::next_id());
        logit_kind_t logit_kind         = plan->logit_kind;

        return std::make_unique<AbstractNode>(AbstractNode{std::move(descendants), std::move(positional_vec), transform_kind, dim, std::move(val_id), std::move(id), logit_kind});
    }

    auto cublas_mono_log2(const std::unique_ptr<AbstractNode>& plan) -> std::unique_ptr<AbstractNode>{

        using namespace syntax_tree;

        if (plan == nullptr){
            throw exception::invalid_argument();
        }

        auto descendants                = std::vector<std::unique_ptr<AbstractNode>>{utility::deepcopy(plan)};
        auto positional_vec             = std::vector<size_t>{0u};
        transform_kind_t transform_kind = transform_kind_log2;
        MatrixDimension dim             = plan->dim;
        std::string val_id              = utility::combine_identifier(plan->val_id, transform_kind_cstr(transform_kind_log2));
        std::string id                  = utility::combine_identifier(plan->id, utility::next_id());
        logit_kind_t logit_kind         = plan->logit_kind;

        return std::make_unique<AbstractNode>(AbstractNode{std::move(descendants), std::move(positional_vec), transform_kind, dim, std::move(val_id), std::move(id), logit_kind});
    }

    auto cublas_mono_log10(const std::unique_ptr<AbstractNode>& plan) -> std::unique_ptr<AbstractNode>{

        using namespace syntax_tree;

        if (plan == nullptr){
            throw exception::invalid_argument();
        }

        auto descendants                = std::vector<std::unique_ptr<AbstractNode>>{utility::deepcopy(plan)};
        auto positional_vec             = std::vector<size_t>{0u}; 
        transform_kind_t transform_kind = transform_kind_log10;
        MatrixDimension dim             = plan->dim;
        std::string val_id              = utility::combine_identifier(plan->val_id, transform_kind_cstr(transform_kind_log10));
        std::string id                  = utility::combine_identifier(plan->id, utility::next_id());
        logit_kind_t logit_kind         = plan->logit_kind;

        return std::make_unique<AbstractNode>(AbstractNode{std::move(descendants), std::move(positional_vec), transform_kind, dim, std::move(val_id), std::move(id), logit_kind});
    } 

    auto cublas_mono_abs(const std::unique_ptr<AbstractNode>& plan) -> std::unique_ptr<AbstractNode>{

        using namespace syntax_tree;

        if (plan == nullptr){
            throw exception::invalid_argument();
        }

        auto descendants                = std::vector<std::unique_ptr<AbstractNode>>{utility::deepcopy(plan)};
        auto positional_vec             = std::vector<size_t>{0u};
        transform_kind_t transform_kind = transform_kind_abs;
        MatrixDimension dim             = plan->dim;
        std::string val_id              = utility::combine_identifier(plan->val_id, transform_kind_cstr(transform_kind_abs));
        std::string id                  = utility::combine_identifier(plan->id, utility::next_id());
        logit_kind_t logit_kind         = plan->logit_kind;

        return std::make_unique<AbstractNode>(AbstractNode{std::move(descendants), std::move(positional_vec), transform_kind, dim, std::move(val_id), std::move(id), logit_kind});
    }

    auto cublas_mono_cos(const std::unique_ptr<AbstractNode>& plan) -> std::unique_ptr<AbstractNode>{

        using namespace syntax_tree;

        if (plan == nullptr){
            throw exception::invalid_argument();
        }

        auto descendants                = std::vector<std::unique_ptr<AbstractNode>>{utility::deepcopy(plan)};
        auto positional_vec             = std::vector<size_t>{0u};
        transform_kind_t transform_kind = transform_kind_cos;
        MatrixDimension dim             = plan->dim;
        std::string val_id              = utility::combine_identifier(plan->val_id, transform_kind_cstr(transform_kind_cos));
        std::string id                  = utility::combine_identifier(plan->id, utility::next_id());
        logit_kind_t logit_kind         = plan->logit_kind;

        return std::make_unique<AbstractNode>(AbstractNode{std::move(descendants), std::move(positional_vec), transform_kind, dim, std::move(val_id), std::move(id), logit_kind});
    }

    auto cublas_mono_acos(const std::unique_ptr<AbstractNode>& plan) -> std::unique_ptr<AbstractNode>{

        using namespace syntax_tree;

        if (plan == nullptr){
            throw exception::invalid_argument{};
        }

        auto descendants                = std::vector<std::unique_ptr<AbstractNode>>{utility::deepcopy(plan)};
        auto positional_vec             = std::vector<size_t>{0u};
        transform_kind_t transform_kind = transform_kind_acos;
        MatrixDimension dim             = plan->dim;
        std::string val_id              = utility::combine_identifier(plan->val_id, transform_kind_cstr(transform_kind_acos));
        std::string id                  = utility::combine_identifier(plan->id, utility::next_id());
        logit_kind_t logit_kind         = plan->logit_kind;

        return std::make_unique<AbstractNode>(AbstractNode{std::move(descendants), std::move(positional_vec), transform_kind, dim, std::move(val_id), std::move(id), logit_kind});
    }

    auto cublas_mono_sin(const std::unique_ptr<AbstractNode>& plan) -> std::unique_ptr<AbstractNode>{

        using namespace syntax_tree;

        if (plan == nullptr){
            throw exception::invalid_argument();
        }

        auto descendants                = std::vector<std::unique_ptr<AbstractNode>>{utility::deepcopy(plan)};
        auto positional_vec             = std::vector<size_t>{0u};
        transform_kind_t transform_kind = transform_kind_sin;
        MatrixDimension dim             = plan->dim;
        std::string val_id              = utility::combine_identifier(plan->val_id, transform_kind_cstr(transform_kind_sin));
        std::string id                  = utility::combine_identifier(plan->id, utility::next_id());
        logit_kind_t logit_kind         = plan->logit_kind;

        return std::make_unique<AbstractNode>(AbstractNode{std::move(descendants), std::move(positional_vec), transform_kind, dim, std::move(val_id), std::move(id), logit_kind});
    }

    auto cublas_mono_asin(const std::unique_ptr<AbstractNode>& plan) -> std::unique_ptr<AbstractNode>{

        using namespace syntax_tree;

        if (plan == nullptr){
            throw exception::invalid_argument();
        }

        auto descendants                = std::vector<std::unique_ptr<AbstractNode>>{utility::deepcopy(plan)};
        auto positional_vec             = std::vector<size_t>{0u};
        transform_kind_t transform_kind = transform_kind_asin;
        MatrixDimension dim             = plan->dim;
        std::string val_id              = utility::combine_identifier(plan->val_id, transform_kind_cstr(transform_kind_asin));
        std::string id                  = utility::combine_identifier(plan->id, utility::next_id());
        logit_kind_t logit_kind         = plan->logit_kind;

        return std::make_unique<AbstractNode>(AbstractNode{std::move(descendants), std::move(positional_vec), transform_kind, dim, std::move(val_id), std::move(id), logit_kind});
    }
 
    auto cublas_mono_tan(const std::unique_ptr<AbstractNode>& plan) -> std::unique_ptr<AbstractNode>{

        using namespace syntax_tree;

        if (plan == nullptr){
            throw exception::invalid_argument();
        }

        auto descendants                = std::vector<std::unique_ptr<AbstractNode>>{utility::deepcopy(plan)};
        auto positional_vec             = std::vector<size_t>{0u};
        transform_kind_t transform_kind = transform_kind_tan;
        MatrixDimension dim             = plan->dim;
        std::string val_id              = utility::combine_identifier(plan->val_id, transform_kind_cstr(transform_kind_tan));
        std::string id                  = utility::combine_identifier(plan->id, utility::next_id());
        logit_kind_t logit_kind         = plan->logit_kind;

        return std::make_unique<AbstractNode>(AbstractNode{std::move(descendants), std::move(positional_vec), transform_kind, dim, std::move(val_id), std::move(id), logit_kind});
    }

    auto cublas_mono_atan(const std::unique_ptr<AbstractNode>& plan) -> std::unique_ptr<AbstractNode>{

        using namespace syntax_tree;

        if (plan == nullptr){
            throw exception::invalid_argument();
        }

        auto descendants                = std::vector<std::unique_ptr<AbstractNode>>{utility::deepcopy(plan)};
        auto positional_vec             = std::vector<size_t>{0u};
        transform_kind_t transform_kind = transform_kind_atan;
        MatrixDimension dim             = plan->dim;
        std::string val_id              = utility::combine_identifier(plan->val_id, transform_kind_cstr(transform_kind_atan));
        std::string id                  = utility::combine_identifier(plan->id, utility::next_id());
        logit_kind_t logit_kind         = plan->logit_kind;

        return std::make_unique<AbstractNode>(AbstractNode{std::move(descendants), std::move(positional_vec), transform_kind, dim, std::move(val_id), std::move(id), logit_kind});
    }

    auto cublas_mono_sqrt(const std::unique_ptr<AbstractNode>& plan) -> std::unique_ptr<AbstractNode>{

        using namespace syntax_tree;

        if (plan == nullptr){
            throw exception::invalid_argument();
        }

        auto descendants                = std::vector<std::unique_ptr<AbstractNode>>{utility::deepcopy(plan)};
        auto positional_vec             = std::vector<size_t>{0u};
        transform_kind_t transform_kind = transform_kind_sqrt;
        MatrixDimension dim             = plan->dim;
        std::string val_id              = utility::combine_identifier(plan->val_id, transform_kind_cstr(transform_kind_sqrt));
        std::string id                  = utility::combine_identifier(plan->id, utility::next_id());
        logit_kind_t logit_kind         = plan->logit_kind;

        return std::make_unique<AbstractNode>(AbstractNode{std::move(descendants), std::move(positional_vec), transform_kind, dim, std::move(val_id), std::move(id), logit_kind});
    }

    auto cublas_mono_invsqrt(const std::unique_ptr<AbstractNode>& plan) -> std::unique_ptr<AbstractNode>{

        using namespace syntax_tree;

        if (plan == nullptr){
            throw exception::invalid_argument();
        }

        auto descendants                = std::vector<std::unique_ptr<AbstractNode>>{utility::deepcopy(plan)};
        auto positional_vec             = std::vector<size_t>{0u};
        transform_kind_t transform_kind = transform_kind_invsqrt;
        MatrixDimension dim             = plan->dim;
        std::string val_id              = utility::combine_identifier(plan->val_id, transform_kind_cstr(transform_kind_invsqrt));
        std::string id                  = utility::combine_identifier(plan->id, utility::next_id());
        logit_kind_t logit_kind         = plan->logit_kind; 

        return std::make_unique<AbstractNode>(AbstractNode{std::move(descendants), std::move(positional_vec), transform_kind, dim, std::move(val_id), std::move(id), logit_kind});
    }

    auto cublas_mono_negative(const std::unique_ptr<AbstractNode>& plan) -> std::unique_ptr<AbstractNode>{

        using namespace syntax_tree;

        if (plan == nullptr){
            throw exception::invalid_argument();
        }

        auto descendants                = std::vector<std::unique_ptr<AbstractNode>>{utility::deepcopy(plan)};
        auto positional_vec             = std::vector<size_t>{0u};
        transform_kind_t transform_kind = transform_kind_negative;
        MatrixDimension dim             = plan->dim;
        std::string val_id              = utility::combine_identifier(plan->val_id, transform_kind_cstr(transform_kind_negative));
        std::string id                  = utility::combine_identifier(plan->id, utility::next_id());
        logit_kind_t logit_kind         = plan->logit_kind;

        return std::make_unique<AbstractNode>(AbstractNode{std::move(descendants), std::move(positional_vec), transform_kind, dim, std::move(val_id), std::move(id), logit_kind});
    }

    auto cublas_mono_negate(const std::unique_ptr<AbstractNode>& plan) -> std::unique_ptr<AbstractNode>{

        using namespace syntax_tree;

        if (plan == nullptr){
            throw exception::invalid_argument();
        }

        auto descendants                = std::vector<std::unique_ptr<AbstractNode>>{utility::deepcopy(plan)};
        auto positional_vec             = std::vector<size_t>{0u};
        transform_kind_t transform_kind = transform_kind_negate;
        MatrixDimension dim             = plan->dim;
        std::string val_id              = utility::combine_identifier(plan->val_id, transform_kind_cstr(transform_kind_negate));
        std::string id                  = utility::combine_identifier(plan->id, utility::next_id());
        logit_kind_t logit_kind         = plan->logit_kind;

        return std::make_unique<AbstractNode>(AbstractNode{std::move(descendants), std::move(positional_vec), transform_kind, dim, std::move(val_id), std::move(id), logit_kind});
    }

    auto cublas_mono_transpose(const std::unique_ptr<AbstractNode>& plan) -> std::unique_ptr<AbstractNode>{

        using namespace syntax_tree;

        if (plan == nullptr){
            throw exception::invalid_argument();
        }

        auto descendants                = std::vector<std::unique_ptr<AbstractNode>>{utility::deepcopy(plan)};
        auto positional_vec             = std::vector<size_t>{0u};
        transform_kind_t transform_kind = transform_kind_transpose;
        MatrixDimension dim             = plan->dim;
        std::string val_id              = utility::combine_identifier(plan->val_id, transform_kind_cstr(transform_kind_transpose));
        std::string id                  = utility::combine_identifier(plan->id, utility::next_id());
        logit_kind_t logit_kind         = plan->logit_kind;

        return std::make_unique<AbstractNode>(AbstractNode{std::move(descendants), std::move(positional_vec), transform_kind, dim, std::move(val_id), std::move(id), logit_kind});
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
        auto positional_vec             = std::vector<size_t>{0u, 1u};
        transform_kind_t transform_kind = transform_kind_linear;
        MatrixDimension dim             = MatrixDimension{lhs->dim.row_sz, rhs->dim.column_sz};
        std::string val_id              = utility::combine_identifier(utility::combine_identifier(lhs->val_id, rhs->val_id), transform_kind_cstr(transform_kind_linear));
        std::string id                  = utility::combine_identifier(utility::combine_identifier(lhs->id, rhs->id), utility::next_id()); 
        logit_kind_t logit_kind         = lhs->logit_kind;

        return std::make_unique<AbstractNode>(AbstractNode{std::move(descendants), std::move(positional_vec), transform_kind, dim, std::move(val_id), std::move(id), logit_kind});
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
        auto positional_vec             = std::vector<size_t>{0u, 1u};
        transform_kind_t transform_kind = transform_kind_dot;
        MatrixDimension dim             = lhs->dim;
        std::string val_id              = utility::combine_identifier(utility::combine_identifier(lhs->val_id, rhs->val_id), transform_kind_cstr(transform_kind_linear));
        std::string id                  = utility::combine_identifier(utility::combine_identifier(lhs->id, rhs->id), utility::next_id());
        logit_kind_t logit_kind         = lhs->logit_kind;

        return std::make_unique<AbstractNode>(AbstractNode{std::move(descendants), std::move(positional_vec), transform_kind, dim, std::move(val_id), std::move(id), logit_kind});
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
        auto positional_vec             = std::vector<size_t>{0u, 1u};
        transform_kind_t transform_kind = transform_kind_add;
        MatrixDimension dim             = lhs->dim;
        std::string val_id              = utility::combine_identifier(utility::combine_identifier(lhs->val_id, rhs->val_id), transform_kind_cstr(transform_kind_add));
        std::string id                  = utility::combine_identifier(utility::combine_identifier(lhs->id, rhs->id), utility::next_id());
        logit_kind_t logit_kind         = lhs->logit_kind;

        return std::make_unique<AbstractNode>(AbstractNode{std::move(descendants), std::move(positional_vec), transform_kind, dim, std::move(val_id), std::move(id), logit_kind});
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
        auto positional_vec             = std::vector<size_t>{0u, 1u};
        transform_kind_t transform_kind = transform_kind_sub;
        MatrixDimension dim             = lhs->dim;
        std::string val_id              = utility::combine_identifier(utility::combine_identifier(lhs->val_id, rhs->val_id), transform_kind_cstr(transform_kind_sub));
        std::string id                  = utility::combine_identifier(utility::combine_identifier(lhs->id, rhs->id), utility::next_id());
        logit_kind_t logit_kind         = lhs->logit_kind;

        return std::make_unique<AbstractNode>(AbstractNode{std::move(descendants), std::move(positional_vec), transform_kind, dim, std::move(val_id), std::move(id), logit_kind});
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
        auto positional_vec             = std::vector<size_t>{0u, 1u};
        transform_kind_t transform_kind = transform_kind_mul;
        MatrixDimension dim             = lhs->dim;
        std::string val_id              = utility::combine_identifier(utility::combine_identifier(lhs->val_id, rhs->val_id), transform_kind_cstr(transform_kind_mul));
        std::string id                  = utility::combine_identifier(utility::combine_identifier(lhs->id, rhs->id), utility::next_id());
        logit_kind_t logit_kind         = lhs->logit_kind;

        return std::make_unique<AbstractNode>(AbstractNode{std::move(descendants), std::move(positional_vec), transform_kind, dim, std::move(val_id), std::move(id), logit_kind});
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
        auto positional_vec             = std::vector<size_t>{0u, 1u};
        transform_kind_t transform_kind = transform_kind_div;
        MatrixDimension dim             = lhs->dim;
        std::string val_id              = utility::combine_identifier(utility::combine_identifier(lhs->val_id, rhs->val_id), transform_kind_cstr(transform_kind_div));
        std::string id                  = utility::combine_identifier(utility::combine_identifier(lhs->id, rhs->id), utility::next_id());
        logit_kind_t logit_kind         = lhs->logit_kind;

        return std::make_unique<AbstractNode>(AbstractNode{std::move(descendants), std::move(positional_vec), transform_kind, dim, std::move(val_id), std::move(id), logit_kind});
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
        auto positional_vec             = std::vector<size_t>{0u, 1u};
        transform_kind_t transform_kind = transform_kind_pow;
        MatrixDimension dim             = lhs->dim;
        std::string val_id              = utility::combine_identifier(utility::combine_identifier(lhs->val_id, rhs->val_id), transform_kind_cstr(transform_kind_div));
        std::string id                  = utility::combine_identifier(utility::combine_identifier(lhs->id, rhs->id), utility::next_id());
        logit_kind_t logit_kind         = lhs->logit_kind;

        return std::make_unique<AbstractNode>(AbstractNode{std::move(descendants), std::move(positional_vec), transform_kind, dim, std::move(val_id), std::move(id), logit_kind});
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
        auto positional_vec             = std::vector<size_t>{0u, 1u};
        transform_kind_t transform_kind = transform_kind_min;
        MatrixDimension dim             = lhs->dim;
        std::string val_id              = utility::combine_identifier(utility::combine_identifier(lhs->val_id, rhs->val_id), transform_kind_cstr(transform_kind_min));
        std::string id                  = utility::combine_identifier(utility::combine_identifier(lhs->id, rhs->id), utility::next_id());
        logit_kind_t logit_kind         = lhs->logit_kind;
        
        return std::make_unique<AbstractNode>(AbstractNode{std::move(descendants), std::move(positional_vec), transform_kind, dim, std::move(val_id), std::move(id), logit_kind});
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
        auto positional_vec             = std::vector<size_t>{0u, 1u};
        transform_kind_t transform_kind = transform_kind_max;
        MatrixDimension dim             = lhs->dim;
        std::string val_id              = utility::combine_identifier(utility::combine_identifier(lhs->val_id, rhs->val_id), transform_kind_cstr(transform_kind_max));
        std::string id                  = utility::combine_identifier(utility::combine_identifier(lhs->id, rhs->id), utility::next_id());
        logit_kind_t logit_kind         = lhs->logit_kind;

        return std::make_unique<AbstractNode>(AbstractNode{std::move(descendants), std::move(positional_vec), transform_kind, dim, std::move(val_id), std::move(id), logit_kind});
    }

    auto cublas_optimize_fast(int device_id, const std::unique_ptr<AbstractNode>& plan, size_t buf_cap) -> std::unique_ptr<AbstractNode>{

    }

    auto cublas_optimize_slow(int device_id, const std::unique_ptr<AbstractNode>& plan, size_t buf_cap) -> std::unique_ptr<AbstractNode>{

    }

    auto cublas_make_executable(const std::unique_ptr<AbstractNode>& plan) -> std::unique_ptr<engine::ExecutorInterface>{

    }
} 

#endif