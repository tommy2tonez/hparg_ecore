#ifndef __NETWORK_KERNELMAP_X_IMPL1_H__
#define __NETWORK_KERNELMAP_X_IMPL1_H__

#include <stdint.h>
#include <stdlib.h>
#include <chrono>
#include <optional>
#include <vector>
#include <unordered_map>
#include <nutex>
#include <atomic>
#include "network_exception.h"
#include "network_log.h"
#include "network_fileio.h"

namespace dg::network_kernelmap_x_impl1::model{

    using fsys_ptr_t = uint64_t;
    using Lock  = std::conditional_t<IS_ATOMIC_OPERATION_PREFERRED,
                                     std::atomic_flag,
                                     std::mutex>;  

    static_assert(dg::is_ptr_v<fsys_ptr_t>);

    struct FSysPtrInfo{
        fsys_ptr_t ptr;
        size_t reference;
    };

    struct MemoryNode{
        std::unique_ptr<char[]> cptr; //deallocator
        std::optional<FSysPtrInfo> fsys_ptr_info;
        std::chrono::nanoseconds timestamp;
    };

    struct HeapNode: MemoryNode{
        size_t idx;
    };

    struct MapResource{
        HeapNode * node;
        size_t off;
    
        auto ptr() const noexcept -> void *{

            using namespace dg::network_genult;
            return dg::memult::advance(safe_ptr_access(safe_ptr_access(this->node)->cptr.get()), this->off);
        }

        auto const_ptr() const noexcept -> const void *{

            using namespace dg::network_genult;
            return dg::memult::advance(safe_ptr_access(safe_ptr_access(this->node)->cptr.get()), this->off);
        }
    };
}

namespace dg::network_kernelmap_x_impl1::interface{

    using namespace network_kernelmap_x_impl1::model; 

    struct FsysLoaderInterface{
        virtual ~FsysLoaderInterface() = default;
        virtual auto load(MemoryNode&, fsys_ptr_t) noexcept -> exception_t = 0;
        virtual void unload(MemoryNode&) noexcept = 0;
    };

    struct MapInterface{
        virtual ~MapInterface() = default;
        virtual auto map(fsys_ptr_t) noexcept -> std::expected<MapResource, exception_t> = 0;
        virtual void map_release(MapResource) noexcept = 0;
    };

    struct MapDistributorInterface{
        virtual ~MapDistributorInterface() = default;
        virtual auto id(fsys_ptr_t) noexcept -> std::expected<size_t, exception_t> = 0;
    };
}

namespace dg::network_kernelmap_x_impl1::implementation{
    
    using namespace network_kernelmap_x_impl1::interface;

    struct HeapNodeCmp{

        constexpr auto operator()(const HeapNode& lhs, const HeapNode& rhs) const noexcept -> int{

            size_t lhs_reference = 0u;
            size_t rhs_reference = 0u;

            if (static_cast<bool>(lhs.fsys_ptr_info)){
                lhs_reference = lhs.fsys_ptr_info->reference;
            } 

            if (static_cast<bool>(rhs.fsys_ptr_info)){
                rhs_reference = rhs.fsys_ptr_info->reference;
            }

            if (lhs_reference < rhs_reference){
                return -1;
            }

            if (lhs_reference > rhs_reference){
                return 1;
            }
            
            if (lhs.timestamp < rhs.timestamp){
                return -1;
            }

            if (lhs.timestamp > rhs.timestamp){
                return 1;
            }

            return 0;
        } 
    };

    class DirectFsysLoader: public virtual FsysLoaderInterface{

        private:

            std::unordered_map<fsys_ptr_t, std::filesystem::path> stable_storage_dict;
            size_t memregion_sz;

        public:

            explicit DirectFsysLoader(std::unordered_map<fsys_ptr_t, std::filesystem::path> stable_storage_dict,
                                      size_t memregion_sz) noexcept: stable_storage_dict(std::move(stable_storage_dict)),
                                                                     memregion_sz(memregion_sz){}

            auto load(MemoryNode& root, fsys_ptr_t region) noexcept -> exception_t{
                
                if constexpr(DEBUG_MODE_FLAG){
                    if (static_cast<bool>(root.fsys_ptr_info)){
                        dg::network_log_stackdump::critical(network_exception::verbose(network_exception::INTERNAL_CORRUPTION));
                        std::abort();
                    }
                }

                auto dict_ptr = this->stable_storage_dict_find_entry(region);

                if (dict_ptr == this->stable_storage_dict_end()){
                    return network_exception::INVALID_DICTIONARY_KEY;
                }

                const std::filesystem::path& inpath = dict_ptr->second;
                const char * cstr_path              = inpath.c_str();
                void * dst                          = root.cptr.get();
                
                dg::network_fileio::dg_read_binary_direct_nothrow(cstr_path, dst, this->memregion_sz);
                
                root.fsys_ptr_info  = VMAPtrInfo{region, 0u};
                root.timestamp      = dg::network_genult::unix_timestamp();

                return dg::network_exception::SUCCESS;
            }

            void unload(MemoryNode& root) noexcept{ //correct: this is a reverse operation of load - should be void (...) noexcept 

                if constexpr(DEBUG_MODE_FLAG){
                    if (!static_cast<bool>(root.fsys_ptr_info)){
                        dg::network_log_stackdump::critical(network_exception::verbose(network_exception::INTERNAL_CORRUPTION));
                        std::abort();
                    }

                    if (root.fsys_ptr_info->reference != 0u){
                        dg::network_log_stackdump::critical(network_exception::verbose(network_exception::INTERNAL_CORRUPTION));
                        std::abort();
                    }
                }

                fsys_ptr_t host_region  = root.fsys_ptr_info->ptr;
                auto dict_ptr           = this->stable_storage_dict_find_entry(host_region);

                if constexpr(DEBUG_MODE_FLAG){
                    if (dict_ptr == this->stable_storage_dict_end()){
                        dg::network_log_stackdump::critical(network_exception::verbose(network_exception::INTERNAL_CORRUPTION));
                        std::abort();
                    }
                }

                const std::filesystem::path& opath  = dict_ptr->second;
                const char * cstr_path              = opath.c_str();
                const void * src                    = root.cptr.get();
                
                dg::network_fileio::dg_write_binary_direct_nothrow(cstr_path, src, this->memregion_sz);

                root.fsys_ptr_info  = std::nullopt;
                root.timestamp      = dg::network_genult::unix_timestamp();
            }

        private:

            inline auto stable_storage_dict_find_entry(fsys_ptr_t key) const noexcept{ //return type

                return this->stable_storage_dict.find(key);
            }

            inline auto stable_storage_dict_end() const noexcept{ //return type

                return this->stable_storage_dict.end();
            }
    };

    template <size_t MEMREGION_SZ>
    class Map: public virtual MapInterface{

        private:

            std::vector<std::unique_ptr<HeapNode>> priority_queue;
            std::unordered_map<fsys_ptr_t, HeapNode *> allocation_dict;
            std::unique_ptr<FsysLoaderInterface> fsys_loader;
            MemoryNode tmp_space;
            std::unique_ptr<Lock> lck;

        public:

            static_assert(dg::memult::is_pow2(MEMREGION_SZ));

            explicit Map(std::vector<std::unique_ptr<HeapNode>> priority_queue, 
                         std::unordered_map<fsys_ptr_t, HeapNode *> allocation_dict,
                         std::unique_ptr<FsysLoaderInterface> fsys_loader,
                         MemoryNode tmp_space,
                         std::unique_ptr<Lock> lck,
                         std::integral_constant<size_t, MEMREGION_SZ>) noexcept: priority_queue(std::move(priority_queue)),
                                                                                 allocation_dict(std::move(allocation_dict)),
                                                                                 fsys_loader(std::move(fsys_loader)),
                                                                                 tmp_space(std::move(tmp_space)),
                                                                                 lck(std::move(lck)){}

            auto map(fsys_ptr_t ptr) noexcept -> std::expected<MapResource, exception_t>{

                auto lck_grd = dg::genult::lock_guard(*this->lck);
                return this->internal_map(ptr);
            }

            void map_release(MapResource map_resource) noexcept{

                auto lck_grd = dg::genult::lock_guard(*this->lck);
                this->internal_map_release(map_resource);
            }

        private:

            void heap_swap(std::unique_ptr<HeapNode>& lhs, std::unique_ptr<HeapNode>& rhs) const noexcept{

                std::swap(lhs->idx, rhs->idx);
                std::swap(lhs, rhs);
            } 

            void heap_push_up_at(size_t idx) noexcept{
                
                if (idx == 0u){
                    return;
                }

                size_t c = (idx - 1) >> 1;

                if (HeapNodeCmp{}(*this->priority_queue[c], *this->priority_queue[idx]) <= 0){
                    return;
                }

                this->heap_swap(this->priority_queue[c], this->priority_queue[idx]);
                this->heap_push_up_at(c);
            }

            void heap_push_down_at(size_t idx) noexcept{

                size_t c = idx * 2 + 1;

                if (c >= this->priority_queue.size()){
                    return;
                }

                if (c + 1 < this->priority_queue.size() && HeapNodeCmp{}(*this->priority_queue[c + 1], *this->priority_queue[c]) < 0){
                    c += 1;
                }

                if (HeapNodeCmp{}(*this->priority_queue[idx], *this->priority_queue[c]) <= 0){
                    return;
                }

                this->heap_swap(this->priority_queue[idx], this->priority_queue[c]);
                this->heap_push_down_at(c);
            }

            void heap_increase_reference(HeapNode * node) noexcept{

                dg::network_genult::safe_optional_access(dg::network_genult::safe_ptr_access(node)->fsys_ptr_info);
                node->fsys_ptr_info->reference += 1;
                heap_push_down_at(node->idx);
            }

            void heap_decrease_reference(HeapNode * node) noexcept{
                
                dg::network_genult::safe_optional_access(dg::network_genult::safe_ptr_access(node)->fsys_ptr_info);

                if constexpr(DEBUG_MODE_FLAG){
                    if (node->fsys_ptr_info->reference == 0u){
                        dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                        std::abort();
                    }
                }

                node->fsys_ptr_info->reference -= 1;
                node->timestamp = dg::network_genult::unix_timestamp();
                heap_push_up_at(node->idx);
            }

            void allocation_dict_remove_entry(fsys_ptr_t key) noexcept{
                
                auto map_ptr = this->allocation_dict.find(key);

                if constexpr(DEBUG_MODE_FLAG){
                    if (map_ptr == this->allocation_dict.end()){
                        dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                        std::abort();
                    }
                }

                this->allocation_dict.erase(map_ptr);
            } 

            void allocation_dict_add_entry(fsys_ptr_t key, HeapNode * value) noexcept{ //correct: good to have a noexcept to avoid bad_alloc

                this->allocation_dict[key] = value;
            }

            auto allocation_dict_find_entry(fsys_ptr_t key) noexcept{ //return type

                return this->allocation_dict.find(key);
            }

            auto allocation_dict_end() noexcept{ //return type

                return this->allocation_dict.end();
            }

            auto internal_map(fsys_ptr_t ptr) noexcept -> std::expected<MapResource, exception_t>{

                fsys_ptr_t ptr_region   = dg::memult::region(ptr, std::integral_constant<size_t, MEMREGION_SZ>{});
                auto dict_ptr           = this->allocation_dict_find_entry(ptr_region);

                if (dict_ptr != this->allocation_dict_end()){
                    heap_increase_reference(dict_ptr->second);
                    return MapResource{dict_ptr->second, dg::memult::region_offset(ptr, std::integral_constant<size_t, MEMREGION_SZ>{})};
                }

                if (static_cast<bool>(this->priority_queue[0u]->fsys_ptr_info)){
                    if (this->priority_queue[0u]->fsys_ptr_info->reference == 0u){
                        exception_t err = this->fsys_loader->load(this->tmp_space, ptr_region);

                        if (dg::network_exception::is_failed(err)){
                            return std::unexpected(err);
                        }

                        fsys_ptr_t removing_region  = this->priority_queue[0u]->fsys_ptr_info->ptr;
                        this->fsys_loader->unload(*priority_queue[0u]);
                        this->allocation_dict_remove_entry(removing_region);
                        this->allocation_dict_add_entry(ptr_region, priority_queue[0u].get());
                        std::swap(static_cast<MemoryNode&>(*this->priority_queue[0u]), this->tmp_space); //fine - does not affect ptr() and const_ptr() - assume that those methods are invoked when user is in charge of the map_resource
                        this->heap_push_down_at(0u);

                        return this->internal_map(ptr);
                    }

                    return std::unexpected(dg::network_exception::OUT_OF_MEMORY);
                }

                exception_t err = this->fsys_loader->load(*priority_queue[0u], ptr_region);
                
                if (dg::network_exception::is_failed(err)){
                    return std::unexpected(err);
                }
                
                this->allocation_dict_add_entry(ptr_region, priority_queue[0u].get());
                this->heap_push_down_at(0u);
                return this->internal_map(ptr);
            }

            void internal_map_release(MapResource map_resource) noexcept{

                heap_decrease_reference(map_resource.node);
            } 
    };
    
    template <size_t MEMREGION_SZ> //fine - inherit a virtual interface - this is not bad practice
    class StdMapDistributor: public virtual MapDistributorInterface{

        private:

            std::unordered_map<fsys_ptr_t, size_t> region_id_dict;
        
        public:

            static_assert(dg::memult::is_pow2(MEMREGION_SZ));

            explicit StdMapDistributor(std::unordered_map<fsys_ptr_t, size_t> region_id_dict,
                                       std::integral_constant<size_t, MEMREGION_SZ>) noexcept: region_id_dict(std::move(region_id_dict)){}

            auto id(fsys_ptr_t ptr) noexcept -> std::expected<size_t, exception_t>{

                fsys_ptr_t ptr_region   = dg::memult::region(ptr, std::integral_constant<size_t, MEMREGION_SZ>{});
                auto dict_ptr           = this->region_id_dict.find(ptr_region);

                if (dict_ptr == this->region_id_dict.end()){
                    return std::unexpected(dg::network_exception::INVALID_DICTIONARY_KEY);
                }

                return dict_ptr->second;
            }
    };

    class ConcurrentMap: public virtual MapInterface{

        private:

            std::vector<std::unique_ptr<MapInterface>> map_table;
            std::unique_ptr<MapDistributorInterface> map_distributor;
        
        public:

            explicit ConcurrentMap(std::vector<std::unique_ptr<MapInterface>> map_table,
                                   std::unique_ptr<MapDistributorInterface> map_distributor) noexcept: map_table(std::move(map_table)),
                                                                                                        map_distributor(std::move(map_distributor)){}

            auto map(fsys_ptr_t ptr) noexcept -> std::expected<MapResource, exception_t>{

                auto rs = this->map_distributor->id(ptr);

                if (!rs.has_value()){
                    return std::unexpected(rs.error());
                }

                return map_table[rs.value()]->map(ptr);
            }
            
            void map_release(MapResource map_resource) noexcept{
                
                fsys_ptr_t ptr  = dg::network_genult::safe_optional_access(dg::network_genult::safe_pointer_access(map_resource.node)->fsys_ptr_info)->ptr;
                auto rs         = this->map_distributor->id(ptr);

                if constexpr(DEBUG_MODE_FLAG){
                    if (!rs.has_value()){
                        dg::network_log_stackdump::critical(network_exception::verbose(network_exception::INTERNAL_CORRUPTION));
                        std::abort();
                    }
                }

                map_table[rs.value()]->map_release(map_resource);
            }
    };
}

namespace dg::network_kernelmap_x_impl1{

    template <size_t MEMREGION_SZ>
    auto make(fsys_ptr_t * region, std::filesystem::path * path, fsys_device_id_t * device_id, size_t n, std::integral_constant<size_t, MEMREGION_SZ>) -> std::unique_ptr<interface::MapInterface>{

    }
} 

#endif