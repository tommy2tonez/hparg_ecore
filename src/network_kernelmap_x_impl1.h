#ifndef __NETWORK_KERNELMAP_X_IMPL1_H__
#define __NETWORK_KERNELMAP_X_IMPL1_H__

#include <stdint.h>
#include <stdlib.h>
#include <chrono>
#include <optional>
#include <vector>
#include <unordered_map>
#include <nutex>
#include "network_exception.h"
#include "network_log.h"

namespace dg::network_kernelmap_x_impl1::model{

    using fsys_ptr_t = uint64_t;

    struct FSysPtrInfo{
        fsys_ptr_t ptr;
        size_t reference;
    };

    struct MemoryNode{
        std::unique_ptr<char[]> cptr; //
        std::optional<FSysPtrInfo> fsys_ptr_info;
        std::chrono::nanoseconds timestamp;
    };

    struct HeapNode: MemoryNode{
        size_t idx;
    };

    struct MapResource{
        HeapNode * node;
        size_t off;
    
        inline auto ptr() const noexcept -> void *{

            return dg::memult::advance(dg::network_genult::safe_ptr_access(this->node)->cptr, this->off);
        }

        inline auto const_ptr() const noexcept -> const void *{

            return dg::memult::advance(dg::network_genult::safe_ptr_access(this->node)->cptr, this->off);
        }
    };
}

namespace dg::network_kernelmap_x_impl1::interface{

    using namespace network_kernelmap_x_impl1::model; 

    struct FsysLoaderInterface{
        virtual ~FsysLoaderInterface() = default;
        virtual auto load(MemoryNode&, fsys_ptr_t) const noexcept -> exception_t = 0;
        virtual auto unload(MemoryNode&) const noexcept -> exception_t = 0;
    };

    struct MapInterface{
        virtual ~MapInterface() = default;
        virtual auto map_try(fsys_ptr_t) noexcept -> std::expected<MapResource, exception_t> = 0;
        virtual auto map_wait(fsys_ptr_t) noexcept -> MapResource = 0;
        virtual void map_release(MapResource) noexcept = 0;
    };

    struct MapDistributionInterface{
        virtual ~MapDistributionInterface() = default;
        virtual auto id(fsys_ptr_t) const noexcept -> std::expected<size_t, exception_t> = 0;
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

    template <size_t MEMREGION_SZ>
    class DirectFsysLoader: public virtual FsysLoaderInterface{

        private:

            std::unordered_map<fsys_ptr_t, std::filesystem::path> stable_storage_dict;

        public:

            static_assert(dg::filesystem::is_met_io_direct_requirement(MEMREGION_SZ));

            explicit DirectFsysLoader(std::unordered_map<fsys_ptr_t, std::filesystem::path> stable_storage_dict,
                                      std::integral_constant<size_t, MEMREGION_SZ>) noexcept: stable_storage_dict(std::move(stable_storage_dict)){}

            auto load(MemoryNode& root, fsys_ptr_t region) const noexcept -> exception_t{

                if (static_cast<bool>(root.fsys_ptr_info)){
                    dg::network_log_stackdump::critical(network_exception::verbose(network_exception::INTERNAL_CORRUPTION));
                    std::abort();
                }

                auto dict_ptr = this->stable_storage_dict_find_entry(region);

                if (dict_ptr == this->stable_storage_dict_end()){
                    return network_exception::INVALID_DICTIONARY_KEY;
                }

                const std::filesystem::path& inpath = dict_ptr->second;
                const char * cstr_path              = inpath.c_str();
                void * dst                          = root.cptr.get();
                exception_t err_code                = dg::filesystem::readfile_binary_direct(cstr_path, dst, MEMREGION_SZ);
                
                if (dg::network_exception::is_failed(err_code)){
                    return err_code;
                }

                root.fsys_ptr_info  = VMAPtrInfo{region, 0u};
                root.timestamp      = dg::utility::unix_timestamp();

                return dg::network_exception::SUCCESS;
            }

            auto unload(MemoryNode& root) const noexcept -> exception_t{

                if (!static_cast<bool>(root.fsys_ptr_info)){
                    dg::network_log_stackdump::critical(network_exception::verbose(network_exception::INTERNAL_CORRUPTION));
                    std::abort();
                }

                if (root.fsys_ptr_info->reference != 0u){
                    dg::network_log_stackdump::critical(network_exception::verbose(network_exception::INTERNAL_CORRUPTION));
                    std::abort();
                }

                fsys_ptr_t host_region  = root.fsys_ptr_info->ptr;
                auto dict_ptr           = this->stable_storage_dict_find_entry(host_region);

                if (dict_ptr == this->stable_storage_dict_end()){
                    dg::network_log_stackdump::critical(network_exception::verbose(network_exception::INTERNAL_CORRUPTION));
                    std::abort();
                }

                const std::filesystem::path& opath  = dict_ptr->second;
                const char * cstr_path              = outpath.c_str();
                const void * src                    = root.cptr.get();
                exception_t err_code                = dg::filesystem::writefile_binary_direct(cstr_path, src, MEMREGION_SZ);

                if (dg::network_exception::is_failed(err_code)){
                    return err_code;
                }

                root.fsys_ptr_info  = std::nullopt;
                root.timestamp      = dg::utility::unix_timestamp();

                return dg::network_exception::SUCCESS;
            }

        private:

            inline auto stable_storage_dict_find_entry(fsys_ptr_t key) const noexcept{

                return this->stable_storage_dict.find(key);
            }

            inline auto stable_storage_dict_end() const noexcept{

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
            std::mutex mtx;

        public:

            static_assert(dg::memult::is_pow2(MEMREGION_SZ));

            explicit Map(std::vector<std::unique_ptr<HeapNode>> priority_queue, 
                         std::unordered_map<fsys_ptr_t, HeapNode *> allocation_dict,
                         std::unique_ptr<FsysLoaderInterface> fsys_loader,
                         MemoryNode tmp_space,
                         std::integral_constant<size_t, MEMREGION_SZ>) noexcept: priority_queue(std::move(priority_queue)),
                                                                                 allocation_dict(std::move(allocation_dict)),
                                                                                 fsys_loader(std::move(fsys_loader)),
                                                                                 tmp_space(std::move(tmp_space)),
                                                                                 mtx(){}

            auto map_try(fsys_ptr_t ptr) noexcept -> std::expected<MapResource, exception_t>{

                auto lck_grd = std::lock_guard<std::mutex>(this->mtx);
                return this->internal_map_try(ptr);
            }

            auto map_wait(fsys_ptr_t ptr) noexcept -> MapResource{

                while (true){
                    auto rs = map_try(ptr);
                     
                    if (rs.has_value()){
                        return rs.value();
                    }

                    dg::network_log_stackdump::critical(dg::network_exception::verbose(rs.error()));
                    std::abort();
                }
            } 

            void map_release(MapResource map_resource) noexcept{

                auto lck_grd = std::lock_guard<std::mutex>(this->mtx);
                this->internal_map_release(map_resource);
            }

        private:

            inline auto region(fsys_ptr_t ptr) noexcept -> fsys_ptr_t{
                
                using ptr_arithmetic_t = typename ptr_info<fsys_ptr_t>::max_unsigned_t;
                constexpr ptr_arithmetic_t BITMASK = ~(static_cast<ptr_arithmetic_t>(MEMREGION_SZ) - 1); 
                return pointer_cast<fsys_ptr_t>(pointer_cast<ptr_arithmetic_t>(ptr) & BITMASK);
            }

            inline auto region_offset(fsys_ptr_t ptr) noexcept -> size_t{

                using ptr_arithmetic_t = typename ptr_info<fsys_ptr_t>::max_unsigned_t;
                return pointer_cast<ptr_arithmetic_t>(ptr) % static_cast<ptr_arithmetic_t>(MEMREGION_SZ);
            }

            inline void heap_swap(std::unique_ptr<HeapNode>& lhs, std::unique_ptr<HeapNode>& rhs) const noexcept{

                std::swap(lhs->idx, rhs->idx);
                std::swap(lhs, rhs);
            } 

            inline void heap_push_up_at(size_t idx) noexcept{
                
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

            inline void heap_push_down_at(size_t idx) noexcept{

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

            inline void heap_push_updown_at(size_t idx) noexcept{

                this->heap_push_up_at(idx);
                this->heap_push_down_at(idx);
            }

            inline void heap_increase_reference(HeapNode * node) noexcept{

                if (!static_cast<bool>(dg::safe_ptr_access(node)->fsys_ptr_info)){
                    dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                    std::abort();
                }

                node->fsys_ptr_info->reference += 1;
                heap_push_updown_at(node->idx);
            }

            inline void heap_decrease_reference(HeapNode * node) noexcept{

                if (!static_cast<bool>(dg::safe_ptr_access(node)->fsys_ptr_info)){
                    dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                    std::abort();
                }

                if (node->fsys_ptr_info->reference == 0u){
                    dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                    std::abort();
                }

                node->fsys_ptr_info->reference -= 1;
                node->timestamp = dg::utility::unix_timestamp();
                heap_push_updown_at(node->idx);
            }

            inline void allocation_dict_remove_entry(fsys_ptr_t key) noexcept{

                this->allocation_dict.erase(key);
            } 

            inline void allocation_dict_add_entry(fsys_ptr_t key, HeapNode * value) noexcept{

                this->allocation_dict[key] = value;
            }

            inline auto allocation_dict_find_entry(fsys_ptr_t key) noexcept{

                return this->allocation_dict.find(key);
            }

            inline auto allocation_dict_end() noexcept{

                return this->allocation_dict.end();
            }

            inline auto internal_map_try(fsys_ptr_t ptr) noexcept -> std::expected<MapResource, exception_t>{

                fsys_ptr_t ptr_region   = this->region(ptr);
                auto dict_ptr           = this->allocation_dict_find_entry(ptr_region);

                if (dict_ptr != this->allocation_dict_end()){
                    heap_increase_reference(dict_ptr->second);
                    return MapResource{dict_ptr->second, region_offset(ptr)};
                }

                if (static_cast<bool>(this->priority_queue[0u]->fsys_ptr_info)){
                    if (this->priority_queue[0u]->fsys_ptr_info->reference == 0u){
                        exception_t create_err_code     = this->fsys_loader->load(this->tmp_space, ptr_region);

                        if (dg::network_exception::is_failed(create_err_code)){
                            return std::unexpected(create_err_code);
                        }

                        fsys_ptr_t removing_region      = this->priority_queue[0u]->fsys_ptr_info->ptr;
                        exception_t release_err_code    = this->fsys_loader->unload(*priority_queue[0u]);
                        
                        if (dg::network_exception::is_failed(release_err_code)){
                            tmp_space.fsys_ptr_info = std::nullopt;
                            return std::unexpected(release_err_code);
                        }

                        this->allocation_dict_remove_entry(removing_region);
                        this->allocation_dict_add_entry(ptr_region, priority_queue[0u].get());
                        std::swap(static_cast<MemoryNode&>(*this->priority_queue[0u]), this->tmp_space);
                        this->heap_push_down_at(0u);

                        return this->internal_map_try(ptr);
                    }

                    return std::unexpected(dg::network_exception::OUT_OF_MEMORY);
                }

                exception_t create_err_code = this->fsys_loader->load(*priority_queue[0u], ptr_region);
                
                if (dg::network_exception::is_failed(create_err_code)){
                    return std::unexpected(create_err_code);
                }
                
                this->allocation_dict_add_entry(ptr_region, priority_queue[0u].get());
                this->heap_push_down_at(0u);
                return this->internal_map_try(ptr);
            }

            inline void internal_map_release(MapResource map_resource) noexcept{

                heap_decrease_reference(map_resource.node);
            } 
    };
    
    template <size_t MEMREGION_SZ>
    class StdMapDistribution: public virtual MapDistributionInterface{

        private:

            std::unordered_map<fsys_ptr_t, size_t> region_id_dict;
        
        public:

            static_assert(dg::memult::is_pow2(MEMREGION_SZ));

            explicit StdMapDistribution(std::unordered_map<fsys_ptr_t, size_t> region_id_dict,
                                        std::integral_constant<size_t, MEMREGION_SZ>) noexcept: region_id_dict(std::move(region_id_dict)){}

            auto id(fsys_ptr_t ptr) const noexcept -> std::expected<size_t, exception_t>{

                auto dict_ptr = this->region_id_dict.find(this->region(ptr));

                if (dict_ptr == this->region_id_dict.end()){
                    return dg::network_exception::INVALID_DICTIONARY_KEY;
                }

                return dict_ptr->second;
            }
        
        private:

            inline auto region(fsys_ptr_t ptr) const noexcept -> fsys_ptr_t{

                using ptr_arithmetic_t = typename dg::ptr_info<fsys_ptr_t>::max_unsigned_t;
                constexpr ptr_arithmetic_t BITMASK  = ~(static_cast<ptr_arithmetic_t>(MEMREGION_SZ) - 1); 
                return pointer_cast<fsys_ptr_t>(pointer_cast<ptr_arithmetic_t>(ptr) & BITMASK); 
            }
    };

    class ConcurrentMap: public virtual MapInterface{

        private:

            std::vector<std::unique_ptr<MapInterface>> map_table;
            std::unique_ptr<MapDistributionInterface> map_distributor;
        
        public:

            explicit ConcurrentMap(std::vector<std::unique_ptr<MapInterface>> map_table,
                                   std::unique_ptr<MapDistributionInterface> map_distributor) noexcept: map_table(std::move(map_table)),
                                                                                                        map_distributor(std::move(map_distributor)){}

            auto map_try(fsys_ptr_t ptr) noexcept -> std::expected<MapResource, exception_t>{

                auto rs = this->map_distributor->id(ptr);

                if (!static_cast<bool>(rs)){
                    return std::unexpected(rs.error());
                }

                return map_table[rs.value()]->map_try(ptr);
            }

            auto map_wait(fsys_ptr_t ptr) noexcept -> MapResource{

                auto rs = this->map_distributor->id(ptr);

                if (!static_cast<bool>(rs)){
                    dg::network_log_stackdump::critical(network_exception::verbose(rs.error()));
                    std::abort();
                }

                return map_table[rs.value()]->map_wait(ptr);
            }

            void map_release(MapResource map_resource) noexcept{
                
                fsys_ptr_t ptr  = dg::safe_pointer_access(map_resource.node)->fsys_ptr_info->ptr;
                auto rs         = this->map_distributor->id(ptr);

                if (!static_cast<bool>(rs)){
                    dg::network_log_stackdump::critical(network_exception::verbose(network_exception::INTERNAL_CORRUPTION));
                    std::abort();
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