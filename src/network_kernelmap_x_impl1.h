#ifndef __NETWORK_KERNELMAP_X_IMPL1_H__
#define __NETWORK_KERNELMAP_X_IMPL1_H__

//define HEADER_CONTROL 9

#include <stdint.h>
#include <stdlib.h>
#include <chrono>
#include <optional>
#include <mutex>
#include <atomic>
#include "network_exception.h"
#include "network_log.h"
#include "network_std_container.h"
#include "network_pointer.h"
#include <memory>
#include "network_fileio.h"
#include "network_memult.h"
#include <random>

namespace dg::network_kernelmap_x_impl1::model{

    static inline constexpr bool IS_SPINLOCK_PREFERRED = true;

    using fsys_ptr_t    = dg::network_pointer::fsys_ptr_t;
    using Lock          = std::conditional_t<IS_SPINLOCK_PREFERRED,
                                             std::atomic_flag,
                                             std::mutex>;  

    static inline constexpr fsys_ptr_t NULL_FSYS_PTR = dg::network_pointer::NULL_FSYS_PTR; 

    struct MemoryNode{
        std::shared_ptr<char[]> cptr;
        fsys_ptr_t fsys_ptr;
        size_t mem_sz;
    };

    struct ReferenceMemoryNode: MemoryNode{
        size_t reference;
        std::chrono::time_point<std::chrono::steady_clock> last_modified;
    };

    struct HeapNode: ReferenceMemoryNode{
        size_t idx;
    };

    struct MapResource{
        HeapNode * node;
        char * mapped_region;
        size_t off;
        fsys_ptr_t fsys_ptr; 

        inline auto ptr() const noexcept -> void *{

            return dg::memult::next(this->mapped_region, this->off);
        }
    };

    struct ConcurrentMapResource{
        MapResource resource;
        size_t map_id;

        inline auto ptr() const noexcept -> void *{

            return this->resource.ptr();
        }
    };
}

namespace dg::network_kernelmap_x_impl1::interface{

    using namespace network_kernelmap_x_impl1::model; 

    struct FsysLoaderInterface{
        virtual ~FsysLoaderInterface() noexcept = default;
        virtual auto load(MemoryNode&, fsys_ptr_t, size_t) noexcept -> exception_t = 0;
        virtual void unload(MemoryNode&) noexcept = 0;
    };

    struct MapInterface{
        virtual ~MapInterface() noexcept = default;
        virtual auto map(fsys_ptr_t) noexcept -> std::expected<MapResource, exception_t> = 0;
        virtual auto remap_try(MapResource, fsys_ptr_t) noexcept -> std::expected<std::optional<MapResource>, exception_t> = 0;
        virtual void unmap(MapResource) noexcept = 0;
    };

    struct ConcurrentMapInterface{
        virtual ~ConcurrentMapInterface() noexcept = default;
        virtual auto map(fsys_ptr_t) noexcept -> std::expected<ConcurrentMapResource, exception_t> = 0;
        virtual auto remap_try(ConcurrentMapResource, fsys_ptr_t) noexcept -> std::expected<std::optional<ConcurrentMapResource>, exception_t> = 0;
        virtual void unmap(ConcurrentMapResource) noexcept = 0;
    };

    struct MapDistributorInterface{
        virtual ~MapDistributorInterface() noexcept = default;
        virtual auto id(fsys_ptr_t) noexcept -> std::expected<size_t, exception_t> = 0;
    };
}

namespace dg::network_kernelmap_x_impl1::implementation{

    using namespace network_kernelmap_x_impl1::interface;

    struct HeapNodeCmpLess{

        constexpr auto operator()(const HeapNode& lhs, const HeapNode& rhs) const noexcept -> bool{

            return std::make_tuple(lhs.reference, lhs.last_modified) < std::make_tuple(rhs.reference, rhs.last_modified); 
        } 
    };

    class FsysLoader: public virtual FsysLoaderInterface{

        private:

            dg::unordered_unstable_map<fsys_ptr_t, std::filesystem::path> alias_dict; //this is actually an easy fix - just have virtual filepath and rid of the pollution problem
            size_t memregion_sz;

        public:

            FsysLoader(dg::unordered_unstable_map<fsys_ptr_t, std::filesystem::path> alias_dict,
                       size_t memregion_sz) noexcept: alias_dict(std::move(alias_dict)),
                                                      memregion_sz(memregion_sz){}

            auto load(MemoryNode& root, fsys_ptr_t region, size_t sz) noexcept -> exception_t{
                
                if (root.cptr == nullptr){
                    return dg::network_exception::INVALID_ARGUMENT;
                }

                if (root.fsys_ptr != NULL_FSYS_PTR){
                    return dg::network_exception::INVALID_ARGUMENT; //fine - this is a subset of MemoryNode valid_states that does not meet the requirement of load - precond qualified
                }

                if (sz > this->memregion_sz){
                    return dg::network_exception::BAD_ACCESS;
                }

                auto dict_ptr = this->alias_dict.find(region);

                if (dict_ptr == this->alias_dict.end()){
                    return dg::network_exception::BAD_ACCESS;
                }

                const std::filesystem::path& inpath = dict_ptr->second;
                const char * cstr_path              = inpath.c_str();
                void * dst                          = root.cptr.get();
                exception_t read_err                = dg::network_fileio::dg_read_binary(cstr_path, dst, sz); 

                if (dg::network_exception::is_failed(read_err)){
                    return read_err;
                }

                root.fsys_ptr   = region;
                root.mem_sz     = sz;

                return dg::network_exception::SUCCESS;
            }

            void unload(MemoryNode& root) noexcept{

                if constexpr(DEBUG_MODE_FLAG){
                    if (root.fsys_ptr == NULL_FSYS_PTR){
                        dg::network_log_stackdump::critical(dg::network_exception::verbose(network_exception::INTERNAL_CORRUPTION));
                        std::abort();
                    }
                }

                fsys_ptr_t host_region  = root.fsys_ptr;
                auto dict_ptr           = this->alias_dict.find(host_region);

                if constexpr(DEBUG_MODE_FLAG){
                    if (dict_ptr == this->alias_dict.end()){
                        dg::network_log_stackdump::critical(network_exception::verbose(network_exception::INTERNAL_CORRUPTION));
                        std::abort();
                    }
                }

                const std::filesystem::path& opath  = dict_ptr->second;
                const char * cstr_path              = opath.c_str();
                void * src                          = root.cptr.get();
                size_t cpy_sz                       = root.mem_sz;
                dg::network_exception_handler::nothrow_log(dg::network_fileio::dg_write_binary(cstr_path, src, cpy_sz)); //this has to be a nothrow-ops - I don't like inverse operation to be throw-able - recoverability is unified-fsys's responsibility
                root.fsys_ptr                       = NULL_FSYS_PTR;
                root.mem_sz                         = 0u;
            }
    };

    class Map: public virtual MapInterface{

        private:

            dg::vector<std::unique_ptr<HeapNode>> priority_queue;
            dg::unordered_unstable_map<fsys_ptr_t, HeapNode *> allocation_dict;
            std::unique_ptr<FsysLoaderInterface> fsys_loader;
            MemoryNode tmp_space;
            std::unique_ptr<Lock> lck;
            stdx::hdi_container<size_t> memregion_sz; //we are accessing this concurrently surrounded by the mutating variables - so we must use hardware destructive interference

        public:

            Map(dg::vector<std::unique_ptr<HeapNode>> priority_queue, 
                dg::unordered_unstable_map<fsys_ptr_t, HeapNode *> allocation_dict,
                std::unique_ptr<FsysLoaderInterface> fsys_loader,
                MemoryNode tmp_space,
                std::unique_ptr<Lock> lck,
                size_t memregion_sz) noexcept: priority_queue(std::move(priority_queue)),
                                               allocation_dict(std::move(allocation_dict)),
                                               fsys_loader(std::move(fsys_loader)),
                                               tmp_space(std::move(tmp_space)),
                                               lck(std::move(lck)),
                                               memregion_sz(stdx::hdi_container<size_t>{memregion_sz}){}

            auto map(fsys_ptr_t ptr) noexcept -> std::expected<MapResource, exception_t>{

                stdx::xlock_guard<Lock> lck_grd(*this->lck);
                return this->internal_map(ptr);
            }

            auto remap_try(MapResource resource, fsys_ptr_t ptr) noexcept -> std::expected<std::optional<MapResource>, exception_t>{

                if (dg::memult::ptrcmp_equal(dg::memult::region(ptr, this->memregion_sz.value), resource.fsys_ptr)){
                    return std::optional<MapResource>(MapResource{.node             = resource.node, 
                                                                  .mapped_region    = resource.mapped_region, 
                                                                  .off              = dg::memult::region_offset(ptr, this->memregion_sz.value),
                                                                  .fsys_ptr         = resource.fsys_ptr});
                }

                return std::optional<MapResource>(std::nullopt);
            }

            void unmap(MapResource map_resource) noexcept{

                stdx::xlock_guard<Lock> lck_grd(*this->lck);
                this->internal_unmap(map_resource);
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

                if (HeapNodeCmpLess{}(*this->priority_queue[idx], *this->priority_queue[c])){
                    this->heap_swap(this->priority_queue[c], this->priority_queue[idx]);
                    this->heap_push_up_at(c);
                }
            }

            void heap_push_down_at(size_t idx) noexcept{

                size_t c = idx * 2 + 1;

                if (c >= this->priority_queue.size()){
                    return;
                }

                if (c + 1 < this->priority_queue.size() && HeapNodeCmpLess{}(*this->priority_queue[c + 1], *this->priority_queue[c])){
                    c += 1;
                }

                if (HeapNodeCmpLess{}(*this->priority_queue[c], *this->priority_queue[idx])){
                    this->heap_swap(this->priority_queue[idx], this->priority_queue[c]);
                    this->heap_push_down_at(c);
                }
            }

            auto internal_map(fsys_ptr_t ptr) noexcept -> std::expected<MapResource, exception_t>{

                fsys_ptr_t ptr_region   = dg::memult::region(ptr, this->memregion_sz.value);
                size_t ptr_offset       = dg::memult::region_offset(ptr, this->memregion_sz.value);
                auto dict_ptr           = this->allocation_dict.find(ptr_region);

                //found cache_map
                //cache map entry guarantees that the page is valid, increment + update mofidication time return MapResource 

                if (dict_ptr != this->allocation_dict.end()){
                    HeapNode * found_node       = dict_ptr->second;
                    found_node->reference       += 1;
                    found_node->last_modified   = std::chrono::steady_clock::now();

                    heap_push_down_at(found_node->idx);

                    return MapResource{.node            = found_node, 
                                       .mapped_region   = found_node->cptr.get(), 
                                       .off             = ptr_offset,
                                       .fsys_ptr        = ptr_region};
                }

                //did not find the associated cache page, the page is in the file system

                HeapNode * cand = this->priority_queue[0].get();

                if (cand->fsys_ptr != NULL_FSYS_PTR){
                    if (cand->reference != 0u){
                        return std::unexpected(dg::network_exception::RESOURCE_EXHAUSTION); //OK
                    }

                    //load into the memory node, memory node is guaranteed to be empty
                    exception_t err = this->fsys_loader->load(this->tmp_space, ptr_region, this->memregion_sz.value);

                    if (dg::network_exception::is_failed(err)){
                        return std::unexpected(err); //OK, no actions on internal states, returns unexpected(err);
                    }

                    //data is loaded
                    //we'll try to make the tx atomic

                    //--confusing--
                    fsys_ptr_t removing_region = cand->fsys_ptr;
                    this->fsys_loader->unload(*cand); //unload the cache page
                    size_t rm_sz = this->allocation_dict.erase(removing_region); //evict the cache page from the allocation dict

                    //at this point, in the perspective of cand, we are in a valid state as if the cache page (or fsys_ptr) was never loaded, this is a perfect inverse operation

                    if constexpr(DEBUG_MODE_FLAG){
                        if (rm_sz != 1u){
                            dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                            std::abort();
                        }
                    }

                    //the current top node is associated with the ptr_region, do cache page registration
                    auto [_, status] = this->allocation_dict.insert(std::make_pair(ptr_region, cand));

                    if constexpr(DEBUG_MODE_FLAG){
                        if (!status){
                            dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                            std::abort();
                        }
                    }

                    //this is the buggy part (we'll attempt to swap the resource of tmp_space and cand, new tmp_space is guaranteed to be empty, and cand to represents the tmp_space)
                    //this only works if <cptr[], mem_sz> is not a problem to other problems, this is a bad assumption
                    std::swap(static_cast<MemoryNode&>(*cand), this->tmp_space);
                    //--end-of-confusing--

                    cand->reference += 1;
                    cand->last_modified = std::chrono::steady_clock::now();

                    this->heap_push_down_at(0u);

                    return MapResource{.node            = cand, 
                                       .mapped_region   = cand->cptr.get(), 
                                       .off             = ptr_offset,
                                       .fsys_ptr        = ptr_region};
                }

                exception_t err = this->fsys_loader->load(*cand, ptr_region, this->memregion_sz.value);

                if (dg::network_exception::is_failed(err)){
                    return std::unexpected(err); //OK, cand internal state atomicity is guaranteed by the callee
                }

                //attempt to register ptr_region -> cand
                auto [_, status] = this->allocation_dict.insert(std::make_pair(ptr_region, cand));

                if constexpr(DEBUG_MODE_FLAG){
                    if (!status){
                        dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                        std::abort();
                    }
                }

                cand->reference += 1;
                cand->last_modified = std::chrono::steady_clock::now();

                this->heap_push_down_at(0u);

                return MapResource{.node            = cand, 
                                   .mapped_region   = cand->cptr.get(), 
                                   .off             = ptr_offset,
                                   .fsys_ptr        = ptr_region};
            }

            void internal_unmap(MapResource map_resource) noexcept{

                map_resource.node->reference        -= 1;
                map_resource.node->last_modified    = std::chrono::steady_clock::now();

                heap_push_up_at(map_resource.node->idx);
            } 
    };
    
    class StdMapDistributor: public virtual MapDistributorInterface{

        private:

            dg::unordered_unstable_map<fsys_ptr_t, size_t> region_id_dict;
            size_t memregion_sz;

        public:

            StdMapDistributor(dg::unordered_unstable_map<fsys_ptr_t, size_t> region_id_dict,
                              size_t memregion_sz) noexcept: region_id_dict(std::move(region_id_dict)),
                                                             memregion_sz(memregion_sz){}

            auto id(fsys_ptr_t ptr) noexcept -> std::expected<size_t, exception_t>{

                fsys_ptr_t ptr_region   = dg::memult::region(ptr, this->memregion_sz);
                auto dict_ptr           = this->region_id_dict.find(ptr_region);

                if (dict_ptr == this->region_id_dict.end()){
                    return std::unexpected(dg::network_exception::INVALID_DICTIONARY_KEY);
                }

                return dict_ptr->second;
            }
    };

    class ConcurrentMap: public virtual ConcurrentMapInterface{

        private:

            dg::vector<std::unique_ptr<MapInterface>> map_table;
            std::unique_ptr<MapDistributorInterface> map_distributor;
        
        public:

            ConcurrentMap(dg::vector<std::unique_ptr<MapInterface>> map_table,
                          std::unique_ptr<MapDistributorInterface> map_distributor) noexcept: map_table(std::move(map_table)),
                                                                                              map_distributor(std::move(map_distributor)){}

            auto map(fsys_ptr_t ptr) noexcept -> std::expected<ConcurrentMapResource, exception_t>{

                auto map_id = this->map_distributor->id(ptr);

                if (!map_id.has_value()){
                    return std::unexpected(map_id.error());
                }

                auto resource = this->map_table[map_id.value()]->map(ptr);
                
                if (!resource.has_value()){
                    return std::unexpected(resource.error());
                }

                return ConcurrentMapResource{.resource  = resource.value(), 
                                             .map_id    = map_id.value()};
            }

            auto remap_try(ConcurrentMapResource map_resource, fsys_ptr_t ptr) noexcept -> std::expected<std::optional<ConcurrentMapResource>, exception_t>{

                auto map_id = this->map_distributor->id(ptr);

                if (!map_id.has_value()){
                    return std::unexpected(map_id.error());
                }

                if (map_resource.map_id != map_id.value()){
                    return std::optional<ConcurrentMapResource>(std::nullopt);
                }

                std::expected<std::optional<MapResource>, exception_t> remap_resource = this->map_table[map_id.value()]->remap_try(map_resource.resource, ptr);

                if (!remap_resource.has_value()){
                    return std::unexpected(remap_resource.error());
                }

                if (!remap_resource.value().has_value()){
                    return std::optional<ConcurrentMapResource>(std::nullopt);
                }

                return std::optional<ConcurrentMapResource>(ConcurrentMapResource{.resource = remap_resource.value().value(), 
                                                                                  .map_id   = map_id.value()});
            }

            void unmap(ConcurrentMapResource map_resource) noexcept{
                
                map_table[map_resource.map_id]->unmap(map_resource.resource);
            }
    };

    //OK
    //----
    //we'll need to do accurate concurrent_mapping to increase performance
    //each page should be 10MB to avoid overheads
    //we'll be dispatching tons of tile on a page to avoid page_fault 

    struct Factory{

        static auto off_sptr(std::shared_ptr<char[]> sptr, size_t off) -> std::shared_ptr<char[]>{

            auto destructor = [sptr](char *) noexcept{};
            char * new_ptr  = stdx::advance(sptr.get(), off);
            
            return std::unique_ptr<char[], decltype(destructor)>(new_ptr, destructor);
        }

        static auto spawn_fsys_loader(const dg::unordered_map<fsys_ptr_t, std::filesystem::path>& alias_map, size_t memregion_sz) -> std::unique_ptr<FsysLoaderInterface>{
            
            if (!dg::memult::is_pow2(memregion_sz)){
                dg::network_exception::throw_exception(dg::network_exception::INVALID_ARGUMENT);
            }

            auto alias_fast_map = dg::unordered_unstable_map<fsys_ptr_t, std::filesystem::path>(alias_map.begin(), alias_map.end());
            return std::make_unique<FsysLoader>(std::move(alias_fast_map), memregion_sz);
        }

        static auto spawn_map(const dg::unordered_map<fsys_ptr_t, std::filesystem::path>& alias_map, size_t memregion_sz, size_t memory_node_count) -> std::unique_ptr<MapInterface>{

            const size_t MIN_MEMORY_NODE_COUNT  = 1u;
            const size_t MAX_MEMORY_NODE_COUNT  = size_t{1} << 30;

            if (!dg::memult::is_pow2(memregion_sz)){
                dg::network_exception::throw_exception(dg::network_exception::INVALID_ARGUMENT);
            }

            if (std::clamp(memory_node_count, MIN_MEMORY_NODE_COUNT, MAX_MEMORY_NODE_COUNT) != memory_node_count){
                dg::network_exception::throw_exception(dg::network_exception::INVALID_ARGUMENT);
            }
    
            auto priority_queue                 = dg::vector<std::unique_ptr<HeapNode>>{};
            std::shared_ptr<char[]> memblk      = std::unique_ptr<char[], decltype(&std::free)>(static_cast<char *>(std::aligned_alloc(memregion_sz, memregion_sz * (memory_node_count + 1))), std::free);

            for (size_t i = 0u; i < memory_node_count; ++i){
                HeapNode node{};
                node.idx            = i;
                node.cptr           = off_sptr(memblk, i * memregion_sz);
                node.fsys_ptr       = NULL_FSYS_PTR;
                node.reference      = 0u;
                node.last_modified  = std::chrono::steady_clock::now();
                priority_queue.push_back(std::make_unique<HeapNode>(std::move(node)));
            }

            dg::unordered_unstable_map<fsys_ptr_t, HeapNode *> allocation_dict{};
            MemoryNode tmp{};
            tmp.cptr            = off_sptr(memblk, memory_node_count * memregion_sz);
            tmp.fsys_ptr        = NULL_FSYS_PTR;
            tmp.reference       = 0u;
            tmp.last_modified   = std::chrono::steady_clock::now(); 
            auto lck            = std::make_unique<Lock>();

            return std::make_unique<Map>(std::move(priority_queue), std::move(allocation_dict), spawn_fsys_loader(alias_map, memregion_sz), std::move(tmp), std::move(lck), memregion_sz);
        }

        static auto spawn_map_distributor(const dg::unordered_map<fsys_ptr_t, size_t>& region_id_dict, size_t memregion_sz) -> std::unique_ptr<MapDistributorInterface>{

            if (!dg::memult::is_pow2(memregion_sz)){
                dg::network_exception::throw_exception(dg::network_exception::INVALID_ARGUMENT);
            }

            dg::unordered_unstable_map<fsys_ptr_t, size_t> region_id_fastdict(region_id_dict.begin(), region_id_dict.end());
            return std::make_unique<StdMapDistributor>(std::move(region_id_fastdict), memregion_sz);
        } 

        static auto spawn_concurrent_map(dg::vector<std::unique_ptr<MapInterface>> map_table, std::unique_ptr<MapDistributorInterface> map_distributor) -> std::unique_ptr<ConcurrentMapInterface>{

            if (std::find(map_table.begin(), map_table.end(), nullptr) != map_table.end()){
                dg::network_exception::throw_exception(dg::network_exception::INVALID_ARGUMENT);
            }

            if (map_distributor == nullptr){
                dg::network_exception::throw_exception(dg::network_exception::INVALID_ARGUMENT);
            }

            return std::make_unique<ConcurrentMap>(std::move(map_table), std::move(map_distributor));
        }
    };

    struct ConcurrentMapMake{

        static auto unifdist_map(const dg::unordered_map<fsys_ptr_t, std::filesystem::path>& alias_map, size_t distribution_factor) -> dg::vector<dg::unordered_map<fsys_ptr_t, std::filesystem::path>>{

            if (alias_map.size() == 0u){
                dg::network_exception::throw_exception(dg::network_exception::INVALID_ARGUMENT);
            }

            const size_t MIN_DISTRIBUTION_FACTOR    = 1u;
            const size_t MAX_DISTRIBUTION_FACTOR    = alias_map.size();

            if (std::clamp(distribution_factor, MIN_DISTRIBUTION_FACTOR, MAX_DISTRIBUTION_FACTOR) != distribution_factor){
                dg::network_exception::throw_exception(dg::network_exception::INVALID_ARGUMENT);
            }

            auto rs             = dg::vector<dg::unordered_map<fsys_ptr_t, std::filesystem::path>>{};
            auto map_vec        = dg::vector<std::pair<fsys_ptr_t, std::filesystem::path>>(alias_map.begin(), alias_map.end());
            size_t segment_sz   = map_vec.size() / distribution_factor;

            std::shuffle(map_vec.begin(), map_vec.end(), std::mt19937{});

            for (size_t i = 0u; i < distribution_factor; ++i){
                size_t first    = segment_sz * i;
                size_t last     = segment_sz * (i + 1);

                if (i + 1 == distribution_factor){
                    last = map_vec.size();
                }

                auto appendee = dg::unordered_map<fsys_ptr_t, std::filesystem::path>(map_vec.begin() + first, map_vec.begin() + last); 
                rs.push_back(std::move(appendee));
            }

            return rs;
        }

        static auto make(const dg::unordered_map<fsys_ptr_t, std::filesystem::path>& bijective_alias_map,
                         size_t memregion_sz,
                         double ram_to_disk_ratio,
                         size_t distribution_factor) -> std::unique_ptr<interface::ConcurrentMapInterface>{
            
            const double MIN_RAM_TO_DISK_RATIO  = double{0.001};
            const double MAX_RAM_TO_DISK_RATIO  = double{0.999};

            if (std::clamp(ram_to_disk_ratio, MIN_RAM_TO_DISK_RATIO, MAX_RAM_TO_DISK_RATIO) != ram_to_disk_ratio){
                dg::network_exception::throw_exception(dg::network_exception::INVALID_ARGUMENT);
            } 

            dg::vector<std::unique_ptr<MapInterface>> map_table{};
            dg::unordered_map<fsys_ptr_t, size_t> region_table_idx_map{};
            dg::vector<dg::unordered_map<fsys_ptr_t, std::filesystem::path>> distributed_map = unifdist_map(bijective_alias_map, distribution_factor); 

            for (const auto& raw_map: distributed_map){
                size_t memory_node_count = raw_map.size() * ram_to_disk_ratio; //this is prolly the most problematic line
                std::unique_ptr<MapInterface> map = Factory::spawn_map(raw_map, memregion_sz, memory_node_count);
                map_table.push_back(std::move(map));

                for (const auto& map_pair: raw_map){
                    region_table_idx_map[std::get<0>(map_pair)] = map_table.size() - 1;
                }
            }

            std::unique_ptr<MapDistributorInterface> map_distributor = Factory::spawn_map_distributor(std::move(region_table_idx_map), memregion_sz);
            return Factory::spawn_concurrent_map(std::move(map_table), std::move(map_distributor));
        }
    };
}

namespace dg::network_kernelmap_x_impl1{

    using fsys_ptr_t = dg::network_pointer::fsys_ptr_t; 

    extern auto make(const dg::unordered_map<fsys_ptr_t, std::filesystem::path>& bijective_alias_map, 
                     size_t memregion_sz, 
                     double ram_to_disk_ratio, 
                     size_t distribution_factor) -> std::unique_ptr<interface::ConcurrentMapInterface>{

        return implementation::ConcurrentMapMake::make(bijective_alias_map, memregion_sz, ram_to_disk_ratio, distribution_factor);
    }
}

#endif