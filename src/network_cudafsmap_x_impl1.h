#ifndef __NETWORK_CUDAFSMAP_X_IMPL1_H__
#define __NETWORK_CUDAFSMAP_X_IMPL1_H__

#include <stdint.h>
#include <stdlib.h>
#include <chrono>
#include <optional>
#include <nutex>
#include <atomic>
#include "network_exception.h"
#include "network_log.h"
#include "network_cufsio.h"
#include "network_allocation_clib.h"
#include "network_std_container.h"
#include "network_pointer.h"
#include "stdx.h"

namespace dg::network_cudafsmap_x_impl1::model{

    static inline constexpr bool IS_ATOMIC_OPERATION_PREFERRED = false;

    using cuda_ptr_t    = dg::network_pointer::cuda_ptr_t;
    using cufs_ptr_t    = dg::network_pointer::cufs_ptr_t;
    using Lock          = std::conditional_t<IS_ATOMIC_OPERATION_PREFERRED,
                                             std::atomic_flag,
                                             std::mutex>;  

    struct MemoryNode{
        std::shared_ptr<cuda_ptr_t> cuptr;
        cufs_ptr_t fsys_ptr;
        size_t mem_sz;
    };

    struct ReferenceMemoryNode: MemoryNode{
        size_t reference; 
        std::chrono::nanoseconds last_modified;
    };

    struct HeapNode: ReferenceMemoryNode{
        size_t idx;
    };

    struct MapResource{
        HeapNode * node;
        cuda_ptr_t mapped_region;
        size_t off;

        inline auto ptr() const noexcept -> cuda_ptr_t{

            return dg::memult:next(this->mapped_region, this->off);
        }
    };

    struct ConcurrentMapResource{
        MapResource resource;
        size_t map_id;

        inline auto ptr() const noexcept -> cuda_ptr_t{

            return resource.ptr();
        }
    };
}

namespace dg::network_cudafsmap_x_impl1::interface{

    using namespace network_cudafsmap_x_impl1::model; 

    struct CuFSLoaderInterface{
        virtual ~CuFSLoaderInterface() noexcept = default;
        virtual auto load(MemoryNode&, cufs_ptr_t, size_t) noexcept -> exception_t = 0;
        virtual void unload(MemoryNode&) noexcept = 0;
    };

    struct MapInterface{
        virtual ~MapInterface() noexcept = default;
        virtual auto map(cufs_ptr_t) noexcept -> std::expected<MapResource, exception_t> = 0;
        virtual auto remap_try(MapResource, cufs_ptr_t) noexcept -> std::expected<std::optional<MapResource>, exception_t> = 0;
        virtual void unmap(MapResource) noexcept = 0;
    };

    struct ConcurrentMapInterface{
        virtual ~ConcurrentMapInterface() noexcept = default;
        virtual auto map(cufs_ptr_t) noexcept -> std::expected<ConcurrentMapResource, exception_t> = 0;
        virtual auto remap_try(ConcurrentMapResource, cufs_ptr_t) noexcept -> std::expected<std::optional<ConcurrentMapResource>, exception_t> = 0;
        virtual void unmap(ConcurrentMapResource) noexcept = 0;
    };

    struct MapDistributorInterface{
        virtual ~MapDistributorInterface() noexcept = default;
        virtual auto id(cufs_ptr_t) noexcept -> std::expected<size_t, exception_t> = 0;
    };
}

namespace dg::network_cudafsmap_x_impl1::implementation{

    using namespace network_cudafsmap_x_impl1::interface;

    struct HeapNodeCmpLess{

        constexpr auto operator()(const HeapNode& lhs, const HeapNode& rhs) const noexcept -> bool{

            auto lhs = std::make_tuple(lhs.reference, lhs.last_modified.count()); //I'm hoping the compiler does a simple memcmp - we might need to leverage trivial serialization + std::array cmp if necessary
            auto rhs = std::make_tuple(rhs.reference, rhs.last_modified.count());

            return lhs < rhs;
        } 
    };

    class FsysLoader: public virtual CuFSLoaderInterface{

        private:

            std::unique_ptr<dg::network_cufsio::FsysIOInterface> fsys_io;
            dg::unordered_unstable_map<cufs_ptr_t, std::filesystem::path> alias_dict; //this is actually an easy fix - just have virtual filepath and rid of the pollution problem
            size_t memregion_sz;

        public:

            FsysLoader(std::unique_ptr<dg::network_cufsio::FsysIOInterface> fsys_io,
                       dg::unordered_unstable_map<cufs_ptr_t, std::filesystem::path> alias_dict,
                       size_t memregion_sz) noexcept: fsys_io(std::move(fsys_io)),
                                                      alias_dict(std::move(alias_dict)),
                                                      memregion_sz(memregion_sz){}

            auto load(MemoryNode& root, cufs_ptr_t region, size_t sz) noexcept -> exception_t{
                
                if (root.cuptr == nullptr){
                    return dg::network_exception::INVALID_ARGUMENT;
                }

                if (root.fsys_ptr != dg::pointer_limits<cufs_ptr_t>::null_value()){
                    return dg::network_exception::INVALID_ARGUMENT; //fine - this is a subset of MemoryNode valid_states that does not meet the requirement of load - precond qualified
                }

                if (sz > this->memregion_sz){
                    return dg::network_exception::BAD_ACCESS;
                }

                auto dict_ptr = this->alias_dict.find(region); //we are assuming NULL_CUFS_PTR is not found 

                if (dict_ptr == this->alias_dict.end()){
                    return dg::network_exception::BAD_ACCESS;
                }

                const std::filesystem::path& inpath = dict_ptr->second;
                const char * cstr_path              = inpath.c_str();
                cuda_ptr_t dst                      = *root.cuptr;
                exception_t read_err                = this->fsys_io->cuda_read_binary(cstr_path, dst, sz); 

                if (dg::network_exception::is_failed(read_err)){
                    return read_err;
                }

                root.fsys_ptr   = region;
                root.mem_sz     = sz;

                return dg::network_exception::SUCCESS;
            }

            void unload(MemoryNode& root) noexcept{

                if constexpr(DEBUG_MODE_FLAG){
                    if (root.fsys_ptr == dg::pointer_limits<cufs_ptr_t>::null_value()){
                        dg::network_log_stackdump::critical(network_exception::verbose(network_exception::INTERNAL_CORRUPTION));
                        std::abort();
                    }
                }

                cufs_ptr_t host_region  = root.fsys_ptr;
                auto dict_ptr           = this->alias_dict.find(host_region);

                if constexpr(DEBUG_MODE_FLAG){
                    if (dict_ptr == this->alias_dict.end()){
                        dg::network_log_stackdump::critical(network_exception::verbose(network_exception::INTERNAL_CORRUPTION));
                        std::abort();
                    }
                }

                const std::filesystem::path& opath  = dict_ptr->second;
                const char * cstr_path              = opath.c_str();
                cuda_ptr_t src                      = *root.cuptr;
                size_t cpy_sz                       = root.mem_sz;
                dg::network_exception_handler::nothrow_log(this->fsys_io->cuda_write_binary(cstr_path, src, cpy_sz));
                root.fsys_ptr                       = dg::pointer_limits<cufs_ptr_t>::null_value();
                root.mem_sz                         = 0u;
            }
    };

    class Map: public virtual MapInterface{

        private:

            dg::vector<std::unique_ptr<HeapNode>> priority_queue;
            dg::unordered_unstable_map<cufs_ptr_t, HeapNode *> allocation_dict;
            std::unique_ptr<CuFSLoaderInterface> fsys_loader;
            MemoryNode tmp_space;
            std::unique_ptr<Lock> lck;
            stdx::hdi_container<size_t> memregion_sz; //we have hardware interference issue

        public:

            Map(dg::vector<std::unique_ptr<HeapNode>> priority_queue, 
                dg::unordered_unstable_map<cufs_ptr_t, HeapNode *> allocation_dict,
                std::unique_ptr<CuFSLoaderInterface> fsys_loader,
                MemoryNode tmp_space,
                std::unique_ptr<Lock> lck,
                size_t memregion_sz) noexcept: priority_queue(std::move(priority_queue)),
                                               allocation_dict(std::move(allocation_dict)),
                                               fsys_loader(std::move(fsys_loader)),
                                               tmp_space(std::move(tmp_space)),
                                               lck(std::move(lck)),
                                               memregion_sz(stdx::hdi_container<size_t>{memregion_sz}){} //we are doing hotfixes - 

            auto map(cufs_ptr_t ptr) noexcept -> std::expected<MapResource, exception_t>{

                stdx::xlock_guard<Lock> lck_grd(*this->lck);
                return this->internal_map(ptr);
            }

            auto remap_try(MapResource map_resource, cufs_ptr_t ptr) noexcept -> std::expected<std::optional<MapResource>, exception_t>{

                if (dg::memult::ptrcmp_equal(dg::memult::region(ptr, this->memregion_sz.value), map_resource.fsys_ptr)){
                    return std::optional<MapResource>(MapResource{map_resource.node, map_resource.mapped_region, dg::memult::region_offset(ptr, this->memregion_sz.value)});
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

            auto internal_map(cufs_ptr_t ptr) noexcept -> std::expected<MapResource, exception_t>{

                //this is called induction coding
                //we assume what we want - and we make sure that every exit strategy would snap what we want -> what we want (valid state -> valid state)
                //let's say I'm assuming allocation dict is storing valid mapped references 

                cufs_ptr_t ptr_region   = dg::memult::region(ptr, this->memregion_sz.value);
                size_t ptr_offset       = dg::memult::region_offset(ptr, this->memregion_sz.value);
                auto dict_ptr           = this->allocation_dict.find(ptr_region);

                if (dict_ptr != this->allocation_dict.end()){
                    HeapNode * found_node = dict_ptr->second;
                    found_node->reference += 1;
                    found_node->last_modified = stdx::unix_timestamp();
                    heap_push_down_at(found_node->idx);
                    return MapResource{found_node, *found_node->cuptr, ptr_offset};
                }

                HeapNode * cand = this->priority_queue[0].get();

                if (cand->fsys_ptr != dg::pointer_limits<cufs_ptr_t>::null_value()){
                    if (cand->reference == 0u){
                        exception_t err = this->fsys_loader->load(this->tmp_space, ptr_region, this->memregion_sz.value);

                        if (dg::network_exception::is_failed(err)){
                            return std::unexpected(err);
                        }

                        cufs_ptr_t removing_region = cand->fsys_ptr;
                        this->fsys_loader->unload(*cand);
                        this->allocation_dict.erase(removing_region);
                        this->allocation_dict.insert(std::make_pair(ptr_region, cand));
                        std::swap(static_cast<MemoryNode&>(*cand), this->tmp_space);
                        cand->reference += 1;
                        cand->last_modified = stdx::unix_timestamp();
                        this->heap_push_down_at(0u);

                        return MapResource{cand, *cand->cuptr, ptr_offset};
                    }

                    return std::unexpected(dg::network_exception::RESOURCE_EXHAUSTION);
                }

                //I was trying to prove that if cand->fsys_ptr == null_value then reference == 0u
                //then I make this an assumption and make sure my every code path follows the rule

                exception_t err = this->fsys_loader->load(*cand, ptr_region, this->memregion_sz.value);

                if (dg::network_exception::is_failed(err)){
                    return std::unexpected(err);
                }

                this->allocation_dict.insert(std::make_pair(ptr_region, cand));
                cand->reference += 1;
                cand->last_modified = stdx::unix_timestamp();
                this->heap_push_down_at(0u);

                return MapResource{cand, *cand->cuptr, ptr_offset};
            }

            void internal_unmap(MapResource map_resource) noexcept{

                map_resource.node->reference -= 1;
                map_resource.node->last_modified = stdx::unix_timestamp();
                heap_push_up_at(map_resource.node->idx);
            } 
    };

    class StdMapDistributor: public virtual MapDistributorInterface{

        private:

            dg::unordered_unstable_map<cufs_ptr_t, size_t> region_id_dict;
            size_t memregion_sz;

        public:

            StdMapDistributor(dg::unordered_unstable_map<cufs_ptr_t, size_t> region_id_dict,
                              size_t memregion_sz) noexcept: region_id_dict(std::move(region_id_dict)),
                                                             memregion_sz(memregion_sz){}

            auto id(cufs_ptr_t ptr) noexcept -> std::expected<size_t, exception_t>{

                cufs_ptr_t ptr_region   = dg::memult::region(ptr, this->memregion_sz);
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

            auto map(cufs_ptr_t ptr) noexcept -> std::expected<ConcurrentMapResource, exception_t>{

                auto map_id = this->map_distributor->id(ptr);

                if (!map_id.has_value()){
                    return std::unexpected(map_id.error());
                }

                auto resource = this->map_table[map_id.value()]->map(ptr);
                
                if (!resource.has_value()){
                    return std::unexpected(resource.error());
                }

                return ConcurrentMapResource{resource.value(), map_id.value()};
            }
            
            auto remap_try(ConcurrentMapResource map_resource, cufs_ptr_t ptr) noexcept -> std::expected<std::optional<ConcurrentMapResource>, exception_t>{

                auto map_id = this->map_distributor->id(ptr);

                if (!map_id.has_value()){
                    return std::unexpected(map_id.error());
                } 

                if (map_resource.map_id != map_id.value()){
                    return std::optional<ConcurrentMapResource>(std::nullopt);
                }

                std::expected<std::optional<MapResource>, exception_t> resource = this->map_table[map_id.value()]->remap(map_resource.resource, ptr);

                if (!resource.has_value()){
                    return std::unexpected(resource.error());
                }

                if (!resource.value().has_value()){
                    return std::optional<ConcurrentMapResource>(std::nullopt);
                }

                return ConcurrentMapResource{resource.value().value(), map_id.value()};
            }

            void unmap(ConcurrentMapResource map_resource) noexcept{
                
                map_table[map_resource.map_id]->unmap(map_resource.resource);
            }
    };

    struct Factory{

        static auto off_cuptr(std::shared_ptr<cuda_ptr_t> sptr, size_t off) -> std::shared_ptr<cuda_ptr_t>{

            auto destructor = [sptr](cuda_ptr_t * cuda_ptr) noexcept{
                delete cuda_ptr;
            };

            cuda_ptr_t new_ptr = dg::memult::badvance(*sptr.get(), off);
            return std::unique_ptr<cuda_ptr_t, decltype(destructor)>(new cuda_ptr_t(new_ptr), destructor);
        }

        static auto spawn_fsys_loader(std::unique_ptr<dg::network_cufsio::FsysIOInterface> fsysio,
                                      const dg::unordered_map<cufs_ptr_t, std::filesystem::path>& alias_map, 
                                      size_t memregion_sz) -> std::unique_ptr<CuFSLoaderInterface>{
            
            if (fsysio == nullptr){
                dg::network_exception::throw_exception(dg::network_exception::INVALID_ARGUMENT);
            }

            if (!dg::memult::is_pow2(memregion_sz)){
                dg::network_exception::throw_exception(dg::network_exception::INVALID_ARGUMENT);
            }

            auto alias_fast_map = dg::unordered_unstable_map<cufs_ptr_t, std::filesystem::path>(alias_map.begin(), alias_map.end()); 
            return std::make_unique<FsysLoader>(std::move(fsysio), std::move(alias_fast_map), memregion_sz);
        }

        static auto spawn_map(const dg::unordered_map<cufs_ptr_t, std::filesystem::path>& alias_map, size_t memregion_sz, int gpu_device, size_t memory_node_count) -> std::unique_ptr<MapInterface>{

            const size_t MIN_MEMORY_NODE_COUNT  = 1u;
            const size_t MAX_MEMORY_NODE_COUNT  = size_t{1} << 30;

            if (!dg::memult::is_pow2(memregion_sz)){
                dg::network_exception::throw_exception(dg::network_exception::INVALID_ARGUMENT);
            }

            if (std::clamp(memory_node_count, MIN_MEMORY_NODE_COUNT, MAX_MEMORY_NODE_COUNT) != memory_node_count){
                dg::network_exception::throw_exception(dg::network_exception::INVALID_ARGUMENT);
            }
    
            auto priority_queue                 = dg::vector<std::unique_ptr<HeapNode>>{};
            std::shared_ptr<cuda_ptr_t> memblk  = dg::network_allocation_cuda_x::cuda_aligned_malloc(gpu_device, memregion_sz, memregion_sz * (memory_node_count + 1));
            auto registered_stable_ptr          = dg::vector<std::pair<cuda_ptr_t, size_t>>{};

            for (size_t i = 0u; i < memory_node_count; ++i){
                HeapNode node{};
                node.idx            = i;
                node.cuptr          = off_cuptr(memblk, i * memregion_sz);
                node.fsys_ptr       = dg::pointer_limits<cufs_ptr_t>::null_value();
                node.reference      = 0u;
                node.last_modified  = stdx::unix_timestamp();
                registered_stable_ptr.push_back(std::make_pair(*node.cuptr, memregion_sz));
                priority_queue.push_back(std::make_unique<HeapNode>(std::move(node)));
            }

            dg::unordered_unstable_map<cufs_ptr_t, HeapNode *> allocation_dict{};
            MemoryNode tmp{};
            tmp.cuptr           = off_cuptr(memblk, memory_node_count * memregion_sz);
            tmp.fsys_ptr        = dg::pointer_limits<cufs_ptr_t>::null_value();
            tmp.reference       = 0u;
            tmp.last_modifed    = stdx::unix_timestamp(); 
            auto lck            = std::make_unique<Lock>();
            registered_stable_ptr.push_back(std::make_pair(*tmp.cuptr, memregion_sz));

            std::unique_ptr<dg::network_cufsio::FsysIOInterface> raw_fsys_loader = dg::network_cufsio::make_cuda_fsys_loader(registered_stable_ptr);
            std::unique_ptr<CuFSLoaderInterface> alias_fsys_loader = spawn_fsys_loader(std::move(raw_fsys_loader), alias_map, memregion_sz);

            return std::make_unique<Map>(std::move(priority_queue), std::move(allocation_dict), std::move(alias_fsys_loader), std::move(tmp), std::move(lck), memregion_sz);
        }

        static auto spawn_map_distributor(const dg::unordered_map<cufs_ptr_t, size_t>& region_id_dict, size_t memregion_sz) -> std::unique_ptr<MapDistributorInterface>{

            if (!dg::memult::is_pow2(memregion_sz)){
                dg::network_exception::throw_exception(dg::network_exception::INVALID_ARGUMENT);
            }

            dg::unordered_unstable_map<cufs_ptr_t, size_t> region_id_fastdict(region_id_dict.begin(), region_id_dict.end());
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

        static auto to_device_memmap(const dg::unordered_map<cufs_ptr_t, std::filesystem::path>& alias_map,
                                     const dg::unordered_map<std::filesystem::path, int>& gpu_platform_map) -> dg::unordered_map<int, dg::unordered_map<cufs_ptr_t, std::filesystem::path>>{

            auto rs = dg::unordered_map<int, dg::unordered_map<cufs_ptr_t, std::filesystem::path>>{};

            for (const auto& map_pair: alias_map){
                auto [ptr, fsys_path]   = map_pair;
                auto gpu_ptr            = gpu_platform_map.find(fsys_path);

                if (gpu_ptr == gpu_platform_map.end()){
                    dg::network_exception::throw_exception(dg::network_exception::INVALID_ARGUMENT);
                }

                int gpu_device_id       = gpu_ptr->second;
                rs[gpu_device_id][ptr]  = fsys_path;
            }

            return rs;
        }

        static auto unifdist_map(const dg::unordered_map<cufs_ptr_t, std::filesystem::path>& map, size_t distribution_factor) -> dg::vector<dg::unordered_map<cufs_ptr_t, std::filesystem::path>>{

            if (map.size() == 0u){
                dg::network_exception::throw_exception(dg::network_exception::INVALID_ARGUMENT);
            }

            const size_t MIN_DISTRIBUTION_FACTOR    = 1u;
            const size_t MAX_DISTRIBUTION_FACTOR    = map.size();

            if (std::clamp(distribution_factor, MIN_DISTRIBUTION_FACTOR, MAX_DISTRIBUTION_FACTOR) != distribution_factor){
                dg::network_exception::throw_exception(dg::network_exception::INVALID_ARGUMENT);
            }

            auto rs             = dg::vector<dg::unordered_map<cufs_ptr_t, std::filesystem::path>>{};
            auto map_vec        = dg::vector<std::pair<cufs_ptr_t, std::filesystem::path>>(map.begin(), map.end());
            size_t segment_sz   = map_vec.size() / distribution_factor;
            auto rand_dev       = std::bind(std::uniform_int_distribution<size_t>{}, std::mt19937{}); 

            std::shuffle(map_vec.begin(), map_vec.end(), rand_dev);

            for (size_t i = 0u; i < distribution_factor; ++i){
                size_t first    = segment_sz * i;
                size_t last     = segment_sz * (i + 1);

                if (i + 1 == distribution_factor){
                    last = map_vec.size();
                }

                auto appendee = dg::unordered_map<cufs_ptr_t, std::filesystem::path>(map_vec.begin() + first, map_vec.begin() + last); 
                rs.push_back(std::move(appendee));
            }

            return rs;
        }

        static auto make(const dg::unordered_map<cufs_ptr_t, std::filesystem::path>& bijective_alias_map, 
                         const dg::unordered_map<std::filesystem::path, int>& gpu_platform_map, 
                         size_t memregion_sz, 
                         double ram_to_disk_ratio, 
                         size_t distribution_factor) -> std::unique_ptr<interface::ConcurrentMapInterface>{
            
            const double MIN_RAM_TO_DISK_RATIO  = double{0.001};
            const double MAX_RAM_TO_DISK_RATIO  = double{0.999};

            if (std::clamp(ram_to_disk_ratio, MIN_RAM_TO_DISK_RATIO, MAX_RAM_TO_DISK_RATIO) != ram_to_disk_ratio){
                dg::network_exception::throw_exception(dg::network_exception::INVALID_ARGUMENT);
            } 

            dg::unordered_map<int, dg::unordered_map<cufs_ptr_t, std::filesystem::path>> device_memmap = to_device_memmap(bijective_alias_map, gpu_platform_map);
            dg::vector<std::unique_ptr<MapInterface>> map_table{};
            dg::unordered_map<cufs_ptr_t, size_t> region_table_idx_map{};

            for (const auto& devicemap_pair: device_memmap){
                dg::vector<dg::unordered_map<cufs_ptr_t, std::filesystem::path>> distributed_map = unifdist_map(std::get<1>(devicemap_pair), distribution_factor); 

                for (const auto& raw_map: distributed_map){
                    size_t memory_node_count = raw_map.size() * ram_to_disk_ratio; //this is prolly the most problematic line
                    std::unique_ptr<MapInterface> map = Factory::spawn_map(raw_map, memregion_sz, std::get<0>(devicemap_pair), memory_node_count);
                    map_table.push_back(std::move(map));

                    for (const auto& map_pair: raw_map){
                        region_table_idx_map[std::get<0>(map_pair)] = map_table.size() - 1;
                    }
                }
            }

            std::unique_ptr<MapDistributorInterface> map_distributor = Factory::spawn_map_distributor(std::move(region_table_idx_map), memregion_sz);
            return Factory::spawn_concurrent_map(std::move(map_table), std::move(map_distributor));
        }
    };
}

namespace dg::network_cudafsmap_x_impl1{

    extern auto make(const dg::unordered_map<cufs_ptr_t, std::filesystem::path>& bijective_alias_map, 
                     const dg::unordered_map<std::filesystem::path, int>& gpu_platform_map, 
                     size_t memregion_sz, 
                     double ram_to_disk_ratio, 
                     size_t distribution_factor) -> std::unique_ptr<interface::ConcurrentMapInterface>{

        return implementation::ConcurrentMapMake::make(bijective_alias_map, gpu_platform_map, memregion_sz, ram_to_disk_ratio, distribution_factor);
    }
}

#endif