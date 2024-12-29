#ifndef __NETWORK_CUDA_MAP_IMPL1_H__
#define __NETWORK_CUDA_MAP_IMPL1_H__

//we are mapping cuda_ptr_t -> cuda_pinned_ptr_t
//we are doing eviction like we are doing with fsys_ptr_t
//this is mainly for transferring data from host -> cuda and cuda -> host - without the cost of cudaMemcpy - by seriallly copy the data from/to an intermediate fast platform
//alright - 2024 and this is still relevant - we'll operate mostly on cuda_pinned_ptr_t to transfer to and from cuda asynchronous devices
//code is clear - we'll be back

namespace dg::network_cudamap_impl1::model{

    struct MemoryNode{
        std::shared_ptr<cuda_pinned_ptr_t> cupinned_ptr;
        cuda_ptr_t cuda_ptr;
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
        cuda_pinned_ptr_t mapped_region; //we are avoiding hardware interference
        size_t off;

        auto ptr() const noexcept -> cuda_pinned_ptr_t{

            return dg::memult::next(this->mapped_region, this->off);
        }
    };

    struct ConcurrentMapResource{
        MapResource resource;
        size_t map_id;

        auto ptr() const noexcept -> cuda_pinned_ptr_t{

            return this->resource.ptr();
        }
    };
}

namespace dg::network_cudamap_impl1::interface{

    //there is a virtue in rewriting these - first is interface polymorphism - we aren't having that - so compiler can actually treat this as a real object rather than a virtual object
    //second is maintainability - we wrote things once - leave things be
    //we'll abstract the memregions for now - assume this interface is a general interface for mapping memories - not implementation interface

    using namespace network_cudamap_impl1::model;

    struct MemoryLoaderInterface{
        virtual ~MemoryLoaderInterface() noexcept = default;
        virtual auto load(MemoryNode&, cuda_ptr_t, size_t) noexcept -> exception_t = 0; //its kinda off without the range here - I admit
        virtual void unload(MemoryNode&) noexcept = 0; //we offload the responsibility unload noexceptability here - there are tricks we aren't applying yet
    };

    struct MapInterface{
        virtual ~MapInterface() noexcept = default;
        virtual auto map(cuda_ptr_t) noexcept -> std::expected<MapResource, exception_t> = 0;
        virtual auto evict_try(cuda_ptr_t) noexcept -> std::expected<bool, exception_t> = 0; //returns true if successfully not existed post the call - we dont care about the actual eviction
        virtual auto remap_try(MapResource, cuda_ptr_t) noexcept -> std::expected<std::optional<MapResource>, exception_t> = 0;
        virtual void unmap(MapResource) noexcept = 0;
    };

    struct ConcurrentMapInterface{
        virtual ~ConcurrentMapInterface() noexcept = default;
        virtual auto map(cuda_ptr_t) noexcept -> std::expected<ConcurrentMapResource, exception_t> = 0;
        virtual auto evict_try(cuda_ptr_t) noexcept -> std::expected<bool, exception_t> = 0;
        virtual auto remap_try(ConcurrentMapResource, cuda_ptr_t) noexcept -> std::expected<std::optional<ConcurrentMapResource>, exception_t> = 0;
        virtual void unmap(ConcurrentMapResource) noexcept = 0;
    };

    struct MapDistributorInterface{
        virtual ~MapDistributorInterface() noexcept = default;
        virtual auto id(cuda_ptr_t) noexcept -> std::expected<size_t, exception_t> = 0;
    };
}

namespace dg::network_cudamap_impl1::implementation{

    class MemoryLoader: public virtual MemoryLoaderInterface{

        private:

            dg::vector<std::shared_ptr<cuda_ptr_t>> reference_bag; //we are storing cuda_ptr_t as reference to avoid dangling pointer issues - shared_ptr erasing the deallocator type so we can use that feature here without being too ambiguous like std::shared_ptr<void> or std::unique_ptr<ObjectInterface>
            dg::unordered_unstable_set<cuda_ptr_t> valid_region_set; 
            size_t memregion_sz;

        public:

            MemoryLoader(dg::vector<std::shared_ptr<cuda_ptr_t>> reference_bag,
                         dg::unordered_unstable_set<cuda_ptr_t> valid_region_set,
                         size_t memregion_sz) noexcept: reference_bag(std::move(reference_bag)),
                                                        valid_region_set(std::move(valid_region_set)),
                                                        memregion_sz(memregion_sz){}

            auto load(MemoryNode& root, cuda_ptr_t region, size_t sz) noexcept -> exception_t{

                if (root.cupinned_ptr == nullptr){
                    return dg::network_exception::INVALID_ARGUMENT;
                }

                if (root.cuda_ptr != dg::pointer_limits<cuda_ptr_t>::null_value()){
                    return dg::network_exception::INVALID_ARGUMENT;
                }

                if (sz > this->memregion_sz){
                    return dg::network_exception::BAD_ACCESS;
                }

                if (!this->valid_region_set.contains(region)){ //we assume valid_region_set does not contain nullptr
                    return dg::network_exception::BAD_ACCESS;
                }

                //I dont know if this is 2024 relevant but I'll add this
                if (dg::network_allocation_cuda_x::pinned_memory_platform(*root.cupinned_ptr) != dg::network_allocation_cuda_x::cuda_memory_platform(region)){
                    return dg::network_exception::INCOMPATIBLE_MEMORY_TRANSFER;
                }

                exception_t err = dg::network_cuda_controller::cuda_memcpy(*root.cupinned_ptr, region, sz, dg::network_cuda_controller::memcpyDefault);

                if (dg::network_exception::is_failed(err)){
                    return err; 
                }

                root.cuda_ptr   = region;
                root.mem_sz     = sz;

                return dg::network_exception::SUCCESS;
            }

            void unload(MemoryNode& root) noexcept{

                //this has to be a reverse operation of load - we aren't taking nullptrs

                if constexpr(DEBUG_MODE_FLAG){
                    if (root.cuda_ptr == dg::pointer_limits<cuda_ptr_t>::null_value()){
                        dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                        std::abort();
                    }
                }

                //we are compromising the err by using replicas in the future - we can't do except reverse operation - it's cluttering the code  
                dg::network_exception_handler::nothrow_log(dg::network_cuda_controller::cuda_memcpy(root.cuda_ptr, *root.cupinned_ptr, root.mem_sz, dg::network_cuda_controller::memcpyDefault));
                root.cuda_ptr   = dg::pointer_limits<cuda_ptr_t>::null_value();
                root.mem_sz     = 0u;
            }
    };

    struct ReferenceMemoryNodeCmpLess{

        constexpr auto operator()(const ReferenceMemoryNode& lhs, const ReferenceMemoryNode& rhs) const noexcept -> bool{

            auto lhs = std::make_tuple(lhs.reference, lhs.cuda_ptr != dg::pointer_limits<cuda_ptr_t>::null_value(), lhs.last_modified);
            auto rhs = std::make_tuple(rhs.reference, rhs.cuda_ptr != dg::pointer_limits<cuda_ptr_t>::null_value(), rhs.last_modified);

            return lhs < rhs;
        }
    };

    class Map: public virtual MapInterface{

        private:

            std::unique_ptr<MemoryLoaderInterface> memory_loader;
            dg::vector<std::unique_ptr<HeapNode>> priority_queue;
            dg::unordered_unstable_map<cuda_ptr_t, HeapNode *> mapped_dict;
            std::unique_ptr<MemoryNode> tmp_memory_node;
            size_t memregion_sz;
            std::unique_ptr<Lock> mtx;

        public:

            Map(std::unique_ptr<MemoryLoaderInterface> memory_loader,
                dg::vector<std::unique_ptr<HeapNode>> priority_queue,
                dg::unordered_unstable_map<cuda_ptr_t, HeapNode *> mapped_dict,
                std::unique_ptr<MemoryNode> tmp_memory_node,
                size_t memregion_sz,
                std::unique_ptr<Lock> mtx) noexcept: memory_loader(std::move(memory_loader)),
                                                     priority_queue(std::move(priority_queue)),
                                                     mapped_dict(std::move(mapped_dict)),
                                                     tmp_memory_node(std::move(tmp_memory_node)),
                                                     memregion_sz(memregion_sz),
                                                     mtx(std::move(mtx)){}

            auto map(cuda_ptr_t cuda_ptr) noexcept -> std::expected<MapResource, exception_t>{

                stdx::lock_guard<Lock> lck_grd(*this->mtx);
                return this->internal_map(cuda_ptr);
            }

            auto evict_try(cuda_ptr_t cuda_ptr) noexcept -> std::expected<bool, exception_t>{

                stdx::lock_guard<Lock> lck_grd(*this->mtx);
                return this->internal_evict_try(cuda_ptr);
            }

            auto remap_try(MapResource resource, cuda_ptr_t new_ptr) noexcept -> std::expected<std::optional<MapResource>, exception_t>{

                if (!dg::memult::ptrcmp_equal(dg::memult::region(new_ptr, this->memregion_sz), resource.mapped_region)){
                    return std::optional<MapResource>(std::nullopt);
                }

                return std::optional<MapResource>(MapResource{resource.node, resource.mapped_region, dg::memult::region_offset(new_ptr, this->memregion_sz)});
            } 

            void unmap(MapResource map_resource) noexcept{

                stdx::lock_guard<Lock> lck_grd(*this->mtx);
                this->internal_unmap(map_resource);
            }
        
        private:

            void push_down_at(size_t idx) noexcept{

                size_t c = idx * 2 + 1;

                if (c >= this->priority_queue.size()){
                    return;
                }

                if (c + 1 < this->priority_queue.size() && ReferenceMemoryNodeCmpLess{}(*this->priority_queue[c + 1], *this->priority_queue[c])){
                    c += 1;
                }

                if (ReferenceMemoryNodeCmpLess{}(*this->priority_queue[c], *this->priority_queue[idx])){
                    std::swap(priority_queue[idx]->idx, priority_queue[c]->idx);
                    std::swap(priority_queue[idx], priority_queue[c]);
                    this->push_down_at(c);
                }
            }

            void push_up_at(size_t idx) noexcept{

                if (idx == 0u){
                    return;
                }

                size_t c = (idx - 1) >> 1;

                if (ReferenceMemoryNodeCmpLess{}(*priority_queue[idx], *priority_queue[c])){
                    std::swap(priority_queue[idx]->idx, priority_queue[c]->idx);
                    std::swap(priority_queue[idx], priority_queue[c]);
                    this->push_up_at(c);
                }
            }

            auto internal_map(cuda_ptr_t cuda_ptr) noexcept -> std::expected<MapResource, exception_t>{

                cuda_ptr_t region   = dg::memult::region(cuda_ptr, this->memregion_sz);
                size_t offset       = dg::memult::region_offset(cuda_ptr, this->memregion_sz); 
                auto map_ptr        = this->mapped_dict.find(region);

                if (map_ptr != this->mapped_dict.end()){
                    HeapNode * heap_node        = map_ptr->second;
                    heap_node->reference       += 1;
                    heap_node->last_modified    = stdx::utc_timestamp();
                    this->push_down_at(heap_node->idx);
                    MapResource rs              = MapResource{heap_node, *heap_node->cupinned_ptr, offset};

                    return rs; 
                }

                HeapNode * front = this->priority_queue.front().get();

                if (front->reference != 0u){
                    return std::unexpected(dg::network_exception::RESOURCE_EXHAUSTION);
                }

                //reference == 0u;

                if (front->cuda_ptr != dg::pointer_limits<cuda_ptr_t>::null_value()){
                    exception_t err = this->memory_loader->load(*this->tmp_memory_node, region, this->memregion_sz);

                    if (dg::network_exception::is_failed(err)){
                        return std::unexpected(err);
                    }

                    cuda_ptr_t removing_region = dg::memult::region(front->cuda_ptr, this->memregion_sz);
                    this->mapped_dict.erase(removing_region); 
                    this->memory_loader->unload(*front);
                    std::swap(*this->tmp_memory_node, static_cast<MemoryNode&>(*front));
                    this->mapped_dict.insert(std::make_pair(region, front)); //we are risking leaks - but we are doing noexcept allocations
                    front->reference    += 1;
                    front->last_modified = stdx::utc_timestamp(); 
                    this->push_down_at(0u);

                    return MapResource{front, *front->cupinned_ptr, offset};
                }

                //cuda_ptr == null
                exception_t err = this->memory_loader->load(*front, region, this->memregion_sz);

                if (dg::network_exception::is_failed(err)){
                    return std::unexpected(err);
                }

                this->mapped_dict.insert(std::make_pair(region, front));
                front->reference       += 1;
                front->last_modified    = stdx::utc_timestamp();
                this->push_down_at(0u);

                return MapResource{front, *front->cupinned_ptr, offset};
            }

            auto internal_evict_try(cuda_ptr_t cuda_ptr) noexcept -> std::expected<bool, exception_t>{

                cuda_ptr_t region   = dg::memult::region(cuda_ptr, this->memregion_sz);
                auto map_ptr        = this->mapped_dict.find(region);

                if (map_ptr == this->mapped_dict.end()){
                    return true;
                }

                HeapNode * heap_node = map_ptr->second; 

                if (heap_node->reference != 0u){
                    return false;
                }

                this->memory_loader->unload(*heap_node);
                this->mapped_dict.erase(map_ptr);
                heap_node->last_modified = stdx::utc_timestamp();
                this->push_up_at(heap_node->idx);

                return true;
            }

            void internal_unmap(MapResource map_resource) noexcept{

                map_resource.node->reference    -= 1;
                map_resource.node->last_modified = stdx::utc_timestamp();
                this->push_up_at(map_resource.node->idx);
            }
    };

    class RegionMapDistributor: public virtual MapDistributorInterface{

        private:

            dg::unordered_unstable_map<cuda_ptr_t, size_t> region_id_map;
            size_t memregion_sz; 

        public:

            RegionMapDistributor(dg::unrodered_unstable_map<cuda_ptr_t, size_t> region_id_map,
                                 size_t memregion_sz) noexcept: region_id_map(std::move(region_id_map)),
                                                                memregion_sz(memregion_sz){}

            auto id(cuda_ptr_t cuda_ptr) noexcept -> std::expected<size_t, exception_t>{

                cuda_ptr_t region = dg::memult::region(cuda_ptr, this->memregion_sz);
                auto map_ptr = this->region_id_map.find(region);

                if (map_ptr == this->region_id_map.end()){
                    return std::unexpected(dg::network_exception::INVALID_REGION);
                }

                return map_ptr->second;
            }
    };

    class ConcurrentMap: public virtual ConcurrentMapInterface{

        private:

            dg::vector<std::unique_ptr<MapInterface>> map_vec;
            std::unique_ptr<MapDistributorInterface> map_distributor;

        public:

            ConcurrentMap(dg::vector<std::unique_ptr<MapInterface>> map_vec,
                          std::unique_ptr<MapDistributorInterface> map_distributor) noexcept: map_vec(std::move(map_vec)),
                                                                                              map_distributor(std::move(map_distributor)){}

            auto map(cuda_ptr_t cuda_ptr) noexcept -> std::expected<ConcurrentMapResource, exception_t>{

                std::expected<size_t, exception_t> mapper_id = this->map_distributor->id(cuda_ptr);

                if (!mapper_id.has_value()){
                    return std::unexpected(mapper_id.error());
                }

                std::expected<MapResource, exception_t> map_resource = this->map_vec[mapper_id.value()]->map(cuda_ptr);

                if (!map_resource.has_value()){
                    return std::unexpected(map_resource.error());
                }

                return ConcurrentMapResource{map_resource.value(), mapper_id.value()};
            }

            auto evict_try(cuda_ptr_t cuda_ptr) noexcept -> std::expected<bool, exception_t>{

                std::expected<size_t, exception_t> mapper_id = this->map_distributor->id(cuda_ptr);

                if (!mapper_id.has_value()){
                    return std::unexpected(mapper_id.error());
                }

                return this->map_vec[mapper_id.value()]->evict_try(cuda_ptr);
            }

            auto remap_try(ConcurrentMapResource old_resource, cuda_ptr_t cuda_ptr) noexcept -> std::expected<std::optional<ConcurrentMapResource>, exception_t>{

                std::expected<size_t, exception_t> mapper_id = this->map_distributor->id(cuda_ptr);

                if (!mapper_id.has_value()){
                    return std::unexpected(mapper_id.error());
                }

                if (mapper_id.value() != old_resource.map_id){
                    return std::optional<ConcurrentMapResource>(std::nullopt);
                }

                std::expected<std::optional<MapResource>, exception_t> remap_resource = this->map_vec[mapper_id.value()]->remap_try(old_resource.resource, cuda_ptr);

                if (!remap_resource.has_value()){
                    return std::unexpected(remap_resource.error());
                }

                if (!remap_resource.value().has_value()){
                    return std::optional<ConcurrentMapResource>(std::nullopt);
                }

                return std::optional<ConcurrentMapResource>(ConcurrentMapResource{remap_resource.value().value(), mapper_id.value()});
            }

            void unmap(ConcurrentMapResource map_resource) noexcept{

                this->map_vec[map_resource.map_id]->unmap(map_resource.resource);
            }
    };
}

namespace dg::network_cudamap_impl1{

}

#endif