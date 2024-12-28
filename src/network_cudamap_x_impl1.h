#ifndef __NETWORK_CUDA_MAP_IMPL1_H__
#define __NETWORK_CUDA_MAP_IMPL1_H__

//we are mapping cuda_ptr_t -> cuda_pinned_ptr_t
//we are doing eviction like we are doing with fsys_ptr_t
//this is mainly for transferring data from host -> cuda and cuda -> host - without the cost of cudaMemcpy - by seriallly copy the data from/to an intermediate fast platform
//alright - 2024 and this is still relevant - we'll operate mostly on cuda_pinned_ptr_t to transfer to and from cuda asynchronous devices

namespace dg::network_cudamap_impl1::model{

    struct MemoryNode{
        std::shared_ptr<cuda_pinned_ptr_t> cupinned_ptr;
        cuda_ptr_t cuda_ptr;
        size_t reference;
        std::chrono::nanoseconds last_modified;
        size_t mem_sz;
    };

    struct HeapNode: MemoryNode{
        size_t idx;
    };

    struct MapResource{
        HeapNode * node;
        size_t off;

        auto ptr() const noexcept -> cuda_pinned_ptr_t{

            return dg::memult::next(*this->node->cupinned_ptr, this->off);
        }
    };

    struct ConcurrentMapResource{
        MapResource resource;
        size_t map_id;

        auto ptr() const noexcept -> cuda_pinned_ptr_t{

            return resource.ptr();
        }
    };
}

namespace dg::network_cudamap_impl1::interface{

    //there is a virtue in rewriting these - first is interface polymorphism - we aren't having that - so compiler can actually treat this as a real object rather than a virtual object
    //second is maintainability - we wrote things once - leave things be

    using namespace network_cudamap_impl1::model;

    struct MemoryLoaderInterface{
        virtual ~MemoryLoaderInterface() noexcept = default;
        virtual auto load(MemoryNode&, cuda_ptr_t, size_t) noexcept -> exception_t = 0; //its kinda off without the range here - I admit
        virtual void unload(MemoryNode&) noexcept = 0; //we offload the responsibility unload noexceptability here - there are tricks we aren't applying yet
    };

    struct MapInterface{
        virtual ~MapInterface() noexcept = default;
        virtual auto map(cuda_ptr_t) noexcept -> std::expected<MapResource, exception_t> = 0;
        virtual auto remap_try(MapResource, cuda_ptr_t) noexcept -> std::expected<std::optional<MapResource>, exception_t> = 0;
        virtual void unmap(MapResource) noexcept = 0;
        virtual auto memregion_size() const noexcept -> size_t = 0;
    };

    struct ConcurrentMapInterface{
        virtual ~ConcurrentMapInterface() noexcept = default;
        virtual auto map(cuda_ptr_t) noexcept -> std::expected<ConcurrentMapResource, exception_t> = 0;
        virtual auto remap_try(ConcurrentMapResource, cuda_ptr_t) noexcept -> std::expected<std::optional<ConcurrentMapResource>, exception_t> = 0;
        virtual void unmap(ConcurrentMapResource) noexcept = 0;
        virtual auto memregion_size() const noexcept -> size_t = 0;
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

            auto load(MemoryNode& root, cuda_ptr_t region) noexcept -> exception_t{

            }

            void unload(MemoryNode& root) noexcept{

            }
    };

    class Map: public virtual MapInterface{

        private:

            std::unique_ptr<MemoryLoaderInterface> memory_loader;
            dg::vector<std::unique_ptr<HeapNode>> priority_queue;
            dg::unordered_unstable_map<cuda_ptr_t, HeapNode *> mapped_dict;
            size_t memregion_sz;
            std::unique_ptr<Lock> mtx;

        public:

            Map(std::unique_ptr<MemoryLoaderInterface> memory_loader,
                dg::vector<std::unique_ptr<HeapNode>> priority_queue,
                dg::unordered_unstable_map<cuda_ptr_t, HeapNode *> mapped_dict,
                size_t memregion_sz,
                std::unique_ptr<Lock> mtx) noexcept: memory_loader(std::move(memory_loader)),
                                                     priority_queue(std::move(priority_queue)),
                                                     mapped_dict(std::move(mapped_dict)),
                                                     memregion_sz(memregion_sz),
                                                     mtx(std::move(mtx)){}

            auto map(cuda_ptr_t cuda_ptr) noexcept -> std::expected<MapResource, exception_t>{

                stdx::lock_guard<Lock> lck_grd(*this->mtx);
                return this->internal_map(cuda_ptr);
            }

            void unmap(MapResource map_resource) noexcept{

                stdx::lock_guard<Lock> lck_grd(*this->mtx);
                this->internal_unmap(map_resource);
            }

            auto memregion_size() const noexcept -> size_t{

                return this->memregion_sz;
            }
    };

    //it's kinda weird - I admit - this should be simply dg::unordered_unstable_map inside ConcurrentMap
    //I'll fix this later
    class MapDistributor: public virtual MapDistributorInterface{

        private:

            dg::unordered_unstable_map<cuda_ptr_t, size_t> region_id_map;
            size_t memregion_sz; 

        public:

            MapDistributor(dg::unrodered_unstable_map<cuda_ptr_t, size_t> region_id_map,
                           size_t memregion_sz) noexcept: region_id_map(std::move(region_id_map)),
                                                          memregion_sz(memregion_sz){}

            auto id(cuda_ptr_t cuda_ptr) noexcept -> std::expected<size_t, exception_t>{

            }
    };

    class ConcurrentMap: public virtual ConcurrentMapInterface{

        private:

            dg::vector<std::unique_ptr<MapInterface>> map_vec;
            std::unique_ptr<MapDistributorInterface> map_distributor;
            size_t memregion_sz;  

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
            
            auto memregion_size() const noexcept -> size_t{

                return this->memregion_sz;
            }
    };
}

namespace dg::network_cudamap_impl1{

}

#endif