#ifndef __DG_NETWORK_TILE_FOREIGN_INJECTION_H__
#define __DG_NETWORK_TILE_FOREIGN_INJECTION_H__

#include <atomic>

namespace dg::network_tile_foreign_injection{

    class TileAllocatorInterface{

        public:
            
            virtual ~TileAllocatorInterface() noexcept = default;
            virtual auto allocate(polymorphic_tile_kind_t) noexcept -> std::expected<uma_ptr_t, exception_t> = 0;
            virtual void deallocate(uma_ptr_t) noexcept = 0;
    };

    class TileForeignInjectionInterface{

        public:

            virtual ~TileForeignInjectionInterface() noexcept = default;
            virtual auto inject(polymorphic_tile_kind_t, const dg::string&) noexcept -> std::expected<uma_ptr_t, exception_t> = 0; 
            virtual void deallocate(uma_ptr_t) noexcept = 0;
    };

    struct CircularTileAllocatorEntry{
        stdx::hdi_container<dg::vector<uma_ptr_t>> allocation_vec;
        stdx::hdi_container<std::atomic<size_t>> cursor_dict;
    };

    class CircularTileAllocator: public virtual TileAllocatorInterface{

        private:

            dg::unordered_unstable_map<polymorphic_tile_kind_t, CircularTileAllocatorEntry> allocation_entry_dict;

        public:

            CircularTileAllocator(dg::unordered_unstable_map<polymorphic_tile_kind_t, CircularTileAllocatorEntry> allocation_entry_dict) noexcept: allocation_entry_dict(std::move(allocation_entry_dict)){}

            auto allocate(polymorphic_tile_kind_t tile_kind) noexcept -> std::expected<uma_ptr_t, exception_t>{

                auto map_ptr = this->allocation_entry_dict.find(tile_kind);

                if (map_ptr == this->allocation_entry_dict.end()){
                    return std::unexpected(dg::network_exception::INVALID_TILE_KIND);
                }

                size_t next_idx = map_ptr->second.cursor_dict.value.fetch_add(1u, std::memory_order_relaxed); //this is relaxed ordering - we are allocating tile addreses - we aren't responsible for the tile content at this level - it's memlock responsibility - we dont want to thrash core here

                if constexpr(DEBUG_MODE_FLAG){
                    if (!stdx::is_pow2(map_ptr->second.allocation_vec.value.size())){
                        dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                        std::abort();
                    }
                }

                size_t idx = next_idx & (map_ptr->second.allocation_vec.value.size() - 1u);  
                return map_ptr->second.allocation_vec.value[idx];
            }

            void deallocate(uma_ptr_t ptr) noexcept{

                (void) ptr; //we aren't abiding to the deallocation laws - this is bad - user might expect this to be a serialized process
            }
    };

    class TileForeignInjection: public virtual TileForeignInjectionInterface{

        private:

            std::unique_ptr<TileAllocatorInterface> allocator;
        
        public:

            TileForeignInjection(std::unique_ptr<TileAllocatorInterface> allocator) noexcept: allocator(std::move(allocator)){}

            auto inject(polymorphic_tile_kind_t tile_kind, const dg::string& serialized_tile) -> std::expected<uma_ptr_t, exception_t>{

                std::expected<uma_ptr_t, exception_t> addr = this->allocator->allocate(tile_kind);

                if (!addr.has_value()){
                    return std::unexpected(addr.error());
                }

                PolymorphicTile polymorphic_tile{};
                exception_t err = dg::network_exception::to_cstyle_function(dg::network_compact_serializer::integrity_deserialize_into<PolymorphicTile>)(polymorphic_tile, serialized_tile.data(), serialized_tile.size(), this->serialization_secret); //we have to use serialization secret to guard 

                if (dg::network_exception::is_failed(err)){
                    this->allocator->deallocate(addr.value());
                    return std::unexpected(err);
                }

                switch (polymorphic_tile.tile_kind){

                }
            }

            void deallocate(uma_ptr_t ptr) noexcept{

                // dg::network_tile_initialization::orphan(ptr);
                this->allocator->deallocate(ptr);
                //
            }
    };

} 

#endif