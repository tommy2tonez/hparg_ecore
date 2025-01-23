#ifndef __DG_NETWORK_TILE_FOREIGN_INJECTION_H__
#define __DG_NETWORK_TILE_FOREIGN_INJECTION_H__

#include <atomic>
#include <memory>
#include <stdint.h>
#include <stddef.h>
#include "stdx.h"

namespace dg::network_tile_foreign_injection{

    class TileAllocatorInterface{

        public:

            virtual ~TileAllocatorInterface() noexcept = default;
            virtual auto allocate(polymorphic_tile_kind_t) noexcept -> std::expected<uma_ptr_t, exception_t> = 0;
            virtual void deallocate(uma_ptr_t) noexcept = 0;
    };

    //this is weird - I'll fix the convention
    class TileForeignInjectionInterface{

        public:

            virtual ~TileForeignInjectionInterface() noexcept = default;
            virtual auto inject(polymorphic_tile_kind_t, const dg::string&) noexcept -> std::expected<uma_ptr_t, exception_t> = 0; 
            virtual void deallocate(uma_ptr_t) noexcept = 0;
    };

    class TileShadowControllerInterface{

        public:

            virtual ~TileShadowControllerInterface() noexcept = default;
            virtual auto shadow(uma_ptr_t, polymorphic_tile_kind_t, const dg::string&) noexcept -> exception_t = 0; 
            virtual auto get_alias(uma_ptr_t) noexcept -> std::expected<uma_ptr_t, exception_t> = 0;
            virtual auto unshadow(uma_ptr_t) noexcept = 0;
    };

    class TileRoomControllerInterface{

        public:

            virtual ~TileRoomControllerInterface() noexcept = default;
            virtual auto enter_room(uma_ptr_t) noexcept -> std::expected<bool, exception_t> = 0;
            virtual auto get_kick_candidate() noexcept -> std::expected<uma_ptr_t, exception_t> = 0;
            virtual auto exit_room(uma_ptr_t) noexcept = 0;
    };

    class TileShadowControllerXInterface{

        public:

            virtual ~TileShadowControllerXInterface() noexcept = default;
            virtual auto shadow(uma_ptr_t, polymorphic_tile_kind_t, const dg::string&) noexcept -> exception_t = 0;
            virtual auto get_alias(uma_ptr_t) noexcept -> std::expected<uma_ptr_t, exception_t> = 0;
    };

    class TileAllocator: public virtual TileAllocatorInterface{

        private:

            dg::unordered_unstable_map<polymorphic_tile_kind_t, dg::vector<uma_ptr_t>> allocation_dict;
        
        public:

            TileAllocator(dg::unordered_unstable_map<polymorphic_tile_kind_t, dg::vector<uma_ptr_t>> allocation_dict) noexcept: allocation_dict(std::move(allocation_dict)){}

            auto allocate(polymorphic_tile_kind_t tile_kind) noexcept -> std::expected<uma_ptr_t, exception_t>{

                auto map_ptr = this->allocation_dict.find(tile_kind);

                if (map_ptr == this->allocation_dict.end()){
                    return std::unexpected(dg::network_exception::BAD_ENTRY);
                }

                if (map_ptr->second.empty()){
                    return std::unexpected(dg::network_exception::RESOURCE_EXHAUSTION);
                }

                uma_ptr_t rs = map_ptr->second.back();
                map_ptr->second.pop_back();

                return rs;
            }

            void deallocate(uma_ptr_t addr) noexcept{

                //alright this is gonna be a spaghetti if not done correctly - whatever
                polymorphic_tile_kind_t tile_kind = dg::network_tile_member_getsetter::get_tile_kind_nothrow(addr);
                allocation_dict.find(tile_kind)->second.push_back(addr);
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

    class TileShadowController: public virtual TileShadowControllerInterface{

        private:

            std::unique_ptr<TileForeignInjectionInterface> foreign_injector;
            dg::unordered_unstable_map<uma_ptr_t, uma_ptr_t> alias_dict;
            size_t alias_dict_capacity; 

        public:

            TileShawdowController(std::unique_ptr<TileForeignInjectionInterface> foreign_injector,
                                  dg::unordered_unstable_map<uma_ptr_t, uma_ptr_t> alias_dict,
                                  size_t alias_dict_capacity) noexcept: foreign_injector(std::move(foreign_injector)),
                                                                        alias_dict(std::move(alias_dict)),
                                                                        alias_dict_capacity(alias_dict_capacity){}

            auto shadow(uma_ptr_t shadowed_addr, polymorphic_tile_kind_t kind, const dg::string& injection_data) noexcept -> exception_t{

                if (this->alias_dict.size() == this->alias_dict_capacity){
                    return dg::network_exception::RESOURCE_EXHAUSTION;
                }

                if (this->alias_dict.contains(shadowed_addr)){
                    return dg::network_exception::BAD_ENTRY;
                } 

                std::expected<uma_ptr_t, exception_t> shadowing_addr = this->foreign_injector->inject(kind, injection_data);

                if (!shadowing_addr.has_value()){
                    return shadowing_addr.error();
                }

                this->alias_dict.insert(std::make_pair(shadowed_addr, shadowing_addr.value()));
                return dg::network_exception::SUCCESS;
            }

            auto get_alias(uma_ptr_t shadowed_addr) noexcept -> std::expected<uma_ptr_t, exception_t>{

                auto map_ptr = this->alias_dict.find(shadowed_addr);

                if (map_ptr == this->alias_dict.end()){
                    return std::unexpected(dg::network_exception::BAD_ENTRY);
                }

                return map_ptr->second;
            }

            void unshadow(uma_ptr_t shadowed_addr) noexcept{

                auto map_ptr = this->alias_dict.find(shadowed_addr);

                if constexpr(DEBUG_MODE_FLAG){
                    if (map_ptr == this->alias_dict.end()){
                        dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                        std::abort();
                    }
                }

                this->foreign_injector->deallocate(map_ptr->second);
                this->alias_dict.erase(map_ptr);
            }
    };

    class TileShadowControllerX: public virtual TileShadowControllerXInterface{

        private:

            std::unique_ptr<TileShadowControllerInterface> base_tile_shadow_controller;
            std::unique_ptr<TileRoomControllerInterface> room_controller;
            std::unique_ptr<std::mutex> mtx;

        public:

            TileShadowControllerX(std::unique_ptr<TileShadowControllerInterface> base_tile_shadow_controller,
                                  std::unique_ptr<TileRoomControllerInterface> room_controller,
                                  std::unique_ptr<std::mutex> mtx): base_tile_shadow_controller(std::move(base_tile_shadow_controller)),
                                                                    room_controller(std::move(room_controller)),
                                                                    mtx(std::move(mtx)){}

            auto shadow(uma_ptr_t shadowed_addr, polymorphic_tile_kind_t tile_kind, const dg::string& init_payload) noexcept -> exception_t{

                stdx::lock_guard<std::mutex> lck_grd(*this->mtx);
                std::expected<bool, exception_t> room_status = this->room_controller->enter_room(shadowed_addr);

                if (!room_status.has_value()){
                    return room_status.error();
                }

                if (!room_status.value()){
                    std::expected<uma_ptr_t, exception_t> cand = this->room_controller->get_kick_candidate();

                    if (!cand.has_value()){
                        return cand.error();
                    }

                    this->base_tile_shadow_controller->unshadow(cand.value());
                    this->room_controller->exit_room(cand.value());
                    dg::network_exception_handler::assert(dg::network_exception_handler::nothrow_log(this->room_controller->enter_room(shadowed_addr)));
                }

                exception_t shadowing_status = this->base_tile_shadow_controller->shadow(shadowed_addr, tile_kind, init_payload);

                if (dg::network_exception::is_failed(shadowing_status)){
                    this->room_controller->exit_room(shadowed_addr);
                    return shadowing_status;
                }

                return dg::network_exception::SUCCESS;;
            }

            auto get_alias(uma_ptr_t shadowed_addr) noexcept -> std::expected<uma_ptr_t, exception_t>{

                stdx::lock_guard<std::mutex> lck_grd(*this->mtx);
                return this->base_tile_shadow_controller->get_alias(shadowed_addr);
            }
    };

    class ConcurrentTileShadowControllerX: public virtual TileShadowControllerXInterface{

        private:

            std::vector<std::unique_ptr<TileShadowControllerXInterface>> base_controller_vec;

        public:

            ConcurrentTileShadowControllerX(std::vector<std::unique_ptr<TileShadowControllerXInterface>> base_controller_vec) noexcept: base_controller_vec(std::move(base_controller_vec)){}

            auto shadow(uma_ptr_t shadowed_addr, polymorphic_tile_kind_t tile_kind, const dg::string& init_payload) noexcept -> exception_t{

                size_t idx = dg::hasher::murmur_hash(dg::pointer_cast<dg::pointer_info<uma_ptr_t>::max_unsigned_t>(shadowed_addr)) & (this->base_controller_vec.size() - 1u);
                return this->base_controller_vec[idx]->shadow(shadowed_addr, tile_kind, init_payload);
            }

            auto get_alias(uma_ptr_t shadowed_addr) noexcept -> std::expected<uma_ptr_t, exception_t>{

                size_t idx = dg::hasher::murmur_hash(dg::pointer_cast<dg::pointer_info<uma_ptr_t>::max_unsigned_t>(shadowed_addr)) & (this->base_controller_vec.size() - 1u);
                return this->base_controller_vec[idx]->get_alias(shadowed_addr);
            }
    };

} 

#endif