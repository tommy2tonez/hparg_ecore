#ifndef __NETWORK_TILE_TLB_H__
#define __NETWORK_TILE_TLB_H__

namespace dg::network_tile_tlb{

    class TileTLBInterface{

        public:

            virtual ~TileTLBInterface() noexcept = default;
            virtual auto add(uma_ptr_t key, uma_ptr_t value) noexcept -> exception_t = 0;
            virtual auto translate(uma_ptr_t key) noexcept -> std::optional<uma_ptr_t> = 0;
            virtual void erase(uma_ptr_t key) noexcept = 0;
    };

}

#endif