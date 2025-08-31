#ifndef __DG_NETWORK_CHRONO_H__
#define __DG_NETWORK_CHRONO_H__

namespace dg{

    template <class Clock>
    class ticking_clock{

        public:

            ticking_clock() = default;
            ticking_clock(size_t){}

            auto get() noexcept -> decltype(auto){

                return Clock::now();
            }
    };
}
#endif