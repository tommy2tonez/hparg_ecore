#define DEBUG_MODE_FLAG true

#include "stdx.h"
#include "network_exception.h"
#include "network_postgres_db.h"
#include <string>
#include <string_view>
#include <optional>
#include "network_raii_x.h"
#include "network_concurrency.h"
#include "network_randomizer.h"
#include "network_fileio_linux.h"
#include "network_fileio.h"
// #include "network_kernelmap_x_impl1.h"
#include "network_allocation.h"
#include "network_std_container.h"
// #include "network_auth.h"

//hopefully the project is within 1 << 20 line of code
//1/2 of project is about heavy cuda optimization
//1/4 of project is about locality and affinity (reimplementation of kernel scheduler if need to)
//1/4 of project is about efficient transportation (socket + fileio + friends)

//once all these compiled nicely then a parser to split header + implementation is required 

int main(){

    // using deregister_t  = void (*)(size_t) noexcept;
    // using pruned_t      = dg::network_type_traits_x::base_type_t<deregister_t>;

    // static_assert(std::is_same_v<deregister_t, pruned_t>); 

}