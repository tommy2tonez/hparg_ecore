#ifndef __DG_NETWORK_OUTBOUND_DROPBOX_H__
#define __DG_NETWORK_OUTBOUND_DROPBOX_H__

#include <stdint.h>
#include <stdlib.h>

namespace dg::network_outbound_dropbox{

    //1hr
    //seems like it makes sense for outbound_dropbox to have (uint8_t, const char *, size_t) interface - for extensibility 
    //kernel_dispatcher would coerce this -> certain serialization format which is deserializable by extmemcommit_kernel 
    //need to actually semantically differentiate passive consumer/ pulling_consumer

    inline void submit(uint8_t, const char *, size_t) noexcept{

    }
} 

#endif