#ifndef __NETWORK_KERNEL_PRODUCER_H__
#define __NETWORK_KERNEL_PRODUCER_H__

namespace dg::kernel_producer{

    //a component that spins on kernel
    //this component read_header and fwd packet -> acked or non_acked handler    
    //if receipt_required - call acked_handler
    //if receipt_not_required - call non-acked handler
    //a component that spins on acked and non_acked handler
    //2 days of work

} 

#endif