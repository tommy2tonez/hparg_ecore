#ifndef __NETWORK_PAYLOAD_INJECTION_H__
#define __NETWORK_PAYLOAD_INJECTION_H__

namespace dg::network_payload_generator{

    auto make_init_leaf_payload(){ //logit init

    }

    auto make_init_mono_payload(){ //addr init

    }

    auto make_init_pair_payload(){

    } 

    auto make_init_uacm_payload(){

    } 

    auto make_init_pacm_payload(){

    }

    auto make_init_crit_payload(){

    }

    auto make_init_virtual_payload(){
        
    } 

    auto make_memevent_crit_ping_payload(){ //user invoke

    } 

    auto make_memevent_init_ping_payload(){ //program invoke

    }

    auto make_memevent_init_pong_payload(){ //program invoke

    }

    auto make_memevent_backprop_ping_payload(){ //program invoke - ping as soon as gradient is updated

    }

    auto make_memevent_backprop_pong_payload(){ //program invoke - pong as soon as scanner hits the ping

    }

    auto make_memevent_backprop_injection_payload(){ //program invoke - send injection as soon as the pong request received

    }
}

#endif