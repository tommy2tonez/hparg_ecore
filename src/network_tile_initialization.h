#ifndef __DG_NETWORK_TILE_INITIALIZATION_H__
#define __DG_NETWORK_TILE_INITIALIZATION_H__

namespace dg::network_tile_initialization{

    //all init by addr
    //leaf + crit logit values are payload dumpable

    void init_leaf(){

    }

    void init_mono(){

    } 

    void init_pair(){ //sub-logic of pacm - should remove

    }

    void init_uacm(){

    }

    void init_pacm(){

    }

    void init_crit(){ //store expected value - propagate gradients as soon as crit on child fwd pong packet

    }

    void init_msgr(){ //clone operation - redirect fwd or bwd operation (redirect -> enduser by protocol injection) - no acked

    }
} 

#endif