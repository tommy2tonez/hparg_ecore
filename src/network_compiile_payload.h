#ifndef __DG_CORE_COMPILE_PAYLOAD_H__
#define __DG_CORE_COMPILE_PAYLOAD_H__

#include <stdint.h>
#include <stdlib.h>
#include <vector>

namespace dg::network_core_compile_payload{

    static inline size_t LOCK_OPTION                                            = {}; 
    static inline size_t PAGE_SZ                                                = size_t{1} << 30;
    static inline size_t GRID_SZ                                                = size_t{1} << 25;
    static inline size_t TILE_SZ                                                = size_t{1} << 16;
    static inline size_t TRANSLATION_UNIT_SZ                                    = size_t{1} << 20; 
    static inline size_t TILE_8BIT_COUNT                                        = size_t{};
    static inline size_t TILE_16BIT_COUNT                                       = size_t{};
    static inline size_t UNORDERED_ACCUM_GROUP_SZ                               = size_t{};
    static inline size_t PAIR_ACCUM_GROUP_SZ                                    = size_t{}; 

    //TILE allocations are on virtual_addr_space 

    static inline std::vector<int> CUDA_COMPUTE_DEVICE                          = {};
    static inline std::vector<int> HOST_COMPUTE_DEVICE                          = {};
    static inline std::vector<int> HOST_COMPUTE_AFFINITY_OPTION                 = {};
    static inline std::vector<int> HOST_NETWORK_DEVICE                          = {};
    static inline std::vector<int> HOST_NETWORK_DEVICE_BOUNCE_BUFFER_SZ         = {};
    static inline std::vector<int> HOST_NETWORK_DEVICE_AFFINITY_OPTION          = {}; 
    static inline std::vector<int> HOST_SCAN_FWD_DEVICE                         = {};
    static inline std::vector<int> HOST_SCAN_FWD_FREQUENCY                      = {};
    static inline std::vector<int> HOST_SCAN_FWD_DEVICE_AFFINITY_OPTION         = {};
    static inline std::vector<int> HOST_SCAN_BWD_DEVICE                         = {};
    static inline std::vector<int> HOST_SCAN_BWD_FREQUENCY                      = {};
    static inline std::vector<int> HOST_SCAN_BWD_DROPOUT_RATE                   = {};
    static inline std::vector<int> HOST_SCAN_BWD_DEVICE_AFFINITY_OPTION         = {};

    static inline uintptr_t COMPUTATION_CLUSTER_VIRTUAL_ADDR_FIRST              = 0;
    static inline uintptr_t COMPUTATION_CLUSTER_VIRTUAL_ADDR_LAST               = 0;
    static inline size_t COMPUTATION_CLUSER_UNIT_COUNT                          = 0; 
    static inline uintptr_t HOST_VIRTUAL_ADDR_FIRST                             = 0;
    static inline std::vector<uint16_t> COMPUTATION_CLUSTER_PEER_ADDRESS        = {}; 
    static inline bool IS_NAT_PUNCH_REQUIRED                                    = {};

    static inline void *** HOST_SCAN_FWD_TLB                                    = {};
    static inline void *** HOST_SCAN_BWD_TLB                                    = {};

    static inline std::vector<std::vector<std::pair<void *, void *>>> CUDA_TLB  = {};
    static inline std::vector<std::vector<std::pair<void *, void *>>> RAM_TLB   = {};

    static inline bool HAS_CPU_UU_8_8                                           = true;
    static inline bool HAS_CPU_UU_8_16                                          = true;
    static inline bool HAS_CPU_UU_16_8                                          = true;
    static inline bool HAS_CPU_UU_16_16                                         = true;

    static inline bool HAS_CUDA_UU_8_8                                          = true;
    static inline bool HAS_CUDA_UU_8_16                                         = true;
    static inline bool HAS_CUDA_UU_16_8                                         = true;
    static inline bool HAS_CUDA_UU_16_16                                        = true;

    static inline bool HAS_CUDA_UF_8_8                                          = true;
    static inline bool HAS_CUDA_UF_8_16                                         = true;
    static inline bool HAS_CUDA_UF_16_8                                         = true;
    static inline bool HAS_CUDA_UF_16_16                                        = true;

    static inline bool HAS_CUDA_FU_8_8                                          = true;
    static inline bool HAS_CUDA_FU_8_16                                         = true;
    static inline bool HAS_CUDA_FU_16_8                                         = true;
    static inline bool HAS_CUDA_FU_16_16                                        = true;

    static inline bool HAS_CUDA_FF_8_8                                          = true;
    static inline bool HAS_CUDA_FF_8_16                                         = true;
    static inline bool HAS_CUDA_FF_16_8                                         = true;
    static inline bool HAS_CUDA_FF_16_16                                        = true;
}

#endif 