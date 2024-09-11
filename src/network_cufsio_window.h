#ifndef __NETWORK_CUFSIO_WINDOW_H__
#define __NETWORK_CUFSIO_WINDOW_H__

namespace dg::network_cufsio_window{
    
    auto dg_read_binary_direct(const char * fp, cuda_ptr_t dst, size_t dst_cap) noexcept -> exception_t{

    }

    void dg_read_binary_direct_nothrow(const char * fp, cuda_ptr_t dst, size_t dst_cap) noexcept{

    }

    auto dg_write_binary_direct(const char * fp, cuda_ptr_t src, size_t src_sz) noexcept -> exception_t{

    }

    void dg_write_binary_direct_nothrow(const char * fp, cuda_ptr_t src, size_t src_sz) noexcept{
        
    }
} 

#endif