def permute_zip(arr: list[list[object]]) -> list[list]:

    if len(arr) == 0:
        return []
    
    if len(arr) == 1:
        return [[e] for e in arr[0]]
    
    successor: list[list]   = permute_zip(arr[1:])
    cur_list: list[object]  = arr[0]
    rs: list[list]          = []

    for tup in successor:
        for obj in cur_list:
            rs += [tup + [obj]]
    
    return rs

def fill_empty(s: str, sz: int) -> str:

    return s + (" " * (sz - len(s)))

def combinatorial_zip(arr: list[list[object]]) -> list[tuple]:
    
    if (len(arr) == 0):
        return []
    
    if (len(arr) == 1):
        return [tuple([obj]) for obj in arr[0]]
    
    successor_arr: list[tuple] = combinatorial_zip(arr[1:])
    rs: list[tuple] = []

    for arr_e in arr[0]:
        for successor_e in successor_arr:
            rs += [tuple([arr_e] + list(successor_e))]
    
    return rs

def launder_script(var_name: str, tile_type: str):
    
    if tile_type == "u8":
        return "stdx::launder_pointer<%s>(%s)" % ("dg::network_tile_metadata::host_u8_t", var_name)

    if tile_type == "u16":
        return "stdx::launder_pointer<%s>(%s)" % ("dg::network_tile_metadata::host_u16_t", var_name)

    if tile_type == "f8":
        return "stdx::launder_pointer<%s>(%s)" % ("dg::network_tile_metadata::host_f8_t", var_name)

    if tile_type == "f16":
        return "stdx::launder_pointer<%s>(%s)" % ("dg::network_tile_metadata::host_f16_t", var_name)

    raise Exception() 

def main():

    arr = combinatorial_zip([["u8",  "u16", "f16"], ["u8",  "u16", "f16"], ["u8", "u16", "f16"], ["act"]])
    dispatch_arr = []

    for  left_t, right_t, casting_t, func_name in arr:
        lhs             = "rs[make_dispatch_code(%s%s%s_%s_%s_%s, ops_%s)]" % (left_t[0], right_t[0], casting_t[0], left_t[1:], right_t[1:], casting_t[1:], func_name)
        rhs             = "[](void * __restrict__ dst, const void * __restrict__ dst_logit, const void * __restrict__ src_grad, const void * __restrict__ other_logit) noexcept{dg::network_tileops_host_static::bwd_pair_rhs_ops_%s_%s_%s_%s_%s::%s(%s, %s, %s, %s);}" % (right_t, right_t, right_t, left_t, casting_t, func_name, launder_script("dst", right_t), launder_script("dst_logit", right_t), launder_script("src_grad", left_t), launder_script("other_logit", right_t))
        init_script     = "%s\t\t= %s" % (fill_empty(lhs, 70), rhs)
        dispatch_arr    += [init_script]

    print(len(dispatch_arr))
    print(";\n".join(dispatch_arr))

    # alias_dict      = {"tile_u8_t": "u8",
    #                   "tile_u16_t": "u16",
    #                   "tile_u32_t": "u32",
    #                   "tile_f8_t": "f8",
    #                   "tile_f16_t": "f16"}
    
    # src_logit_arr   = ["tile_u8_t", "tile_u16_t", "tile_f16_t"]
    # src_grad_arr    = ["tile_u8_t", "tile_u16_t", "tile_f16_t"]
    # dst_logit_arr   = ["tile_u8_t", "tile_u16_t", "tile_f16_t"]
    # dst_grad_arr    = ["tile_u8_t", "tile_u16_t", "tile_f16_t"]
    # casting_ops_arr = ["tile_u8_t", "tile_u16_t", "tile_f16_t"]

    # fwd_mono_dict       = dict()
    # fwd_uacm_dict       = dict()
    # fwd_pacm_dict       = dict()
    # fwd_pair_dict       = dict()

    # bwd_mono_dict       = dict()
    # bwd_uacm_dict       = dict()
    # bwd_pair_lhs_dict   = dict()
    # bwd_pair_rhs_dict   = dict()
    # zip_arr             = permute_zip([src_logit_arr, src_grad_arr, src_logit_arr, src_grad_arr, dst_logit_arr, dst_grad_arr, casting_ops_arr])

    # for (src_logit, src_grad, other_src_logit, other_src_grad, dst_logit, dst_grad, casting_ops) in zip_arr:
    #     fwd_mono_key    = "fwd_mono_ops_%s_%s_%s" % (alias_dict[dst_logit], alias_dict[src_logit], alias_dict[casting_ops])
    #     fwd_mono_value  = "templated_ops::fwd_mono_restrict_aligned_ops<%s, %s, %s, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>" % (dst_logit, src_logit, casting_ops)
        
    #     fwd_uacm_key    = "fwd_uacm_ops_%s_%s_%s" % (alias_dict[dst_logit], alias_dict[src_logit], alias_dict[casting_ops])
    #     fwd_uacm_value  = "templated_ops::fwd_uacm_restrict_aligned_ops<%s, %s, %s, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>" % (dst_logit, src_logit, casting_ops)

    #     fwd_pair_key    = "fwd_pair_ops_%s_%s_%s_%s" % (alias_dict[dst_logit], alias_dict[src_logit], alias_dict[other_src_logit], alias_dict[casting_ops])
    #     fwd_pair_value  = "templated_ops::fwd_pair_restrict_aligned_ops<%s, %s, %s, %s, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>" % (dst_logit, src_logit, other_src_logit, casting_ops)

    #     fwd_pacm_key    = "fwd_pacm_ops_%s_%s_%s_%s" % (alias_dict[dst_logit], alias_dict[src_logit], alias_dict[other_src_logit], alias_dict[casting_ops])
    #     fwd_pacm_value  = "templated_ops::fwd_pacm_restrict_aligned_ops<%s, %s, %s, %s, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>" % (dst_logit, src_logit, other_src_logit, casting_ops)

    #     bwd_mono_key    = "bwd_mono_ops_%s_%s_%s_%s" % (alias_dict[dst_logit], alias_dict[dst_grad], alias_dict[src_grad], alias_dict[casting_ops])
    #     bwd_mono_value  = "templated_ops::bwd_mono_restrict_aligned_ops<%s, %s, %s, %s, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>" % (dst_logit, dst_grad, src_grad, casting_ops)

    #     bwd_uacm_key    = "bwd_uacm_ops_%s_%s_%s_%s_%s" % (alias_dict[dst_logit], alias_dict[dst_grad], alias_dict[src_logit], alias_dict[src_grad], alias_dict[casting_ops])
    #     bwd_uacm_value  = "templated_ops::bwd_uacm_restrict_aligned_ops<%s, %s, %s, %s, %s, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>" % (dst_logit, dst_grad, src_logit, src_grad, casting_ops)

    #     bwd_plhs_key    = "bwd_pair_lhs_ops_%s_%s_%s_%s_%s" % (alias_dict[dst_logit], alias_dict[dst_grad], alias_dict[other_src_logit], alias_dict[src_grad], alias_dict[casting_ops])
    #     bwd_plhs_value  = "templated_ops::bwd_pair_lhs_restrict_aligned_ops<%s, %s, %s, %s, %s, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>" % (dst_logit, dst_grad, other_src_logit, src_grad, casting_ops)

    #     bwd_prhs_key    = "bwd_pair_rhs_ops_%s_%s_%s_%s_%s" % (alias_dict[dst_logit], alias_dict[dst_grad], alias_dict[other_src_logit], alias_dict[src_grad], alias_dict[casting_ops])
    #     bwd_prhs_value  = "templated_ops::bwd_pair_rhs_restrict_aligned_ops<%s, %s, %s, %s, %s, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>" % (dst_logit, dst_grad, other_src_logit, src_grad, casting_ops)

    #     fwd_mono_dict[fwd_mono_key]     = fwd_mono_value
    #     fwd_uacm_dict[fwd_uacm_key]     = fwd_uacm_value
    #     fwd_pacm_dict[fwd_pacm_key]     = fwd_pacm_value
    #     fwd_pair_dict[fwd_pair_key]     = fwd_pair_value 
    #     bwd_mono_dict[bwd_mono_key]     = bwd_mono_value
    #     bwd_uacm_dict[bwd_uacm_key]     = bwd_uacm_value
    #     bwd_pair_lhs_dict[bwd_plhs_key] = bwd_plhs_value
    #     bwd_pair_rhs_dict[bwd_prhs_key] = bwd_prhs_value

    # l = (fwd_mono_dict | fwd_uacm_dict | fwd_pacm_dict | fwd_pair_dict | bwd_mono_dict | bwd_uacm_dict | bwd_pair_lhs_dict | bwd_pair_rhs_dict).items()

    # out_script  = "\n".join(["using %s\t\t= %s;" % (fill_empty(k, 50), v) for (k, v) in l])

    # with open("output.txt", "w") as f_out:
    #     f_out.write(out_script)

main()
