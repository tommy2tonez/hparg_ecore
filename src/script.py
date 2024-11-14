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

def main():

    alias_dict      = {"tile_u8_t": "u8",
                      "tile_u16_t": "u16",
                      "tile_u32_t": "u32",
                      "tile_f8_t": "f8",
                      "tile_f16_t": "f16"}
    
    src_logit_arr   = ["tile_u8_t", "tile_u16_t", "tile_f16_t"]
    src_grad_arr    = ["tile_u8_t", "tile_u16_t", "tile_f16_t"]
    dst_logit_arr   = ["tile_u8_t", "tile_u16_t", "tile_f16_t"]
    dst_grad_arr    = ["tile_u8_t", "tile_u16_t", "tile_f16_t"]
    casting_ops_arr = ["tile_u8_t", "tile_u16_t", "tile_f16_t"]

    fwd_mono_dict       = dict()
    fwd_uacm_dict       = dict()
    fwd_pacm_dict       = dict()
    fwd_pair_dict       = dict()

    bwd_mono_dict       = dict()
    bwd_uacm_dict       = dict()
    bwd_pair_lhs_dict   = dict()
    bwd_pair_rhs_dict   = dict()
    zip_arr             = permute_zip([src_logit_arr, src_grad_arr, src_logit_arr, src_grad_arr, dst_logit_arr, dst_grad_arr, casting_ops_arr])

    for (src_logit, src_grad, other_src_logit, other_src_grad, dst_logit, dst_grad, casting_ops) in zip_arr:
        
        fwd_mono_key    = "fwd_mono_ops_%s_%s_%s" % (alias_dict[dst_logit], alias_dict[src_logit], alias_dict[casting_ops])
        fwd_mono_value  = "templated_ops::fwd_mono_restrict_aligned_ops<%s, %s, %s, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>" % (dst_logit, src_logit, casting_ops)
        
        fwd_uacm_key    = "fwd_uacm_ops_%s_%s_%s" % (alias_dict[dst_logit], alias_dict[src_logit], alias_dict[casting_ops])
        fwd_uacm_value  = "templated_ops::fwd_uacm_restrict_aligned_ops<%s, %s, %s, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>" % (dst_logit, src_logit, casting_ops)

        fwd_pair_key    = "fwd_pair_ops_%s_%s_%s_%s" % (alias_dict[dst_logit], alias_dict[src_logit], alias_dict[other_src_logit], alias_dict[casting_ops])
        fwd_pair_value  = "templated_ops::fwd_pair_restrict_aligned_ops<%s, %s, %s, %s, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>" % (dst_logit, src_logit, other_src_logit, casting_ops)

        fwd_pacm_key    = "fwd_pacm_ops_%s_%s_%s_%s" % (alias_dict[dst_logit], alias_dict[src_logit], alias_dict[other_src_logit], alias_dict[casting_ops])
        fwd_pacm_value  = "templated_ops::fwd_pacm_restrict_aligned_ops<%s, %s, %s, %s, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>" % (dst_logit, src_logit, other_src_logit, casting_ops)

        bwd_mono_key    = "bwd_mono_ops_%s_%s_%s_%s" % (alias_dict[dst_logit], alias_dict[dst_grad], alias_dict[src_grad], alias_dict[casting_ops])
        bwd_mono_value  = "templated_ops::bwd_mono_restrict_aligned_ops<%s, %s, %s, %s, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>" % (dst_logit, dst_grad, src_grad, casting_ops)

        bwd_uacm_key    = "bwd_uacm_ops_%s_%s_%s_%s_%s" % (alias_dict[dst_logit], alias_dict[dst_grad], alias_dict[src_logit], alias_dict[src_grad], alias_dict[casting_ops])
        bwd_uacm_value  = "templated_ops::bwd_uacm_restrict_aligned_ops<%s, %s, %s, %s, %s, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>" % (dst_logit, dst_grad, src_logit, src_grad, casting_ops)

        bwd_plhs_key    = "bwd_pair_lhs_ops_%s_%s_%s_%s_%s" % (alias_dict[dst_logit], alias_dict[dst_grad], alias_dict[other_src_logit], alias_dict[src_grad], alias_dict[casting_ops])
        bwd_plhs_value  = "templated_ops::bwd_pair_lhs_restrict_aligned_ops<%s, %s, %s, %s, %s, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>" % (dst_logit, dst_grad, other_src_logit, src_grad, casting_ops)

        bwd_prhs_key    = "bwd_pair_rhs_ops_%s_%s_%s_%s_%s" % (alias_dict[dst_logit], alias_dict[dst_grad], alias_dict[other_src_logit], alias_dict[src_grad], alias_dict[casting_ops])
        bwd_prhs_value  = "templated_ops::bwd_pair_rhs_restrict_aligned_ops<%s, %s, %s, %s, %s, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>" % (dst_logit, dst_grad, other_src_logit, src_grad, casting_ops)

        fwd_mono_dict[fwd_mono_key]     = fwd_mono_value
        fwd_uacm_dict[fwd_uacm_key]     = fwd_uacm_value
        fwd_pacm_dict[fwd_pacm_key]     = fwd_pacm_value
        fwd_pair_dict[fwd_pair_key]     = fwd_pair_value 
        bwd_mono_dict[bwd_mono_key]     = bwd_mono_value
        bwd_uacm_dict[bwd_uacm_key]     = bwd_uacm_value
        bwd_pair_lhs_dict[bwd_plhs_key] = bwd_plhs_value
        bwd_pair_rhs_dict[bwd_prhs_key] = bwd_prhs_value

    l = (fwd_mono_dict | fwd_uacm_dict | fwd_pacm_dict | fwd_pair_dict | bwd_mono_dict | bwd_uacm_dict | bwd_pair_lhs_dict | bwd_pair_rhs_dict).items()

    out_script  = "\n".join(["using %s\t\t= %s;" % (fill_empty(k, 50), v) for (k, v) in l])

    with open("output.txt", "w") as f_out:
        f_out.write(out_script)

main()
