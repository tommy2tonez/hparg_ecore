#define DEBUG_MODE_FLAG false
#include <unordered_set>
#include <atomic>
#include <memory>
#include <chrono>
#include <iterator>

int main(){

    //okay - there are numerous complains about code management, spaghetti + friends 
    //in this application, we assume all memory operations are serialized through: memlock -> uma_ptr_t -> vma_ptr_t -> cuda_ptr_t host_ptr_t, etc.
    //we assume pointer reachability is guaranteed by allocators (tile_member_access or actual allocations - this is to avoid segment mapping for memcpy or memset - which is bad desicison - we rather allocate on different memregion_size)
    //we assume that uma_ptr_t -> tile_kind, rcu_addr are fixed for program-duration 
    //we assume maintainers are rational and read the design decisions (not doing std::make_unique<dg::network_stack_allocation::Allocation<char[]>> for example)
    //we assume that most heavy-lifting workloads are done by cuda, host is only responsible for network_packets + memmapping + asynchronous device dispatching + initializations + msgrfwd + msgrbwds 
    //I think we could reach 500 cuda TBs/core * s easily - if there are enough combinatorial operations (linear)  
    //things aren't hard if you get the basics

    //the hierarchical of memory operations are memlock -> uma_ptr_t
    //every memlock operations must be in single payload and have clear exit strategies
    //every uma_ptr_t maps must be in single payload and have clear exit strategies
    //every uma_ptr_t acquistions if coupled with memlock acquisitions must preceded by memlock acquitions

    //we mostly do cuda memory transfer serially - via cuda_transfer_ptr_t (pinned_memory) -> cuda_ptr_t (device_memory)
    //the actual step of doing memory operation is to transfer device_memory to pinned_memory - write - transfer back to device_memory if requested or evicted (like fsys_ptr_t)
    //we mutual exclude the uma_ptr_t (-> vma_ptr_1, vma_ptr_t2, vma_ptr_t3) by using the serialization controller network_uma_tlb

    //we'll close 2024 with a demo guys - stay tuned - this is gonna be epic
    //alright - we are doing hybrid collectors
    //we still do frequency but we kinda improve the speed by partially serialize the accesses
    //minimum viable products:  - 100 cuda TB flops/ host_core*s (fwd + bwd) - this is mandatory
    //                          - user designated logs - maybe rewrite postgres (or reconfig) if interfering too much with filesystem writing
    //                          - 32 socket ports sat NIC 
    //                          - allocating 1 << 30 tiles/s - ingesting leaf + immu + crit at 5GBs/ sec

    //side notes: we are reaching 5MBs of raw code - this is a major milestone

    //alrights - we are brainstorming the collector ideas
    //we have the idea of frequency on memregion - and we have to adhere to the advertised frequency - there are computation limits, overheads, delays, kernel scheduling, etc.
    //so we have to split the memregions into groups, which we serially process, so we can circumscribe the maximum possible flops and hit the advertised frequency
    //and for each group, we use power law to scale, says 1.5 Hz on memregion0, 2.25 on memregion1, 3.375 on memregion2, etc, you get the idea
    //this is one training strategy

    //alrights
    //we must compromise cuda device
    //we might not compromise host_asynchronous because it's rarely faulty
    //we'll try 1k - 1k5 LOC per day in 2025 guys

    //alrights
    //been trying to think long and hard on not circular logic for uma_ptr_t
    //there isn't
    //the current logic is probably the cleanest possible (we want the serialized access that uma_ptr_t access provides)
    //as long as the cuda_ptr_t that is referenced by the vma_ptr_t1 is cuda_ptr_t referenced by cutf_ptr_t (vma_ptr_t2), and they have injective property with uma_ptr_t
    //the optimization of transfer and evict is not optional in this case

    //we've been getting feedbacks about overfitting - we'll try to increase the compression_rate/neural_network_size
    //we aren't using linear in this projects - only addnear - we assume addnear is an arbitrary operation that is superior to linear for now - we'll invent the operation
    //we'll do paths + friends and train this model at roughly 1/10000 cost of the current models

    //we got 50% of the infra done guys - including memories - transportation - initializations - burning tiles - network packet infrastructure - allocations - memlock infrastructure 
    //the other 50% must be from cuda wizards
}