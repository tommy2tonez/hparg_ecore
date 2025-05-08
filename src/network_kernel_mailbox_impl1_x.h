#ifndef __NETWORK_KERNEL_MAILBOX_IMPL1_X_H__
#define __NETWORK_KERNEL_MAILBOX_IMPL1_X_H__

//define HEADER_CONTROL 10

#include <memory>
#include "network_kernel_mailbox_impl1.h"
#include "network_trivial_serializer.h"
#include "network_concurrency.h"
#include "network_std_container.h"
#include <chrono>
#include "network_log.h"
#include "network_concurrency_x.h"
#include "stdx.h"
#include "network_exception_handler.h"
#include <chrono>

namespace dg::network_kernel_mailbox_impl1_meterlogx{

    using Address           = dg::network_kernel_mailbox_impl1::model::Address; 
    using MailBoxArgument   = dg::network_kernel_mailbox_impl1::model::MailBoxArgument;

    struct MeterInterface{
        virtual ~MeterInterface() noexcept = default;
        virtual void tick(size_t) noexcept = 0;
        virtual auto get_count() noexcept -> size_t = 0;
        virtual auto get_count_since() noexcept -> std::chrono::time_point<std::chrono::steady_clock> = 0;  
        virtual void reset() noexcept = 0;
    };

    struct MessageStreamerInterface{
        virtual ~MessageStreamerInterface() noexcept = default;
        virtual void stream(std::string_view) noexcept = 0;
    };

    class AtomicMeter: public virtual MeterInterface{

        private:

            //alright, people would argue that this is hardware_destructive_interference_sz, for best practices, I agree

            stdx::inplace_hdi_container<std::atomic<size_t>> count;
            stdx::inplace_hdi_container<std::atomic<std::chrono::time_point<std::chrono::steady_clock>>> since;

        public:

            AtomicMeter() noexcept: count(std::in_place_t{}, 0u),
                                    since(std::in_place_t{}, std::chrono::steady_clock::now()){};

            void tick(size_t incoming_sz) noexcept{

                this->count.value.fetch_add(incoming_sz, std::memory_order_relaxed);
            }

            auto get_count() noexcept -> size_t{

                return this->count.value.load(std::memory_order_relaxed);
            }

            auto get_count_since() noexcept -> std::chrono::time_point<std::chrono::steady_clock>{

                return this->since.value.load(std::memory_order_relaxed);
            }

            void reset() noexcept{

                stdx::seq_cst_guard seqcst_tx;

                this->count.value.exchange(0u, std::memory_order_relaxed);
                this->since.value.exchange(std::chrono::steady_clock::now(), std::memory_order_relaxed);
            }
    };

    class RandomDistributedMeter: public virtual MeterInterface{

        private:

            std::unique_ptr<std::unique_ptr<MeterInterface>[]> base_arr;
            size_t pow2_base_arr_sz;
        
        public:

            RandomDistributedMeter(std::unique_ptr<std::unique_ptr<MeterInterface>[]> base_arr,
                                   size_t pow2_base_arr_sz) noexcept: base_arr(std::move(base_arr)),
                                                                      pow2_base_arr_sz(pow2_base_arr_sz){}

            void tick(size_t incoming_sz) noexcept{

                size_t random_value = dg::network_randomizer::randomize_int<size_t>();
                size_t random_idx   = random_value & (this->pow2_base_arr_sz - 1u);

                this->base_arr[random_idx]->tick(incoming_sz);
            }

            auto get_count() noexcept -> size_t{

                size_t total_sz = 0u;

                for (size_t i = 0u; i < this->pow2_base_arr_sz; ++i){
                    total_sz += this->base_arr[i]->get_count();
                }

                return total_sz;
            }

            auto get_count_since() noexcept -> std::chrono::time_point<std::chrono::steady_clock>{

                std::chrono::time_point<std::chrono::steady_clock> since = this->base_arr[0]->get_count_since();

                for (size_t i = 1u; i < this->pow2_base_arr_sz; ++i){
                    since = std::min(since, this->base_arr[i]->get_count_since());
                }

                return since;
            }

            void reset() noexcept{

                for (size_t i = 0u; i < this->pow2_base_arr_sz; ++i){
                    this->base_arr[i]->reset();
                }
            }
    };

    class MeterLogWorker: public virtual dg::network_concurrency::WorkerInterface{

        private:

            dg::string device_id;
            std::shared_ptr<MeterInterface> send_meter;
            std::shared_ptr<MeterInterface> recv_meter;
            std::shared_ptr<MessageStreamerInterface> msg_streamer;
            std::chrono::nanoseconds meter_dur;

        public:

            MeterLogWorker(dg::string device_id,
                           std::shared_ptr<MeterInterface> send_meter, 
                           std::shared_ptr<MeterInterface> recv_meter,
                           std::shared_ptr<MessageStreamerInterface> msg_streamer,
                           std::chrono::nanoseconds meter_dur) noexcept: device_id(std::move(device_id)),
                                                                         send_meter(std::move(send_meter)),
                                                                         recv_meter(std::move(recv_meter)),
                                                                         msg_streamer(std::move(msg_streamer)),
                                                                         meter_dur(meter_dur){}

            bool run_one_epoch() noexcept{
 
                std::expected<dg::string, exception_t> send_msg = this->make_send_meter_msg(this->send_meter->get_count(), this->send_meter->get_count_since());
                std::expected<dg::string, exception_t> recv_msg = this->make_recv_meter_msg(this->recv_meter->get_count(), this->recv_meter->get_count_since());

                if (!send_msg.has_value()){
                    dg::network_log_stackdump::error_fast(dg::network_exception::verbose(send_msg.error()));
                    return false;
                }

                if (!recv_msg.has_value()){
                    dg::network_log_stackdump::error_fast(dg::network_exception::verbose(recv_msg.error()));
                    return false;
                }

                this->msg_streamer->stream(send_msg.value());
                this->msg_streamer->stream(recv_msg.value());

                this->send_meter->reset();
                this->recv_meter->reset();
                std::this_thread::sleep_for(this->meter_dur);

                return true;
            }
        
        private:

            auto make_send_meter_msg(size_t bsz, std::chrono::time_point<std::chrono::steady_clock> since) noexcept -> std::expected<dg::string, exception_t>{

                auto now        = std::chrono::steady_clock::now();
                auto dur_in_ms  = std::chrono::duration_cast<std::chrono::milliseconds>(now - since);
                size_t tick_sz  = dur_in_ms.count();

                if (tick_sz == 0u){
                    return stdx::expected_format("[METER_REPORT] low meter precision resolution (device_id: {}, part: send_meter)", this->device_id);
                } 

                size_t bsz_per_ms   = bsz / tick_sz;
                size_t bsz_per_s    = bsz_per_ms * 1000u;

                return stdx::expected_format("[METER_REPORT] {} bytes/s sent to {}", bsz_per_s, this->device_id);
            }

            auto make_recv_meter_msg(size_t bsz, std::chrono::time_point<std::chrono::steady_clock> since) noexcept -> std::expected<dg::string, exception_t>{

                auto now        = std::chrono::steady_clock::now();
                auto dur_in_ms  = std::chrono::duration_cast<std::chrono::milliseconds>(now - since);
                size_t tick_sz  = dur_in_ms.count();

                if (tick_sz == 0u){
                    return stdx::expected_format("[METER_REPORT] low meter precision resolution (device_id: {}, part: recv_meter)", this->device_id);
                }

                size_t bsz_per_ms   = bsz / tick_sz;
                size_t bsz_per_s    = bsz_per_ms * 1000u;

                return stdx::expected_format("[METER_REPORT] {} bytes/s recv from {}", bsz_per_s, this->device_id);
            }
    };

    class MeteredMailBox: public virtual dg::network_kernel_mailbox_impl1::core::MailboxInterface{

        private:

            dg::vector<dg::network_concurrency::daemon_raii_handle_t> daemon_vec;
            std::shared_ptr<dg::network_kernel_mailbox_impl1::core::MailboxInterface> base;
            std::shared_ptr<MeterInterface> send_meter;
            std::shared_ptr<MeterInterface> recv_meter;
        
        public:

            MeteredMailBox(dg::vector<dg::network_concurrency::daemon_raii_handle_t> daemon_vec, 
                           std::shared_ptr<dg::network_kernel_mailbox_impl1::core::MailboxInterface> base,
                           std::shared_ptr<MeterInterface> send_meter,
                           std::shared_ptr<MeterInterface> recv_meter): daemon_vec(std::move(daemon_vec)),
                                                                        base(std::move(base)),
                                                                        send_meter(std::move(send_meter)),
                                                                        recv_meter(std::move(recv_meter)){}

            void send(std::move_iterator<MailBoxArgument *> data_arr, size_t sz, exception_t * exception_arr) noexcept{
                
                MailBoxArgument * base_data_arr = data_arr.base();
                size_t total_sz = 0u;

                for (size_t i = 0u; i < sz; ++i){
                    total_sz += base_data_arr[i].content.size();
                }

                this->base->send(std::make_move_iterator(base_data_arr), sz, exception_arr);

                for (size_t i = 0u; i < sz; ++i){
                    if (dg::network_exception::is_failed(exception_arr[i])){
                        total_sz -= base_data_arr[i].content.size();
                    }
                }

                this->send_meter->tick(total_sz);
            }

            void recv(dg::string * output_arr, size_t& output_arr_sz, size_t output_arr_cap) noexcept{

                this->base->recv(output_arr, output_arr_sz, output_arr_cap);
                size_t total_sz = 0u;

                for (size_t i = 0u; i < output_arr_sz; ++i){
                    total_sz += output_arr[i].size();
                }

                this->recv_meter->tick(total_sz);
            }

            auto max_consume_size() noexcept -> size_t{

                return this->base->max_consume_size();
            }
    };

    struct ComponentFactory{

        template <class T, class ...Args>
        static auto to_dg_vector(std::vector<T, Args...> vec) -> dg::vector<T>{

            dg::vector<T> rs{};

            for (size_t i = 0u; i < vec.size(); ++i){
                rs.emplace_back(std::move(vec[i]));
            }

            return rs;
        }

        static auto to_dg_string(std::string_view inp) -> dg::string{

            dg::string rs{};
            std::copy(inp.begin(), inp.end(), std::back_inserter(rs));

            return rs;
        }

        static auto get_meter() -> std::unique_ptr<MeterInterface>{

            return std::make_unique<AtomicMeter>();
        }

        static auto get_distributed_meter(size_t tentative_concurrent_meter_sz) -> std::unique_ptr<MeterInterface>{

            const size_t MIN_CONCURRENT_METER_SZ    = size_t{1};
            const size_t MAX_CONCURRENT_METER_SZ    = size_t{1} << 20; 

            if (std::clamp(tentative_concurrent_meter_sz, MIN_CONCURRENT_METER_SZ, MAX_CONCURRENT_METER_SZ) != tentative_concurrent_meter_sz){
                dg::network_exception::throw_exception(dg::network_exception::INVALID_ARGUMENT);
            }

            size_t meter_arr_sz = stdx::ceil2(tentative_concurrent_meter_sz);
            auto meter_arr      = std::make_unique<std::unique_ptr<MeterInterface>[]>(meter_arr_sz);

            for (size_t i = 0u; i < meter_arr_sz; ++i){
                meter_arr[i] = ComponentFactory::get_meter();
            }

            return std::make_unique<RandomDistributedMeter>(std::move(meter_arr), meter_arr_sz);
        }

        static auto get_meter_log_worker(std::string device_id,
                                         std::shared_ptr<MeterInterface> send_meter,
                                         std::shared_ptr<MeterInterface> recv_meter,
                                         std::shared_ptr<MessageStreamerInterface> msg_stream,
                                         std::chrono::nanoseconds meter_dur) -> std::unique_ptr<dg::network_concurrency::WorkerInterface>{
            
            const size_t MIN_DEVICE_ID_SZ                   = size_t{0u};
            const size_t MAX_DEVICE_ID_SZ                   = size_t{1} << 10;
            const std::chrono::nanoseconds MIN_METER_DUR    = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::microseconds(1));
            const std::chrono::nanoseconds MAX_METER_DUR    = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::hours(1));

            if (std::clamp(static_cast<size_t>(device_id.size()), MIN_DEVICE_ID_SZ, MAX_DEVICE_ID_SZ) != device_id.size()){
                dg::network_exception::throw_exception(dg::network_exception::INTERNAL_CORRUPTION);
            }

            if (send_meter == nullptr){
                dg::network_exception::throw_exception(dg::network_exception::INTERNAL_CORRUPTION);
            }

            if (recv_meter == nullptr){
                dg::network_exception::throw_exception(dg::network_exception::INTERNAL_CORRUPTION);
            }

            if (msg_stream == nullptr){
                dg::network_exception::throw_exception(dg::network_exception::INTERNAL_CORRUPTION);
            }

            if (std::clamp(meter_dur, MIN_METER_DUR, MAX_METER_DUR) != meter_dur){
                dg::network_exception::throw_exception(dg::network_exception::INTERNAL_CORRUPTION);
            }

            return std::make_unique<MeterLogWorker>(ComponentFactory::to_dg_string(device_id), 
                                                    std::move(send_meter), 
                                                    std::move(recv_meter),
                                                    std::move(msg_stream),
                                                    meter_dur);
        }

        static auto get_metered_mailbox(std::vector<dg::network_concurrency::daemon_raii_handle_t> daemon_vec,
                                        std::shared_ptr<dg::network_kernel_mailbox_impl1::core::MailboxInterface> base,
                                        std::shared_ptr<MeterInterface> send_meter,
                                        std::shared_ptr<MeterInterface> recv_meter) -> std::unique_ptr<dg::network_kernel_mailbox_impl1::core::MailboxInterface>{

            if (daemon_vec.empty()){
                dg::network_exception::throw_exception(dg::network_exception::INTERNAL_CORRUPTION);
            }

            if (base == nullptr){
                dg::network_exception::throw_exception(dg::network_exception::INTERNAL_CORRUPTION);
            }

            if (send_meter == nullptr){
                dg::network_exception::throw_exception(dg::network_exception::INTERNAL_CORRUPTION);
            }

            if (recv_meter == nullptr){
                dg::network_exception::throw_exception(dg::network_exception::INTERNAL_CORRUPTION);
            }

            return std::make_unique<MeteredMailBox>(ComponentFactory::to_dg_vector(std::move(daemon_vec)),
                                                    std::move(base),
                                                    std::move(send_meter),
                                                    std::move(recv_meter));
        }
    };

    struct Config{
        std::shared_ptr<dg::network_kernel_mailbox_impl1::core::MailboxInterface> base;
        uint32_t concurrent_recv_meter_sz;
        uint32_t concurrent_send_meter_sz;
        std::string device_id;
        std::shared_ptr<MessageStreamerInterface> msg_streamer;
        std::chrono::nanoseconds meter_dur;
    };

    struct ConfigMaker{
        
        private:

            static auto make_recv_meter(Config config) -> std::unique_ptr<MeterInterface>{
                
                if (config.concurrent_recv_meter_sz == 0u){
                    dg::network_exception::throw_exception(dg::network_exception::INVALID_ARGUMENT);
                }

                if (config.concurrent_recv_meter_sz == 1u){
                    return ComponentFactory::get_meter();
                }

                return ComponentFactory::get_distributed_meter(config.concurrent_recv_meter_sz);
            }

            static auto make_send_meter(Config config) -> std::unique_ptr<MeterInterface>{

                if (config.concurrent_send_meter_sz == 0u){
                    dg::network_exception::throw_exception(dg::network_exception::INVALID_ARGUMENT);
                }

                if (config.concurrent_send_meter_sz == 1u){
                    return ComponentFactory::get_meter();
                }

                return ComponentFactory::get_distributed_meter(config.concurrent_send_meter_sz);
            }

            static auto make_workers(Config config, 
                                     std::shared_ptr<MeterInterface> send_meter, 
                                     std::shared_ptr<MeterInterface> recv_meter) -> std::vector<dg::network_concurrency::daemon_raii_handle_t>{

                std::unique_ptr<dg::network_concurrency::WorkerInterface> worker = ComponentFactory::get_meter_log_worker(config.device_id, send_meter, recv_meter, config.msg_streamer, config.meter_dur);

                auto daemon_vec     = std::vector<dg::network_concurrency::daemon_raii_handle_t>();
                auto daemon_handle  = dg::network_exception_handler::throw_nolog(dg::network_concurrency::daemon_saferegister(dg::network_concurrency::HEARTBEAT_DAEMON, std::move(worker)));
                daemon_vec.emplace_back(std::move(daemon_handle));

                return daemon_vec;
            }

        public:

            static auto make(Config config) -> std::unique_ptr<dg::network_kernel_mailbox_impl1::core::MailboxInterface>{

                std::shared_ptr<MeterInterface> send_meter  = ConfigMaker::make_send_meter(config);
                std::shared_ptr<MeterInterface> recv_meter  = ConfigMaker::make_recv_meter(config);

                return ComponentFactory::get_metered_mailbox(ConfigMaker::make_workers(config, send_meter, recv_meter),
                                                             std::move(config.base),
                                                             send_meter,
                                                             recv_meter);
            }
    };

    extern auto make(Config config) -> std::unique_ptr<dg::network_kernel_mailbox_impl1::core::MailboxInterface>{

        return ConfigMaker::make(std::move(config));
    }
}

namespace dg::network_kernel_mailbox_impl1_flash_streamx{

    //alright, we'll be back to implement if there are issues or requests
    //this is good enough, we dont have time
    //the only problem is the problem of affinity, we'll write an affined function to do customized mapping of containers
    //the problem quickly resolves if are to increase latency -> 10ms per consumption or production
    //yet we are very laggy, this is probably for the best because there is literally nothing free
    
    //whenever I felt like I did something wrong
    //let's see the options

    //low latency, low thruput
    //low latency, high thruput (this is what we want)
        //low latency, high thruput low cap switch signal
            //a lot of memory orderings (unavoidable)

        //low latency, high thruput high cap switch signal
            //OK, no <a lot of memory orderings>, no-busy state might result in the worst case of <worst_case_latency> * <bouncing_container_sz>
                //we are interested in offsetting the cost of <worst_case_latency> * <bouncing_container_sz>
                //this is doable via padding requests (empty requests without doing actual things to push the order thru with low latency)

    //high latency, low thruput (this is not what we want)
    //high latency, high thruput

    //it seems that 10ms is best for logit density mining
    //after consulted with my friends in the STD, they preferred we staying in the nanoseconds territory, I explained that has a lot to do with request pads
    //because low_latency + low cap ::wait is a recipe for disaster, we are polluting the system with cmpexchg instructions

    //we are awared of the std::memory_order_relaxed serialization at the subscribe function
    //we dont have a way to work around this yet, because the anomaly dampener for the uniform distribution is important in producer + consumer pattern
    //we are aiming for at most 1s / latency_time memory orders per second
    //or 1MB - 10MB of digesting data/ memory order
    //the latter would sat the RAM bandwidth faster so the memory ordering is no longer an issue

    //we use the digestion size of roughly 1 << 16/ load to offset the cost of access synchronization mechanisms
    //we use random hash table to increase compute
    //those are two very different things
    //one is CPU efficiency by increasing producing + consuming sz
    //another is task thruput by using more CPU resource concurrently

    //------
    //my std friends liked the idea of no synchronization computation + high frequency computing
    //yet ... they still want to have certain mechanisms to do tile synchronizations (backward + forward by using immediate semaphore tiles) 
    //------

    //I feel like doing containers (we'll be doing containers before getting back to this problem), node_fast_map + node_fast_set containers 

    //I was thinking about the scheduling techniques
    //let's skim through the options:
    
    //static number:
        //exact scheduler forward + backward
        //  - discretized enumerated exact scheduler
        //  - exact exact scheduler

        //frequency scheduler forward + backward
        //  - discretized enumerated extract frequency scheduler
        //  - exact frequency scheduler

    //adaptive number:
        //  - ... 

    //alright, we can literally list 1024 ways to do the scheduled forward + backward thing
    //we dont care, we want to prove that doing things our way is complete and the overhead is within a certain acceptable compute flops

    //assume that there is only one forward tile on the memregion
        //- we can use empty reference tile to bring the acummulated uncertainty between frequencied regions -> our expecting scheduled time
        //- the overhead of doing so is bouncing_memregion * lock_overhead + memevent queueing overheads + etc.
        //- the overhead of the mempress collector is fixed, it's always ticking
        //- we can't guarantee that things would be frequencied as specified, this is a statistics fling that we have to collect + use machine learning to tune our engine
        //- what we can guarantee is that the machine will snap into certain states after a certain time, abs_window + fixed retransmission_window + fixed_retransmission_sz are to serve the purpose
        //- and the frequency is not adaptive, it is utc_clock::discretized_time_range, we can implicitly expect (or soft_synchronize) that the machine will behave correctly if certain empty state has been reached

    //assume that there are more than one forward tile on the memregion:
        //- only statistic calibration can give you the answer

    //alright, I was thinking about delvrsrv + hash_table injection
    //it's a radix of radix sort + distributed hash_table invoke
    //we want to fit the hash table -> 64KB of cache memory before we load/ read the table(s) in batches
    //alright fellas, I literally dont have time to argue with yall

    //I have a timeline to push this to the mainframe, by using fully detached infrastructure + fixed window transmission (we have seeked advices, it's the best route for this kind of forever detached tree training computation, there is literally not a single explicit synchronization instruction, we just peek the tree to test for integrity)
    //flash_streamx is only for 64KB transmission packet, with the event trigger size of 1MB/consumption
    //the hashtable is the best for temporal cache retrieval, with the sole operations are insert and clear 
    //the hashtable is also best for delvrsrv_kv hash_table, with the sole operations are insert, iteration and clear
    //the hashtable cannot be further optimized, can be used concurrently on multicore with no SIMD instructions available, it's the best memory footprint wise speaking, we are literally considering uint8_t as virtual_addr_t for kv_feeder
    //our client is very stingent on the time and the usability of this
    //so we have tons of work to do

    //alright, I have gotten three specific feedbacks
    //first is to drain the swamp periodically for distributed container
    //second is to think about the requirements of destroy
    //let's see the variables that we could collect for our priority queue
    //- user-specified index
    //- stream start_time
    //- stream absolute timeout
    //- stream collected latency array
    //- stream progress
    //- system current state
    //- from IP
    //- # of current concurrent IPs
    //third is to be considerate about the temporal unordered_map + temporal unordered set memory access pattern, is there something we could do on the sender side to offset this cost, we also dont want to be predictable for the reason being predictable == exploitable 

    //we implemented (1)
    //we are thinking about implementing (3) without actually overstepping into the-micro-optimization-and-enables-exploitation territory
    //it's a hashing problem, we can clue the region of frequent accesses by using IP, and their packet id should be incremental
    //alright, one last request is to implement a select-like feature of std by using encapsulated global affined variables injection. Our client does not want to be forced to wait to get data. We live in a free world fellas
    //assume the reactor threshold is N, we can expect that the latency of all the guys above the threshold to be in the nanoseconds range
    //we can assume the guys below the threshold to be not in the nanoseconds range. We must specify that N to be the peeking size for each of the subcontainer because thats the worst latency scenerio, and the consumer must have the consume size of subcontainer_sz * N
    //we have seeked advices that the reactor threshold must be of 10 MB, and the worst draining latency must be of 10ms. We want the water pipe to run all the time to push data through, to achieve the best performance scenerio of 1GB/(100*k memory orderings)
    //we will work on improving the memory footprint of maps and sets. its super very complicated

    //we are only worried about the force WAIT (both send and recv, external interface), not the select feature of reaching a certain threshold, implementing that is super hard to be efficient
    //so we'll implement a global static affined injection of timeout
    //that's good enough for MVP, we'll move on for now, we'll circle back to do profile guided optimizations
    //this is not a good answer, this seems more like a patch
    //because it really is

    template <class ID>
    class AffinedTimeoutController{

        private:

            using self = AffinedTimeoutController;

            static inline std::vector<stdx::hdi_container<std::optional<std::chrono::nanoseconds>>> timeout_container = []{

                size_t container_sz = dg::network_concurrency::THREAD_COUNT;
                auto rs = std::vector<stdx::hdi_container<std::optional<std::chrono::nanoseconds>>>(container_sz);

                for (size_t i = 0u; i < container_sz; ++i){
                    rs[i].value = std::nullopt;
                }

                return rs;
            }();

        public:

            static consteval auto get_min_timeout() noexcept -> std::chrono::nanoseconds{

                return std::chrono::nanoseconds::min();
            }

            static consteval auto get_max_timeout() noexcept -> std::chrono::nanoseconds{

                return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::minutes(1));
            }

            static void abs_set(std::optional<std::chrono::nanoseconds> timeout_period) noexcept{

                self::timeout_container[dg::network_concurrency::this_thread_idx()].value = timeout_period;
            } 

            static void set(std::chrono::nanoseconds timeout_period) noexcept{

                self::timeout_container[dg::network_concurrency::this_thread_idx()].value = std::clamp(timeout_period, self::get_min_timeout(), self::get_max_timeout());                
            }

            static void clear() noexcept{

                self::timeout_container[dg::network_concurrency::this_thread_idx()].value = std::nullopt;
            }

            static auto get() noexcept -> std::optional<std::chrono::nanoseconds>{

                return self::timeout_container[dg::network_concurrency::this_thread_idx()].value;
            }
    };

    struct ClientRecvSignature{};
    using client_affined_timeout_controller = AffinedTimeoutController<ClientRecvSignature>;  

    template <class ID>
    struct AffinedBlockOnExhaustionController{

        private:

            using self = AffinedBlockOnExhaustionController;

            static inline std::vector<stdx::hdi_container<std::optional<bool>>> flag_container = []{

                size_t container_sz = dg::network_concurrency::THREAD_COUNT;
                auto rs = std::vector<stdx::hdi_container<std::optional<bool>>>(container_sz);

                for (size_t i = 0u; i < container_sz; ++i){
                    rs[i].value = std::nullopt;
                }

                return rs;
            }();

        public:

            static void abs_set(std::optional<bool> state) noexcept{

                self::flag_container[dg::network_concurrency::this_thread_idx()].value = state;
            }

            static void set(bool flag) noexcept{

                self::flag_container[dg::network_concurrency::this_thread_idx()].value = flag; 
            }

            static void clear() noexcept{

                self::flag_container[dg::network_concurrency::this_thread_idx()].value = std::nullopt;
            }

            static auto get() noexcept -> std::optional<bool>{

                return self::flag_container[dg::network_concurrency::this_thread_idx()].value;
            }
    };

    struct ClientSendSignature{};
    using client_affined_blocksend_controller = AffinedBlockOnExhaustionController<ClientSendSignature>; 

    using Address           = dg::network_kernel_mailbox_impl1::model::Address; 
    using MailBoxArgument   = dg::network_kernel_mailbox_impl1::model::MailBoxArgument;

    static inline constexpr size_t DEFAULT_KEYVALUE_FEED_SIZE               = size_t{1} << 10; 
    static inline constexpr size_t DEFAULT_KEY_FEED_SIZE                    = size_t{1} << 8;
    static inline constexpr uint32_t PACKET_SEGMENT_SERIALIZATION_SECRET    = 3036322422ULL; //we randomize the secret within the uint32_t range to make sure that we dont have internal corruptions
    static inline constexpr uint32_t PACKET_INTEGRITY_SECRET                = 2203221141ULL;

    //OK
    struct GlobalIdentifier{
        Address addr;
        std::pair<uint64_t, uint64_t> local_id; //avoid __uint128_t

        template <class Reflector>
        constexpr void dg_reflect(const Reflector& reflector) const noexcept{
            reflector(addr, local_id);
        }

        template <class Reflector>
        constexpr void dg_reflect(const Reflector& reflector) noexcept{
            reflector(addr, local_id);
        }
    };

    //OK
    struct PacketSegment{
        dg::string buf;
        GlobalIdentifier id;
        uint64_t segment_idx;
        uint64_t segment_sz;

        uint64_t mm_integrity_value;
        bool has_mm_integrity_value;
    };

    //OK
    struct AssembledPacket{
        dg::vector<PacketSegment> data;
        size_t collected_segment_sz;
        size_t total_segment_sz;
    };

    //OK
    static auto serialize_packet_segment(PacketSegment&& segment) noexcept -> std::expected<dg::string, exception_t>{

        using header_t      = std::tuple<GlobalIdentifier, uint64_t, uint64_t, uint64_t, bool>;

        size_t header_sz    = dg::network_compact_serializer::integrity_size(header_t{});
        size_t old_sz       = segment.buf.size();
        size_t new_sz       = old_sz + header_sz;
        auto header         = header_t{segment.id, segment.segment_idx, segment.segment_sz, segment.mm_integrity_value, segment.has_mm_integrity_value};

        try{
            segment.buf.resize(new_sz);
            dg::network_compact_serializer::integrity_serialize_into(std::next(segment.buf.data(), old_sz), header, PACKET_SEGMENT_SERIALIZATION_SECRET);
            return std::expected<dg::string, exception_t>(std::move(segment.buf));
        } catch (...){
            return std::unexpected(dg::network_exception::wrap_std_exception(std::current_exception()));
        }
    }

    //OK
    static auto deserialize_packet_segment(dg::string&& buf) noexcept -> std::expected<PacketSegment, exception_t>{

        using header_t      = std::tuple<GlobalIdentifier, uint64_t, uint64_t, uint64_t, bool>;

        size_t header_sz    = dg::network_compact_serializer::integrity_size(header_t{});
        size_t buf_sz       = buf.size();
        auto header         = header_t{}; 

        if (buf_sz < header_sz){
            return std::unexpected(dg::network_exception::SOCKET_STREAM_BAD_SEGMENT);           
        }

        size_t hdr_off      = buf.size() - header_sz;
        char * hdr_buf      = std::next(buf.data(), hdr_off);  

        exception_t err     = dg::network_exception::to_cstyle_function(dg::network_compact_serializer::integrity_deserialize_into<header_t>)(header, hdr_buf, header_sz, PACKET_SEGMENT_SERIALIZATION_SECRET);

        if (dg::network_exception::is_failed(err)){
            return std::unexpected(err);
        }

        buf.resize(hdr_off);

        PacketSegment rs    = {};
        rs.buf              = std::move(buf);
        std::tie(rs.id, rs.segment_idx, rs.segment_sz, rs.mm_integrity_value, mm.has_mm_integrity_value) = header;

        return rs;
    }

    //OK
    static auto internal_assembled_packet_to_buffer(AssembledPacket&& pkt) noexcept -> std::expected<dg::string, exception_t>{

        if (pkt.total_segment_sz != pkt.collected_segment_sz){
            return std::unexpected(dg::network_exception::INVALID_ARGUMENT);
        }

        if (pkt.total_segment_sz != pkt.data.size()){
            return std::unexpected(dg::network_exception::INVALID_ARGUMENT);
        }

        if (pkt.total_segment_sz == 0u){
            return dg::string();
        }

        if (pkt.total_segment_sz == 1u){
            return std::expected<dg::string, exception_t>(std::move(pkt.data.front().buf));
        }

        size_t total_bsz = 0u;

        for (size_t i = 0u; i < pkt.total_segment_sz; ++i){
            total_bsz += pkt.data[i].buf.size();
        }

        std::expected<dg::string, exception_t> rs = dg::network_exception::cstyle_initialize<dg::string>(total_bsz, 0);

        if (!rs.has_value()){
            return rs;
        }

        char * out_it = rs.value().data(); 

        for (size_t i = 0u; i < pkt.total_segment_sz; ++i){
            out_it = std::copy(pkt.data[i].buf.begin(), pkt.data[i].buf.end(), out_it);
        }

        return rs;
    }

    //OK
    static auto internal_integrity_assembled_packet_to_buffer(AssembledPacket&& pkt) noexcept -> std::expected<dg::string, exception_t>{

        if (pkt.data.size() == 0u){
            return internal_assembled_packet_to_buffer(static_cast<AssembledPacket&&>(pkt));
        }

        uint64_t mm_integrity_value                 = pkt.data.front().mm_integrity_value;
        bool has_mm_integrity_value                 = pkt.data.front().has_mm_integrity_value;
        std::expected<dg::string, exception_t> rs   = internal_assembled_packet_to_buffer(static_cast<AssembledPacket&&>(pkt));

        if (!rs.has_value()){
            return rs;
        }

        if (has_mm_integrity_value){
            uint64_t integrity_value = dg::network_hash::murmur_hash(rs->data(), rs->size(), PACKET_INTEGRITY_SECRET);

            if (mm_integrity_value != integrity_value){
                return std::unexpected(dg::network_exception::SOCKET_STREAM_CORRUPTED);
            }
        }

        return rs;
    }

    //OK
    struct ExhaustionControllerInterface{
        virtual ~ExhaustionControllerInterface() noexcept = default;
        virtual auto is_should_wait() noexcept -> bool = 0;
        virtual auto update_waiting_size(size_t) noexcept -> exception_t = 0;
    };

    //OK
    struct PacketIDGeneratorInterface{
        virtual ~PacketIDGeneratorInterface() noexcept = default;
        virtual auto get_id() noexcept -> GlobalIdentifier = 0;
    };

    //OK
    struct ProducerDrainerPredicateInterface{
        virtual ~ProducerDrainerPredicateInterface() noexcept = default;
        virtual auto is_should_drain() noexcept -> bool = 0;
        virtual void reset() noexcept = 0;
    };

    //OK
    struct PacketizerInterface{
        virtual ~PacketizerInterface() noexcept = default;
        virtual auto packetize(dg::string&&) noexcept -> std::expected<dg::vector<PacketSegment>, exception_t> = 0;
        virtual auto segment_byte_size() const noexcept -> size_t = 0;
        virtual auto max_packet_size() const noexcept -> size_t = 0;
        virtual auto max_segment_count() const noexcept -> size_t = 0;
    };

    //OK
    struct InBoundGateInterface{
        virtual ~InBoundGateInterface() noexcept = default;
        virtual void thru(GlobalIdentifier * global_id_arr, size_t sz, exception_t * exception_arr) noexcept = 0;
        virtual auto max_consume_size() noexcept -> size_t = 0;
    };

    //OK
    struct BlackListGateInterface: virtual InBoundGateInterface{
        virtual ~BlackListGateInterface() noexcept = default;
        virtual void blacklist(GlobalIdentifier * global_id_arr, size_t sz, exception_t * exception_arr) noexcept = 0;
    };

    //OK
    struct EntranceControllerInterface{
        virtual ~EntranceControllerInterface() noexcept = default;
        virtual void tick(GlobalIdentifier * global_id_arr, size_t sz, exception_t * exception_arr) noexcept = 0; 
        virtual void get_expired_id(GlobalIdentifier * output_arr, size_t& sz, size_t cap) noexcept = 0;
        virtual auto max_consume_size() noexcept -> size_t = 0;
    };

    //OK
    struct PacketAssemblerInterface{
        virtual ~PacketAssemblerInterface() noexcept = default;
        virtual void assemble(std::move_iterator<PacketSegment *> segment_arr, size_t sz, std::expected<AssembledPacket, exception_t> * assembled_arr) noexcept = 0; //it seems like this breaks the rule of unexpected == no_consume, we have to alter the semantic, exception_t is only for exceptions, we dont really know if std::optional<> is worth it
        virtual void destroy(GlobalIdentifier * id_arr, size_t sz) noexcept = 0;
        virtual auto max_consume_size() noexcept -> size_t = 0;
    };

    //OK
    struct InBoundContainerInterface{
        virtual ~InBoundContainerInterface() noexcept = default;
        virtual void push(std::move_iterator<dg::string *> arr, size_t sz, exception_t * exception_arr) noexcept = 0;
        virtual void pop(dg::string * output_arr, size_t& sz, size_t cap) noexcept = 0;
        virtual auto max_consume_size() noexcept -> size_t = 0;
    };

    //OK
    struct OutBoundRuleInterface{
        virtual ~OutBoundRuleInterface() noexcept = default;
        virtual auto thru(const Address&) noexcept -> std::expected<bool, exception_t> = 0;
    };

    //OK
    class NoExhaustionController: public virtual ExhaustionControllerInterface{

        public:

            auto is_should_wait() noexcept -> bool{

                return false;
            }

            auto update_waiting_size(size_t) noexcept -> exception_t{

                return dg::network_exception::SUCCESS;
            }
    };

    class EmptyOutBoundRule: public virtual OutBoundRuleInterface{

        public:

            auto thru(const Address&) noexcept -> std::expected<bool, exception_t>{

                return true;
            }
    };

    class TimingDrainerPredicate: public virtual ProducerDrainerPredicateInterface{

        private:

            stdx::hdi_container<std::atomic<std::chrono::time_point<std::chrono::steady_clock>>> since;
            std::chrono::nanoseconds timeout;

        public:

            TimingDrainerPredicate(std::chrono::nanoseconds timeout){

                this->since.value.exchange(std::chrono::steady_clock::now(), std::memory_order_seq_cst);
                this->timeout = timeout;
            }

            auto is_should_drain() noexcept -> bool{

                std::chrono::time_point<std::chrono::steady_clock> now = std::chrono::steady_clock::now(); //bad
                std::chrono::nanoseconds lapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(now - this->since.value.load(std::memory_order_relaxed));

                return lapsed > this->timeout;
            }

            void reset() noexcept{

                this->since.value.exchange(std::chrono::steady_clock::now(), std::memory_order_relaxed); //bad
            }
    };

    template <class Key, class Value>
    class temporal_finite_unordered_map{

        private:

            dg::unordered_unstable_map<Key, Value> map_container;
            dg::pow2_cyclic_queue<std::pair<Key, Value>> cyclic_queue;

        public:

            temporal_finite_unordered_map(size_t tentative_cap){

                size_t upcap            = stdx::ceil2(std::max(size_t{2}, tentative_cap));
                size_t pow2_exponent    = stdx::ulog2(upcap);
                this->map_container.reserve(upcap);
                this->cyclic_queue      = dg::pow2_cyclic_queue<std::pair<Key, Value>>(pow2_exponent);
            }

            template <class KeyLike>
            constexpr auto find(const KeyLike& keylike) noexcept{

                return this->map_container.find(keylike);
            }

            template <class KeyLike>
            constexpr auto find(const KeyLike& keylike) const noexcept{

                return this->map_container.find(keylike);
            }

            constexpr auto begin() noexcept{

                return this->map_container.begin();
            }

            constexpr auto begin() const noexcept{

                return this->map_container.begin();
            }

            constexpr auto end() noexcept{
                
                return this->map_container.end();
            }

            constexpr auto end() const noexcept{

                return this->map_container.end();
            }

            template <class ...Args>
            constexpr auto emplace(Args&& ...args) noexcept{

                if (this->cyclic_queue.size() == this->cyclic_queue.capacity()) [[unlikely]]{
                    this->cut_half();
                }

                std::pair<const Key, Value> kv_pair(std::forward<Args>(args)...);

                auto rs = this->map_container.insert(kv_pair);
                dg::network_exception_handler::nothrow_log(this->cyclic_queue.push_back(std::move(kv_pair)));

                return rs;
            }

            template <class KeyArg, class ...Args>
            constexpr auto try_emplace(KeyArg&& key, Args&&... value_args) noexcept{

                if (this->cyclic_queue.size() == this->cyclic_queue.capacity()) [[unlikely]]{
                    this->cut_half();
                }

                std::pair<const Key, Value> kv_pair(std::piecewise_construct,
                                                    std::forward_as_tuple(std::forward<KeyArg>(key)),
                                                    std::forward_as_tuple(std::forward<Args>(value_args)...));

                auto rs = this->map_container.insert(kv_pair);
                dg::network_exception_handler::nothrow_log(this->cyclic_queue.push_back(std::move(kv_pair)));

                return rs;
            }

        private:

            void cut_half() noexcept{

                size_t half_sz      = this->cyclic_queue.size() >> 1;
                size_t removable_sz = std::min(static_cast<size_t>(this->cyclic_queue.size()), half_sz);

                this->cyclic_queue.erase_front_range(removable_sz);
                this->map_container.clear();

                for (size_t i = 0u; i < this->cyclic_queue.size(); ++i){
                    this->map_container.insert(this->cyclic_queue[i]);
                } 
            }
    };

    //we need to research our way through this
    //distributed_hash_set by using radix tree
    //done
    //simd membership cmp
    //done
    //population extraction
    //done
    //we are not awared of other options

    template <class Key>
    class bloom_filter{

        private:

            dg::vector<bool> bloom_table;
            dg::vector<uint32_t> murmur_hash_secret_vec;
            size_t sz;

        public:

            bloom_filter(size_t bloom_table_tentative_cap, size_t rehash_sz){

                if (rehash_sz == 0u){
                    throw std::invalid_argument("bad bloom_filter constructor's arguments");
                }

                size_t upcap    = stdx::ceil2(bloom_table_tentative_cap);
                auto rand_gen   = std::bind(std::uniform_int_distribution<uint32_t>{}, std::mt19937{static_cast<uint32_t>(std::chrono::high_resolution_clock::now().time_since_epoch().count())});
                this->sz        = 0u; 

                this->bloom_table.resize(upcap, false);
                this->murmur_hash_secret_vec.resize(rehash_sz);

                std::generate(this->murmur_hash_secret_vec.begin(), this->murmur_hash_secret_vec.end(), rand_gen);
            }

            auto not_contains(const Key& key) const noexcept -> bool{ //bloom filter only works for not_contains, true negative, so the interface should be built on such, not not contains does not equal to contains, logically speaking it is, yet we are returning false if we cant determine the result

                for (uint32_t secret: this->murmur_hash_secret_vec){
                    size_t hashed_value = dg::network_hash::hash_reflectible(key, secret);

                    if (!this->internal_numerical_key_search(hashed_value)){
                        return true;
                    }
                }

                return false;
            }

            void insert(const Key& key) noexcept{

                for (uint32_t secret: this->murmur_hash_secret_vec){
                    size_t hashed_value = dg::network_hash::hash_reflectible(key, secret);
                    this->internal_numerical_key_insert(hashed_value);
                }

                this->sz += 1;
            }

            void clear() noexcept{

                std::fill(this->bloom_table.begin(), this->bloom_table.end(), false);
                this->sz = 0u;
            }

            auto size() const noexcept{

                return this->sz;
            }

            auto capacity() const noexcept{

                return this->bloom_table.size();
            }
        
        private:

            auto internal_numerical_key_search(size_t key) const noexcept -> bool{

                size_t idx = key & (this->bloom_table.size() - 1u); 
                return this->bloom_table[idx];
            }

            void internal_numerical_key_insert(size_t key) noexcept{

                size_t idx = key & (this->bloom_table.size() - 1u);
                this->bloom_table[idx] = true;
            }
    };

    template <class Key>
    class temporal_switching_bloom_filter{

        private:

            bloom_filter<Key> bloom_filter1;
            bloom_filter<Key> bloom_filter2;
            size_t switchfoot_sz;
            size_t clearfoot_sz;
            size_t footstep_sz;
            bool foot;

        public:

            temporal_switching_bloom_filter(size_t cap, size_t rehash_sz, size_t reliability_decay): bloom_filter1(std::max(size_t{2}, cap), rehash_sz),
                                                                                                     bloom_filter2(std::max(size_t{2}, cap), rehash_sz){

                this->switchfoot_sz     = std::max(size_t{1}, this->bloom_filter1.capacity() >> reliability_decay);
                this->clearfoot_sz      = switchfoot_sz  * 2u;
                this->footstep_sz       = 0u;
                this->foot              = true;
            }

            auto not_contains(const Key& key) const noexcept -> bool{

                return this->bloom_filter1.not_contains(key) && this->bloom_filter2.not_contains(key);
            }

            void insert(const Key& key) noexcept{

                if (this->bloom_filter1.size() + this->bloom_filter2.size() == this->clearfoot_sz){
                    if (this->foot){
                        this->bloom_filter2.clear();
                    } else{
                        this->bloom_filter1.clear();
                    }
                }

                if (this->footstep_sz == this->switchfoot_sz){
                    this->foot          = !this->foot;
                    this->footstep_sz   = 0u;
                }

                if (this->foot){
                    this->bloom_filter1.insert(key);
                } else{
                    this->bloom_filter2.insert(key);
                }

                this->footstep_sz += 1;
            }
    };

    //OK, version control by internalizing dependencies, we wont have version control problems
    class dg_binary_semaphore{

        private:

            std::binary_semaphore base;

        public:

            constexpr dg_binary_semaphore(std::ptrdiff_t state): base(state){}

            auto acquire() noexcept -> exception_t{

                try{
                    this->base.acquire();
                    return dg::network_exception::SUCCESS;
                } catch (...){
                    return dg::network_exception::wrap_std_exception(std::current_exception());
                }

            }

            auto release() noexcept -> exception_t{

                try{
                    this->base.release(1u);
                    return dg::network_exception::SUCCESS;
                } catch(...){
                    return dg::network_exception::wrap_std_exception(std::current_exception());
                }
            }

            template <class Rep, class Period>
            auto try_acquire_for(std::chrono::duration<Rep, Period> dur) noexcept -> std::expected<bool, exception_t>{

                try{
                    return base.try_acquire_for(dur);
                } catch (...){
                    return std::unexpected(dg::network_exception::wrap_std_exception(std::current_exception()));
                }
            }

            template <class Clock, class Duration>
            auto try_acquire_until(std::chrono::time_point<Clock, Duration> abs_time) noexcept -> std::expected<bool, exception_t>{

                try{
                    return base.try_acquire_until(abs_time);
                } catch (...){
                    return std::unexpected(dg::network_exception::wrap_std_exception(std::current_exception()));
                }
            }
    };

    //OK, version control by internalizing dependencies, we wont have version control problems
    //the complex reactor will try to subscribe -> the smp_queue serialized access state, and notify most of the times (not all of the time if the threshold is reached)

    class ComplexReactor{

        private:

            dg::vector<std::shared_ptr<dg_binary_semaphore>> smp_queue; //smp_queue serialized access state
            size_t smp_queue_cap; //const states
            std::mutex mtx_smp_queue; //smp_queue serialized access state 

            stdx::inplace_hdi_container<std::atomic<intmax_t>> counter; //counter states 
            stdx::inplace_hdi_container<std::atomic<intmax_t>> wakeup_threshold; //mostly_const (preinitialized)

            stdx::inplace_hdi_container<std::atomic<size_t>> smp_queue_sz; //smp_queue serialized access states

        public:

            ComplexReactor(size_t smp_queue_cap): smp_queue(),
                                                  smp_queue_cap(smp_queue_cap), 
                                                  mtx_smp_queue(),
                                                  counter(std::in_place_t{}, 0),
                                                  wakeup_threshold(std::in_place_t{}, 0),
                                                  smp_queue_sz(std::in_place_t{}, 0u){}

            void increment(size_t sz) noexcept{

                constexpr size_t INCREMENT_RETRY_SZ                 = 4u;
                const std::chrono::nanoseconds FAILED_LOCK_SLEEP    = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::microseconds{1});

                this->counter.value.fetch_add(sz, std::memory_order_relaxed); //increment
                intmax_t expected = this->wakeup_threshold.value.load(std::memory_order_relaxed);

                for (size_t epoch = 0u; epoch < INCREMENT_RETRY_SZ; ++epoch){
                    intmax_t current = this->counter.value.load(std::memory_order_relaxed);

                    if (current < expected){
                        return;
                    }

                    std::atomic_signal_fence(std::memory_order_seq_cst);
                    size_t current_queue_sz = this->smp_queue_sz.value.load(std::memory_order_relaxed);

                    if (current_queue_sz == 0u){
                        return;
                    }

                    bool try_lock_rs = this->mtx_smp_queue.try_lock(); 

                    if (!try_lock_rs){
                        stdx::lock_yield(FAILED_LOCK_SLEEP);
                        continue;
                    }

                    this->unsafe_notify_subscribers();
                    this->mtx_smp_queue.unlock();

                    return;
                }
            }

            void decrement(size_t sz) noexcept{

                this->counter.value.fetch_sub(sz, std::memory_order_relaxed);
            }

            void reset() noexcept{

                this->counter.value.exchange(intmax_t{0}, std::memory_order_relaxed);
            } 

            auto set_wakeup_threshold(intmax_t arg) noexcept{

                this->wakeup_threshold.value.exchange(arg, std::memory_order_relaxed);
            } 

            void subscribe(std::chrono::nanoseconds waiting_time) noexcept{

                constexpr uint8_t ACTION_NO         = 0u;
                constexpr uint8_t ACTION_ACQUIRE    = 1u;

                intmax_t current    = this->counter.value.load(std::memory_order_relaxed);
                intmax_t expected   = this->wakeup_threshold.value.load(std::memory_order_relaxed);

                if (current >= expected){
                    return; //OK
                }

                std::shared_ptr<dg_binary_semaphore> spinning_smp = dg::network_allocation::make_shared<dg_binary_semaphore>(0);

                uint8_t action = [&, this]() noexcept{
                    stdx::xlock_guard<std::mutex> lck_grd(this->mtx_smp_queue);

                    //register the subscriber to the queue
                    //hope that the subscriber will be notified if a certain threshold is reached, not guaranteed 
                    //let's see where the notifications are
                    //alright, wait is very complicated to implement, the problem we are facing is the unit problem
                    //I'm telling yall that the unit in host_asynchronous can be solved, yet the unit problem in the mailbox or UDP 8K cannot be solved
                    //this is an entire different radix of things, our entire system only point of compromision is THIS

                    exception_t register_err    = this->unsafe_register_subscriber(spinning_smp);
                    
                    if (dg::network_exception::is_failed(register_err)){
                        return ACTION_NO; //OK
                    }

                    intmax_t new_current        = this->counter.value.load(std::memory_order_relaxed); 

                    if (new_current >= expected){
                        this->unsafe_notify_subscribers();
                    }

                    return ACTION_ACQUIRE;
                }();

                switch (action){
                    case ACTION_NO:
                    {
                        std::this_thread::sleep_for(waiting_time);
                        break;
                    }
                    case ACTION_ACQUIRE:
                    {
                        std::expected<bool, exception_t> smp_err = spinning_smp->try_acquire_for(waiting_time);

                        if (!smp_err.has_value()){
                            dg::network_log_stackdump::error(dg::network_exception::verbose(smp_err.error()));
                        }

                        break;
                    }
                    default:
                    {
                        if constexpr(DEBUG_MODE_FLAG){
                            dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                            std::abort();
                        } else{
                            std::unreachable();
                        }
                    }
                }
            }

        private:

            auto unsafe_register_subscriber(std::shared_ptr<dg_binary_semaphore> smp) noexcept -> exception_t{

                if (smp == nullptr){
                    return dg::network_exception::INVALID_ARGUMENT;
                }

                if (this->smp_queue.size() == this->smp_queue_cap){
                    return dg::network_exception::QUEUE_FULL;
                }

                this->smp_queue.push_back(std::move(smp));
                stdx::seq_cst_guard seqcst_tx;
                this->smp_queue_sz.value.fetch_add(1u, std::memory_order_relaxed);
            }

            void unsafe_notify_subscribers() noexcept{

                stdx::seq_cst_guard seqcst_tx;
                size_t old_sz = this->smp_queue_sz.value.exchange(0u, std::memory_order_relaxed);

                for (size_t i = 0u; i < old_sz; ++i){ //we are inferring the sz read to establish implicit memory ordering
                    exception_t smp_err = this->smp_queue[i]->release();

                    if (dg::network_exception::is_failed(smp_err)){
                        dg::network_log_stackdump::error(dg::network_exception::verbose(smp_err));
                    }
                }

                this->smp_queue.clear();
            }
    };

    //OK, this is too expensive, we have to increase the <incrementing_id> -> __uint128_t to chance better than RAM, whenever we have a question about random accuracy, we need to know whether it chances better than RAM or not
    class PacketIDGenerator: public virtual PacketIDGeneratorInterface{

        private:

            Address factory_addr;
            std::atomic<uint64_t> incrementing_id;

        public:

            PacketIDGenerator(Address factory_addr,
                              uint64_t incrementing_id) noexcept: factory_addr(std::move(factory_addr)),
                                                                  incrementing_id(incrementing_id){}

            auto get_id() noexcept -> GlobalIdentifier{

                GlobalIdentifier id = {};
                id.addr             = this->factory_addr;
                id.local_id         = std::make_pair(this->incrementing_id.fetch_add(1u, std::memory_order_relaxed), uint64_t{});

                return id;
            }
    };

    //OK
    class RandomPacketIDGenerator: public virtual PacketIDGeneratorInterface{

        private:

            Address factory_addr;
        
        public:

            RandomPacketIDGenerator(Address factory_addr) noexcept: factory_addr(std::move(factory_addr)){}

            auto get_id() noexcept -> GlobalIdentifier{

                GlobalIdentifier id = {};
                id.addr             = this->factory_addr;
                id.local_id         = std::make_pair(dg::network_randomizer::randomize_int<uint64_t>(), dg::network_randomizer::randomize_int<uint64_t>()); //increase locality of packet_id to help with the memory pattern (maybe or maybe not)

                return id;
            }
    };

    //OK
    class TemporalAbsoluteTimeoutInBoundGate: public virtual InBoundGateInterface{

        private:

            temporal_finite_unordered_map<GlobalIdentifier, std::chrono::time_point<std::chrono::steady_clock>> abstimeout_map;
            const std::chrono::nanoseconds abs_timeout_dur;
            std::unique_ptr<std::mutex> mtx;
            stdx::hdi_container<size_t> thru_sz_per_load;

        public:

            TemporalAbsoluteTimeoutInBoundGate(temporal_finite_unordered_map<GlobalIdentifier, std::chrono::time_point<std::chrono::steady_clock>> abstimeout_map,
                                               std::chrono::nanoseconds abs_timeout_dur,
                                               std::unique_ptr<std::mutex> mtx,
                                               stdx::hdi_container<size_t> thru_sz_per_load) noexcept: abstimeout_map(std::move(abstimeout_map)),
                                                                                                       abs_timeout_dur(abs_timeout_dur),
                                                                                                       mtx(std::move(mtx)),
                                                                                                       thru_sz_per_load(std::move(thru_sz_per_load)){}

            void thru(GlobalIdentifier * global_id_arr, size_t sz, exception_t * exception_arr) noexcept{

                if constexpr(DEBUG_MODE_FLAG){
                    if (sz > this->max_consume_size()){
                        dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                        std::abort();
                    }
                }

                stdx::xlock_guard<std::mutex> lck_grd(*this->mtx);

                std::chrono::time_point<std::chrono::steady_clock> now              = std::chrono::steady_clock::now();
                std::chrono::time_point<std::chrono::steady_clock> timeout_value    = now + this->abs_timeout_dur;

                for (size_t i = 0u; i < sz; ++i){
                    auto map_ptr = this->abstimeout_map.find(global_id_arr[i]);

                    if (map_ptr == this->abstimeout_map.end()){
                        auto [emplace_ptr, status] = this->abstimeout_map.try_emplace(global_id_arr[i], timeout_value);
                        dg::network_exception_handler::dg_assert(status);
                        exception_arr[i] = dg::network_exception::SUCCESS; 
                        continue;
                    }

                    if (map_ptr->second <= now){
                        exception_arr[i] = dg::network_exception::SOCKET_STREAM_TIMEOUT;
                        continue;                        
                    }

                    exception_arr[i] = dg::network_exception::SUCCESS;
                }
            }

            auto max_consume_size() noexcept -> size_t{

                return this->thru_sz_per_load.value;
            }        
    };

    //OK
    class RandomHashDistributedInBoundGate: public virtual InBoundGateInterface{

        private:

            std::unique_ptr<std::unique_ptr<InBoundGateInterface>[]> base_arr;
            size_t pow2_base_arr_sz;
            size_t keyvalue_feed_cap;
            size_t thru_sz_per_load;

        public:

            RandomHashDistributedInBoundGate(std::unique_ptr<std::unique_ptr<InBoundGateInterface>[]> base_arr,
                                             size_t pow2_base_arr_sz,
                                             size_t keyvalue_feed_cap,
                                             size_t thru_sz_per_load) noexcept: base_arr(std::move(base_arr)),
                                                                                pow2_base_arr_sz(pow2_base_arr_sz),
                                                                                keyvalue_feed_cap(keyvalue_feed_cap),
                                                                                thru_sz_per_load(thru_sz_per_load){}

            void thru(GlobalIdentifier * global_id_arr, size_t sz, exception_t * exception_arr) noexcept{

                if constexpr(DEBUG_MODE_FLAG){
                    if (sz > this->max_consume_size()){
                        dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                        std::abort();
                    }
                }

                auto feed_resolutor                 = InternalFeedResolutor{};
                feed_resolutor.dst                  = this->base_arr.get();

                size_t trimmed_keyvalue_feed_cap    = std::min(this->keyvalue_feed_cap, sz);
                size_t feeder_allocation_cost       = dg::network_producer_consumer::delvrsrv_kv_allocation_cost(&feed_resolutor, trimmed_keyvalue_feed_cap);
                dg::network_stack_allocation::NoExceptRawAllocation<char[]> feeder_mem(feeder_allocation_cost);
                auto feeder                         = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_kv_open_preallocated_raiihandle(&feed_resolutor, trimmed_keyvalue_feed_cap, feeder_mem.get()));

                std::fill(exception_arr, std::next(exception_arr, sz), dg::network_exception::SUCCESS);

                for (size_t i = 0u; i < sz; ++i){
                    size_t hashed_value         = dg::network_hash::hash_reflectible(global_id_arr[i]);
                    size_t partitioned_idx      = hashed_value & (this->pow2_base_arr_sz - 1u);
                    auto feed_arg               = InternalFeedArgument{};
                    feed_arg.id                 = global_id_arr[i];
                    feed_arg.bad_exception_ptr  = std::next(exception_arr, i); 

                    dg::network_producer_consumer::delvrsrv_kv_deliver(feeder.get(), partitioned_idx, feed_arg);
                }

            }

            auto max_consume_size() noexcept -> size_t{

                return this->thru_sz_per_load;
            }
        
        private:

            struct InternalFeedArgument{
                GlobalIdentifier id;
                exception_t * bad_exception_ptr;
            };

            struct InternalFeedResolutor: dg::network_producer_consumer::KVConsumerInterface<size_t, InternalFeedArgument>{

                std::unique_ptr<InBoundGateInterface> * dst;

                void push(const size_t& idx, std::move_iterator<InternalFeedArgument *> data_arr, size_t sz) noexcept{

                    InternalFeedArgument * base_data_arr = data_arr.base();

                    dg::network_stack_allocation::NoExceptAllocation<GlobalIdentifier[]> global_id_arr(sz);
                    dg::network_stack_allocation::NoExceptAllocation<exception_t[]> exception_arr(sz);

                    for (size_t i = 0u; i < sz; ++i){
                        global_id_arr[i] = base_data_arr[i].id;
                    }

                    this->dst[idx]->thru(global_id_arr.get(), sz, exception_arr.get());

                    for (size_t i = 0u; i < sz; ++i){
                        if (dg::network_exception::is_failed(exception_arr[i])){
                            *base_data_arr[i].bad_exception_ptr = exception_arr[i];
                        }
                    }
                }
            };
    };

    //OK
    class TemporalBlackListGate: public virtual BlackListGateInterface{

        private:

            temporal_switching_bloom_filter<GlobalIdentifier> blacklist_set;
            std::unique_ptr<std::mutex> mtx;
            stdx::hdi_container<size_t> thru_sz_per_load;

        public:

            TemporalBlackListGate(temporal_switching_bloom_filter<GlobalIdentifier> blacklist_set,
                                  std::unique_ptr<std::mutex> mtx,
                                  stdx::hdi_container<size_t> thru_sz_per_load) noexcept: blacklist_set(std::move(blacklist_set)),
                                                                                          mtx(std::move(mtx)),
                                                                                          thru_sz_per_load(thru_sz_per_load){}

            void thru(GlobalIdentifier * global_id_arr, size_t sz, exception_t * exception_arr) noexcept{

                stdx::xlock_guard<std::mutex> lck_grd(*this->mtx);

                for (size_t i = 0u; i < sz; ++i){
                    if (this->blacklist_set.not_contains(global_id_arr[i])){
                        exception_arr[i] = dg::network_exception::SUCCESS;
                    } else{
                        exception_arr[i] = dg::network_exception::SOCKET_STREAM_MIGHT_BE_BLACKLISTED;
                    }
                }
            }

            void blacklist(GlobalIdentifier * global_id_arr, size_t sz, exception_t * exception_arr) noexcept{

                if constexpr(DEBUG_MODE_FLAG){
                    if (sz > this->max_consume_size()){
                        dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                        std::abort();        
                    }
                }

                stdx::xlock_guard<std::mutex> lck_grd(*this->mtx);

                for (size_t i = 0u; i < sz; ++i){
                    this->blacklist_set.insert(global_id_arr[i]);
                    exception_arr[i] = dg::network_exception::SUCCESS;
                }
            }

            auto max_consume_size() noexcept -> size_t{

                return this->thru_sz_per_load.value;
            }
    };

    //OK
    class RandomHashDistributedBlackListGate: public virtual BlackListGateInterface{

        private:

            std::unique_ptr<std::unique_ptr<BlackListGateInterface>[]> base_arr;
            size_t pow2_base_arr_sz;
            size_t keyvalue_feed_cap;
            size_t consume_sz_per_load;
        
        public:

            RandomHashDistributedBlackListGate(std::unique_ptr<std::unique_ptr<BlackListGateInterface>[]> base_arr,
                                               size_t pow2_base_arr_sz,
                                               size_t keyvalue_feed_cap,
                                               size_t consume_sz_per_load) noexcept: base_arr(std::move(base_arr)),
                                                                                     pow2_base_arr_sz(pow2_base_arr_sz),
                                                                                     keyvalue_feed_cap(keyvalue_feed_cap),
                                                                                     consume_sz_per_load(consume_sz_per_load){}

            void thru(GlobalIdentifier * global_id_arr, size_t sz, exception_t * exception_arr) noexcept{

                auto thru_feed_resolutor            = InternalThruFeedResolutor{};
                thru_feed_resolutor.dst             = this->base_arr.get();

                size_t trimmed_keyvalue_feed_cap    = std::min(this->keyvalue_feed_cap, sz);
                size_t thru_feeder_allocation_cost  = dg::network_producer_consumer::delvrsrv_kv_allocation_cost(&thru_feed_resolutor, trimmed_keyvalue_feed_cap);
                dg::network_stack_allocation::NoExceptRawAllocation<char[]> thru_feeder_mem(thru_feeder_allocation_cost);
                auto thru_feeder                    = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_kv_open_preallocated_raiihandle(&thru_feed_resolutor, trimmed_keyvalue_feed_cap, thru_feeder_mem.get()));

                std::fill(exception_arr, std::next(exception_arr, sz), dg::network_exception::SUCCESS);

                for (size_t i = 0u; i < sz; ++i){
                    size_t hashed_value             = dg::network_hash::hash_reflectible(global_id_arr[i]);
                    size_t partitioned_idx          = hashed_value & (this->pow2_base_arr_sz - 1u);
                    auto thru_feed_arg              = InternalThruFeedArgument{};
                    thru_feed_arg.id                = global_id_arr[i];
                    thru_feed_arg.bad_exception_ptr = std::next(exception_arr, i);

                    dg::network_producer_consumer::delvrsrv_kv_deliver(thru_feeder.get(), partitioned_idx, thru_feed_arg);
                }
            }

            void blacklist(GlobalIdentifier * global_id_arr, size_t sz, exception_t * exception_arr) noexcept{

                if constexpr(DEBUG_MODE_FLAG){
                    if (sz > this->max_consume_size()){
                        dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                        std::abort();
                    }
                }

                auto blacklist_feed_resolutor           = InternalBlackListFeedResolutor{};
                blacklist_feed_resolutor.dst            = this->base_arr.get();

                size_t trimmed_keyvalue_feed_cap        = std::min(this->keyvalue_feed_cap, sz);
                size_t blacklist_feeder_allocation_cost = dg::network_producer_consumer::delvrsrv_kv_allocation_cost(&blacklist_feed_resolutor, trimmed_keyvalue_feed_cap);
                dg::network_stack_allocation::NoExceptRawAllocation<char[]> blacklist_feeder_mem(blacklist_feeder_allocation_cost);
                auto blacklist_feeder                   = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_kv_open_preallocated_raiihandle(&blacklist_feed_resolutor, trimmed_keyvalue_feed_cap, blacklist_feeder_mem.get()));

                std::fill(exception_arr, std::next(exception_arr, sz), dg::network_exception::SUCCESS);

                for (size_t i = 0u ; i < sz; ++i){
                    size_t hashed_value                     = dg::network_hash::hash_reflectible(global_id_arr[i]);
                    size_t partitioned_idx                  = hashed_value & (this->pow2_base_arr_sz - 1u);
                    auto blacklist_feed_arg                 = InternalBlackListFeedArgument{};
                    blacklist_feed_arg.id                   = global_id_arr[i];
                    blacklist_feed_arg.bad_exception_ptr    = std::next(exception_arr, i); 

                    dg::network_producer_consumer::delvrsrv_kv_deliver(blacklist_feeder.get(), partitioned_idx, blacklist_feed_arg);
                }
            }

            auto max_consume_size() noexcept -> size_t{

                return this->consume_sz_per_load;
            }
        
        private:

            struct InternalThruFeedArgument{
                GlobalIdentifier id;
                exception_t * bad_exception_ptr;
            };

            struct InternalThruFeedResolutor: dg::network_producer_consumer::KVConsumerInterface<size_t, InternalThruFeedArgument>{

                std::unique_ptr<BlackListGateInterface> * dst;

                void push(const size_t& idx, std::move_iterator<InternalThruFeedArgument *> data_arr, size_t sz) noexcept{

                    InternalThruFeedArgument * base_data_arr = data_arr.base();

                    dg::network_stack_allocation::NoExceptAllocation<exception_t[]> exception_arr(sz);
                    dg::network_stack_allocation::NoExceptAllocation<GlobalIdentifier[]> global_id_arr(sz);

                    for (size_t i = 0u; i < sz; ++i){
                        global_id_arr[i] = base_data_arr[i].id;
                    }

                    this->dst[idx]->thru(global_id_arr.get(), sz, exception_arr.get());

                    for (size_t i = 0u; i < sz; ++i){
                        if (dg::network_exception::is_failed(exception_arr[i])){
                            *base_data_arr[i].bad_exception_ptr = exception_arr[i];
                        }
                    }
                }
            };

            struct InternalBlackListFeedArgument{
                GlobalIdentifier id;
                exception_t * bad_exception_ptr;
            };

            struct InternalBlackListFeedResolutor: dg::network_producer_consumer::KVConsumerInterface<size_t, InternalBlackListFeedArgument>{

                std::unique_ptr<BlackListGateInterface> * dst;

                void push(const size_t& idx, std::move_iterator<InternalBlackListFeedArgument *> data_arr, size_t sz) noexcept{

                    InternalBlackListFeedArgument * base_data_arr = data_arr.base();

                    dg::network_stack_allocation::NoExceptAllocation<exception_t[]> exception_arr(sz);
                    dg::network_stack_allocation::NoExceptAllocation<GlobalIdentifier[]> global_id_arr(sz);

                    for (size_t i = 0u; i < sz; ++i){
                        global_id_arr[i] = base_data_arr[i].id;
                    }

                    this->dst[idx]->blacklist(global_id_arr.get(), sz, exception_arr.get());

                    for (size_t i = 0u; i < sz; ++i){
                        if (dg::network_exception::is_failed(exception_arr[i])){
                            *base_data_arr[i].bad_exception_ptr = exception_arr[i];
                        }
                    }
                }
            };
    };

    //OK
    class Packetizer: public virtual PacketizerInterface{

        private:

            std::unique_ptr<PacketIDGeneratorInterface> packet_id_gen;
            const size_t segment_bsz_pow2_exponent;
            const size_t pow2_max_stream_bsz;
            bool has_mm_integrity;

        public:

            Packetizer(std::unique_ptr<PacketIDGeneratorInterface> packet_id_gen,
                       size_t segment_bsz_pow2_exponent,
                       size_t pow2_max_stream_bsz,
                       bool has_mm_integrity) noexcept: packet_id_gen(std::move(packet_id_gen)),
                                                        segment_bsz_pow2_exponent(segment_bsz_pow2_exponent),
                                                        pow2_max_stream_bsz(pow2_max_stream_bsz),
                                                        has_mm_integrity(has_mm_integrity){}

            auto packetize(dg::string&& buf) noexcept -> std::expected<dg::vector<PacketSegment>, exception_t>{

                if (buf.size() > this->pow2_max_stream_bsz){
                    return std::unexpected(dg::network_exception::SOCKET_STREAM_BAD_BUFFER_LENGTH);
                }

                size_t segment_bsz      = size_t{1} << this->segment_bsz_pow2_exponent;
                size_t segment_even_sz  = buf.size() >> this->segment_bsz_pow2_exponent; //alright it seems silly but we have to not use division under every circumstances, a division operation is very costly on CPU, it is equivalent to a L2 cache fetch instruction (this will bring our program -> python speed)
                size_t segment_odd_sz   = size_t{(static_cast<size_t>(buf.size()) & (segment_bsz - 1u)) != 0u};
                size_t segment_sz       = segment_even_sz + segment_odd_sz;

                std::expected<dg::vector<PacketSegment>, exception_t> rs = dg::network_exception::cstyle_initialize<dg::vector<PacketSegment>>(std::max(size_t{1}, segment_sz));

                if (!rs.has_value()){
                    return std::unexpected(rs.error());
                }

                GlobalIdentifier pkt_stream_id = this->packet_id_gen->get_id();

                if (segment_sz == 0u){
                    PacketSegment segment           = {};
                    segment.buf                     = {};
                    segment.id                      = pkt_stream_id;
                    segment.segment_idx             = 0u; //this is logically incorrect, we are not protected by range bro anymore
                    segment.segment_sz              = 0u;
                    segment.mm_integrity_value      = this->integrity_encode(segment.buf, this->has_mm_integrity);
                    segment.has_mm_integrity_value  = this->has_mm_integrity;
                    rs.value()[0]                   = std::move(segment);

                    return rs;
                }

                if (segment_sz == 1u){ //premature very useful optimization, effectively skip 1 buffer iteration
                    PacketSegment segment           = {};
                    segment.buf                     = std::move(buf);
                    segment.id                      = pkt_stream_id;
                    segment.segment_idx             = 0u;
                    segment.segment_sz              = 1u;
                    segment.mm_integrity_value      = this->integrity_encode(segment.buf, this->has_mm_integrity);
                    segment.has_mm_integrity_value  = this->has_mm_integrity;
                    rs.value()[0]                   = std::move(segment);

                    return rs;
                }

                uint64_t integrity_value = this->integrity_encode(buf, this->has_mm_integrity); 

                for (size_t i = 0u; i < segment_sz; ++i){
                    size_t first                                    = segment_bsz * i;
                    size_t last                                     = std::min(segment_bsz * (i + 1), buf.size()); 
                    PacketSegment segment                           = {};
                    segment.id                                      = pkt_stream_id;
                    segment.segment_idx                             = i;
                    segment.segment_sz                              = segment_sz;
                    segment.mm_integrity_value                      = integrity_value;
                    segment.has_mm_integrity_value                  = this->has_mm_integrity;
                    std::expected<dg::string, exception_t> app_buf  = dg::network_exception::cstyle_initialize<dg::string>((last - first), 0); 

                    if (!app_buf.has_value()){
                        return std::unexpected(app_buf.error()); //leaking ids
                    }

                    std::copy(std::next(buf.begin(), first), std::next(buf.begin(), last), app_buf.value().begin());

                    segment.buf                                     = std::move(app_buf.value());
                    rs.value()[i]                                   = std::move(segment);
                }

                return rs;
            }

            auto segment_byte_size() const noexcept -> size_t{

                return size_t{1} << this->segment_bsz_pow2_exponent;
            }

            auto max_packet_size() const noexcept -> size_t{

                return this->pow2_max_stream_bsz;
            }

            auto max_segment_count() const noexcept -> size_t{

                return std::max(size_t{1}, this->pow2_max_stream_bsz >> this->segment_bsz_pow2_exponent);
            }
        
        private:

            template <class ...Args>
            auto integrity_encode(const std::basic_string<Args...>& buf, bool has_integrity_encode) noexcept -> uint64_t{

                if (!has_integrity_encode){
                    return 0u;
                }

                return dg::network_hash::murmur_hash(buf.data(), buf.size(), PACKET_INTEGRITY_SECRET);
            }
    };

    //OK
    struct EntranceEntry{
        std::chrono::time_point<std::chrono::steady_clock> timestamp; //steady clock is a clock that never goes back in time, only true if we are on the same thread of execution, to avoid exotic error or errors that chance worse than RAM, we are to make sure that our queue state is accurate
        GlobalIdentifier key;
        __uint128_t entry_id;
    };

    //OK
    class EntranceController: public virtual EntranceControllerInterface{

        private:

            dg::pow2_cyclic_queue<EntranceEntry> entrance_entry_pq; //no exhausted container
            const size_t entrance_entry_pq_cap; 
            dg::unordered_unstable_map<GlobalIdentifier, __uint128_t> key_id_map; //no exhausted container
            const size_t key_id_map_cap;
            __uint128_t id_ticker;
            const std::chrono::nanoseconds expiry_period;
            std::unique_ptr<std::mutex> mtx;
            stdx::hdi_container<size_t> tick_sz_per_load;

        public:

            EntranceController(dg::pow2_cyclic_queue<EntranceEntry> entrance_entry_pq,
                               size_t entrance_entry_pq_cap,
                               dg::unordered_unstable_map<GlobalIdentifier, __uint128_t> key_id_map,
                               size_t key_id_map_cap,
                               __uint128_t id_ticker,
                               std::chrono::nanoseconds expiry_period,
                               std::unique_ptr<std::mutex> mtx,
                               stdx::hdi_container<size_t> tick_sz_per_load) noexcept: entrance_entry_pq(std::move(entrance_entry_pq)),
                                                                                       entrance_entry_pq_cap(entrance_entry_pq_cap),
                                                                                       key_id_map(std::move(key_id_map)),
                                                                                       key_id_map_cap(key_id_map_cap),
                                                                                       id_ticker(id_ticker),
                                                                                       expiry_period(std::move(expiry_period)),
                                                                                       mtx(std::move(mtx)),
                                                                                       tick_sz_per_load(std::move(tick_sz_per_load)){}

            void tick(GlobalIdentifier * global_id_arr, size_t sz, exception_t * exception_arr) noexcept{

                if constexpr(DEBUG_MODE_FLAG){
                    if (sz > this->max_consume_size()){
                        dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION)); //exceeding clock + queue accuracy
                        std::abort();
                    }
                }

                stdx::xlock_guard<std::mutex> lck_grd(*this->mtx);

                auto now = this->get_now();

                for (size_t i = 0u; i < sz; ++i){
                    if (this->entrance_entry_pq.size() == this->entrance_entry_pq_cap){
                        exception_arr[i] = dg::network_exception::QUEUE_FULL;
                        continue;
                    }

                    if (this->key_id_map.size() == this->key_id_map_cap){
                        exception_arr[i] = dg::network_exception::QUEUE_FULL;
                        continue;
                    }

                    auto entry      = EntranceEntry{};
                    entry.timestamp = now;
                    entry.key       = global_id_arr[i];
                    entry.entry_id  = this->id_ticker++;

                    this->entrance_entry_pq.push_back(entry);
                    this->key_id_map[global_id_arr[i]] = entry.entry_id;
                    exception_arr[i] = dg::network_exception::SUCCESS;
                }
            }

            //assume finite sorted queue of entrance_entry_pq
            //assume key_id_map points to the lastest GlobalIdentifier guy in the entrance_entry_pq

            void get_expired_id(GlobalIdentifier * output_arr, size_t& sz, size_t cap) noexcept{ //bad

                stdx::xlock_guard<std::mutex> lck_grd(*this->mtx);

                auto bar_time   = std::chrono::steady_clock::now() - this->expiry_period;
                sz              = 0u;

                while (true){
                    if (sz == cap){
                        return;
                    }

                    if (this->entrance_entry_pq.empty()){
                        return;
                    }

                    if (this->entrance_entry_pq.front().timestamp > bar_time){ //bad
                        return;
                    }

                    EntranceEntry entry = this->entrance_entry_pq.front();
                    this->entrance_entry_pq.pop_front();
                    auto map_ptr        = this->key_id_map.find(entry.key); 

                    if constexpr(DEBUG_MODE_FLAG){
                        if (map_ptr == this->key_id_map.end()){
                            dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                            std::abort();
                        }
                    }

                    if (map_ptr->second == entry.entry_id){
                        output_arr[sz++] = entry.key;
                        this->key_id_map.erase(map_ptr);
                    }
                }
            }

            auto max_consume_size() noexcept -> size_t{

                return this->tick_sz_per_load.value;
            }

        private:

            auto get_now() const noexcept -> std::chrono::time_point<std::chrono::steady_clock>{

                if (this->entrance_entry_pq.empty()){
                    return std::chrono::steady_clock::now();
                }

                return std::max(this->entrance_entry_pq.back().timestamp, std::chrono::steady_clock::now());
            }
    };

    //OK
    class RandomHashDistributedEntranceController: public virtual EntranceControllerInterface{

        private:

            std::unique_ptr<std::unique_ptr<EntranceControllerInterface>[]> base_arr;
            const size_t pow2_base_arr_sz;
            const size_t keyvalue_feed_cap;
            const size_t zero_bounce_sz;
            std::unique_ptr<ProducerDrainerPredicateInterface> drainer_pred;
            const size_t drain_peek_cap_per_container;
            const size_t max_tick_per_load; 

        public:

            RandomHashDistributedEntranceController(std::unique_ptr<std::unique_ptr<EntranceControllerInterface>[]> base_arr,
                                                    size_t pow2_base_arr_sz,
                                                    size_t keyvalue_feed_cap,
                                                    size_t zero_bounce_sz,
                                                    std::unique_ptr<ProducerDrainerPredicateInterface> drainer_pred,
                                                    size_t drain_peek_cap_per_container,
                                                    size_t max_tick_per_load) noexcept: base_arr(std::move(base_arr)),
                                                                                        pow2_base_arr_sz(pow2_base_arr_sz),
                                                                                        keyvalue_feed_cap(keyvalue_feed_cap),
                                                                                        zero_bounce_sz(zero_bounce_sz),
                                                                                        drainer_pred(std::move(drainer_pred)),
                                                                                        drain_peek_cap_per_container(drain_peek_cap_per_container),
                                                                                        max_tick_per_load(max_tick_per_load){}

            void tick(GlobalIdentifier * global_id_arr, size_t sz, exception_t * exception_arr) noexcept{

                auto feed_resolutor                 = InternalFeedResolutor{};
                feed_resolutor.dst                  = this->base_arr.get();

                size_t trimmed_keyvalue_feed_cap    = std::min(this->keyvalue_feed_cap, sz);
                size_t feed_allocation_cost         = dg::network_producer_consumer::delvrsrv_kv_allocation_cost(&feed_resolutor, trimmed_keyvalue_feed_cap);
                dg::network_stack_allocation::NoExceptRawAllocation<char[]> feed_mem(feed_allocation_cost);
                auto feeder                         = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_kv_open_preallocated_raiihandle(&feed_resolutor, trimmed_keyvalue_feed_cap, feed_mem.get()));

                std::fill(exception_arr, std::next(exception_arr, sz), dg::network_exception::SUCCESS);

                for (size_t i = 0u; i < sz; ++i){
                    size_t hashed_value     = dg::network_hash::hash_reflectible(global_id_arr[i]);
                    size_t partitioned_idx  = hashed_value & (this->pow2_base_arr_sz - 1u); //bad
                    auto feed_arg           = InternalFeedArgument{};
                    feed_arg.id             = global_id_arr[i];
                    feed_arg.failed_err_ptr = std::next(exception_arr, i);

                    dg::network_producer_consumer::delvrsrv_kv_deliver(feeder.get(), partitioned_idx, feed_arg);
                }
            }

            void get_expired_id(GlobalIdentifier * output_arr, size_t& sz, size_t cap) noexcept{

                if (this->drainer_pred->is_should_drain()){
                    this->internal_drain(output_arr, sz, cap);
                    this->drainer_pred->reset();
                } else{
                    this->internal_curious_pop(output_arr, sz, cap);
                }
            }

            auto max_consume_size() noexcept -> size_t{

                return this->max_tick_per_load;
            }
        
        private:

            struct InternalFeedArgument{
                GlobalIdentifier id;
                exception_t * failed_err_ptr;
            };

            struct InternalFeedResolutor: dg::network_producer_consumer::KVConsumerInterface<size_t, InternalFeedArgument>{

                std::unique_ptr<EntranceControllerInterface> * dst;

                void push(const size_t& idx, std::move_iterator<InternalFeedArgument *> data_arr, size_t sz) noexcept{

                    InternalFeedArgument * base_data_arr = data_arr.base();

                    dg::network_stack_allocation::NoExceptAllocation<GlobalIdentifier[]> global_id_arr(sz);
                    dg::network_stack_allocation::NoExceptAllocation<exception_t[]> exception_arr(sz);

                    for (size_t i = 0u; i < sz; ++i){
                        global_id_arr[i] = base_data_arr[i].id;
                    }

                    this->dst[idx]->tick(global_id_arr.get(), sz, exception_arr.get());

                    for (size_t i = 0u; i < sz; ++i){
                        if (dg::network_exception::is_failed(exception_arr[i])){
                            *base_data_arr[i].failed_err_ptr = exception_arr[i];
                        }
                    }
                }
            };

            void internal_curious_pop(GlobalIdentifier * output_arr, size_t& sz, size_t cap) noexcept{

                sz = 0u;

                for (size_t i = 0u; i < this->zero_bounce_sz; ++i){ //we are bouncing to keep the total expired id under the reactor's threshold
                    size_t random_value = dg::network_randomizer::randomize_int<size_t>();
                    size_t idx          = random_value & (this->pow2_base_arr_sz - 1u);
                    this->base_arr[idx]->get_expired_id(output_arr, sz, cap);
                    
                    if (sz != 0u){
                        return;
                    }
                }
            }

            void internal_drain(GlobalIdentifier * output_arr, size_t& sz, size_t cap) noexcept{

                //we are draining the leftover under the reactor's threshold
                //we are increasing fairness of probing by using random start

                sz = 0u;
                GlobalIdentifier * current_output_arr = output_arr;
                size_t current_cap  = cap;
                size_t start_seek   = dg::network_randomizer::randomize_int<size_t>() >> 1; //alright this is one of the myths, ceause the array is already pow2, wraparound should work, because its aligned at 0 

                for (size_t i = 0u; i < this->pow2_base_arr_sz; ++i){
                    if (current_cap == 0u){
                        return;
                    }

                    size_t peeking_cap  = std::min(current_cap, this->drain_peek_cap_per_container);
                    size_t adjusted_idx = (start_seek + i) & (this->pow2_base_arr_sz - 1u);
                    size_t current_sz   = {};
                    this->base_arr[adjusted_idx]->get_expired_id(current_output_arr, current_sz, peeking_cap);
                    current_cap         -= current_sz;
                    sz                  += current_sz;
                    std::advance(current_output_arr, current_sz);
                }
            }

    };

    //OK
    class ExhaustionControlledEntranceController: public virtual EntranceControllerInterface{

        private:

            std::unique_ptr<EntranceControllerInterface> base;
            std::shared_ptr<dg::network_concurrency_infretry_x::ExecutorInterface> executor;
            std::shared_ptr<ExhaustionControllerInterface> exhaustion_controller;

        public:

            ExhaustionControlledEntranceController(std::unique_ptr<EntranceControllerInterface> base,
                                                   std::shared_ptr<dg::network_concurrency_infretry_x::ExecutorInterface> executor,
                                                   std::shared_ptr<ExhaustionControllerInterface> exhaustion_controller) noexcept: base(std::move(base)),
                                                                                                                                   executor(std::move(executor)),
                                                                                                                                   exhaustion_controller(std::move(exhaustion_controller)){}

            void tick(GlobalIdentifier * global_id_arr, size_t sz, exception_t * exception_arr) noexcept{

                GlobalIdentifier * first_id_ptr     = global_id_arr;
                GlobalIdentifier * last_id_ptr      = std::next(first_id_ptr, sz);
                exception_t * first_exception_ptr   = exception_arr;
                exception_t * last_exception_ptr    = std::next(exception_arr, sz);
                size_t sliding_window_sz            = sz;

                //let's try to grasp what's going on here
                //we are protected by range bro
                //assume we are to resolve every QUEUE_FULL, our state is correct at all time
                //we are to maintain the state of peeking the next sliding_window of QUEUE_FULL
                //such is we are to decay one state -> another state of lesser std::distance(first_id_ptr, last_id_ptr)

                auto task = [&, this]() noexcept{
                    this->base->tick(first_id_ptr, sliding_window_sz, first_exception_ptr);
                    size_t queueing_sz                          = std::count(first_exception_ptr, last_exception_ptr, dg::network_exception::QUEUE_FULL);
                    exception_t err                             = this->exhaustion_controller->update_waiting_size(queueing_sz);

                    if (dg::network_exception::is_failed(err)){
                        dg::network_log_stackdump::error_fast_optional(dg::network_exception::verbose(err));
                    }

                    exception_t * first_retriable_exception_ptr = std::find(first_exception_ptr, last_exception_ptr, dg::network_exception::QUEUE_FULL);
                    exception_t * last_retriable_exception_ptr  = std::find_if(first_retriable_exception_ptr, last_exception_ptr, [](exception_t err){return err != dg::network_exception::QUEUE_FULL;});
                    sliding_window_sz                           = std::distance(first_retriable_exception_ptr, last_retriable_exception_ptr);
                    size_t relative_offset                      = std::distance(first_exception_ptr, first_retriable_exception_ptr);

                    std::advance(first_id_ptr, relative_offset);
                    std::advance(first_exception_ptr, relative_offset);

                    return !this->exhaustion_controller->is_should_wait() || first_id_ptr == last_id_ptr;
                };

                auto virtual_task = dg::network_concurrency_infretry_x::ExecutableWrapper<decltype(task)>(std::move(task));
                this->executor->exec(virtual_task);
            }

            void get_expired_id(GlobalIdentifier * output_arr, size_t& sz, size_t cap) noexcept{

                this->base->get_expired_id(output_arr, sz, cap);
            }

            auto max_consume_size() noexcept -> size_t{

                return this->base->max_consume_size();
            }
    };

    //OK
    class PacketAssembler: public virtual PacketAssemblerInterface{

        private:

            dg::unordered_unstable_map<GlobalIdentifier, AssembledPacket> packet_map;
            size_t packet_map_cap;
            size_t global_packet_segment_cap;
            size_t global_packet_segment_counter;
            size_t max_segment_sz_per_stream; 
            std::unique_ptr<std::mutex> mtx;
            stdx::hdi_container<size_t> consume_sz_per_load;

        public:

            PacketAssembler(dg::unordered_unstable_map<GlobalIdentifier, AssembledPacket> packet_map,
                            size_t packet_map_cap,
                            size_t global_packet_segment_cap,
                            size_t global_packet_segment_counter,
                            size_t max_segment_sz_per_stream,
                            std::unique_ptr<std::mutex> mtx,
                            stdx::hdi_container<size_t> consume_sz_per_load) noexcept: packet_map(std::move(packet_map)),
                                                                                       packet_map_cap(packet_map_cap),
                                                                                       global_packet_segment_cap(global_packet_segment_cap),
                                                                                       global_packet_segment_counter(global_packet_segment_counter),
                                                                                       max_segment_sz_per_stream(max_segment_sz_per_stream),
                                                                                       mtx(std::move(mtx)),
                                                                                       consume_sz_per_load(std::move(consume_sz_per_load)){}

            void assemble(std::move_iterator<PacketSegment *> segment_arr, size_t sz, std::expected<AssembledPacket, exception_t> * assembled_arr) noexcept{

                if constexpr(DEBUG_MODE_FLAG){
                    if (sz > this->max_consume_size()){
                        dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                        std::abort();
                    }
                }

                stdx::xlock_guard<std::mutex> lck_grd(*this->mtx);
                PacketSegment * base_segment_arr = segment_arr.base();

                for (size_t i = 0u; i < sz; ++i){
                    if (base_segment_arr[i].segment_sz == 0u){
                        assembled_arr[i] = this->make_empty_assembled_packet(0u);
                        continue;
                    }

                    auto map_ptr = this->packet_map.find(base_segment_arr[i].id);

                    if (map_ptr == this->packet_map.end()){
                        if (base_segment_arr[i].segment_sz > this->max_segment_sz_per_stream){
                            assembled_arr[i] = std::unexpected(dg::network_exception::SOCKET_STREAM_BAD_SEGMENT_SIZE);
                            continue;
                        }

                        if (this->packet_map.size() == this->packet_map_cap){
                            assembled_arr[i] = std::unexpected(dg::network_exception::QUEUE_FULL);
                            continue;
                        }

                        if (this->global_packet_segment_counter + base_segment_arr[i].segment_sz > this->global_packet_segment_cap){
                            assembled_arr[i] = std::unexpected(dg::network_exception::QUEUE_FULL);
                            continue;
                        }

                        std::expected<AssembledPacket, exception_t> waiting_pkt = this->make_empty_assembled_packet(base_segment_arr[i].segment_sz);

                        if (!waiting_pkt.has_value()){
                            assembled_arr[i] = std::unexpected(waiting_pkt.error());
                            continue;
                        }

                        auto [emplace_ptr, status]          = this->packet_map.try_emplace(base_segment_arr[i].id, std::move(waiting_pkt.value()));
                        dg::network_exception_handler::dg_assert(status);
                        map_ptr                             = emplace_ptr;
                        this->global_packet_segment_counter += base_segment_arr[i].segment_sz; 
                    }

                    //this is fine, because it is object <dig_in> operator = (object&&), there wont be issues, yet we want to make this explicit

                    size_t segment_idx                      = base_segment_arr[i].segment_idx;
                    map_ptr->second.data[segment_idx]       = std::move(base_segment_arr[i]);
                    map_ptr->second.collected_segment_sz    += 1;

                    if (map_ptr->second.collected_segment_sz == map_ptr->second.total_segment_sz){
                        this->global_packet_segment_counter -= map_ptr->second.total_segment_sz; 
                        assembled_arr[i]                    = std::move(map_ptr->second);
                        this->packet_map.erase(map_ptr);
                    } else{
                        assembled_arr[i] = std::unexpected(dg::network_exception::SOCKET_STREAM_SEGMENT_FILLING);
                    }
                }
            }

            void destroy(GlobalIdentifier * id_arr, size_t sz) noexcept{

                if constexpr(DEBUG_MODE_FLAG){
                    if (sz > this->max_consume_size()){
                        dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                        std::abort();
                    }
                }

                stdx::xlock_guard<std::mutex> lck_grd(*this->mtx);

                for (size_t i = 0u; i < sz; ++i){
                    auto map_ptr = this->packet_map.find(id_arr[i]);

                    if (map_ptr == this->packet_map.end()){
                        continue;
                    }

                    this->global_packet_segment_counter -= map_ptr->second.total_segment_sz;
                    this->packet_map.erase(map_ptr);
                }
            }

            auto max_consume_size() noexcept -> size_t{

                return this->consume_sz_per_load.value;
            }

        private:

            auto make_empty_assembled_packet(size_t segment_sz) noexcept -> std::expected<AssembledPacket, exception_t>{

                auto vec            = dg::network_exception::cstyle_initialize<dg::vector<PacketSegment>>(segment_sz);

                if (!vec.has_value()){
                    return std::unexpected(vec.error());
                }

                size_t collected    = 0u;
                size_t total        = segment_sz;

                return AssembledPacket{std::move(vec.value()), collected, total};
            }
    };

    //OK
    class RandomHashDistributedPacketAssembler: public virtual PacketAssemblerInterface{

        private:

            std::unique_ptr<std::unique_ptr<PacketAssemblerInterface>[]> base_arr;
            const size_t pow2_base_arr_sz;
            const size_t keyvalue_feed_cap;
            const size_t consume_sz_per_load;

        public:

            RandomHashDistributedPacketAssembler(std::unique_ptr<std::unique_ptr<PacketAssemblerInterface>[]> base_arr,
                                                 size_t pow2_base_arr_sz,
                                                 size_t keyvalue_feed_cap,
                                                 size_t consume_sz_per_load) noexcept: base_arr(std::move(base_arr)),
                                                                                       pow2_base_arr_sz(pow2_base_arr_sz),
                                                                                       keyvalue_feed_cap(keyvalue_feed_cap),
                                                                                       consume_sz_per_load(consume_sz_per_load){}

            void assemble(std::move_iterator<PacketSegment *> segment_arr, size_t sz, std::expected<AssembledPacket, exception_t> * assembled_arr) noexcept{

                if constexpr(DEBUG_MODE_FLAG){
                    if (sz > this->max_consume_size()){
                        dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                        std::abort();
                    }
                }

                PacketSegment * base_segment_arr    = segment_arr.base();

                auto feed_resolutor                 = InternalAssembleFeedResolutor{};
                feed_resolutor.dst                  = this->base_arr.get(); 

                size_t trimmed_keyvalue_feed_cap    = std::min(this->keyvalue_feed_cap, sz);
                size_t feed_allocation_cost         = dg::network_producer_consumer::delvrsrv_kv_allocation_cost(&feed_resolutor, trimmed_keyvalue_feed_cap);
                dg::network_stack_allocation::NoExceptRawAllocation<char[]> feed_mem(feed_allocation_cost);
                auto feeder                         = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_kv_open_preallocated_raiihandle(&feed_resolutor, trimmed_keyvalue_feed_cap, feed_mem.get()));

                for (size_t i = 0u; i < sz; ++i){
                    size_t hashed_value             = dg::network_hash::hash_reflectible(base_segment_arr[i].id);
                    size_t partitioned_idx          = hashed_value & (this->pow2_base_arr_sz - 1u);

                    auto feed_arg                   = InternalAssembleFeedArgument{}; //bad
                    feed_arg.segment                = std::move(base_segment_arr[i]);
                    feed_arg.fallback_segment_ptr   = std::next(base_segment_arr, i);
                    feed_arg.rs                     = std::next(assembled_arr, i); 

                    dg::network_producer_consumer::delvrsrv_kv_deliver(feeder.get(), partitioned_idx, std::move(feed_arg));
                }
            }

            void destroy(GlobalIdentifier * id_arr, size_t sz) noexcept{

                if constexpr(DEBUG_MODE_FLAG){
                    if (sz > this->max_consume_size()){
                        dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                        std::abort();
                    }
                }

                auto feed_resolutor                 = InternalDestroyFeedResolutor{};
                feed_resolutor.dst                  = this->base_arr.get();

                size_t trimmed_keyvalue_feed_cap    = std::min(this->keyvalue_feed_cap, sz);
                size_t feed_allocation_cost         = dg::network_producer_consumer::delvrsrv_kv_allocation_cost(&feed_resolutor, trimmed_keyvalue_feed_cap);
                dg::network_stack_allocation::NoExceptRawAllocation<char[]> feed_mem(feed_allocation_cost);
                auto feeder                         = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_kv_open_preallocated_raiihandle(&feed_resolutor, trimmed_keyvalue_feed_cap, feed_mem.get()));

                for (size_t i = 0u; i < sz; ++i){
                    size_t hashed_value             = dg::network_hash::hash_reflectible(id_arr[i]);
                    size_t partitioned_idx          = hashed_value & (this->pow2_base_arr_sz - 1u);

                    dg::network_producer_consumer::delvrsrv_kv_deliver(feeder.get(), partitioned_idx, id_arr[i]);
                } 
            }

            auto max_consume_size() noexcept -> size_t{

                return this->consume_sz_per_load;
            }
        
        private:

            struct InternalAssembleFeedArgument{
                PacketSegment segment;
                PacketSegment * fallback_segment_ptr;
                std::expected<AssembledPacket, exception_t> * rs;
            };

            struct InternalAssembleFeedResolutor: dg::network_producer_consumer::KVConsumerInterface<size_t, InternalAssembleFeedArgument>{

                std::unique_ptr<PacketAssemblerInterface> * dst;

                void push(const size_t& idx, std::move_iterator<InternalAssembleFeedArgument *> data_arr, size_t sz) noexcept{

                    InternalAssembleFeedArgument * base_data_arr = data_arr.base();

                    dg::network_stack_allocation::NoExceptAllocation<PacketSegment[]> segment_arr(sz);
                    dg::network_stack_allocation::NoExceptAllocation<std::expected<AssembledPacket, exception_t>[]> rs_arr(sz);

                    for (size_t i = 0u; i < sz; ++i){
                        segment_arr[i] = std::move(base_data_arr[i].segment);
                    }

                    this->dst[idx]->assemble(std::make_move_iterator(segment_arr.get()), sz, rs_arr.get());

                    for (size_t i = 0u; i < sz; ++i){
                        if (rs_arr[i].has_value()){
                            *base_data_arr[i].rs = std::move(rs_arr[i]); //thru
                            continue;   
                        }

                        if (rs_arr[i].error() == dg::network_exception::SOCKET_STREAM_SEGMENT_FILLING){
                            *base_data_arr[i].rs = std::move(rs_arr[i]); //thru
                            continue;
                        }

                        //fallback, move semantic does not apply to failed exception_t
                        *base_data_arr[i].rs                    = std::move(rs_arr[i]);
                        *base_data_arr[i].fallback_segment_ptr  = std::move(segment_arr[i]);
                    }
                }
            };

            struct InternalDestroyFeedResolutor: dg::network_producer_consumer::KVConsumerInterface<size_t, GlobalIdentifier>{

                std::unique_ptr<PacketAssemblerInterface> * dst;

                void push(const size_t& idx, std::move_iterator<GlobalIdentifier *> data_arr, size_t sz) noexcept{

                    this->dst[idx]->destroy(data_arr.base(), sz);
                }
            };
    };

    //OK
    class ExhaustionControlledPacketAssembler: public virtual PacketAssemblerInterface{

        private:

            std::unique_ptr<PacketAssemblerInterface> base;
            std::shared_ptr<dg::network_concurrency_infretry_x::ExecutorInterface> executor;
            std::shared_ptr<ExhaustionControllerInterface> exhaustion_controller;

        public:

            ExhaustionControlledPacketAssembler(std::unique_ptr<PacketAssemblerInterface> base,
                                                std::shared_ptr<dg::network_concurrency_infretry_x::ExecutorInterface> executor,
                                                std::shared_ptr<ExhaustionControllerInterface> exhaustion_controller) noexcept: base(std::move(base)),
                                                                                                                                executor(std::move(executor)),
                                                                                                                                exhaustion_controller(std::move(exhaustion_controller)){}

            void assemble(std::move_iterator<PacketSegment *> segment_arr, size_t sz, std::expected<AssembledPacket, exception_t> * assembled_arr) noexcept{

                PacketSegment * base_segment_arr                                    = segment_arr.base();

                PacketSegment * first_segment_ptr                                   = base_segment_arr;
                PacketSegment * last_segment_ptr                                    = std::next(first_segment_ptr, sz);
                std::expected<AssembledPacket, exception_t> * first_assembled_ptr   = assembled_arr;
                std::expected<AssembledPacket, exception_t> * last_assembled_ptr    = std::next(first_assembled_ptr, sz);
                size_t sliding_window_sz                                            = sz;
                auto full_signal                                                    = std::expected<AssembledPacket, exception_t>(std::unexpected(dg::network_exception::QUEUE_FULL)); 

                auto task = [&, this]() noexcept{
                    this->base->assemble(std::make_move_iterator(first_segment_ptr), sliding_window_sz, first_assembled_ptr);
                    size_t queueing_sz  = std::count_if(first_assembled_ptr, last_assembled_ptr, [](const auto& e){return e.has_value() == false && e.error() == dg::network_exception::QUEUE_FULL;});
                    exception_t err     = this->exhaustion_controller->update_waiting_size(queueing_sz);

                    if (dg::network_exception::is_failed(err)){
                        dg::network_log_stackdump::error_fast_optional(dg::network_exception::verbose(err));
                    }

                    std::expected<AssembledPacket, exception_t> * first_retriable_ptr   = std::find_if(first_assembled_ptr, last_assembled_ptr, [](const auto& e){return e.has_value() == false && e.error() == dg::network_exception::QUEUE_FULL;});
                    std::expected<AssembledPacket, exception_t> * last_retriable_ptr    = std::find_if(first_retriable_ptr, last_assembled_ptr, [](const auto& e){return e.has_value() == true || e.error() != dg::network_exception::QUEUE_FULL;});

                    sliding_window_sz                                                   = std::distance(first_retriable_ptr, last_retriable_ptr);
                    size_t relative_offset                                              = std::distance(first_assembled_ptr, first_retriable_ptr);

                    std::advance(first_segment_ptr, relative_offset);
                    std::advance(first_assembled_ptr, relative_offset);

                    return !this->exhaustion_controller->is_should_wait() || first_segment_ptr == last_segment_ptr;
                };

                auto virtual_task = dg::network_concurrency_infretry_x::ExecutableWrapper<decltype(task)>(task);
                this->executor->exec(virtual_task);
            }

            void destroy(GlobalIdentifier * id_arr, size_t sz) noexcept{

                this->base->destroy(id_arr, sz);
            }

            auto max_consume_size() noexcept -> size_t{

                return this->base->max_consume_size();
            }
    };

    //OK
    class BufferFIFOContainer: public virtual InBoundContainerInterface{

        private:

            dg::pow2_cyclic_queue<dg::string> buffer_vec;
            size_t buffer_vec_capacity;
            std::unique_ptr<std::mutex> mtx;
            stdx::hdi_container<size_t> consume_sz_per_load;

        public:

            BufferFIFOContainer(dg::pow2_cyclic_queue<dg::string> buffer_vec,
                                size_t buffer_vec_capacity,
                                std::unique_ptr<std::mutex> mtx,
                                stdx::hdi_container<size_t> consume_sz_per_load) noexcept: buffer_vec(std::move(buffer_vec)),
                                                                                           buffer_vec_capacity(buffer_vec_capacity),
                                                                                           mtx(std::move(mtx)),
                                                                                           consume_sz_per_load(std::move(consume_sz_per_load)){}

            void push(std::move_iterator<dg::string *> buffer_arr, size_t sz, exception_t * exception_arr) noexcept{

                stdx::xlock_guard<std::mutex> lck_grd(*this->mtx);

                if constexpr(DEBUG_MODE_FLAG){
                    if (sz > this->max_consume_size()){
                        dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                        std::abort();
                    }
                }

                size_t app_cap  = this->buffer_vec_capacity - this->buffer_vec.size();
                size_t app_sz   = std::min(sz, app_cap);
                size_t old_sz   = this->buffer_vec.size();
                size_t new_sz   = old_sz + app_sz;

                this->buffer_vec.resize(new_sz);

                std::copy(buffer_arr, std::next(buffer_arr, app_sz), std::next(this->buffer_vec.begin(), old_sz));
                std::fill(exception_arr, std::next(exception_arr, app_sz), dg::network_exception::SUCCESS);
                std::fill(std::next(exception_arr, app_sz), std::next(exception_arr, sz), dg::network_exception::QUEUE_FULL);
            }

            void pop(dg::string * output_buffer_arr, size_t& sz, size_t output_buffer_arr_cap) noexcept{

                stdx::xlock_guard<std::mutex> lck_grd(*this->mtx);

                sz          = std::min(output_buffer_arr_cap, this->buffer_vec.size());
                auto first  = this->buffer_vec.begin();
                auto last   = std::next(first, sz);

                std::copy(std::make_move_iterator(first), std::make_move_iterator(last), output_buffer_arr);
                this->buffer_vec.erase_front_range(sz);
            }

            auto max_consume_size() noexcept -> size_t{

                return this->consume_sz_per_load.value;
            }
    };

    //OK
    class ExhaustionControlledBufferContainer: public virtual InBoundContainerInterface{

        private:

            std::unique_ptr<InBoundContainerInterface> base;
            std::shared_ptr<dg::network_concurrency_infretry_x::ExecutorInterface> executor;
            std::shared_ptr<ExhaustionControllerInterface> exhaustion_controller;

        public:

            ExhaustionControlledBufferContainer(std::unique_ptr<InBoundContainerInterface> base,
                                                std::shared_ptr<dg::network_concurrency_infretry_x::ExecutorInterface> executor,
                                                std::shared_ptr<ExhaustionControllerInterface> exhaustion_controller) noexcept: base(std::move(base)),
                                                                                                                                executor(std::move(executor)),
                                                                                                                                exhaustion_controller(std::move(exhaustion_controller)){}

            void push(std::move_iterator<dg::string *> buffer_arr, size_t sz, exception_t * exception_arr) noexcept{

                dg::string * buffer_arr_base        = buffer_arr.base();
                dg::string * buffer_arr_first       = buffer_arr_base;
                dg::string * buffer_arr_last        = std::next(buffer_arr_first, sz);
                exception_t * exception_arr_first   = exception_arr;
                exception_t * exception_arr_last    = std::next(exception_arr_first, sz);
                size_t sliding_window_sz            = sz;

                auto task = [&, this]() noexcept{
                    this->base->push(std::make_move_iterator(buffer_arr_first), sliding_window_sz, exception_arr_first);

                    size_t waiting_sz                   = std::count(exception_arr_first, exception_arr_last, dg::network_exception::QUEUE_FULL);
                    exception_t err                     = this->exhaustion_controller->update_waiting_size(waiting_sz);

                    if (dg::network_exception::is_failed(err)){
                        dg::network_log_stackdump::error_fast_optional(dg::network_exception::verbose(err));
                    }

                    exception_t * retriable_arr_first   = std::find(exception_arr_first, exception_arr_last, dg::network_exception::QUEUE_FULL);
                    exception_t * retriable_arr_last    = std::find_if(retriable_arr_first, exception_arr_last, [](exception_t err){return err != dg::network_exception::QUEUE_FULL;});

                    size_t relative_offset              = std::distance(exception_arr_first, retriable_arr_first);
                    sliding_window_sz                   = std::distance(retriable_arr_first, retriable_arr_last);

                    std::advance(buffer_arr_first, relative_offset);
                    std::advance(exception_arr_first, relative_offset); 

                    return !this->exhaustion_controller->is_should_wait() || (buffer_arr_first == buffer_arr_last); //TODOs: we want to subscribe these guys to a load_balancer system
                };

                auto virtual_task = dg::network_concurrency_infretry_x::ExecutableWrapper<decltype(task)>(std::move(task));
                this->executor->exec(virtual_task);
            }

            void pop(dg::string * output_buffer_arr, size_t& sz, size_t output_buffer_arr_cap) noexcept{

                this->base->pop(output_buffer_arr, sz, output_buffer_arr_cap);
            }

            auto max_consume_size() noexcept -> size_t{

                return this->base->max_consume_size();
            }
    };

    //OK
    class HashDistributedBufferContainer: public virtual InBoundContainerInterface{

        private:

            std::unique_ptr<std::unique_ptr<InBoundContainerInterface>[]> buffer_container_vec;
            size_t pow2_buffer_container_vec_sz;
            size_t zero_buffer_retry_sz;
            std::unique_ptr<ProducerDrainerPredicateInterface> drainer_pred;
            size_t drain_peek_cap_per_container;
            size_t consume_sz_per_load; 

        public:

            HashDistributedBufferContainer(std::unique_ptr<std::unique_ptr<InBoundContainerInterface>[]> buffer_container_vec,
                                           size_t pow2_buffer_container_vec_sz,
                                           size_t zero_buffer_retry_sz,
                                           std::unique_ptr<ProducerDrainerPredicateInterface> drainer_pred,
                                           size_t drain_peek_cap_per_container,
                                           size_t consume_sz_per_load) noexcept: buffer_container_vec(std::move(buffer_container_vec)),
                                                                                 pow2_buffer_container_vec_sz(pow2_buffer_container_vec_sz),
                                                                                 zero_buffer_retry_sz(zero_buffer_retry_sz),
                                                                                 drainer_pred(std::move(drainer_pred)),
                                                                                 drain_peek_cap_per_container(drain_peek_cap_per_container),
                                                                                 consume_sz_per_load(consume_sz_per_load){}

            void push(std::move_iterator<dg::string *> buffer_arr, size_t sz, exception_t * exception_arr) noexcept{

                if constexpr(DEBUG_MODE_FLAG){
                    if (sz > this->max_consume_size()){
                        dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                        std::abort();
                    }
                }

                size_t random_value = dg::network_randomizer::randomize_int<size_t>();
                size_t random_idx   = random_value & (this->pow2_buffer_container_vec_sz - 1u);

                this->buffer_container_vec[random_idx]->push(buffer_arr, sz, exception_arr);
            }

            void pop(dg::string * output_buffer_arr, size_t& sz, size_t output_buffer_arr_cap) noexcept{

                if (this->drainer_pred->is_should_drain()){
                    this->internal_drain(output_buffer_arr, sz, output_buffer_arr_cap);
                    this->drainer_pred->reset();
                } else{
                    this->internal_curious_pop(output_buffer_arr, sz, output_buffer_arr_cap);
                }
            }

            auto max_consume_size() noexcept -> size_t{

                return this->consume_sz_per_load;
            }
        
        private:

            void internal_curious_pop(dg::string * output_buffer_arr, size_t& sz, size_t output_buffer_arr_cap) noexcept{

                sz = 0u;

                for (size_t i = 0u; i < this->zero_buffer_retry_sz; ++i){
                    size_t random_value = dg::network_randomizer::randomize_int<size_t>();
                    size_t random_idx   = random_value & (this->pow2_buffer_container_vec_sz - 1u);

                    this->buffer_container_vec[random_idx]->pop(output_buffer_arr, sz, output_buffer_arr_cap);

                    if (sz != 0u){
                        return;
                    }
                }
            }

            void internal_drain(dg::string * output_buffer_arr, size_t& sz, size_t output_buffer_arr_cap) noexcept{

                sz = 0u;
                dg::string * current_output_arr = output_buffer_arr;
                size_t current_cap  = output_buffer_arr_cap;
                size_t start_seek   = dg::network_randomizer::randomize_int<size_t>() >> 1; //alright this is one of the myths, because container_vec_sz is already pow2 

                for (size_t i = 0u; i < this->pow2_buffer_container_vec_sz; ++i){
                    if (current_cap == 0u){
                        return;
                    }

                    size_t peeking_cap  = std::min(current_cap, this->drain_peek_cap_per_container);
                    size_t adjusted_idx = (start_seek + i) & (this->pow2_buffer_container_vec_sz - 1u);
                    size_t current_sz   = {};
                    this->buffer_container_vec[adjusted_idx]->pop(current_output_arr, current_sz, peeking_cap);
                    current_cap         -= current_sz;
                    sz                  += current_sz;
                    std::advance(current_output_arr, current_sz);
                }
            }
    };

    //OK

    template <class T>
    class ReactingBufferContainer{};

    template <class ID>
    class ReactingBufferContainer<AffinedTimeoutController<ID>>: public virtual InBoundContainerInterface{

        private:

            std::unique_ptr<InBoundContainerInterface> base;
            std::unique_ptr<ComplexReactor> reactor;
            std::chrono::nanoseconds max_wait_time;

        public:

            using timeout_controller = AffinedTimeoutController<ID>; //its incredibly hard to not use dep injection, if we are to go down the route of including this as part of the interface, it's not gonna last 

            ReactingBufferContainer(std::unique_ptr<InBoundContainerInterface> base,
                                    std::unique_ptr<ComplexReactor> reactor,
                                    std::chrono::nanoseconds max_wait_time) noexcept: base(std::move(base)),
                                                                                      reactor(std::move(reactor)),
                                                                                      max_wait_time(max_wait_time){}

            void push(std::move_iterator<dg::string *> buffer_arr, size_t sz, exception_t * exception_arr) noexcept{

                this->base->push(buffer_arr, sz, exception_arr);
                size_t thru_sz = std::count(exception_arr, std::next(exception_arr, sz), dg::network_exception::SUCCESS);
                this->reactor->increment(thru_sz);
            }

            void pop(dg::string * output_buffer_arr, size_t& sz, size_t output_buffer_arr_cap) noexcept{

                std::atomic_signal_fence(std::memory_order_seq_cst);
                std::optional<std::chrono::nanoseconds> user_specified_timeout = timeout_controller::get();

                if (user_specified_timeout.has_value()){
                    if (user_specified_timeout.value() == std::chrono::nanoseconds::min()){
                        (void) user_specified_timeout;
                    } else{
                        this->reactor->subscribe(user_specified_timeout.value());
                    }
                } else{
                    this->reactor->subscribe(this->max_wait_time);
                }

                std::atomic_signal_fence(std::memory_order_seq_cst);

                this->base->pop(output_buffer_arr, sz, output_buffer_arr_cap);
                this->reactor->decrement(sz);
            }

            auto max_consume_size() noexcept -> size_t{

                return this->base->max_consume_size();
            }
    };

    //OK
    class ExpiryWorker: public virtual dg::network_concurrency::WorkerInterface{

        private:

            std::shared_ptr<PacketAssemblerInterface> packet_assembler;
            std::shared_ptr<EntranceControllerInterface> entrance_controller;
            std::shared_ptr<BlackListGateInterface> blacklist_gate;
            size_t packet_assembler_vectorization_sz;
            size_t expired_id_consume_sz;
            size_t busy_consume_sz;

        public:

            ExpiryWorker(std::shared_ptr<PacketAssemblerInterface> packet_assembler,
                         std::shared_ptr<EntranceControllerInterface> entrance_controller,
                         std::shared_ptr<BlackListGateInterface> blacklist_gate,
                         size_t packet_assembler_vectorization_sz,
                         size_t expired_id_consume_sz,
                         size_t busy_consume_sz) noexcept: packet_assembler(std::move(packet_assembler)),
                                                           entrance_controller(std::move(entrance_controller)),
                                                           blacklist_gate(std::move(blacklist_gate)),
                                                           packet_assembler_vectorization_sz(packet_assembler_vectorization_sz),
                                                           expired_id_consume_sz(expired_id_consume_sz),
                                                           busy_consume_sz(busy_consume_sz){}

            bool run_one_epoch() noexcept{

                size_t id_arr_cap   = this->expired_id_consume_sz;
                size_t id_arr_sz    = {};
                dg::network_stack_allocation::NoExceptAllocation<GlobalIdentifier[]> id_arr(id_arr_cap);
                this->entrance_controller->get_expired_id(id_arr.get(), id_arr_sz, id_arr_cap);

                auto pa_feed_resolutor              = InternalPacketAssemblerDestroyFeedResolutor{};
                pa_feed_resolutor.dst               = this->packet_assembler.get(); 
                pa_feed_resolutor.blacklist_gate    = this->blacklist_gate.get();

                size_t trimmed_pa_feed_sz           = std::min(std::min(std::min(this->packet_assembler_vectorization_sz, this->packet_assembler->max_consume_size()), id_arr_sz), this->blacklist_gate->max_consume_size());
                size_t pa_feeder_allocation_cost    = dg::network_producer_consumer::delvrsrv_allocation_cost(&pa_feed_resolutor, trimmed_pa_feed_sz);
                dg::network_stack_allocation::NoExceptRawAllocation<char[]> pa_feeder_mem(pa_feeder_allocation_cost);
                auto pa_feeder                      = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_preallocated_raiihandle(&pa_feed_resolutor, trimmed_pa_feed_sz, pa_feeder_mem.get()));

                for (size_t i = 0u; i < id_arr_sz; ++i){
                    dg::network_producer_consumer::delvrsrv_deliver(pa_feeder.get(), id_arr[i]);
                }

                return id_arr_sz >= this->busy_consume_sz;
            }

        private:

            struct InternalPacketAssemblerDestroyFeedResolutor: dg::network_producer_consumer::ConsumerInterface<GlobalIdentifier>{

                PacketAssemblerInterface * dst;
                BlackListGateInterface * blacklist_gate;

                void push(std::move_iterator<GlobalIdentifier *> id_arr, size_t id_arr_sz) noexcept{

                    dg::network_stack_allocation::NoExceptAllocation<exception_t[]> exception_arr(id_arr_sz);

                    GlobalIdentifier * base_id_arr = id_arr.base();
                    this->dst->destroy(base_id_arr, id_arr_sz);
                    this->blacklist_gate->blacklist(base_id_arr, id_arr_sz, exception_arr.get());

                    for (size_t i = 0u; i < id_arr_sz; ++i){
                        if (dg::network_exception::is_failed(exception_arr[i])){
                            dg::network_log_stackdump::error_fast_optional(dg::network_exception::verbose(exception_arr[i]));
                        }
                    }
                }
            };     
    };

    //OK
    class InBoundWorker: public virtual dg::network_concurrency::WorkerInterface{

        private:

            std::shared_ptr<PacketAssemblerInterface> packet_assembler;
            std::shared_ptr<InBoundGateInterface> inbound_gate;
            std::shared_ptr<InBoundGateInterface> blacklist_gate;
            std::shared_ptr<InBoundContainerInterface> inbound_container;
            std::shared_ptr<EntranceControllerInterface> entrance_controller;
            std::shared_ptr<dg::network_kernel_mailbox_impl1::core::MailboxInterface> base;
            size_t packet_assembler_vectorization_sz;
            size_t inbound_gate_vectorization_sz;
            size_t blacklist_gate_vectorization_sz;
            size_t entrance_controller_vectorization_sz;
            size_t inbound_container_vectorization_sz;
            size_t upstream_consume_sz;
            size_t busy_consume_sz;

        public:

            InBoundWorker(std::shared_ptr<PacketAssemblerInterface> packet_assembler,
                          std::shared_ptr<InBoundGateInterface> inbound_gate,
                          std::shared_ptr<InBoundGateInterface> blacklist_gate,
                          std::shared_ptr<InBoundContainerInterface> inbound_container,
                          std::shared_ptr<EntranceControllerInterface> entrance_controller,
                          std::shared_ptr<dg::network_kernel_mailbox_impl1::core::MailboxInterface> base,
                          size_t packet_assembler_vectorization_sz,
                          size_t inbound_gate_vectorization_sz,
                          size_t blacklist_gate_vectorization_sz,
                          size_t entrance_controller_vectorization_sz,
                          size_t inbound_container_vectorization_sz,
                          size_t upstream_consume_sz,
                          size_t busy_consume_sz) noexcept: packet_assembler(std::move(packet_assembler)),
                                                            inbound_gate(std::move(inbound_gate)),
                                                            blacklist_gate(std::move(blacklist_gate)),
                                                            inbound_container(std::move(inbound_container)),
                                                            entrance_controller(std::move(entrance_controller)),
                                                            base(std::move(base)),
                                                            packet_assembler_vectorization_sz(packet_assembler_vectorization_sz),
                                                            inbound_gate_vectorization_sz(inbound_gate_vectorization_sz),
                                                            blacklist_gate_vectorization_sz(blacklist_gate_vectorization_sz),
                                                            entrance_controller_vectorization_sz(entrance_controller_vectorization_sz),
                                                            inbound_container_vectorization_sz(inbound_container_vectorization_sz),
                                                            upstream_consume_sz(upstream_consume_sz),
                                                            busy_consume_sz(busy_consume_sz){}

            bool run_one_epoch() noexcept{

                size_t consuming_sz     = 0u;
                size_t consuming_cap    = this->upstream_consume_sz;
                dg::network_stack_allocation::NoExceptAllocation<dg::string[]> buf_arr(consuming_cap);

                this->base->recv(buf_arr.get(), consuming_sz, consuming_cap); 

                auto et_feed_resolutor                      = InternalEntranceFeedResolutor{};
                et_feed_resolutor.entrance_controller       = this->entrance_controller.get();

                size_t trimmed_et_feed_cap                  = std::min(std::min(this->entrance_controller_vectorization_sz, consuming_sz), this->entrance_controller->max_consume_size());
                size_t et_feeder_allocation_cost            = dg::network_producer_consumer::delvrsrv_allocation_cost(&et_feed_resolutor, trimmed_et_feed_cap);
                dg::network_stack_allocation::NoExceptRawAllocation<char[]> et_feeder_mem(et_feeder_allocation_cost);
                auto et_feeder                              = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_preallocated_raiihandle(&et_feed_resolutor, trimmed_et_feed_cap, et_feeder_mem.get()));

                auto ib_feed_resolutor                      = InternalInBoundFeedResolutor{};
                ib_feed_resolutor.inbound_container         = this->inbound_container.get();

                size_t trimmed_ib_feed_cap                  = std::min(std::min(this->inbound_container_vectorization_sz, consuming_sz), this->inbound_container->max_consume_size());
                size_t ib_feeder_allocation_cost            = dg::network_producer_consumer::delvrsrv_allocation_cost(&ib_feed_resolutor, trimmed_ib_feed_cap);
                dg::network_stack_allocation::NoExceptRawAllocation<char[]> ib_feeder_mem(ib_feeder_allocation_cost);
                auto ib_feeder                              = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_preallocated_raiihandle(&ib_feed_resolutor, trimmed_ib_feed_cap, ib_feeder_mem.get())); 

                auto ps_feed_resolutor                      = InternalPacketSegmentFeedResolutor{};
                ps_feed_resolutor.packet_assembler          = this->packet_assembler.get();
                ps_feed_resolutor.entrance_feeder           = et_feeder.get();
                ps_feed_resolutor.inbound_feeder            = ib_feeder.get();

                size_t trimmed_ps_feed_cap                  = std::min(std::min(this->packet_assembler_vectorization_sz, consuming_sz), this->packet_assembler->max_consume_size());
                size_t ps_feeder_allocation_cost            = dg::network_producer_consumer::delvrsrv_allocation_cost(&ps_feed_resolutor, trimmed_ps_feed_cap);
                dg::network_stack_allocation::NoExceptRawAllocation<char[]> ps_feeder_mem(ps_feeder_allocation_cost);
                auto ps_feeder                              = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_preallocated_raiihandle(&ps_feed_resolutor, trimmed_ps_feed_cap, ps_feeder_mem.get()));  

                auto gt_feed_resolutor                      = InternalInBoundGateFeedResolutor{};
                gt_feed_resolutor.inbound_gate              = this->inbound_gate.get();
                gt_feed_resolutor.downstream_feeder         = ps_feeder.get();                    

                size_t trimmed_gt_feed_cap                  = std::min(std::min(this->inbound_gate_vectorization_sz, consuming_sz), this->inbound_gate->max_consume_size());
                size_t gt_feeder_allocation_cost            = dg::network_producer_consumer::delvrsrv_allocation_cost(&gt_feed_resolutor, trimmed_gt_feed_cap);
                dg::network_stack_allocation::NoExceptRawAllocation<char[]> gt_feeder_mem(gt_feeder_allocation_cost);
                auto gt_feeder                              = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_preallocated_raiihandle(&gt_feed_resolutor, trimmed_gt_feed_cap, gt_feeder_mem.get()));

                auto bl_feed_resolutor                      = InternalInBoundGateFeedResolutor{};
                bl_feed_resolutor.inbound_gate              = this->blacklist_gate.get();
                bl_feed_resolutor.downstream_feeder         = gt_feeder.get();

                size_t trimmed_bl_feed_cap                  = std::min(std::min(this->blacklist_gate_vectorization_sz, consuming_sz), this->blacklist_gate->max_consume_size());
                size_t bl_feeder_allocation_cost            = dg::network_producer_consumer::delvrsrv_allocation_cost(&bl_feed_resolutor, trimmed_bl_feed_cap);
                dg::network_stack_allocation::NoExceptRawAllocation<char[]> bl_feeder_mem(bl_feeder_allocation_cost);
                auto bl_feeder                              = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_preallocated_raiihandle(&bl_feed_resolutor, trimmed_bl_feed_cap, bl_feeder_mem.get()));

                for (size_t i = 0u; i < consuming_sz; ++i){
                    std::expected<PacketSegment, exception_t> pkt = deserialize_packet_segment(std::move(buf_arr[i]));

                    if (!pkt.has_value()){
                        dg::network_log_stackdump::error_fast_optional(dg::network_exception::verbose(pkt.error()));
                        continue;
                    }

                    dg::network_producer_consumer::delvrsrv_deliver(bl_feeder.get(), std::move(pkt.value()));
                }

                return consuming_sz >= this->busy_consume_sz;
            }
        
        private:

            struct InternalEntranceFeedResolutor: dg::network_producer_consumer::ConsumerInterface<GlobalIdentifier>{

                EntranceControllerInterface * entrance_controller;

                void push(std::move_iterator<GlobalIdentifier *> id_arr, size_t sz) noexcept{

                    dg::network_stack_allocation::NoExceptAllocation<exception_t[]> exception_arr(sz);
                    this->entrance_controller->tick(id_arr.base(), sz, exception_arr.get());

                    for (size_t i = 0u; i < sz; ++i){
                        if (dg::network_exception::is_failed(exception_arr[i])){
                            dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::SOCKET_STREAM_LEAK));
                        }
                    }
                }
            };

            struct InternalInBoundFeedResolutor: dg::network_producer_consumer::ConsumerInterface<dg::string>{

                InBoundContainerInterface * inbound_container;

                void push(std::move_iterator<dg::string *> data_arr, size_t sz) noexcept{

                    dg::network_stack_allocation::NoExceptAllocation<exception_t[]> exception_arr(sz);
                    this->inbound_container->push(data_arr, sz, exception_arr.get());

                    for (size_t i = 0u; i < sz; ++i){
                        if (dg::network_exception::is_failed(exception_arr[i])){
                            dg::network_log_stackdump::error_fast_optional(dg::network_exception::verbose(exception_arr[i]));
                        }
                    }
                }
            };

            struct InternalPacketSegmentFeedResolutor: dg::network_producer_consumer::ConsumerInterface<PacketSegment>{

                PacketAssemblerInterface * packet_assembler;
                dg::network_producer_consumer::DeliveryHandle<GlobalIdentifier> * entrance_feeder;
                dg::network_producer_consumer::DeliveryHandle<dg::string> * inbound_feeder;

                void push(std::move_iterator<PacketSegment *> data_arr, size_t sz) noexcept{
                    
                    PacketSegment * base_data_arr = data_arr.base();

                    dg::network_stack_allocation::NoExceptAllocation<GlobalIdentifier[]> global_id_arr(sz);
                    dg::network_stack_allocation::NoExceptAllocation<std::expected<AssembledPacket, exception_t>[]> assembled_arr(sz);

                    for (size_t i = 0u; i < sz; ++i){
                        global_id_arr[i] = base_data_arr[i].id; 
                    }

                    this->packet_assembler->assemble(std::make_move_iterator(base_data_arr), sz, assembled_arr.get());
                    
                    std::atomic_signal_fence(std::memory_order_seq_cst);

                    for (size_t i = 0u; i < sz; ++i){
                        dg::network_producer_consumer::delvrsrv_deliver(this->entrance_feeder, global_id_arr[i]);
                    }

                    for (size_t i = 0u; i < sz; ++i){
                        if (assembled_arr[i].has_value()){
                            std::expected<dg::string, exception_t> buf = internal_integrity_assembled_packet_to_buffer(static_cast<AssembledPacket&&>(assembled_arr[i].value())); 

                            if (!buf.has_value()){
                                dg::network_log_stackdump::error_fast_optional(dg::network_exception::verbose(buf.error()));
                            } else{
                                dg::network_producer_consumer::delvrsrv_deliver(this->inbound_feeder, std::move(buf.value()));
                            }
                        } else{
                            if (assembled_arr[i].error() != dg::network_exception::SOCKET_STREAM_SEGMENT_FILLING){
                                dg::network_log_stackdump::error_fast_optional(dg::network_exception::verbose(assembled_arr[i].error()));
                            }
                        }
                    }
                }
            };

            struct InternalInBoundGateFeedResolutor: dg::network_producer_consumer::ConsumerInterface<PacketSegment>{

                InBoundGateInterface * inbound_gate;
                dg::network_producer_consumer::DeliveryHandle<PacketSegment> * downstream_feeder;

                void push(std::move_iterator<PacketSegment *> data_arr, size_t sz) noexcept{

                    PacketSegment * base_data_arr = data_arr.base();

                    dg::network_stack_allocation::NoExceptAllocation<GlobalIdentifier[]> global_id_arr(sz);
                    dg::network_stack_allocation::NoExceptAllocation<exception_t[]> exception_arr(sz);

                    for (size_t i = 0u; i < sz; ++i){
                        global_id_arr[i] = base_data_arr[i].id;
                    }

                    this->inbound_gate->thru(global_id_arr.get(), sz, exception_arr.get());

                    for (size_t i = 0u; i < sz; ++i){
                        if (dg::network_exception::is_failed(exception_arr[i])){
                            dg::network_log_stackdump::error_fast_optional(dg::network_exception::verbose(exception_arr[i]));
                            continue;
                        }

                        dg::network_producer_consumer::delvrsrv_deliver(this->downstream_feeder, std::move(base_data_arr[i]));
                    }
                }  
            };
    };

    //OK
    class MailBox: public virtual dg::network_kernel_mailbox_impl1::core::MailboxInterface{

        private:

            dg::vector<dg::network_concurrency::daemon_raii_handle_t> daemon_vec;
            std::unique_ptr<PacketizerInterface> packetizer;
            std::shared_ptr<dg::network_kernel_mailbox_impl1::core::MailboxInterface> base;
            std::shared_ptr<InBoundContainerInterface> inbound_container;
            std::shared_ptr<OutBoundRuleInterface> outbound_rule;
            size_t transmission_vectorization_sz;

        public:

            MailBox(dg::vector<dg::network_concurrency::daemon_raii_handle_t> daemon_vec,
                    std::unique_ptr<PacketizerInterface> packetizer,
                    std::shared_ptr<dg::network_kernel_mailbox_impl1::core::MailboxInterface> base,
                    std::shared_ptr<InBoundContainerInterface> inbound_container,
                    std::shared_ptr<OutBoundRuleInterface> outbound_rule,
                    size_t transmission_vectorization_sz) noexcept: daemon_vec(std::move(daemon_vec)),
                                                                    packetizer(std::move(packetizer)),
                                                                    base(std::move(base)),
                                                                    inbound_container(std::move(inbound_container)),
                                                                    outbound_rule(std::move(outbound_rule)),
                                                                    transmission_vectorization_sz(transmission_vectorization_sz){}

            void send(std::move_iterator<MailBoxArgument *> data_arr, size_t sz, exception_t * exception_arr) noexcept{
 
                MailBoxArgument * base_data_arr = data_arr.base();

                auto feed_resolutor             = InternalFeedResolutor{};
                feed_resolutor.dst              = this->base.get(); 

                size_t trimmed_mailbox_feed_sz  = std::min(std::min(this->transmission_vectorization_sz, sz * this->packetizer->max_segment_count()), this->base->max_consume_size());
                size_t feeder_allocation_cost   = dg::network_producer_consumer::delvrsrv_kv_allocation_cost(&feed_resolutor, trimmed_mailbox_feed_sz);
                dg::network_stack_allocation::NoExceptRawAllocation<char[]> feeder_mem(feeder_allocation_cost);
                auto feeder                     = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_kv_open_preallocated_raiihandle(&feed_resolutor, trimmed_mailbox_feed_sz, feeder_mem.get()));

                std::fill(exception_arr, std::next(exception_arr, sz), dg::network_exception::SUCCESS);

                for (size_t i = 0u; i < sz; ++i){
                    std::expected<bool, exception_t> outbound_thru_status = this->outbound_rule->thru(base_data_arr[i].to);

                    if (!outbound_thru_status.has_value()){
                        exception_arr[i] = outbound_thru_status.error();
                        continue;
                    }

                    if (!outbound_thru_status.value()){
                        exception_arr[i] = dg::network_exception::SOCKET_STREAM_BAD_OUTBOUND_RULE;
                        continue;
                    }

                    std::expected<dg::vector<PacketSegment>, exception_t> segment_vec = this->packetizer->packetize(static_cast<dg::string&&>(base_data_arr[i].content));

                    if (!segment_vec.has_value()){
                        exception_arr[i] = segment_vec.error();
                        continue;
                    }

                    for (size_t j = 0u; j < segment_vec.value().size(); ++j){
                        dg::network_producer_consumer::delvrsrv_kv_deliver(feeder.get(), base_data_arr[i].to, std::move(segment_vec.value()[j])); //we are to attempt to temporally group the <to> transmission, to increase ack vectorization chances
                    }
                }
            }

            void recv(dg::string * output_arr, size_t& output_arr_sz, size_t output_arr_cap) noexcept{

                return this->inbound_container->pop(output_arr, output_arr_sz, output_arr_cap);
            }

            auto max_consume_size() noexcept -> size_t{

                return this->base->max_consume_size();
            }

        private:

            struct InternalFeedResolutor: dg::network_producer_consumer::KVConsumerInterface<Address, PacketSegment>{

                dg::network_kernel_mailbox_impl1::core::MailboxInterface * dst;

                void push(const Address& to, std::move_iterator<PacketSegment *> data_arr, size_t sz) noexcept{

                    PacketSegment * base_data_arr   = data_arr.base();
                    size_t arr_cap                  = sz;
                    size_t arr_sz                   = 0u;

                    dg::network_stack_allocation::NoExceptAllocation<MailBoxArgument[]> mailbox_arg_arr(arr_cap);
                    dg::network_stack_allocation::NoExceptAllocation<exception_t[]> exception_arr(arr_cap);

                    for (size_t i = 0u; i < sz; ++i){
                        std::expected<dg::string, exception_t> serialized = serialize_packet_segment(static_cast<PacketSegment&&>(base_data_arr[i]));

                        if (!serialized.has_value()){
                            dg::network_log_stackdump::error_fast_optional(dg::network_exception::verbose(serialized.error()));
                            continue;
                        }

                        mailbox_arg_arr[arr_sz].content = std::move(serialized.value());
                        mailbox_arg_arr[arr_sz].to      = to;
                        arr_sz                          += 1;
                    }

                    this->dst->send(std::make_move_iterator(mailbox_arg_arr.get()), arr_sz, exception_arr.get());

                    for (size_t i = 0u; i < arr_sz; ++i){
                        if (dg::network_exception::is_failed(exception_arr[i])){
                            dg::network_log_stackdump::error_fast_optional(dg::network_exception::verbose(exception_arr[i]));
                        }
                    }
                }
            };

    };

    struct ComponentFactory{

        template <class T, class ...Args>
        static auto to_dg_vector(std::vector<T, Args...> vec) -> dg::vector<T>{

            dg::vector<T> rs{};

            for (size_t i = 0u; i < vec.size(); ++i){
                rs.emplace_back(std::move(vec[i]));
            }

            return rs;
        }

        static auto get_timing_drainer(std::chrono::nanoseconds drain_period) -> std::unique_ptr<ProducerDrainerPredicateInterface>{

            const std::chrono::nanoseconds MIN_DRAIN_PERIOD = std::chrono::nanoseconds(1);
            const std::chrono::nanoseconds MAX_DRAIN_PERIOD = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::hours(1));

            if (std::clamp(drain_period, MIN_DRAIN_PERIOD, MAX_DRAIN_PERIOD) != drain_period){
                dg::network_exception::throw_exception(dg::network_exception::INVALID_ARGUMENT);
            }

            return std::make_unique<TimingDrainerPredicate>(drain_period);
        }

        static auto get_complex_reactor(intmax_t wakeup_threshold,
                                        size_t concurrent_subscriber_cap) -> std::unique_ptr<ComplexReactor>{
            
            const size_t MIN_CONCURRENT_SUBSCRIBER_CAP  = 1u;
            const size_t MAX_CONCURRENT_SUBSCRIBER_CAP  = size_t{1} << 25;

            if (std::clamp(concurrent_subscriber_cap, MIN_CONCURRENT_SUBSCRIBER_CAP, MAX_CONCURRENT_SUBSCRIBER_CAP) != concurrent_subscriber_cap){
                dg::network_exception::throw_exception(dg::network_exception::INVALID_ARGUMENT);
            }

            auto rs = std::make_unique<ComplexReactor>(concurrent_subscriber_cap);
            rs->set_wakeup_threshold(wakeup_threshold);

            return rs;
        } 

        static auto get_no_exhaustion_controller() -> std::unique_ptr<ExhaustionControllerInterface>{

            return std::make_unique<NoExhaustionController>();
        }

        static auto get_empty_outbound_rule() -> std::unique_ptr<OutBoundRuleInterface>{

            return std::make_unique<EmptyOutBoundRule>();
        }

        static auto get_incremental_packet_id_generator(Address factory_addr) -> std::unique_ptr<PacketIDGeneratorInterface>{

            uint64_t random_counter = dg::network_randomizer::randomize_int<uint64_t>();
            return std::make_unique<PacketIDGenerator>(std::move(factory_addr), random_counter);
        } 
        
        static auto get_random_packet_id_generator(Address factory_addr) -> std::unique_ptr<PacketIDGeneratorInterface>{

            return std::make_unique<RandomPacketIDGenerator>(factory_addr);
        }

        static auto get_temporal_absolute_timeout_inbound_gate(size_t map_capacity, 
                                                               std::chrono::nanoseconds abs_timeout_dur,
                                                               size_t max_consume_decay_factor = 2u) -> std::unique_ptr<InBoundGateInterface>{

            const size_t MIN_MAP_CAPACITY                       = size_t{1};
            const size_t MAX_MAP_CAPACITY                       = size_t{1} << 40;
            const std::chrono::nanoseconds MIN_ABS_TIMEOUT_DUR  = std::chrono::nanoseconds(1u);
            const std::chrono::nanoseconds MAX_ABS_TIMEOUT_DUR  = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::hours(1));

            if (std::clamp(map_capacity, MIN_MAP_CAPACITY, MAX_MAP_CAPACITY) != map_capacity){
                dg::network_exception::throw_exception(dg::network_exception::INVALID_ARGUMENT);
            }

            if (std::clamp(abs_timeout_dur, MIN_ABS_TIMEOUT_DUR, MAX_ABS_TIMEOUT_DUR) != abs_timeout_dur){
                dg::network_exception::throw_exception(dg::network_exception::INVALID_ARGUMENT);
            }

            size_t tentative_max_consume_sz = map_capacity >> max_consume_decay_factor;
            size_t max_consume_sz           = std::max(size_t{1}, tentative_max_consume_sz); 
            auto abstimeout_map             = temporal_finite_unordered_map<GlobalIdentifier, std::chrono::time_point<std::chrono::steady_clock>>(map_capacity); 

            return std::make_unique<TemporalAbsoluteTimeoutInBoundGate>(std::move(abstimeout_map),
                                                                        abs_timeout_dur,
                                                                        std::make_unique<std::mutex>(),
                                                                        stdx::hdi_container<size_t>{max_consume_sz});
        }

        static auto get_random_hash_distributed_inbound_gate(std::vector<std::unique_ptr<InBoundGateInterface>> inbound_gate_vec,
                                                             size_t keyvalue_feed_cap = DEFAULT_KEYVALUE_FEED_SIZE) -> std::unique_ptr<InBoundGateInterface>{

            const size_t MIN_KEYVALUE_FEED_CAP  = size_t{1};
            const size_t MAX_KEYVALUE_FEED_CAP  = size_t{1} << 25;

            if (!stdx::is_pow2(inbound_gate_vec.size())){
                dg::network_exception::throw_exception(dg::network_exception::INVALID_ARGUMENT);
            }

            if (std::clamp(keyvalue_feed_cap, MIN_KEYVALUE_FEED_CAP, MAX_KEYVALUE_FEED_CAP) != keyvalue_feed_cap){
                dg::network_exception::throw_exception(dg::network_exception::INVALID_ARGUMENT);
            }

            auto inbound_gate_arr       = std::make_unique<std::unique_ptr<InBoundGateInterface>[]>(inbound_gate_vec.size());
            size_t inbound_gate_arr_sz  = inbound_gate_vec.size();
            size_t max_consume_sz       = std::numeric_limits<size_t>::max(); 

            for (size_t i = 0u; i < inbound_gate_vec.size(); ++i){
                if (inbound_gate_vec[i] == nullptr){
                    dg::network_exception::throw_exception(dg::network_exception::INVALID_ARGUMENT);
                }

                max_consume_sz      = std::min(max_consume_sz, inbound_gate_vec[i]->max_consume_size());
                inbound_gate_arr[i] = std::move(inbound_gate_vec[i]);
            }

            return std::make_unique<RandomHashDistributedInBoundGate>(std::move(inbound_gate_arr),
                                                                      inbound_gate_arr_sz,
                                                                      keyvalue_feed_cap,
                                                                      max_consume_sz);
        }

        static auto get_temporal_blacklist_gate(size_t set_cap,
                                                size_t rehash_sz,
                                                size_t reliability_decay_factor,
                                                size_t max_consume_decay_factor = 2u) -> std::unique_ptr<BlackListGateInterface>{
            
            const size_t MIN_SET_CAP    = size_t{1};
            const size_t MAX_SET_CAP    = size_t{1} << 40;

            if (std::clamp(set_cap, MIN_SET_CAP, MAX_SET_CAP) != set_cap){
                dg::network_exception::throw_exception(dg::network_exception::INVALID_ARGUMENT);
            }

            auto blacklist_set              = temporal_switching_bloom_filter<GlobalIdentifier>(set_cap, rehash_sz, reliability_decay_factor);
            size_t tentative_max_consume_sz = set_cap >> max_consume_decay_factor;
            size_t max_consume_sz           = std::max(size_t{1}, tentative_max_consume_sz); 

            return std::make_unique<TemporalBlackListGate>(std::move(blacklist_set),
                                                           std::make_unique<std::mutex>(),
                                                           stdx::hdi_container<size_t>{max_consume_sz});
        }

        static auto get_random_hash_distributed_blacklist_gate(std::vector<std::unique_ptr<BlackListGateInterface>> blacklist_gate_vec,
                                                               size_t keyvalue_feed_cap = DEFAULT_KEYVALUE_FEED_SIZE) -> std::unique_ptr<BlackListGateInterface>{
            
            const size_t MIN_KEYVALUE_FEED_CAP  = size_t{1};
            const size_t MAX_KEYVALUE_FEED_CAP  = size_t{1} << 25;

            if (!stdx::is_pow2(blacklist_gate_vec.size())){
                dg::network_exception::throw_exception(dg::network_exception::INVALID_ARGUMENT);
            }

            if (std::clamp(keyvalue_feed_cap, MIN_KEYVALUE_FEED_CAP, MAX_KEYVALUE_FEED_CAP) != keyvalue_feed_cap){
                dg::network_exception::throw_exception(dg::network_exception::INVALID_ARGUMENT);
            }

            auto blacklist_gate_arr         = std::make_unique<std::unique_ptr<BlackListGateInterface>[]>(blacklist_gate_vec.size());
            size_t blacklist_gate_arr_sz    = blacklist_gate_vec.size();
            size_t max_consume_sz           = std::numeric_limits<size_t>::max();

            for (size_t i = 0u; i < blacklist_gate_vec.size(); ++i){
                if (blacklist_gate_vec[i] == nullptr){
                    dg::network_exception::throw_exception(dg::network_exception::INVALID_ARGUMENT);
                }

                max_consume_sz          = std::min(max_consume_sz, blacklist_gate_vec[i]->max_consume_size());
                blacklist_gate_arr[i]   = std::move(blacklist_gate_vec[i]);
            }

            return std::make_unique<RandomHashDistributedBlackListGate>(std::move(blacklist_gate_arr),
                                                                        blacklist_gate_arr_sz,
                                                                        keyvalue_feed_cap,
                                                                        max_consume_sz);
        }

        static auto get_packetizer(Address factory_addr, 
                                   size_t segment_bsz,
                                   size_t max_packet_bsz,
                                   bool has_integrity_transmit) -> std::unique_ptr<PacketizerInterface>{

            const size_t MIN_SEGMENT_BYTE_SZ    = size_t{1};
            const size_t MAX_SEGMENT_BYTE_SZ    = size_t{1} << 30;  
            const size_t MIN_MAX_PACKET_BSZ     = 0u;
            const size_t MAX_MAX_PACKET_BSZ     = size_t{1} << 40; 

            if (std::clamp(segment_bsz, MIN_SEGMENT_BYTE_SZ, MAX_SEGMENT_BYTE_SZ) != segment_bsz){
                dg::network_exception::throw_exception(dg::network_exception::INVALID_ARGUMENT);
            }

            if (!stdx::is_pow2(segment_bsz)){
                dg::network_exception::throw_exception(dg::network_exception::INVALID_ARGUMENT);
            }

            if (std::clamp(max_packet_bsz, MIN_MAX_PACKET_BSZ, MAX_MAX_PACKET_BSZ) != max_packet_bsz){
                dg::network_exception::throw_exception(dg::network_exception::INVALID_ARGUMENT);
            }

            if (!stdx::is_pow2(max_packet_bsz)){
                dg::network_exception::throw_exception(dg::network_exception::INVALID_ARGUMENT);
            }

            return std::make_unique<Packetizer>(get_random_packet_id_generator(factory_addr),
                                                stdx::ulog2(segment_bsz),
                                                max_packet_bsz,
                                                has_integrity_transmit);
        }  

        static auto get_entrance_controller(size_t queue_cap,
                                            size_t unique_id_cap,
                                            std::chrono::nanoseconds expiry_period,
                                            size_t max_consume_decay_factor = 2u) -> std::unique_ptr<EntranceControllerInterface>{
            
            const size_t MIN_QUEUE_CAP                          = size_t{1};
            const size_t MAX_QUEUE_CAP                          = size_t{1} << 30;
            const size_t MIN_UNIQUE_ID_CAP                      = size_t{1};
            const size_t MAX_UNIQUE_ID_CAP                      = size_t{1} << 30;

            const std::chrono::nanoseconds MIN_EXPIRY_PERIOD    = std::chrono::nanoseconds(1);
            const std::chrono::nanoseconds MAX_EXPIRY_PERIOD    = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::hours(1));

            if (std::clamp(queue_cap, MIN_QUEUE_CAP, MAX_QUEUE_CAP) != queue_cap){
                dg::network_exception::throw_exception(dg::network_exception::INVALID_ARGUMENT);
            }

            if (std::clamp(unique_id_cap, MIN_UNIQUE_ID_CAP, MAX_UNIQUE_ID_CAP) != unique_id_cap){
                dg::network_exception::throw_exception(dg::network_exception::INVALID_ARGUMENT);
            }

            if (std::clamp(expiry_period, MIN_EXPIRY_PERIOD, MAX_EXPIRY_PERIOD) != expiry_period){
                dg::network_exception::throw_exception(dg::network_exception::INVALID_ARGUMENT);
            }

            size_t upqueue_cap              = stdx::ceil2(queue_cap);
            size_t tentative_max_consume_sz = std::min(upqueue_cap, unique_id_cap) >> max_consume_decay_factor;
            size_t max_consume_sz           = std::max(size_t{1}, tentative_max_consume_sz);
            auto entrance_entry_pq          = dg::pow2_cyclic_queue<EntranceEntry>(stdx::ulog2(upqueue_cap));
            auto key_id_map                 = dg::unordered_unstable_map<GlobalIdentifier, __uint128_t>{};

            key_id_map.reserve(unique_id_cap);

            return std::make_unique<EntranceController>(std::move(entrance_entry_pq),
                                                        upqueue_cap,
                                                        std::move(key_id_map),
                                                        unique_id_cap,
                                                        __uint128_t{0u},
                                                        expiry_period,
                                                        std::make_unique<std::mutex>(),
                                                        stdx::hdi_container<size_t>{max_consume_sz});
        }

        static auto get_random_hash_distributed_entrance_controller(std::vector<std::unique_ptr<EntranceControllerInterface>> base_vec,
                                                                    std::chrono::nanoseconds drain_period,
                                                                    size_t drain_peek_cap,
                                                                    size_t keyvalue_feed_cap    = DEFAULT_KEYVALUE_FEED_SIZE,
                                                                    size_t zero_bounce_sz       = 8u) -> std::unique_ptr<EntranceControllerInterface>{

            const size_t MIN_ZERO_BOUNCE_SZ     = size_t{1};
            const size_t MAX_ZERO_BOUNCE_SZ     = size_t{1} << 20;
            const size_t MIN_KEYVALUE_FEED_CAP  = size_t{1};
            const size_t MAX_KEYVALUE_FEED_CAP  = size_t{1} << 25;
            const size_t MIN_DRAIN_PEEK_CAP     = size_t{1};
            const size_t MAX_DRAIN_PEEK_CAP     = size_t{1} << 30;

            if (!stdx::is_pow2(base_vec.size())){
                dg::network_exception::throw_exception(dg::network_exception::INVALID_ARGUMENT);
            }

            if (std::clamp(zero_bounce_sz, MIN_ZERO_BOUNCE_SZ, MAX_ZERO_BOUNCE_SZ) != zero_bounce_sz){
                dg::network_exception::throw_exception(dg::network_exception::INVALID_ARGUMENT);
            }

            if (std::clamp(keyvalue_feed_cap, MIN_KEYVALUE_FEED_CAP, MAX_KEYVALUE_FEED_CAP) != keyvalue_feed_cap){
                dg::network_exception::throw_exception(dg::network_exception::INVALID_ARGUMENT);
            }

            if (std::clamp(drain_peek_cap, MIN_DRAIN_PEEK_CAP, MAX_DRAIN_PEEK_CAP) != drain_peek_cap){
                dg::network_exception::throw_exception(dg::network_exception::INVALID_ARGUMENT);
            }

            auto base_arr           = std::make_unique<std::unique_ptr<EntranceControllerInterface>[]>(base_vec.size());
            size_t base_arr_sz      = base_vec.size();
            size_t max_consume_sz   = std::numeric_limits<size_t>::max(); 

            for (size_t i = 0u; i < base_arr_sz; ++i){
                if (base_vec[i] == nullptr){
                    dg::network_exception::throw_exception(dg::network_exception::INVALID_ARGUMENT);
                }

                max_consume_sz  = std::min(max_consume_sz, base_vec[i]->max_consume_size());
                base_arr[i]     = std::move(base_vec[i]);
            }

            size_t trimmed_feed_cap = std::min(max_consume_sz, keyvalue_feed_cap); 

            return std::make_unique<RandomHashDistributedEntranceController>(std::move(base_arr),
                                                                             base_arr_sz,
                                                                             trimmed_feed_cap,
                                                                             zero_bounce_sz,
                                                                             get_timing_drainer(drain_period),
                                                                             drain_peek_cap,
                                                                             max_consume_sz);
        }

        static auto get_exhaustion_controlled_entrance_controller(std::unique_ptr<EntranceControllerInterface> base,
                                                                  std::shared_ptr<dg::network_concurrency_infretry_x::ExecutorInterface> executor,
                                                                  std::shared_ptr<ExhaustionControllerInterface> exhaustion_controller) -> std::unique_ptr<EntranceControllerInterface>{
            
            if (base == nullptr){
                dg::network_exception::throw_exception(dg::network_exception::INVALID_ARGUMENT);
            }

            if (executor == nullptr){
                dg::network_exception::throw_exception(dg::network_exception::INVALID_ARGUMENT);
            }

            if (exhaustion_controller == nullptr){
                dg::network_exception::throw_exception(dg::network_exception::INVALID_ARGUMENT);
            }

            return std::make_unique<ExhaustionControlledEntranceController>(std::move(base),
                                                                            std::move(executor),
                                                                            std::move(exhaustion_controller));
        }

        static auto get_packet_assembler(size_t packet_map_cap,
                                         size_t global_packet_segment_cap,
                                         size_t max_segment_sz_per_stream,
                                         size_t max_consume_decay_factor = 2u) -> std::unique_ptr<PacketAssemblerInterface>{

            const size_t MIN_PACKET_MAP_CAP             = size_t{1};
            const size_t MAX_PACKET_MAP_CAP             = size_t{1} << 30;
            const size_t MIN_GLOBAL_PACKET_SEGMENT_CAP  = size_t{1};
            const size_t MAX_GLOBAL_PACKET_SEGMENT_CAP  = size_t{1} << 30;
            const size_t MIN_MAX_SEGMENT_SZ_PER_STREAM  = size_t{1};
            const size_t MAX_MAX_SEGMENT_SZ_PER_STREAM  = size_t{1} << 30; 

            if (std::clamp(packet_map_cap, MIN_PACKET_MAP_CAP, MAX_PACKET_MAP_CAP) != packet_map_cap){
                dg::network_exception::throw_exception(dg::network_exception::INVALID_ARGUMENT);
            }

            if (std::clamp(global_packet_segment_cap, MIN_GLOBAL_PACKET_SEGMENT_CAP, MAX_GLOBAL_PACKET_SEGMENT_CAP) != global_packet_segment_cap){
                dg::network_exception::throw_exception(dg::network_exception::INVALID_ARGUMENT);
            }

            if (std::clamp(max_segment_sz_per_stream, MIN_MAX_SEGMENT_SZ_PER_STREAM, MAX_MAX_SEGMENT_SZ_PER_STREAM) != max_segment_sz_per_stream){
                dg::network_exception::throw_exception(dg::network_exception::INVALID_ARGUMENT);
            }

            size_t tentative_max_consume_sz = packet_map_cap >> max_consume_decay_factor;
            size_t max_consume_sz           = std::max(size_t{1}, tentative_max_consume_sz);
            
            auto packet_map                 = dg::unordered_unstable_map<GlobalIdentifier, AssembledPacket>();
            packet_map.reserve(packet_map_cap); 

            return std::make_unique<PacketAssembler>(std::move(packet_map),
                                                    packet_map_cap,
                                                    global_packet_segment_cap,
                                                    size_t{0u},
                                                    max_segment_sz_per_stream,
                                                    std::make_unique<std::mutex>(),
                                                    stdx::hdi_container<size_t>{max_consume_sz});
        }

        static auto get_random_hash_distributed_packet_assembler(std::vector<std::unique_ptr<PacketAssemblerInterface>> base_vec,
                                                                 size_t keyvalue_feed_cap = DEFAULT_KEYVALUE_FEED_SIZE){
            
            const size_t MIN_KEYVALUE_FEED_CAP  = size_t{1};
            const size_t MAX_KEYVALUE_FEED_CAP  = size_t{1} << 25;

            if (!stdx::is_pow2(base_vec.size())){
                dg::network_exception::throw_exception(dg::network_exception::INVALID_ARGUMENT);
            }

            if (std::clamp(keyvalue_feed_cap, MIN_KEYVALUE_FEED_CAP, MAX_KEYVALUE_FEED_CAP) != keyvalue_feed_cap){
                dg::network_exception::throw_exception(dg::network_exception::INVALID_ARGUMENT);
            }

            size_t max_consume_sz   = std::numeric_limits<size_t>::max(); 
            auto base_arr           = std::make_unique<std::unique_ptr<PacketAssemblerInterface>[]>(base_vec.size());
            size_t base_arr_sz      = base_vec.size(); 

            for (size_t i = 0u; i < base_arr_sz; ++i){
                if (base_vec[i] == nullptr){
                    dg::network_exception::throw_exception(dg::network_exception::INVALID_ARGUMENT);
                }

                max_consume_sz  = std::min(max_consume_sz, base_vec[i]->max_consume_size());
                base_arr[i]     = std::move(base_vec[i]);
            }

            size_t trimmed_feed_cap = std::min(keyvalue_feed_cap, max_consume_sz);

            return std::make_unique<RandomHashDistributedPacketAssembler>(std::move(base_arr),
                                                                          base_arr_sz,
                                                                          trimmed_feed_cap,
                                                                          max_consume_sz);
        }

        static auto get_exhaustion_controlled_packet_assembler(std::unique_ptr<PacketAssemblerInterface> base,
                                                               std::shared_ptr<dg::network_concurrency_infretry_x::ExecutorInterface> executor,
                                                               std::shared_ptr<ExhaustionControllerInterface> exhaustion_controller) -> std::unique_ptr<PacketAssemblerInterface>{
            
            if (base == nullptr){
                dg::network_exception::throw_exception(dg::network_exception::INVALID_ARGUMENT);
            }

            if (executor == nullptr){
                dg::network_exception::throw_exception(dg::network_exception::INVALID_ARGUMENT);
            }

            if (exhaustion_controller == nullptr){
                dg::network_exception::throw_exception(dg::network_exception::INVALID_ARGUMENT);
            }

            return std::make_unique<ExhaustionControlledPacketAssembler>(std::move(base),
                                                                         std::move(executor),
                                                                         std::move(exhaustion_controller));

        }

        static auto get_buffer_fifo_container(size_t buffer_capacity,
                                              size_t consume_factor = 4u) -> std::unique_ptr<InBoundContainerInterface>{
            
            const size_t MIN_BUFFER_CAPACITY    = 1u;
            const size_t MAX_BUFFER_CAPACITY    = size_t{1} << 25;

            if (std::clamp(buffer_capacity, MIN_BUFFER_CAPACITY, MAX_BUFFER_CAPACITY) != buffer_capacity){
                dg::network_exception::throw_exception(dg::network_exception::INVALID_ARGUMENT);
            }

            size_t tentative_consume_sz         = buffer_capacity >> consume_factor;
            size_t consume_sz                   = std::max(tentative_consume_sz, size_t{1u});
            size_t upqueue_cap                  = stdx::ceil2(buffer_capacity);

            return std::make_unique<BufferFIFOContainer>(dg::pow2_cyclic_queue<dg::string>(stdx::ulog2(upqueue_cap)),
                                                         upqueue_cap,
                                                         std::make_unique<std::mutex>(),
                                                         stdx::hdi_container<size_t>{consume_sz});
        }

        static auto get_exhaustion_controlled_buffer_container(std::unique_ptr<InBoundContainerInterface> base,
                                                               std::shared_ptr<dg::network_concurrency_infretry_x::ExecutorInterface> executor,
                                                               std::shared_ptr<ExhaustionControllerInterface> exhaustion_controller) -> std::unique_ptr<InBoundContainerInterface>{
            
            if (base == nullptr){
                dg::network_exception::throw_exception(dg::network_exception::INVALID_ARGUMENT);
            }

            if (executor == nullptr){
                dg::network_exception::throw_exception(dg::network_exception::INVALID_ARGUMENT);
            }

            if (exhaustion_controller == nullptr){
                dg::network_exception::throw_exception(dg::network_exception::INVALID_ARGUMENT);
            }

            return std::make_unique<ExhaustionControlledBufferContainer>(std::move(base),
                                                                         std::move(executor),
                                                                         std::move(exhaustion_controller));
        }

        template <class Controller>
        static auto get_reacting_buffer_container(std::unique_ptr<InBoundContainerInterface> base,
                                                  size_t reacting_threshold,
                                                  size_t concurrent_subscriber_cap,
                                                  std::chrono::nanoseconds wait_time) -> std::unique_ptr<InBoundContainerInterface>{

            const size_t MIN_REACTING_THRESHOLD             = size_t{1};
            const size_t MAX_REACTING_THRESHOLD             = size_t{1} << 30;
            const std::chrono::nanoseconds MIN_WAIT_TIME    = std::chrono::nanoseconds{1}; 
            const std::chrono::nanoseconds MAX_WAIT_TIME    = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::minutes{1});

            if (base == nullptr){
                dg::network_exception::throw_exception(dg::network_exception::INVALID_ARGUMENT);
            }

            if (std::clamp(reacting_threshold, MIN_REACTING_THRESHOLD, MAX_REACTING_THRESHOLD) != reacting_threshold){
                dg::network_exception::throw_exception(dg::network_exception::INVALID_ARGUMENT);
            }

            if (std::clamp(wait_time, MIN_WAIT_TIME, MAX_WAIT_TIME) != wait_time){
                dg::network_exception::throw_exception(dg::network_exception::INVALID_ARGUMENT);
            }

            return std::make_unique<ReactingBufferContainer<Controller>>(std::move(base),
                                                                         get_complex_reactor(reacting_threshold, concurrent_subscriber_cap),
                                                                         wait_time);
        }

        static auto get_random_hash_distributed_buffer_container(std::vector<std::unique_ptr<InBoundContainerInterface>> base_vec,
                                                                 std::chrono::nanoseconds drain_period,
                                                                 size_t drain_peek_cap,
                                                                 size_t zero_buffer_retry_sz = 8u) -> std::unique_ptr<InBoundContainerInterface>{

            const size_t MIN_BASE_VEC_SZ            = size_t{1};
            const size_t MAX_BASE_VEC_SZ            = size_t{1} << 20;
            const size_t MIN_ZERO_BUFFER_RETRY_SZ   = size_t{1};
            const size_t MAX_ZERO_BUFFER_RETRY_SZ   = size_t{1} << 20;
            const size_t MIN_DRAIN_PEEK_CAP         = size_t{1};
            const size_t MAX_DRAIN_PEEK_CAP         = size_t{1} << 30;

            if (std::clamp(base_vec.size(), MIN_BASE_VEC_SZ, MAX_BASE_VEC_SZ) != base_vec.size()){
                dg::network_exception::throw_exception(dg::network_exception::INVALID_ARGUMENT);
            }

            if (!stdx::is_pow2(base_vec.size())){
                dg::network_exception::throw_exception(dg::network_exception::INVALID_ARGUMENT);
            }

            if (std::clamp(drain_peek_cap, MIN_DRAIN_PEEK_CAP, MAX_DRAIN_PEEK_CAP) != drain_peek_cap){
                dg::network_exception::throw_exception(dg::network_exception::INVALID_ARGUMENT);
            }

            auto base_vec_up        = std::make_unique<std::unique_ptr<InBoundContainerInterface>[]>(base_vec.size());
            size_t consumption_sz   = std::numeric_limits<size_t>::max(); 

            for (size_t i = 0u; i < base_vec.size(); ++i){
                if (base_vec[i] == nullptr){
                    dg::network_exception::throw_exception(dg::network_exception::INVALID_ARGUMENT);
                }

                consumption_sz  = std::min(consumption_sz, base_vec[i]->max_consume_size());
                base_vec_up[i]  = std::move(base_vec[i]);
            }

            if (std::clamp(zero_buffer_retry_sz, MIN_ZERO_BUFFER_RETRY_SZ, MAX_ZERO_BUFFER_RETRY_SZ) != zero_buffer_retry_sz){
                dg::network_exception::throw_exception(dg::network_exception::INVALID_ARGUMENT);
            }

            return std::make_unique<HashDistributedBufferContainer>(std::move(base_vec_up),
                                                                    base_vec.size(),
                                                                    zero_buffer_retry_sz,
                                                                    get_timing_drainer(drain_period),
                                                                    drain_peek_cap,
                                                                    consumption_sz);
        }

        static auto get_expiry_worker(std::shared_ptr<PacketAssemblerInterface> packet_assembler,
                                      std::shared_ptr<EntranceControllerInterface> entrance_controller,
                                      std::shared_ptr<BlackListGateInterface> blacklist_gate,
                                      size_t packet_assembler_vectorization_sz,
                                      size_t consumption_sz,
                                      size_t busy_consumption_sz) -> std::unique_ptr<dg::network_concurrency::WorkerInterface>{
            
            const size_t MIN_PACKET_ASSEMBLER_VECTORIZATION_SZ  = 1u;
            const size_t MAX_PACKET_ASSEMBLER_VECTORIZATION_SZ  = size_t{1} << 25; 
            const size_t MIN_CONSUMPTION_SZ                     = 1u;
            const size_t MAX_CONSUMPTION_SZ                     = size_t{1} << 25;
            const size_t MIN_BUSY_CONSUMPTION_SZ                = 0u;
            const size_t MAX_BUSY_CONSUMPTION_SZ                = size_t{1} << 25;

            if (packet_assembler == nullptr){
                dg::network_exception::throw_exception(dg::network_exception::INVALID_ARGUMENT);
            }

            if (entrance_controller == nullptr){
                dg::network_exception::throw_exception(dg::network_exception::INVALID_ARGUMENT);
            }

            if (blacklist_gate == nullptr){
                dg::network_exception::throw_exception(dg::network_exception::INVALID_ARGUMENT);
            }

            if (std::clamp(packet_assembler_vectorization_sz, MIN_PACKET_ASSEMBLER_VECTORIZATION_SZ, MAX_PACKET_ASSEMBLER_VECTORIZATION_SZ) != packet_assembler_vectorization_sz){
                dg::network_exception::throw_exception(dg::network_exception::INVALID_ARGUMENT);
            }

            if (std::clamp(consumption_sz, MIN_CONSUMPTION_SZ, MAX_CONSUMPTION_SZ) != consumption_sz){
                dg::network_exception::throw_exception(dg::network_exception::INVALID_ARGUMENT);
            }

            if (std::clamp(busy_consumption_sz, MIN_BUSY_CONSUMPTION_SZ, MAX_BUSY_CONSUMPTION_SZ) != busy_consumption_sz){
                dg::network_exception::throw_exception(dg::network_exception::INVALID_ARGUMENT);
            }

            return std::make_unique<ExpiryWorker>(std::move(packet_assembler), std::move(entrance_controller), std::move(blacklist_gate), 
                                                  packet_assembler_vectorization_sz,
                                                  consumption_sz,
                                                  busy_consumption_sz);
        }

        static auto get_inbound_worker(std::shared_ptr<PacketAssemblerInterface> packet_assembler,
                                       std::shared_ptr<InBoundGateInterface> inbound_gate,
                                       std::shared_ptr<InBoundGateInterface> blacklist_gate,
                                       std::shared_ptr<InBoundContainerInterface> inbound_container,
                                       std::shared_ptr<EntranceControllerInterface> entrance_controller,
                                       std::shared_ptr<dg::network_kernel_mailbox_impl1::core::MailboxInterface> base,
                                       size_t packet_assembler_vectorization_sz,
                                       size_t inbound_gate_vectorization_sz,
                                       size_t blacklist_gate_vectorization_sz,
                                       size_t entrance_controller_vectorization_sz,
                                       size_t inbound_container_vectorization_sz,
                                       size_t consumption_sz,
                                       size_t busy_consumption_sz) -> std::unique_ptr<dg::network_concurrency::WorkerInterface>{

            const size_t MIN_PACKET_ASSEMBLER_VECTORIZATION_SZ      = 1u;
            const size_t MAX_PACKET_ASSEMBLER_VECTORIZATION_SZ      = size_t{1} << 25;
            const size_t MIN_INBOUND_GATE_VECTORIZATION_SZ          = 1u;
            const size_t MAX_INBOUND_GATE_VECTORIZATION_SZ          = size_t{1} << 25;
            const size_t MIN_BLACKLIST_GATE_VECTORIZATION_SZ        = 1u;
            const size_t MAX_BLACKLIST_GATE_VECTORIZATION_SZ        = size_t{1} << 25;
            const size_t MIN_ENTRANCE_CONTROLLER_VECTORIZATION_SZ   = 1u;
            const size_t MAX_ENTRANCE_CONTROLLER_VECTORIZATION_SZ   = size_t{1} << 25;
            const size_t MIN_INBOUND_CONTAINER_VECTORIZATION_SZ     = 1u;
            const size_t MAX_INBOUND_CONTAINER_VECTORIZATION_SZ     = size_t{1} << 25;
            const size_t MIN_CONSUMPTION_SZ                         = 1u;
            const size_t MAX_CONSUMPTION_SZ                         = size_t{1} << 25;
            const size_t MIN_BUSY_CONSUMPTION_SZ                    = 0u;
            const size_t MAX_BUSY_CONSUMPTION_SZ                    = size_t{1} << 25;

            if (packet_assembler == nullptr){
                dg::network_exception::throw_exception(dg::network_exception::INVALID_ARGUMENT);
            }

            if (inbound_gate == nullptr){
                dg::network_exception::throw_exception(dg::network_exception::INVALID_ARGUMENT);
            }

            if (blacklist_gate == nullptr){
                dg::network_exception::throw_exception(dg::network_exception::INVALID_ARGUMENT);
            }

            if (inbound_container == nullptr){
                dg::network_exception::throw_exception(dg::network_exception::INVALID_ARGUMENT);
            }

            if (entrance_controller == nullptr){
                dg::network_exception::throw_exception(dg::network_exception::INVALID_ARGUMENT);
            }

            if (base == nullptr){
                dg::network_exception::throw_exception(dg::network_exception::INVALID_ARGUMENT);
            }

            if (std::clamp(packet_assembler_vectorization_sz, MIN_PACKET_ASSEMBLER_VECTORIZATION_SZ, MAX_PACKET_ASSEMBLER_VECTORIZATION_SZ) != packet_assembler_vectorization_sz){
                dg::network_exception::throw_exception(dg::network_exception::INVALID_ARGUMENT);
            }

            if (std::clamp(inbound_gate_vectorization_sz, MIN_INBOUND_GATE_VECTORIZATION_SZ, MAX_INBOUND_GATE_VECTORIZATION_SZ) != inbound_gate_vectorization_sz){
                dg::network_exception::throw_exception(dg::network_exception::INVALID_ARGUMENT);
            }

            if (std::clamp(blacklist_gate_vectorization_sz, MIN_BLACKLIST_GATE_VECTORIZATION_SZ, MAX_BLACKLIST_GATE_VECTORIZATION_SZ) != blacklist_gate_vectorization_sz){
                dg::network_exception::throw_exception(dg::network_exception::INVALID_ARGUMENT);
            }

            if (std::clamp(entrance_controller_vectorization_sz, MIN_ENTRANCE_CONTROLLER_VECTORIZATION_SZ, MAX_ENTRANCE_CONTROLLER_VECTORIZATION_SZ) != entrance_controller_vectorization_sz){
                dg::network_exception::throw_exception(dg::network_exception::INVALID_ARGUMENT);
            }

            if (std::clamp(inbound_container_vectorization_sz, MIN_INBOUND_CONTAINER_VECTORIZATION_SZ, MAX_INBOUND_CONTAINER_VECTORIZATION_SZ) != inbound_container_vectorization_sz){
                dg::network_exception::throw_exception(dg::network_exception::INVALID_ARGUMENT);
            }

            if (std::clamp(consumption_sz, MIN_CONSUMPTION_SZ, MAX_CONSUMPTION_SZ) != consumption_sz){
                dg::network_exception::throw_exception(dg::network_exception::INVALID_ARGUMENT);
            }

            if (std::clamp(busy_consumption_sz, MIN_BUSY_CONSUMPTION_SZ, MAX_BUSY_CONSUMPTION_SZ) != busy_consumption_sz){
                dg::network_exception::throw_exception(dg::network_exception::INVALID_ARGUMENT);
            }

            return std::make_unique<InBoundWorker>(std::move(packet_assembler),
                                                   std::move(inbound_gate),
                                                   std::move(blacklist_gate),
                                                   std::move(inbound_container),
                                                   std::move(entrance_controller),
                                                   std::move(base),
                                                   packet_assembler_vectorization_sz,
                                                   inbound_gate_vectorization_sz,
                                                   blacklist_gate_vectorization_sz,
                                                   entrance_controller_vectorization_sz,
                                                   inbound_container_vectorization_sz,
                                                   consumption_sz,
                                                   busy_consumption_sz);
        }

        static auto get_flash_streamx_mailbox(std::vector<dg::network_concurrency::daemon_raii_handle_t> daemon_vec,
                                              std::unique_ptr<PacketizerInterface> packetizer,
                                              std::shared_ptr<dg::network_kernel_mailbox_impl1::core::MailboxInterface> base,
                                              std::shared_ptr<InBoundContainerInterface> inbound_container,
                                              std::shared_ptr<OutBoundRuleInterface> outbound_rule,
                                              size_t transmission_vectorization_sz) -> std::unique_ptr<dg::network_kernel_mailbox_impl1::core::MailboxInterface>{

            const size_t MIN_TRANSMISSION_VECTORIZATION_SZ  = 1u;
            const size_t MAX_TRANSMISSION_VECTORIZATION_SZ  = size_t{1} << 25; 

            if (packetizer == nullptr){
                dg::network_exception::throw_exception(dg::network_exception::INVALID_ARGUMENT);
            }

            if (base == nullptr){
                dg::network_exception::throw_exception(dg::network_exception::INVALID_ARGUMENT);
            }

            if (inbound_container == nullptr){
                dg::network_exception::throw_exception(dg::network_exception::INVALID_ARGUMENT);
            }

            if (outbound_rule == nullptr){
                dg::network_exception::throw_exception(dg::network_exception::INVALID_ARGUMENT);
            }

            if (std::clamp(transmission_vectorization_sz, MIN_TRANSMISSION_VECTORIZATION_SZ, MAX_TRANSMISSION_VECTORIZATION_SZ) != transmission_vectorization_sz){
                dg::network_exception::throw_exception(dg::network_exception::INVALID_ARGUMENT);
            }

            return std::make_unique<MailBox>(to_dg_vector(std::move(daemon_vec)),
                                             std::move(packetizer),
                                             std::move(base),
                                             std::move(inbound_container),
                                             std::move(outbound_rule),
                                             transmission_vectorization_sz);
        }
    };

    struct Config{
        Address factory_addr;
        uint32_t packetizer_segment_bsz;
        uint32_t packetizer_max_bsz;
        bool packetizer_has_integrity_transmit;

        uint32_t gate_controller_ato_component_sz;
        uint32_t gate_controller_ato_map_capacity;
        std::chrono::nanoseconds gate_controller_ato_dur;
        uint32_t gate_controller_ato_keyvalue_feed_cap;

        uint32_t gate_controller_blklst_component_sz;
        uint32_t gate_controller_blklst_bloomfilter_cap;
        uint32_t gate_controller_blklst_bloomfilter_rehash_sz;
        uint32_t gate_controller_blklst_bloomfilter_reliability_decay_factor;
        uint32_t gate_controller_blklst_keyvalue_feed_cap; 

        uint32_t latency_controller_component_sz;
        uint32_t latency_controller_bounce_sz;
        std::chrono::nanoseconds latency_controller_drain_period;
        uint32_t latency_controller_drain_probe_peek_cap;
        uint32_t latency_controller_queue_cap;
        uint32_t latency_controller_unique_id_cap;
        std::chrono::nanoseconds latency_controller_expiry_period;
        uint32_t latency_controller_keyvalue_feed_cap;
        bool latency_controller_has_exhaustion_control; 

        uint32_t packet_assembler_component_sz;
        uint32_t packet_assembler_map_cap;
        uint32_t packet_assembler_global_segment_cap;
        uint32_t packet_assembler_max_segment_per_stream;
        uint32_t packet_assembler_keyvalue_feed_cap;
        bool packet_assembler_has_exhaustion_control; 

        uint32_t inbound_container_component_sz;
        uint32_t inbound_container_bounce_sz;
        std::chrono::nanoseconds inbound_container_drain_period;
        uint32_t inbound_container_drain_probe_peek_cap;
        uint32_t inbound_container_cap;
        bool inbound_container_has_exhaustion_control;
        bool inbound_container_has_react_pattern;
        uint32_t inbound_container_react_sz;
        uint32_t inbound_container_subscriber_cap;
        std::chrono::nanoseconds inbound_container_react_latency;

        uint32_t expiry_worker_count;
        uint32_t expiry_worker_packet_assembler_vectorization_sz;
        uint32_t expiry_worker_consume_sz;
        uint32_t expiry_worker_busy_consume_sz;

        uint32_t inbound_worker_count;
        uint32_t inbound_worker_packet_assembler_vectorization_sz;
        uint32_t inbound_worker_inbound_gate_vectorization_sz;
        uint32_t inbound_worker_blacklist_gate_vectorization_sz;
        uint32_t inbound_worker_latency_controller_vectorization_sz;
        uint32_t inbound_worker_inbound_container_vectorization_sz;
        uint32_t inbound_worker_consume_sz;
        uint32_t inbound_worker_busy_consume_sz;

        uint32_t mailbox_transmission_vectorization_sz; 

        std::shared_ptr<OutBoundRuleInterface> outbound_rule;
        std::shared_ptr<dg::network_concurrency_infretry_x::ExecutorInterface> infretry_device;
        std::shared_ptr<dg::network_kernel_mailbox_impl1::core::MailboxInterface> base; //alright people might not like std::unique_ptr<> for throwing reasons, yet it is practice of extensibility to make this std::unique<>, we have to break practice for now
    };

    struct ConfigMaker{

        private:

            static auto make_packetizer(Config config) -> std::unique_ptr<PacketizerInterface>{

                return ComponentFactory::get_packetizer(config.factory_addr, config.packetizer_segment_bsz, config.packetizer_max_bsz, config.packetizer_has_integrity_transmit);
            }

            static auto make_ato_gate_controller(Config config) -> std::unique_ptr<InBoundGateInterface>{
                
                if (config.gate_controller_ato_component_sz == 0u){
                    dg::network_exception::throw_exception(dg::network_exception::INVALID_ARGUMENT);
                }

                if (config.gate_controller_ato_component_sz == 1u){
                    return ComponentFactory::get_temporal_absolute_timeout_inbound_gate(config.gate_controller_ato_map_capacity, 
                                                                                        config.gate_controller_ato_dur);
                }

                std::vector<std::unique_ptr<InBoundGateInterface>> inbound_gate_vec{};

                for (size_t i = 0u; i < config.gate_controller_ato_component_sz; ++i){
                    inbound_gate_vec.push_back(ComponentFactory::get_temporal_absolute_timeout_inbound_gate(config.gate_controller_ato_map_capacity,
                                                                                                            config.gate_controller_ato_dur));
                }

                return ComponentFactory::get_random_hash_distributed_inbound_gate(std::move(inbound_gate_vec), config.gate_controller_ato_keyvalue_feed_cap);
            }

            static auto make_blklst_gate_controller(Config config) -> std::unique_ptr<BlackListGateInterface>{

                if (config.gate_controller_blklst_component_sz == 0u){
                    dg::network_exception::throw_exception(dg::network_exception::INVALID_ARGUMENT);
                }

                if (config.gate_controller_blklst_component_sz == 1u){
                    return ComponentFactory::get_temporal_blacklist_gate(config.gate_controller_blklst_bloomfilter_cap,
                                                                         config.gate_controller_blklst_bloomfilter_rehash_sz,
                                                                         config.gate_controller_blklst_bloomfilter_reliability_decay_factor);
                }

                std::vector<std::unique_ptr<BlackListGateInterface>> blklst_gate_vec{};

                for (size_t i = 0u; i < config.gate_controller_blklst_component_sz; ++i){
                    blklst_gate_vec.push_back(ComponentFactory::get_temporal_blacklist_gate(config.gate_controller_blklst_bloomfilter_cap,
                                                                                            config.gate_controller_blklst_bloomfilter_rehash_sz,
                                                                                            config.gate_controller_blklst_bloomfilter_reliability_decay_factor));
                }

                return ComponentFactory::get_random_hash_distributed_blacklist_gate(std::move(blklst_gate_vec), config.gate_controller_blklst_keyvalue_feed_cap);
            }

            static auto make_packet_assembler(Config config) -> std::unique_ptr<PacketAssemblerInterface>{

                if (config.packet_assembler_component_sz == 0u){
                    dg::network_exception::throw_exception(dg::network_exception::INVALID_ARGUMENT);
                }

                if (config.packet_assembler_component_sz == 1u){
                    return ComponentFactory::get_packet_assembler(config.packet_assembler_map_cap,
                                                                  config.packet_assembler_global_segment_cap,
                                                                  config.packet_assembler_max_segment_per_stream);   
                }

                std::vector<std::unique_ptr<PacketAssemblerInterface>> pktasmblr_vec{};

                for (size_t i = 0u; i < config.packet_assembler_component_sz; ++i){
                    pktasmblr_vec.push_back(ComponentFactory::get_packet_assembler(config.packet_assembler_map_cap,
                                                                                   config.packet_assembler_global_segment_cap,
                                                                                   config.packet_assembler_max_segment_per_stream));
                }

                return ComponentFactory::get_random_hash_distributed_packet_assembler(std::move(pktasmblr_vec), config.packet_assembler_keyvalue_feed_cap);
            }

            static auto make_inbound_container(Config config) -> std::unique_ptr<InBoundContainerInterface>{

                if (config.inbound_container_component_sz == 0u){
                    dg::network_exception::throw_exception(dg::network_exception::INVALID_ARGUMENT);
                }

                if (config.inbound_container_has_exhaustion_control){
                    if (config.inbound_container_has_react_pattern){
                        if (config.inbound_container_component_sz == 1u){
                            return ComponentFactory::get_exhaustion_controlled_buffer_container(ComponentFactory::get_reacting_buffer_container<client_affined_timeout_controller>(ComponentFactory::get_buffer_fifo_container(config.inbound_container_cap),
                                                                                                                                                                                   config.inbound_container_react_sz,
                                                                                                                                                                                   config.inbound_container_subscriber_cap,
                                                                                                                                                                                   config.inbound_container_react_latency),
                                                                                                config.infretry_device,
                                                                                                ComponentFactory::get_no_exhaustion_controller());
                        } else{
                            auto inbound_container_vec = std::vector<std::unique_ptr<InBoundContainerInterface>>(config.inbound_container_component_sz);
                            std::generate(inbound_container_vec.begin(), inbound_container_vec.end(), std::bind_front(ComponentFactory::get_buffer_fifo_container, config.inbound_container_cap, 4u)); //

                            return ComponentFactory::get_exhaustion_controlled_buffer_container(ComponentFactory::get_reacting_buffer_container<client_affined_timeout_controller>(ComponentFactory::get_random_hash_distributed_buffer_container(std::move(inbound_container_vec),
                                                                                                                                                                                                                                                  config.inbound_container_drain_period, 
                                                                                                                                                                                                                                                  config.inbound_container_drain_probe_peek_cap,
                                                                                                                                                                                                                                                  config.inbound_container_bounce_sz),
                                                                                                                                                config.inbound_container_react_sz,
                                                                                                                                                config.inbound_container_subscriber_cap,
                                                                                                                                                config.inbound_container_react_latency),
                                                                                                config.infretry_device,
                                                                                                ComponentFactory::get_no_exhaustion_controller());
                        }
                    } else{
                        if (config.inbound_container_component_sz == 1u){
                            return ComponentFactory::get_exhaustion_controlled_buffer_container(ComponentFactory::get_buffer_fifo_container(config.inbound_container_cap),
                                                                                                config.infretry_device,
                                                                                                ComponentFactory::get_no_exhaustion_controller());
                        } else{
                            auto inbound_container_vec = std::vector<std::unique_ptr<InBoundContainerInterface>>(config.inbound_container_component_sz);
                            std::generate(inbound_container_vec.begin(), inbound_container_vec.end(), std::bind_front(ComponentFactory::get_buffer_fifo_container, config.inbound_container_cap, 4u)); //

                            return ComponentFactory::get_exhaustion_controlled_buffer_container(ComponentFactory::get_random_hash_distributed_buffer_container(std::move(inbound_container_vec),
                                                                                                                                                               config.inbound_container_drain_period,
                                                                                                                                                               config.inbound_container_drain_probe_peek_cap,
                                                                                                                                                               config.inbound_container_bounce_sz),
                                                                                                config.infretry_device,
                                                                                                ComponentFactory::get_no_exhaustion_controller());
                        }
                    }
                } else{
                    if (config.inbound_container_has_react_pattern){
                        if (config.inbound_container_component_sz == 1u){
                            return ComponentFactory::get_reacting_buffer_container<client_affined_timeout_controller>(ComponentFactory::get_buffer_fifo_container(config.inbound_container_cap),
                                                                                                                      config.inbound_container_react_sz,
                                                                                                                      config.inbound_container_subscriber_cap,
                                                                                                                      config.inbound_container_react_latency);
                        } else{
                            auto inbound_container_vec = std::vector<std::unique_ptr<InBoundContainerInterface>>(config.inbound_container_component_sz);
                            std::generate(inbound_container_vec.begin(), inbound_container_vec.end(), std::bind_front(ComponentFactory::get_buffer_fifo_container, config.inbound_container_cap, 4u)); //

                            return ComponentFactory::get_reacting_buffer_container<client_affined_timeout_controller>(ComponentFactory::get_random_hash_distributed_buffer_container(std::move(inbound_container_vec),
                                                                                                                                                                                     config.inbound_container_drain_period,
                                                                                                                                                                                     config.inbound_container_drain_probe_peek_cap,
                                                                                                                                                                                     config.inbound_container_bounce_sz),
                                                                                   config.inbound_container_react_sz,
                                                                                   config.inbound_container_subscriber_cap,
                                                                                   config.inbound_container_react_latency);
                        }
                    } else{
                        if (config.inbound_container_component_sz == 1u){
                            return ComponentFactory::get_buffer_fifo_container(config.inbound_container_cap);
                        } else{
                            auto inbound_container_vec = std::vector<std::unique_ptr<InBoundContainerInterface>>(config.inbound_container_component_sz);
                            std::generate(inbound_container_vec.begin(), inbound_container_vec.end(), std::bind_front(ComponentFactory::get_buffer_fifo_container, config.inbound_container_cap, 4u)); //

                            return ComponentFactory::get_random_hash_distributed_buffer_container(std::move(inbound_container_vec),
                                                                                                  config.inbound_container_drain_period,
                                                                                                  config.inbound_container_drain_probe_peek_cap,
                                                                                                  config.inbound_container_bounce_sz);
                        }
                    }
                }
            }

            static auto make_latency_controller(Config config) -> std::unique_ptr<EntranceControllerInterface>{

                if (config.latency_controller_component_sz == 0u){
                    dg::network_exception::throw_exception(dg::network_exception::INVALID_ARGUMENT);
                }

                if (config.latency_controller_has_exhaustion_control){
                    if (config.latency_controller_component_sz == 1u){
                        return ComponentFactory::get_exhaustion_controlled_entrance_controller(ComponentFactory::get_entrance_controller(config.latency_controller_queue_cap, 
                                                                                                                                         config.latency_controller_unique_id_cap,
                                                                                                                                         config.latency_controller_expiry_period),
                                                                                               config.infretry_device,
                                                                                               ComponentFactory::get_no_exhaustion_controller());
                    } else{
                        std::vector<std::unique_ptr<EntranceControllerInterface>> entrance_controller_vec(config.latency_controller_component_sz);
                        auto gen = std::bind_front(ComponentFactory::get_entrance_controller, config.latency_controller_queue_cap, config.latency_controller_unique_id_cap, config.latency_controller_expiry_period, 2u);
                        std::generate(entrance_controller_vec.begin(), entrance_controller_vec.end(), gen);

                        return ComponentFactory::get_exhaustion_controlled_entrance_controller(ComponentFactory::get_random_hash_distributed_entrance_controller(std::move(entrance_controller_vec),
                                                                                                                                                                 config.latency_controller_drain_period,
                                                                                                                                                                 config.latency_controller_drain_probe_peek_cap,
                                                                                                                                                                 config.latency_controller_keyvalue_feed_cap,
                                                                                                                                                                 config.latency_controller_bounce_sz),
                                                                                               config.infretry_device,
                                                                                               ComponentFactory::get_no_exhaustion_controller());
                    }
                } else{
                    if (config.latency_controller_component_sz == 1u){
                        return ComponentFactory::get_entrance_controller(config.latency_controller_queue_cap,
                                                                         config.latency_controller_unique_id_cap,
                                                                         config.latency_controller_expiry_period);
                    } else{
                        std::vector<std::unique_ptr<EntranceControllerInterface>> entrance_controller_vec(config.latency_controller_component_sz);
                        auto gen = std::bind_front(ComponentFactory::get_entrance_controller, config.latency_controller_queue_cap, config.latency_controller_unique_id_cap, config.latency_controller_expiry_period, 2u);
                        std::generate(entrance_controller_vec.begin(), entrance_controller_vec.end(), gen);

                        return ComponentFactory::get_random_hash_distributed_entrance_controller(std::move(entrance_controller_vec),
                                                                                                 config.latency_controller_drain_period,
                                                                                                 config.latency_controller_drain_probe_peek_cap,
                                                                                                 config.latency_controller_keyvalue_feed_cap,
                                                                                                 config.latency_controller_bounce_sz);
                    }
                }
            }

            static auto make_mailbox_x(std::shared_ptr<dg::network_kernel_mailbox_impl1::core::MailboxInterface> base,
                                       std::shared_ptr<OutBoundRuleInterface> outbound_rule,
                                       std::unique_ptr<PacketizerInterface> packetizer,
                                       std::unique_ptr<InBoundGateInterface> inbound_gate,
                                       std::unique_ptr<BlackListGateInterface> blacklist_gate,
                                       std::unique_ptr<PacketAssemblerInterface> packet_assembler,
                                       std::unique_ptr<InBoundContainerInterface> inbound_container,
                                       std::unique_ptr<EntranceControllerInterface> entrance_controller,
                                       size_t expiry_worker_count,
                                       size_t expiry_worker_packet_assembler_vectorization_sz,
                                       size_t expiry_worker_consume_sz,
                                       size_t expiry_worker_busy_consume_sz,
                                       size_t inbound_worker_count,
                                       size_t inbound_worker_packet_assembler_vectorization_sz,
                                       size_t inbound_worker_inbound_gate_vectorization_sz,
                                       size_t inbound_worker_blacklist_gate_vectorization_sz,
                                       size_t inbound_worker_entrance_controller_vectorization_sz,
                                       size_t inbound_worker_inbound_container_vectorization_sz,
                                       size_t inbound_worker_consume_sz,
                                       size_t inbound_worker_busy_consume_sz,
                                       size_t mailbox_transmission_vectorization_sz,
                                       std::shared_ptr<dg::network_concurrency_infretry_x::ExecutorInterface> infretry_device) -> std::unique_ptr<dg::network_kernel_mailbox_impl1::core::MailboxInterface>{

                const size_t MIN_EXPIRY_WORKER_COUNT        = 1u;
                const size_t MAX_EXPIRY_WORKER_COUNT        = 1024u; 
                const size_t MIN_INBOUND_WORKER_COUNT       = 1u;
                const size_t MAX_INBOUND_WORKER_COUNT       = 1024u;

                if (base == nullptr){
                    dg::network_exception::throw_exception(dg::network_exception::INVALID_ARGUMENT);
                }

                if (packetizer == nullptr){
                    dg::network_exception::throw_exception(dg::network_exception::INVALID_ARGUMENT);
                }

                if (inbound_gate == nullptr){
                    dg::network_exception::throw_exception(dg::network_exception::INVALID_ARGUMENT);
                }

                if (blacklist_gate == nullptr){
                    dg::network_exception::throw_exception(dg::network_exception::INVALID_ARGUMENT);
                }

                if (packet_assembler == nullptr){
                    dg::network_exception::throw_exception(dg::network_exception::INVALID_ARGUMENT);
                }

                if (inbound_container == nullptr){
                    dg::network_exception::throw_exception(dg::network_exception::INVALID_ARGUMENT);
                }

                if (entrance_controller == nullptr){
                    dg::network_exception::throw_exception(dg::network_exception::INVALID_ARGUMENT);
                }

                if (std::clamp(expiry_worker_count, MIN_EXPIRY_WORKER_COUNT, MAX_EXPIRY_WORKER_COUNT) != expiry_worker_count){
                    dg::network_exception::throw_exception(dg::network_exception::INVALID_ARGUMENT);
                }

                if (std::clamp(inbound_worker_count, MIN_INBOUND_WORKER_COUNT, MAX_INBOUND_WORKER_COUNT) != inbound_worker_count){
                    dg::network_exception::throw_exception(dg::network_exception::INVALID_ARGUMENT);
                }

                if (infretry_device == nullptr){
                    dg::network_exception::throw_exception(dg::network_exception::INVALID_ARGUMENT);
                }

                std::shared_ptr<InBoundGateInterface> inbound_gate_sp                   = std::move(inbound_gate);
                std::shared_ptr<BlackListGateInterface> blacklist_gate_sp               = std::move(blacklist_gate);
                std::shared_ptr<PacketAssemblerInterface> packet_assembler_sp           = std::move(packet_assembler);
                std::shared_ptr<InBoundContainerInterface> inbound_container_sp         = std::move(inbound_container);
                std::shared_ptr<EntranceControllerInterface> entrance_controller_sp     = std::move(entrance_controller);
                std::vector<dg::network_concurrency::daemon_raii_handle_t> daemon_vec   = {};

                for (size_t i = 0u; i < inbound_worker_count; ++i){
                    auto worker = ComponentFactory::get_inbound_worker(packet_assembler_sp, inbound_gate_sp, blacklist_gate_sp, 
                                                                       inbound_container_sp, entrance_controller_sp, base,
                                                                       inbound_worker_packet_assembler_vectorization_sz,
                                                                       inbound_worker_inbound_gate_vectorization_sz,
                                                                       inbound_worker_blacklist_gate_vectorization_sz,
                                                                       inbound_worker_entrance_controller_vectorization_sz,
                                                                       inbound_worker_inbound_container_vectorization_sz,
                                                                       inbound_worker_consume_sz,
                                                                       inbound_worker_busy_consume_sz);

                    auto daemon_handle  = dg::network_exception_handler::throw_nolog(dg::network_concurrency::daemon_saferegister(dg::network_concurrency::IO_DAEMON, std::move(worker)));
                    daemon_vec.emplace_back(std::move(daemon_handle));
                }

                for (size_t i = 0u; i < expiry_worker_count; ++i){
                    auto worker = ComponentFactory::get_expiry_worker(packet_assembler_sp, entrance_controller_sp, blacklist_gate_sp,
                                                                      expiry_worker_packet_assembler_vectorization_sz,
                                                                      expiry_worker_consume_sz,
                                                                      expiry_worker_busy_consume_sz);

                    auto daemon_handle  = dg::network_exception_handler::throw_nolog(dg::network_concurrency::daemon_saferegister(dg::network_concurrency::IO_DAEMON, std::move(worker)));
                    daemon_vec.emplace_back(std::move(daemon_handle));
                }

                return ComponentFactory::get_flash_streamx_mailbox(std::move(daemon_vec),
                                                                   std::move(packetizer),
                                                                   base,
                                                                   inbound_container_sp,
                                                                   outbound_rule,
                                                                   mailbox_transmission_vectorization_sz);
            }

        public:

            static auto make(Config config) -> std::unique_ptr<dg::network_kernel_mailbox_impl1::core::MailboxInterface>{

                return make_mailbox_x(config.base,
                                      config.outbound_rule,
                                      make_packetizer(config),
                                      make_ato_gate_controller(config),
                                      make_blklst_gate_controller(config),
                                      make_packet_assembler(config),
                                      make_inbound_container(config),
                                      make_latency_controller(config),
                                      config.expiry_worker_count,
                                      config.expiry_worker_packet_assembler_vectorization_sz,
                                      config.expiry_worker_consume_sz,
                                      config.expiry_worker_busy_consume_sz,
                                      config.inbound_worker_count,
                                      config.inbound_worker_packet_assembler_vectorization_sz,
                                      config.inbound_worker_inbound_gate_vectorization_sz,
                                      config.inbound_worker_blacklist_gate_vectorization_sz,
                                      config.inbound_worker_latency_controller_vectorization_sz,
                                      config.inbound_worker_inbound_container_vectorization_sz,
                                      config.inbound_worker_consume_sz,
                                      config.inbound_worker_busy_consume_sz,
                                      config.mailbox_transmission_vectorization_sz,
                                      config.infretry_device);
            }
    };

    extern auto get_empty_outbound_rule() -> std::unique_ptr<OutBoundRuleInterface>{

        return ComponentFactory::get_empty_outbound_rule();
    } 

    extern auto spawn(Config config) -> std::unique_ptr<dg::network_kernel_mailbox_impl1::core::MailboxInterface>{

        return ConfigMaker::make(config);
    }

    static auto internal_get_affined_blockonsend_permission_guard() noexcept{

        auto prev_state             = client_affined_blocksend_controller::get();
        auto resource_scope_guard   = [prev_state]() noexcept{
            client_affined_blocksend_controller::abs_set(prev_state);
        };

        return stdx::resource_guard(resource_scope_guard);
    } 

    extern void send_2(dg::network_kernel_mailbox_impl1::core::MailboxInterface * obj, std::move_iterator<MailBoxArgument *> data_arr, size_t sz, exception_t * exception_arr, bool is_block) noexcept{

        auto permission_guard = internal_get_affined_blockonsend_permission_guard();
        client_affined_blocksend_controller::set(is_block);
        stdx::safe_ptr_access(obj)->send(data_arr, sz, exception_arr);        
    }

    static auto internal_get_affined_timeout_guard() noexcept{

        auto prev_state             = client_affined_timeout_controller::get();
        auto resource_scope_guard   = [prev_state]() noexcept{
            client_affined_timeout_controller::abs_set(prev_state);
        };

        return stdx::resource_guard(resource_scope_guard);
    }

    extern void recv_2(dg::network_kernel_mailbox_impl1::core::MailboxInterface * obj, dg::string * output_arr, size_t& output_arr_sz, size_t output_arr_cap, std::chrono::nanoseconds timeout) noexcept{

        auto timeout_guard = internal_get_affined_timeout_guard();
        client_affined_timeout_controller::set(timeout);
        stdx::safe_ptr_access(obj)->recv(output_arr, output_arr_sz, output_arr_cap);
    }
}

// namespace dg::network_kernel_mailbox_impl1_radixx{

//     //this is hard to write

//     using Address = dg::network_kernel_mailbox_impl1::model::Address; 
//     using radix_t = uint32_t; 

//     struct RadixMessage{
//         radix_t radix;
//         dg::string content;

//         template <class Reflector>
//         void dg_reflect(const Reflector& reflector) const{
//             reflector(radix, content);
//         }

//         template <class Reflector>
//         void dg_reflect(const Reflector& reflector){
//             reflector(radix, content);
//         }
//     };

//     static auto serialize_radixmsg(RadixMessage&& inp) noexcept -> std::expected<dg::string, exception_t>{

//     }

//     static auto deserialize_radixmsg(dg::string&& inp) noexcept -> std::expected<RadixMessage, exception_t>{

//     }

//     struct RadixMailBoxArgument{
//         Address to;
//         dg::string content;
//         radix_t radix;
//     }; 

//     struct RadixMailboxInterface{
//         virtual ~RadixMailboxInterface() noexcept = default;
//         virtual void send(std::move_iterator<RadixMailBoxArgument *> data_arr, size_t data_arr_sz, exception_t * exception_arr) noexcept = 0;
//         virtual void recv(dg::string * output_arr, size_t& output_arr_sz, size_t output_arr_cap, radix_t radix) noexcept = 0;
//     };

//     struct InBoundContainerInterface{
//         virtual ~InBoundContainerInterface() noexcept = default;
//         virtual void push(std::move_iterator<dg::string *>, size_t, exception_t *) noexcept = 0;
//         virtual void pop(dg::string *, size_t&, size_t) noexcept = 0;
//     };

//     struct RadixInBoundContainerInterface{
//         virtual ~RadixInBoundContainerInterface() noexcept = default;
//         virtual void push(radix_t, std::move_iterator<dg::string *>, size_t, exception_t *) noexcept = 0;
//         virtual void pop(radix_t, dg::string *, size_t&, size_t) noexcept = 0;
//     };

//     class RadixMailBox: public virtual RadixMailboxInterface{

//         private:

//             dg::vector<dg::network_concurrency::daemon_raii_handle_t> daemon_vec; 
//             std::shared_ptr<dg::network_kernel_mailbox_impl1::core::MailboxInterface> base;
//             std::shared_ptr<InBoundContainerInterface> inbound_container;

//         public:

//             RadixMailBox(dg::vector<dg::network_concurrency::daemon_raii_handle_t> daemon_vec, 
//                          std::shared_ptr<dg::network_kernel_mailbox_impl1::core::MailboxInterface> base,
//                          std::shared_ptr<InBoundContainerInterface> inbound_container) noexcept: daemon_vec(std::move(daemon_vec)),
//                                                                                                  base(std::move(base)),
//                                                                                                  inbound_container(std::move(inbound_container)){}
            
//     };

//     struct Factory{

//         static auto spawn_exhaustion_controller(size_t capacity) -> std::unique_ptr<ExhaustionControllerInterface>{

//             const size_t MIN_CAPACITY   = 1u;
//             const size_t MAX_CAPACITY   = size_t{1} << 20;

//             if (std::clamp(capacity, MIN_CAPACITY, MAX_CAPACITY) != capacity){
//                 dg::network_exception::throw_exception(dg::network_exception::INVALID_ARGUMENT);
//             }

//             auto mtx = std::make_unique<std::mutex>(); 
//             return std::make_unique<StdExhaustionController>(0u, capacity, std::move(mtx));
//         }

//         static auto spawn_inbound_container() -> std::unique_ptr<InBoundContainerInterface>{

//             auto map    = dg::unordered_map<radix_t, dg::deque<dg::string>>{};
//             auto mtx    = std::make_unique<std::mutex>(); 

//             return std::make_unique<InBoundContainer>(std::move(map), std::move(mtx));
//         }

//         static auto spawn_exhaustion_controlled_inbound_container(dg::unordered_map<radix_t, size_t> exhaustion_map,
//                                                                   std::shared_ptr<dg::network_concurrency_infretry_x::ExecutorInterface> executor) -> std::unique_ptr<InBoundContainerInterface>{
            
//             if (executor == nullptr){
//                 dg::network_exception::throw_exception(dg::network_exception::INVALID_ARGUMENT);
//             }

//             dg::unordered_map<radix_t, std::unique_ptr<ExhaustionControllerInterface>> exhaustion_controller_map{};

//             for (const auto& pair: exhaustion_map){
//                 exhaustion_controller_map.insert(std::make_pair(std::get<0>(pair), spawn_exhaustion_controller(std::get<1>(pair))));
//             }

//             std::unique_ptr<InBoundContainerInterface> base = spawn_inbound_container();
//             std::unique_ptr<std::mutex> mtx = std::make_unique<std::mutex>(); 

//             return std::make_unique<ExhaustionControlledInBoundContainer>(std::move(base), std::move(exhaustion_controller_map), std::move(executor), std::move(mtx));
//         }

//         static auto spawn_mailbox_radixx(std::unique_ptr<dg::network_kernel_mailbox_impl1::core::MailboxInterface> base,
//                                          std::unique_ptr<InBoundContainerInterface> inbound_container,
//                                          size_t num_inbound_worker) -> std::unique_ptr<RadixMailboxInterface>{
            
//             const size_t MIN_INBOUND_WORKER = 1u;
//             const size_t MAX_INBOUND_WORKER = 1024u; 

//             if (base == nullptr){
//                 dg::network_exception::throw_exception(dg::network_exception::INVALID_ARGUMENT);
//             }

//             if (inbound_container == nullptr){
//                 dg::network_exception::throw_exception(dg::network_exception::INVALID_ARGUMENT);
//             }

//             if (std::clamp(num_inbound_worker, MIN_INBOUND_WORKER, MAX_INBOUND_WORKER) != num_inbound_worker){
//                 dg::network_exception::throw_exception(dg::network_exception::INVALID_ARGUMENT);
//             }

//             dg::vector<dg::network_concurrency::daemon_raii_handle_t> daemon_vec{};
//             std::shared_ptr<dg::network_kernel_mailbox_impl1::core::MailboxInterface> base_sp = std::move(base);
//             std::shared_ptr<InBoundContainerInterface> inbound_container_sp = std::move(inbound_container);

//             //TODO: this is buggy - this will be DL if exception raises - not to worry about this now
//             for (size_t i = 0u; i < num_inbound_worker; ++i){
//                 auto worker = std::make_unique<InBoundWorker>(base_sp, inbound_container_sp);
//                 auto handle = dg::network_exception_handler::throw_nolog(dg::network_concurrency::daemon_saferegister(dg::network_concurrency::IO_DAEMON, std::move(worker)));
//                 daemon_vec.push_back(std::move(handle));
//             }

//             return std::make_unique<RadixMailBox>(std::move(daemon_vec), base_sp, inbound_container_sp);
//         }
//     };

//     struct Config{
//         dg::unordered_map<radix_t, size_t> exhaustion_capacity_map;
//         std::unique_ptr<dg::network_kernel_mailbox_impl1::core::MailboxInterface> base;
//         std::shared_ptr<dg::network_concurrency_infretry_x::ExecutorInterface> retry_device;
//         size_t num_inbound_worker;
//     };

//     auto spawn(Config config) -> std::unique_ptr<RadixMailboxInterface>{

//         auto inbound_container = Factory::spawn_exhaustion_controlled_inbound_container(config.exhaustion_capacity_map, config.retry_device);
//         auto rs = Factory::spawn_mailbox_radixx(std::move(config.base), std::move(inbound_container), config.num_inbound_worker); 

//         return rs;
//     }
// };

/*
namespace dg::network_kernel_mailbox_impl1_heartbeatx{

    using radix_t = network_kernel_mailbox_impl1_radixx::radix_t; 

    struct ObserverInterface{
        virtual ~ObserverInterface() noexcept = default;
        virtual void notify() noexcept = 0; //this has to be an exitable-in-all-scenerios-invoke - to enforce this - return a std::atomic<bool>& for notifier to set - it's the observer's worker responsibility to observe the std::atomic<bool>& intervally  
    };

    struct HeartBeatMonitorInterface{
        virtual ~HeartBeatMonitorInterface() noexcept = default;
        virtual void recv_signal(const Address&) noexcept = 0;
        virtual bool check() noexcept = 0;
    };

    class HeartBeatMonitor: public virtual HeartBeatMonitorInterface{

        private:

            dg::unordered_map<Address, std::chrono::nanoseconds> address_ts_dict;
            std::chrono::nanoseconds error_threshold;
            std::chrono::nanoseconds termination_threshold;
            std::unique_ptr<std::mutex> mtx;

        public:

            HeartBeatMonitor(dg::unordered_map<Address, std::chrono::nanoseconds> address_ts_dict,
                             std::chrono::nanoseconds error_threshold,
                             std::chrono::nanoseconds termination_threshold,
                             std::unique_ptr<std::mutex> mtx) noexcept: address_ts_dict(std::move(address_ts_dict)),
                                                                        error_threshold(std::move(error_threshold)),
                                                                        termination_threshold(std::move(termination_threshold)),
                                                                        mtx(std::move(mtx)){}

            void recv_signal(const Address& addr) noexcept{

                stdx::xlock_guard<std::mutex> lck_grd(*this->mtx);
                auto ptr = this->address_ts_dict.find(addr);

                if (ptr == this->address_ts_dict.end()){
                    auto err_msg = this->make_foreign_heartbeat_error_msg(addr);
                    dg::network_log::error_fast(err_msg.c_str());
                    return;
                }

                ptr->second = dg::network_genult::unix_timestamp();
            }

            bool check() noexcept{

                stdx::xlock_guard<std::mutex> lck_grd(*this->mtx);
                std::chrono::nanoseconds now = dg::network_genult::unix_timestamp(); 
                bool status = true; 

                for (const auto& pair: this->address_ts_dict){
                    if (dg::network_genult::timelapsed(pair.second, now) > this->error_threshold){
                        auto err_msg = this->make_missing_heartbeat_error_msg(pair.first); 
                        dg::network_log::error_fast(err_msg.c_str());
                        status = false;
                    }

                    if (dg::network_genult::timelapsed(pair.second, now) > this->termination_threshold){
                        auto err_msg = this->make_missing_heartbeat_error_msg(pair.first);
                        dg::network_log::critical(err_msg.c_str());
                        std::abort();
                    }
                }

                return status;
            }
        
        private:

            auto make_missing_heartbeat_error_msg(const Address& addr) const noexcept -> dg::string{ //global memory pool - better to be noexcept here

                const char * fmt = "[NETWORKSTACK_HEARTBEAT] heartbeat not detected from {}:{}"; //ip-resolve is done externally - via log_reading - virtual ip is required to spawn proxy (if a node is not responding)
                return std::format(fmt, addr.ip, size_t{addr.port});
            }

            auto make_foreign_heartbeat_error_msg(const Address& addr) const noexcept -> dg::string{

                const char * fmt = "[NETWORKSTACK_HEARTBEAT] foreign heartbeat from {}:{}";
                return std::format(fmt, addr.ip, size_t{addr.port});
            }
    };

    class HeartBeatBroadcaster: public virtual dg::network_concurrency::WorkerInterface{

        private:

            dg::vector<Address> addr_table;
            Address host_addr;
            std::shared_ptr<network_kernel_mailbox_impl1_radixx::MailBoxInterface> mailbox;
            radix_t heartbeat_channel; 

        public:

            HeartBeatBroadcaster(dg::vector<Address> addr_table,
                                 Address host_addr,
                                 std::shared_ptr<network_kernel_mailbox_impl1_radixx::MailBoxInterface> mailbox,
                                 radix_t heartbeat_channel) noexcept addr_table(std::move(addr_table)),
                                                                     host_addr(std::move(host_addr)),
                                                                     mailbox(std::move(mailbox)),
                                                                     heartbeat_channel(std::move(heartbeat_channel)){}

            bool run_one_epoch() noexcept{
                
                size_t host_addr_sz = dg::network_compact_serializer::size(this->host_addr);
                dg::string serialized_host_addr(host_addr_sz);
                dg::network_compact_serializer::serialize_into(serialized_host_addr.data(), this->host_addr); 

                for (const auto& addr: this->addr_table){
                     this->mailbox->send(addr, serialized_host_addr, this->heartbeat_channel);
                }

                return true;
            }
    };

    class HeartBeatReceiver: public virtual dg::network_concurrency::WorkerInterface{

        private:

            std::unique_ptr<HeartBeatMonitorInterface> heartbeat_monitor;
            std::shared_ptr<network_kernel_mailbox_impl1_radixx::MailBoxInterface> mailbox;
            std::shared_ptr<ObserverInterface> observer;
            radix_t heartbeat_channel; 

        public:

            HeartBeatReceiver(std::unique_ptr<HeartBeatMonitorInterface> heartbeat_monitor,
                              std::shared_ptr<network_kernel_mailbox_impl1_radixx::MailBoxInterface> mailbox,
                              std::shared_ptr<ObserverInterface> observer,
                              radix_t heartbeat_channel) noexcept: heartbeat_monitor(std::move(heartbeat_monitor)),
                                                                   mailbox(std::move(mailbox)),
                                                                   observer(std::move(observer)),
                                                                   heartbeat_channel(std::move(heartbeat_channel)){}
            
            bool run_one_epoch() noexcept{

                if (!this->heartbeat_monitor->check()){
                    this->observer->notify();
                }

                std::optional<dg::string> buf = this->mailbox->recv(this->heartbeat_channel);
                
                if (!static_cast<bool>(buf)){
                    return false;
                }

                Address heartbeat_addr{};
                dg::network_compact_serializer::deserialize_into(hearbeat_addr, buf->data()); 
                this->heartbeat_monitor->recv_signal(heartbeat_addr);

                return true;
            }
    };

    class MailBox: public virtual network_kernel_mailbox_impl1_radixx::MailBoxInterface{

        private:

            dg::vector<dg::network_concurrency::daemon_raii_handle_t> daemons;
            std::shared_ptr<network_kernel_mailbox_impl1_radixx::MailBoxInterface> mailbox;

        public:

            MailBox(dg::vector<dg::network_concurrency::daemon_raii_handle_t> daemons,
                    std::shared_ptr<dg::network_kernel_mailbox_impl1::core::MailboxInterface> mailbox) noexcept: daemons(std::move(daemons)),
                                                                                                                 mailbox(std::move(mailbox)){}


            void send(Address addr, dg::string buf, radix_t radix) noexcept{
                
                this->mailbox->send(std::move(addr), std::move(buf), radix);
            }

            auto recv(radix_t radix) noexcept -> std::optional<dg::string>{

                return this->mailbox->recv(radix);
            }
    };
}

namespace dg::network_kernel_mailbox_impl1_concurrentx{


    using radix_t = dg::network_kernel_mailbox_impl1_radixx::radix_t; 

    template <size_t CONCURRENCY_SZ>
    class ConcurrentMailBox: public virtual dg::network_kernel_mailbox_impl1_radixx::MailBoxInterface{

        private:

            dg::vector<std::unique_ptr<dg::network_kernel_mailbox_impl1_radixx::MailBoxInterface>> mailbox_vec;
        
        public:

            ConcurrentMailBox(dg::vector<std::unique_ptr<dg::network_kernel_mailbox_impl1_radixx::MailBoxInterface>> mailbox_vec,
                              std::integral_constant<size_t, CONCURRENCY_SZ>) noexcept: mailbox_vec(std::move(mailbox_vec)){}

            void send(Address addr, dg::string buf, radix_t radix) noexcept{

                size_t idx = dg::network_randomizer::randomize_range(std::integral_constant<size_t, CONCURRENCY_SZ>{}); 
                this->mailbox_vec[idx]->send(std::move(addr), std::move(buf), radix);
            }

            auto recv(radix_t radix) noexcept -> std::optional<std::network_std_container::string>{

                size_t idx = dg::network_randomizer::randomize_range(std::integral_constant<size_t, CONCURRENCY_SZ>{}); 
                return this->mailbox_vec[idx]->recv(radix);
            }
    };
}
*/

#endif
