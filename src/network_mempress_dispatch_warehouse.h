#ifndef __NETWORK_MEMPRESS_DISPATCH_WAREHOUSE_H__
#define __NETWORK_MEMPRESS_DISPATCH_WAREHOUSE_H__

namespace dg::network_mempress_dispatch_warehouse{

    //in this component, we can't latencize, we can't do workorder as a ProducerConsumer pointer approach
    //such is that a unit is an absolute unit of consumption
    //collector collects from mailbox -> dispatch warehouse -> the resolutor
    //dispatch warehouse must store things in dg::vector<>, dg::vector<> as an absolute unit of consumption
    //I know I've been rewriting this a thousand + 1 times
    //this is because we literally dont know what's possible to optimize, and there is dependency + version control problems
    //each container has their own virtue of optimizations that only that container can provide the optimization 

    using event_t = uint64_t;

    class WareHouseInterface{

        public:

            virtual ~WareHouseInterface() noexcept = default;
            virtual auto push(dg::vector<event_t>&& event_arr) noexcept -> std::expected<bool, exception_t> = 0;
            virtual auto pop() noexcept -> dg::vector<event_t> = 0; 
            virtual auto max_consume_size() noexcept -> size_t = 0;
    };

    class ClassRoomInterface{

        public:

            virtual ~ClassRoomInterface() noexcept = default;
            virtual auto in(size_t idx) noexcept -> exception_t = 0;
            virtual void out(size_t idx) noexcept = 0;
            virtual auto peek() noexcept -> std::optional<size_t> = 0; 
    };

    //I feel very empty without actually specifying the digesting event_sz in this component

    class NormalWareHouse: public virtual WareHouseInterface{

        private:

            dg::pow2_cyclic_queue<dg::vector<event_t>> production_queue;
            dg::pow2_cyclic_queue<std::pair<std::binary_semaphore *, std::optional<dg::vector<event_t>> *>> waiting_queue;
            std::unique_ptr<std::mutex> mtx;
            stdx::inplace_hdi_container<std::atomic<bool>> is_empty_concurrent_var;
            stdx::hdi_container<size_t> event_consume_sz;

        public:

            NormalWareHouse(dg::pow2_cyclic_queue<dg::vector<event_t>> production_queue,
                            dg::pow2_cyclic_queue<std::pair<std::binary_semaphore *, std::optional<dg::vector<event_t>> * >> waiting_queue,
                            std::unique_ptr<std::mutex> mtx,
                            bool is_empty_concurrent_var,
                            size_t event_consume_sz) noexcept: production_queue(std::move(production_queue)),
                                                               waiting_queue(std::move(waiting_queue)),
                                                               mtx(std::move(mtx)),
                                                               is_empty_concurrent_var(std::in_place_t{}, is_empty_concurrent_var),
                                                               event_consume_sz(stdx::hdi_container<size_t>{event_consume_sz}){}

            auto push(dg::vector<event_t>&& event_vec) noexcept -> std::expected<bool, exception_t>{

                //I'm tempted not to do an empty check here, because it's out of the scope of this component, yet ...

                if (event_vec.empty()){
                    return std::unexpected(dg::network_exception::INVALID_ARGUMENT);
                }

                if (event_vec.size() > this->max_consume_size()){
                    return std::unexpected(dg::network_exception::INVALID_ARGUMENT);
                }

                std::binary_semaphore * releasing_smp   = nullptr;

                std::expected<bool, exception_t> rs     = [&, this]() noexcept{
                    stdx::xlock_guard<std::mutex> lck_grd(*this->mtx);

                    if (!this->waiting_queue.empty()){
                        auto [fetching_smp, fetching_addr] = this->waiting_queue.front();
                        this->waiting_queue.pop_front();
                        *fetching_addr  = std::optional<dg::vector<event_t>>(std::move(event_vec));
                        std::atomic_signal_fence(std::memory_order_seq_cst);
                        releasing_smp   = fetching_smp; 

                        return std::expected<bool, exception_t>(true);
                    }

                    if (this->production_queue.size() == this->production_queue.capacity()){
                        return std::expected<bool, exception_t>(false);
                    }

                    this->production_queue.push_back(std::move(event_vec));
                    std::atomic_signal_fence(std::memory_order_seq_cst);
                    this->is_empty_concurrent_var.value.exchange(false, std::memory_order_relaxed);

                    return std::expected<bool, exception_t>(true);
                }();

                if (releasing_smp != nullptr){
                    releasing_smp->release();
                }

                return rs;
            }

            auto pop() noexcept -> dg::vector<event_t>{

                std::binary_semaphore fetching_smp(0);
                std::optional<dg::vector<event_t>> fetching_data(std::nullopt); 

                while (true){
                    stdx::xlock_guard<std::mutex> lck_grd(*this->mtx);

                    if (!this->production_queue.empty()){
                        auto rs         = std::move(this->production_queue.front());
                        this->production_queue.pop_front();

                        if (this->production_queue.empty()){
                            std::atomic_signal_fence(std::memory_order_seq_cst);
                            this->is_empty_concurrent_var.value.exchange(true, std::memory_order_relaxed);
                        }

                        return rs;
                    }

                    if (this->waiting_queue.size() == this->waiting_queue.capacity()){
                        continue;
                    }

                    this->waiting_queue.push_back(std::make_pair(&fetching_smp, &fetching_data));
                    break;
                }

                fetching_smp.acquire();
                return dg::vector<event_t>(std::move(fetching_data.value()));
            }

            auto max_consume_size() noexcept -> size_t{

                return this->event_consume_sz.value;
            }

            auto is_empty() const noexcept -> bool{

                return this->is_empty_concurrent_var.value.load(std::memory_order_relaxed);
            }
    };

    //what are the chances ?
    //we'll work on this later
    //this is a hard component to work on
    //normally, we'd want to keep the number of this < 4 or < 8
    //just to have a scalable number, not actually to scale this to like 1024 or 2048 or 4096 for that matter 
    //we'd want to use an unordered set, or the free ticket, bitset memberships, next available bucket + etc.
    //we'd want to reduce the chances as much as possible, we'll leave the others into God's hand
    //plus, we'd want to have a heartbeat memevent to rescue the resolutors

    class NiceClassRoom: public virtual ClassRoomInterface{

        private:

            //64 bits
            //8 concurrent warehouse
            //256 concurrent accessors
            //8 * 8 == 64 bits

            size_t bucket_bit_sz;
            size_t suffix_array_sz;
            stdx::inplace_hdi_container<uint64_t> state;

        public:

            static inline constexpr size_t STATE_BIT_CAP = sizeof(uint64_t) * CHAR_BIT;

            NiceClassRoom(size_t bucket_bit_sz,
                          size_t suffix_array_sz): bucket_bit_sz(bucket_bit_sz),
                                                   suffix_array_sz(suffix_array_sz),
                                                   state(std::in_place_t{}, 0u){

                size_t required_bit_sz = bucket_bit_sz * suffix_array_sz;  

                if (required_bit_sz > STATE_BIT_CAP){
                    dg::network_exception::throw_exception(dg::network_exception::INVALID_ARGUMENT);
                }
            }

            auto in(size_t idx) noexcept -> exception_t{

                if (idx >= this->suffix_array_sz){
                    return dg::network_exception::INVALID_ARGUMENT;
                }

                size_t offset       = idx * this->bucket_bit_sz;
                size_t inc_payload  = size_t{1} << offset;
                
                this->state.value.fetch_add(inc_payload, std::memory_order_relaxed);
                std::atomic_signal_fence(std::memory_order_seq_cst);

                return dg::network_exception::SUCCESS;
            }

            void out(size_t idx) noexcept{

                if constexpr(DEBUG_MODE_FLAG){
                    if (idx >= this->suffix_array_sz){
                        dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                        std::abort();
                    }
                }

                size_t offset       = idx * this->bucket_bit_sz;
                size_t sub_payload  = size_t{1} << offset;

                this->state.value.fetch_sub(sub_payload, std::memory_order_relaxed);
                std::atomic_signal_fence(std::memory_order_seq_cst);
            }

            auto peek() noexcept -> std::optional<size_t>{

                uint64_t membership = this->state.value.load(std::memory_order_relaxed);
                
                if (membership == 0u){
                    return std::nullopt;
                }

                return std::countr_zero(membership) / this->bucket_bit_sz;
            }
    };

    class DistributedWareHouse: public virtual WareHouseInterface{

        private:

            std::unique_ptr<std::unique_ptr<NormalWareHouse>[]> base_arr;
            size_t pow2_base_arr_sz;
            size_t empty_curious_pop_sz;
            std::unique_ptr<ClassRoomInterface> classroom;
            stdx::hdi_container<size_t> event_consume_sz;

        public:

            DistributedWareHouse(std::unique_ptr<std::unique_ptr<NormalWareHouse>[]> base_arr,
                                 size_t pow2_base_arr_sz,
                                 size_t empty_curious_pop_sz,
                                 std::unique_ptr<ClassRoomInterface> classroom,
                                 size_t event_consume_sz) noexcept: base_arr(std::move(base_arr)),
                                                                    pow2_base_arr_sz(pow2_base_arr_sz),
                                                                    empty_curious_pop_sz(empty_curious_pop_sz),
                                                                    classroom(std::move(classroom)),
                                                                    event_consume_sz(stdx::hdi_container<size_t>{event_consume_sz}){}

            auto push(dg::vector<event_t>&& event_vec) noexcept -> std::expected<bool, exception_t>{

                std::optional<size_t> cand = this->classroom->peek();

                if (cand.has_value()){
                    return this->base_arr[cand.value()]->push(std::move(event_vec));
                }

                size_t random_clue  = dg::network_randomizer::randomize_int<size_t>();
                size_t base_arr_idx = random_clue & (this->pow2_base_arr_sz - 1u);

                return this->base_arr[base_arr_idx]->push(std::move(event_vec));
            }

            auto pop() noexcept -> dg::vector<event_t>{

                for (size_t i = 0u; i < this->empty_curious_pop_sz; ++i){
                    size_t random_clue  = dg::network_randomizer::randomize_int<size_t>();
                    size_t base_arr_idx = random_clue & (this->pow2_base_arr_sz - 1u);

                    if (!this->base_arr[base_arr_idx]->is_empty()){
                        return this->pop_at(base_arr_idx);
                    }
                }

                size_t random_clue  = dg::network_randomizer::randomize_int<size_t>();
                size_t base_arr_idx = random_clue & (this->pow2_base_arr_sz - 1u);

                return this->pop_at(base_arr_idx);
            }

            auto max_consume_size() noexcept -> size_t{

                return this->event_consume_sz.value;
            }
        
        private:
            
            auto pop_at(size_t idx) noexcept -> dg::vector<event_t>{

                stdx::seq_cst_guard tx_grd;

                this->classroom->in(idx);
                std::atomic_signal_fence(std::memory_order_seq_cst);
                auto rs = this->base_arr[idx]->pop();
                std::atomic_signal_fence(std::memory_order_seq_cst);
                this->classroom->out(idx);

                return rs;
            }
    };

    class RescueWorker: public virtual dg::network_concurrency::WorkerInterface{

        private:

            std::shared_ptr<WareHouseInterface> warehouse;
            std::chrono::nanoseconds rescue_latency;
            size_t rescue_packet_sz;
            std::unique_ptr<RescuePayloadGenerator> rescue_payload_gen;
        
        public:

            RescueWorker(std::shared_ptr<WareHouseInterface> warehouse,
                         std::chrono::nanoseconds rescue_latency,
                         size_t rescue_packet_sz,
                         std::unique_ptr<RescuePayloadGenerator> rescue_payload_gen) noexcept: warehouse(std::move(warehouse)),
                                                                                               rescue_latency(rescue_latency),
                                                                                               rescue_packet_sz(rescue_packet_sz),
                                                                                               rescue_payload_gen(std::move(rescue_payload_gen)){}
            
            bool run_one_epoch() noexcept{

            }
    };
}

#endif