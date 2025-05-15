#ifndef __NETWORK_MEMPRESS_DISPATCH_WAREHOUSE_H__
#define __NETWORK_MEMPRESS_DISPATCH_WAREHOUSE_H__

namespace dg::network_mempress_dispatch_warehouse{

    //in this component, we can't latencize, we can't do workorder as a ProducerConsumer pointer approach
    //such is that a unit is an absolute unit of consumption
    //collector collects from mailbox -> dispatch warehouse -> the resolutor
    //dispatch warehouse must store things in dg::vector<>, dg::vector<> as an absolute unit of consumption

    using event_t = uint64_t;

    class WareHouseInterface{

        public:

            virtual ~WareHouseInterface() = default;
            virtual auto push(dg::vector<event_t>&& event_arr) noexcept -> std::expected<bool, exception_t> = 0;
            virtual auto pop() noexcept -> dg::vector<event_t> = 0; 
    };

    class NormalWareHouse: public virtual WareHouseInterface{

        private:

            dg::pow2_cyclic_queue<dg::vector<event_t>> production_queue;
            dg::pow2_cyclic_queue<std::pair<std::binary_semaphore *, std::optional<dg::vector<event_t>> *>> waiting_queue;
            std::unique_ptr<std::mutex> mtx;
            stdx::inplace_hdi_container<std::atomic<bool>> is_empty_concurrent_var;

        public:

            NormalWareHouse(dg::pow2_cyclic_queue<dg::vector<event_t>> production_queue,
                            dg::pow2_cyclic_queue<std::pair<std::binary_semaphore *, std::optional<dg::vector<event_t>> * >> waiting_queue,
                            std::unique_ptr<std::mutex> mtx,
                            bool is_empty_concurrent_var) noexcept: production_queue(std::move(production_queue)),
                                                                    waiting_queue(std::move(waiting_queue)),
                                                                    mtx(std::move(mtx)),
                                                                    is_empty_concurrent_var(std::in_place_t{}, is_empty_concurrent_var){}

            auto push(dg::vector<event_t>&& event_vec) noexcept -> std::expected<bool, exception_t>{

                std::binary_semaphore * releasing_smp   = nullptr;

                std::expected<bool, exception_t> rs     = [&, this]() noexcept{
                    stdx::xlock_guard<std::mutex> lck_grd(*this->mtx);

                    if (!this->waiting_queue.empty()){
                        auto [fetching_smp, fetching_addr] = this->waiting_queue.front();
                        this->waiting_queue.pop_front();
                        *fetching_addr  = std::optional<dg::vector<event_t>>(std::move(event_vec));
                        releasing_smp   = fetching_smp; 

                        return std::expected<bool, exception_t>(true);
                    }

                    if (this->production_queue.size() == this->production_queue.capacity()){
                        return std::expected<bool, exception_t>(false);
                    }

                    this->production_queue.push_back(std::move(event_vec));
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
                        auto rs = std::move(this->production_queue.front());
                        this->production_queue.pop_front();

                        if (this->production_queue.empty()){
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

            auto is_empty() const noexcept -> bool{

                return this->is_empty_concurrent_var.value.load(std::memory_order_relaxed);
            }
    };

    class DistributedWareHouse: public virtual WareHouseInterface{

        private:

            std::unique_ptr<std::unique_ptr<NormalWareHouse>[]> base_arr;
            size_t pow2_base_arr_sz;
            size_t empty_curious_pop_sz;
        
        public:

            DistributedWareHouse(std::unique_ptr<std::unique_ptr<NormalWareHouse>[]> base_arr,
                                 size_t pow2_base_arr_sz,
                                 size_t empty_curious_pop_sz) noexcept: base_arr(std::move(base_arr)),
                                                                        pow2_base_arr_sz(pow2_base_arr_sz),
                                                                        empty_curious_pop_sz(empty_curious_pop_sz){}

            auto push(dg::vector<event_t>&& event_vec) noexcept -> std::expected<bool, exception_t>{

            }

            auto pop() noexcept -> dg::vector<event_t>{

            }
    };
}

#endif