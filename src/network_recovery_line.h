#ifndef __DG_NETWORK_RECOVERY_LINE_H__
#define __DG_NETWORK_RECOVERY_LINE_H__

//define HEADER_CONTROL 6

#include "network_exception.h"
#include <memory>
#include <optional>
#include "network_concurrency.h"
#include "network_std_container.h"
#include <optional>
#include "stdx.h"
#include "network_raii_x.h" 

namespace dg::network_recovery_line{

    struct RecoverableInterface{
        virtual ~RecoverableInterface() noexcept = default;
        virtual void recover() noexcept = 0;
    };

    struct RecoveryLineInterface{
        virtual ~RecoveryLineInterface() noexcept = default;
        virtual auto get_recovery_line() noexcept -> std::expected<size_t, exception_t> = 0;
        virtual auto set_resource(size_t, std::shared_ptr<RecoverableInterface>) noexcept -> exception_t = 0;
        virtual auto get_resource(size_t) noexcept -> std::expected<std::shared_ptr<RecoverableInterface>, exception_t> = 0;
        virtual void close_recovery_line(size_t) noexcept = 0;
    };

    struct DispatchingLineContainerInterface{
        virtual ~DispatchingLineContainerInterface() noexcept = default;
        virtual auto push(size_t) noexcept -> exception_t = 0;
        virtual auto pop() noexcept -> std::optional<size_t> = 0;
    };

    struct RecoveryControllerInterface{
        virtual ~RecoveryControllerInterface() noexcept = default;
        virtual auto get_recovery_line(std::unique_ptr<RecoverableInterface>) noexcept -> std::expected<size_t, exception_t> = 0;  
        virtual auto notify(size_t) noexcept -> exception_t = 0;
        virtual void close_recovery_line(size_t) noexcept = 0;
    };

    class RecoveryWorker: public virtual dg::network_concurrency::WorkerInterface{

        private:

            std::shared_ptr<RecoveryLineInterface> recovery_line;
            std::shared_ptr<DispatchingLineContainerInterface> dispatching_ticket_container;
        
        public:

            RecoveryWorker(std::shared_ptr<RecoveryLineInterface> recovery_line,
                           std::shared_ptr<DispatchingLineContainerInterface> dispatching_ticket_container) noexcept: recovery_line(std::move(recovery_line)),
                                                                                                                      dispatching_ticket_container(std::move(dispatching_ticket_container)){}
            
            bool run_one_epoch() noexcept{

                std::optional<size_t> ticket_id = this->dispatching_ticket_container->pop();

                if (!ticket_id.has_value()){
                    return false;
                }

                std::expected<std::shared_ptr<RecoverableInterface>, exception_t> recoverable = this->recovery_line->get_resource(ticket_id.value());
                
                if (!recoverable.has_value()){
                    return true;
                }

                if (recoverable.value() == nullptr){
                    return true;
                }

                recoverable.value()->recover();
                return true;
            }
    };

    class DispatchingLineContainer: public virtual DispatchingLineContainerInterface{

        private:

            dg::deque<size_t> ticket_vec;
            size_t capacity;
            std::unique_ptr<std::mutex> mtx;

        public:

            DispatchingLineContainer(dg::deque<size_t> ticket_vec,
                                     size_t capacity,
                                     std::unique_ptr<std::mutex> mtx) noexcept: ticket_vec(std::move(ticket_vec)),
                                                                                capacity(capacity),
                                                                                mtx(std::move(mtx)){}

            auto push(size_t ticket_id) noexcept -> exception_t{

                auto lck_grd = stdx::lock_guard(*this->mtx);

                if (this->ticket_vec.size() == this->capacity){
                    return dg::network_exception::RESOURCE_EXHAUSTION;
                }

                this->ticket_vec.push_back(ticket_id);
                return dg::network_exception::SUCCESS;
            }

            auto pop() noexcept -> std::optional<size_t>{

                auto lck_grd = stdx::lock_guard(*this->mtx);

                if (this->ticket_vec.empty()){
                    return std::nullopt;
                }

                size_t rs = this->ticket_vec.front();
                this->ticket_vec.pop_front();

                return rs;
            }
    };

    class RecoveryLine: public virtual RecoveryLineInterface{

        private:

            dg::deque<size_t> line_vec;
            dg::unordered_map<size_t, std::shared_ptr<RecoverableInterface>> line_resource_map;
            std::unique_ptr<std::mutex> mtx;
        
        public:

            RecoveryLine(dg::deque<size_t> line_vec,
                         dg::unordered_map<size_t, std::shared_ptr<RecoverableInterface>> line_resource_map,
                         std::unique_ptr<std::mutex> mtx) noexcept: line_vec(std::move(line_vec)),
                                                                    line_resource_map(std::move(line_resource_map)),
                                                                    mtx(std::move(mtx)){}

            auto get_recovery_line() noexcept -> std::expected<size_t, exception_t>{

                auto lck_grd    = stdx::lock_guard(*this->mtx);
                
                if (this->line_vec.empty()){
                    return std::unexpected(dg::network_exception::RESOURCE_EXHAUSTION);
                }

                size_t nxt_id   = this->line_vec.front();
                this->line_vec.pop_front();
                auto [map_ptr, status] = this->line_resource_map.emplace(std::make_pair(nxt_id, nullptr));

                if constexpr(DEBUG_MODE_FLAG){
                    if (!status){
                        std::abort();
                    }
                }

                return nxt_id;
            }

            auto set_resource(size_t line_id, std::shared_ptr<RecoverableInterface> recoverable) noexcept -> exception_t{

                auto lck_grd    = stdx::lock_guard(*this->mtx);
                auto map_ptr    = this->line_resource_map.find(line_id);

                if (map_ptr == this->line_resource_map.end()){
                    return dg::network_exception::INVALID_ARGUMENT;
                }

                map_ptr->second = std::move(recoverable);
                return dg::network_exception::SUCCESS;
            }

            auto get_resource(size_t line_id) noexcept -> std::expected<std::shared_ptr<RecoverableInterface>, exception_t>{

                auto lck_grd    = stdx::lock_guard(*this->mtx);
                auto map_ptr    = this->line_resource_map.find(line_id);

                if (map_ptr == this->line_resource_map.end()){
                    return std::unexpected(dg::network_exception::INVALID_ARGUMENT);
                }

                return map_ptr->second;
            }

            void close_recovery_line(size_t line_id) noexcept{

                auto lck_grd    = stdx::lock_guard(*this->mtx);
                auto map_ptr    = this->line_resource_map.find(line_id);

                if constexpr(DEBUG_MODE_FLAG){
                    if (map_ptr == this->line_resource_map.end()){
                        std::abort();
                    }
                }

                this->line_resource_map.erase(map_ptr);
                this->line_vec.push_back(line_id);
            }
    };

    class RecoveryController: public virtual RecoveryControllerInterface{

        private:

            std::vector<std::shared_ptr<std::thread>> worker_vec;      
            std::shared_ptr<DispatchingLineContainerInterface> dispatch_ticket_container;
            std::shared_ptr<RecoveryLineInterface> recovery_line;
        
        public:

            RecoveryController(std::vector<std::shared_ptr<std::thread>> worker_vec,
                               std::shared_ptr<DispatchingLineContainerInterface> dispatch_ticket_container,
                               std::shared_ptr<RecoveryLineInterface> recovery_line) noexcept: worker_vec(std::move(worker_vec)),
                                                                                               dispatch_ticket_container(std::move(dispatch_ticket_container)),
                                                                                               recovery_line(std::move(recovery_line)){}
            
            auto get_recovery_line(std::unique_ptr<RecoverableInterface> recover_runnable) noexcept -> std::expected<size_t, exception_t>{
                
                if (!recover_runnable){
                    return std::unexpected(dg::network_exception::INVALID_ARGUMENT);
                }

                std::expected<size_t, exception_t> line_id = this->recovery_line->get_recovery_line();

                if (!line_id.has_value()){
                    return std::unexpected(line_id.error());
                }

                exception_t err = this->recovery_line->set_resource(line_id.value(), std::move(recover_runnable));

                if (dg::network_exception::is_failed(err)){
                    this->recovery_line->close_recovery_line(line_id.value());
                    return std::unexpected(err);
                }

                return line_id.value();
            }

            auto notify(size_t line_id) noexcept -> exception_t{

                return this->dispatch_ticket_container->push(line_id);
            }

            void close_recovery_line(size_t line_id) noexcept{
                
                this->recovery_line->close_recovery_line(line_id);
            }
    };

    inline std::unique_ptr<RecoveryControllerInterface> recovery_controller;

    void init(){

    }

    void deinit() noexcept{

        recovery_controller = nullptr;
    }

    auto get_recovery_line(std::unique_ptr<RecoverableInterface> recoverable) noexcept -> std::expected<size_t, exception_t>{

        return recovery_controller->get_recovery_line(std::move(recoverable));
    }

    auto notify(size_t line_id) noexcept -> exception_t{

        return recovery_controller->notify(line_id);
    }

    void close_recovery_line(size_t line_id) noexcept{
        
        recovery_controller->close_recovery_line(line_id);
    }

    auto get_raii_recovery_line(std::unique_ptr<RecoverableInterface> recoverable) noexcept -> std::expected<dg::nothrow_immutable_unique_raii_wrapper<size_t, decltype(&network_recovery_line::close_recovery_line)>, exception_t>{

        std::expected<size_t, exception_t> line_id = network_recovery_line::get_recovery_line(std::move(recoverable));

        if (!line_id.has_value()){
            return std::unexpected(line_id.error());
        }

        return dg::nothrow_immutable_unique_raii_wrapper<size_t, decltype(&network_recovery_line::close_recovery_line)>(line_id.value(), network_recovery_line::close_recovery_line);
    }
}

#endif