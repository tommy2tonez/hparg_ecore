#ifndef __DG_NETWORK_RECOVERY_LINE_H__
#define __DG_NETWORK_RECOVERY_LINE_H__

#include "network_exception.h"
#include <memory>
#include <optional>
#include <network_concurrency.h>
#include "network_std_container.h"
#include <optional>
#include "stdx.h"

namespace dg::network_recovery_line{

    struct RecoveryExecutableInterface{
        virtual ~RecoveryExecutableInterface() noexcept = default;
        virtual void recover() noexcept = 0;
    };

    struct RecoveryLineResourceContainerInterface{
        virtual ~RecoveryLineInterface() noexcept = default;
        virtual void set_recovery_line_resource(size_t, std::shared_ptr<RecoveryExecutableInterface>) noexcept = 0; 
        virtual void get_recovery_line_resource(size_t) noexcept -> std::shared_ptr<RecoveryExecutableInterface> = 0; //even though i'm tempted to do std::optional here - because the program rule is all std::<>_ptr are valid_ptr 
    };

    struct RecoveryLineContainerInterface{
        virtual ~RecoveryLineContainerInterface() noexcept = default;
        virtual void push(size_t) noexcept = 0;
        virtual auto pop() noexcept -> std::optional<size_t> = 0;
    };

    struct RecoveryControllerInterface{
        virtual ~RecoveryControllerInterface() noexcept = default;
        virtual auto get_recovery_line(std::shared_ptr<RecoveryExecutableInterface>) noexcept -> std::expected<size_t, exception_t> = 0;  
        virtual void notify(size_t) noexcept = 0;
        virtual void close(size_t) noexcept = 0;
    };

    class RecoveryWorker: public virtual dg::network_concurrency::WorkerInterface{

        private:

            std::shared_ptr<RecoveryLineInterface> recovery_line;
            std::shared_ptr<RecoveryLineContainerInterface> recovery_ticket_container;
        
        public:

            RecoveryWorker(std::shared_ptr<RecoveryLineInterface> recovery_line,
                           std::shared_ptr<RecoveryLineContainerInterface> recovery_ticket_container) noexcept: recovery_line(std::move(recovery_line)),
                                                                                                                recovery_ticket_container(std::move(recovery_ticket_container)){}
            
            bool run_one_epoch() noexcept{

                std::optional<size_t> ticket_id = this->recovery_ticket_container->pop();

                if (!static_cast<bool>(ticket_id)){
                    return false;
                }

                std::shared_ptr<RecoveryExecutableInterface> executable = this->recovery_line->get_recovery_line(ticket_id.value());

                if (!static_cast<bool>(executable)){
                    return false;
                }

                executable->recover();
                return true;
            }
    };

    class TicketContainer: public virtual RecoveryLineContainerInterface{

        private:

            dg::vector<size_t> tickets;
            std::unique_ptr<std::mutex> mtx;

        public:

            TicketContainer(dg::vector<size_t> tickets,
                            std::unique_ptr<std::mutex> mtx) noexcept: tickets(std::move(tickets)),
                                                                       mtx(std::move(mtx)){}

            void push(size_t ticket_id) noexcept{

                auto lck_grd = stdx::lock_guard(*this->mtx);
                this->tickets.push_back(ticket_id); 
            }

            auto pop() noexcept -> std::optional<size_t>{

                auto lck_grd = stdx::lock_guard(*this->mtx);

                if (this->tickets.empty()){
                    return std::nullopt;
                }

                size_t rs = this->tickets.back();
                this->tickets.pop_back();

                return rs;
            }
    };

    class RecoveryLineResourceContainer: public virtual RecoveryLineResourceContainerInterface{

        private:

            dg::unordered_map<size_t, std::shared_ptr<RecoveryExecutableInterface>> resource_map;
            std::unique_ptr<std::mutex> mtx;
        
        public:

            RecoveryLineResourceContainer(dg::unordered_map<size_t, std::shared_ptr<RecoveryExecutableInterface>> resource_map,
                                          std::unique_ptr<std::mutex> mtx) noexcept: resource_map(std::move(resource_map)),
                                                                                     mtx(std::move(mtx)){}

            void set_recovery_line_resource(size_t line_id, std::shared_ptr<RecoveryExecutableInterface> resource) noexcept{

                auto lck_grd    = stdx::lock_guard(*this->mtx);
                auto map_ptr    = this->resource_map.find(line_id);

                if constexpr(DEBUG_MODE_FLAG){
                    if (map_ptr == this-.resource_map.end()){
                        dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                        std::abort();
                    }
                }

                map_ptr->second = std::move(resource);
            }

            auto get_recovery_line_resource(size_t line_id) noexcept -> std::shared_ptr<RecoveryExecutableInterface>{

                auto lck_grd    = stdx::lock_guard(*this->mtx);
                auto map_ptr    = this->resource_map.find(line_id);

                if constexpr(DEBUG_MODE_FLAG){
                    if (map_ptr == this->resource_map.end()){
                        dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                        std::abort();
                    }
                }

                return map_ptr->second;
            }
    };

    class RecoveryController: public virtual RecoveryControllerInterface{

        private:

            std::unique_ptr<RecoveryLineContainerInterface> token_container;
            std::shared_ptr<RecoveryLineContainerInterface> dispatch_ticket_container;
            std::shared_ptr<RecoveryLineResourceContainerInterface> recovery_line_resource;
        
        public:

            RecoveryController(std::unique_ptr<RecoveryLineContainerInterface> token_container,
                               std::shared_ptr<RecoveryLineContainerInterface> dispatch_ticket_container,
                               std::shared_ptr<RecoveryLineResourceContainerInterface> recovery_line_resource) noexcept: token_container(std::move(token_container)),
                                                                                                                         dispatch_ticket_container(std::move(dispatch_ticket_container)),
                                                                                                                         recovery_line_resource(std::move(recovery_line_resource)){}
            
            auto get_recovery_line(std::shared_ptr<RecoveryExecutableInterface> recover_runnable) noexcept -> std::expected<size_t, exception_t>{
                
                std::optional<size_t> line_id = this->token_container->pop();

                if (!static_cast<bool>(line_id)){
                    return std::unexpected(dg::network_exception::EXHAUSTED_RECOVERY_LINE);
                }

                this->recovery_line_resource->set_recovery_line_resource(line_id.value(), std::move(recover_runnable));
                return line_id.value();
            }

            void notify(size_t line_id) noexcept{

                this->dispatch_ticket_container->push(line_id); //this will call recovery for the wrong component - yet it's expected - all recovery operation should not alter the program state - even if the component is not corrupted
            }

            void close(size_t line_id) noexcept{

                this->recovery_line_resource->set_recovery_line_resource(line_id, nullptr);
                this->token_container->push(line_id);
            }
    };

    inline std::unique_ptr<RecoveryControllerInterface> recovery_controller;

    void init(){

    }

    //recovery_executable_interface is a defined-invokable-in-all-scenerios, exitable-in-all-scenerios component (even if the component does not invoke notify - this is strange - maybe there's a fix - yet it's a stricter req)
    auto get_recovery_line(std::shared_ptr<RecoveryExecutableInterface> recover_runnable) noexcept -> std::expected<size_t, exception_t>{

        return recovery_controller->get_recovery_line(std::move(recover_runnable));
    }

    //notify the dispatcher to recover the component - dispatcher might or might not arrive - it's the caller and friends responsibility to compromise the component + abort the program after a certain threshold  
    void notify(size_t line_id) noexcept{

        recovery_controller->notify(line_id);
    }

    //this will shadow legacy file close - important to not <use namespace> here
    void close(size_t line_id) noexcept{
        
        recovery_controller->close(line_id);
    }

    using recovery_line_dclose_t = void (*)(size_t *) noexcept; 

    auto safeget_recovery_line(std::shared_ptr<RecoveryExecutableInterface> recover_runnable) noexcept -> std::expected<std::unique_ptr<size_t, recovery_line_dclose_t>, exception_t>{

        auto destructor = [](size_t * line_id) noexcept{
            close(*line_id);
            delete line_id;
        }

        std::expected<size_t, exception_t> line = get_recovery_line(std::move(recover_runnable));

        if (dg::network_exception::is_failed(line)){
            return std::unexpected(line.error());
        }

        return std::unique_ptr<size_t, recovery_line_dclose_t>(new size_t{line.value()}, destructor);
    } 
}

#endif