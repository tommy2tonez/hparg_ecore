#ifndef __NETWORK_LOGGER_H__
#define __NETWORK_LOGGER_H__

#include <string>
#include <filesystem> 
#include <fstream>
#include <utility>
#include <algorithm>
#include <format>
#include <memory>
#include <array>
#include <string_view>
#include <mutex>
#include "stdx.h"
#include "network_postgres_db.h"
#include <thread>
#include "network_std_container.h"

// namespace dg::network_log::last_mohican{

// }

namespace dg::network_log::implementation{

    struct LoggerInterface{
        virtual ~LoggerInterface() noexcept = default;
        virtual void log(const char * content, const char * kind) = 0;
        virtual void flush() = 0;
    };

    struct KindLoggerInterface{
        virtual ~KindLoggerInterface() noexcept = default;
        virtual void critical(const char *) noexcept = 0;
        virtual void error(const char *) noexcept = 0;
        virtual void error_optional(const char *) noexcept = 0;
        virtual void error_fast(const char *) noexcept = 0;
        virtual void error_fast_optional(const char *) noexcept = 0;
        virtual void journal(const char *) noexcept = 0;
        virtual void journal_optional(const char *) noexcept = 0;
        virtual void journal_fast(const char *) noexcept = 0;
        virtual void journal_fast_optional(const char *) noexcept = 0;
        virtual void flush() noexcept = 0;
        virtual void flush_optional() noexcept = 0;
    };

    class InstantLogger: public virtual LoggerInterface{

        public:

            void log(const char * content, const char * kind){
                
                std::expected<dg::network_postgres_db::model::SystemLog, exception_t> syslog = dg::network_postgres_db::model_factory::make_systemlog(content, kind, stdx::utc_timestamp());

                if (!syslog.has_value()){
                    dg::network_exception::throw_exception(syslog.error());
                }

                auto commitable = dg::network_postgres_db::make_commitable_create_systemlog(syslog.value());

                if (!commitable.has_value()){
                    dg::network_exception::throw_exception(commitable.error());
                }

                exception_t err = dg::network_postgres_db::commit(stdx::make_vector_convertible(std::move(commitable.value())));

                if (dg::network_exception::is_failed(err)){
                    dg::network_exception::throw_exception(err);
                }
            }

            void flush(){

                // (void) ; //TODOs:
            }
    };

    class BatchLogger: public virtual LoggerInterface{
        
        private:

            dg::vector<dg::network_postgres_db::model::SystemLog> syslog_vec;
            std::unique_ptr<std::atomic_flag> lck;

        public:

            BatchLogger(dg::vector<dg::network_postgres_db::model::SystemLog> syslog_vec,
                        std::unique_ptr<std::atomic_flag> lck) noexcept: syslog_vec(std::move(syslog_vec)),
                                                                         lck(std::move(lck)){} 

            ~BatchLogger() noexcept{

                try{
                    this->flush();
                } catch (...){
                    this->force_flush();
                }
            }

            void log(const char * content, const char * kind){

                auto lck_grd = stdx::lock_guard(*this->lck); 

                if (this->syslog_vec.size() == this->syslog_vec.capacity()){
                    this->flush();
                }

                std::expected<dg::network_postgres_db::model::SystemLog, exception_t> model = dg::network_postgres_db::model_factory::make_systemlog(content, kind, stdx::utc_timestamp());
                
                if (!model.has_value()){
                    dg::network_exception::throw_exception(model.error());
                }

                this->syslog_vec.push_back(std::move(model.value()));
            }

            void flush(){
                
                auto lck_grd = stdx::lock_guard(*this->lck); 

                dg::vector<std::unique_ptr<dg::network_postgres_db::CommitableInterface>> commitable_vec{};

                for (auto& syslog: this->syslog_vec){
                    auto commitable = dg::network_postgres_db::make_commitable_create_systemlog(syslog);
                    if (!commitable.has_value()){
                        dg::network_exception::throw_exception(commitable.error());
                    }
                    commitable_vec.emplace_back(std::move(commitable.value()));
                }

                exception_t err = dg::network_postgres_db::commit(std::move(commitable_vec));

                if (dg::network_exception::is_failed(err)){
                    dg::network_exception::throw_exception(err);
                }

                this->syslog_vec.clear();            
            }
        
        private:

            void force_flush() noexcept{

                //TODOs:
                // auto lck_grd = stdx::lock_guard(this->lck);

                // try{
                //     dg::string bstream = dg::network_compact_serializer::serialize<dg::string>(this->syslog_vec);
                //     dg::network_log::last_mohican::write(bstream, "syslog_vec_compact_serialization_format");
                //     this->syslog_vec.clear();
                // } catch (...){}
            }
    };

    //an extension is required to do error recovery -  

    class ConcurrentLogger: public virtual LoggerInterface{

        private:

            dg::vector<std::unique_ptr<LoggerInterface>> logger_vec;
        
        public:

            ConcurrentLogger(dg::vector<std::unique_ptr<LoggerInterface>> logger_vec) noexcept: logger_vec(std::move(logger_vec)){}

            void log(const char * content, const char * kind){
                
                size_t thr_id       = std::bit_cast<size_t>(std::this_thread::get_id());
                size_t logger_sz    = this->logger_vec.size();
                size_t idx          = stdx::pow2mod_unsigned(thr_id, logger_sz);
;
                this->logger_vec[idx]->log(content, kind);
            }

            void flush(){
                
                size_t thr_id       = std::bit_cast<size_t>(std::this_thread::get_id());
                size_t logger_sz    = this->logger_vec.size();
                size_t idx          = stdx::pow2mod_unsigned(thr_id, logger_sz);

                this->logger_vec[idx]->flush();
            }
    };

    struct KindLogger: public virtual KindLoggerInterface{

        private:

            std::unique_ptr<LoggerInterface> instant_logger;
            std::unique_ptr<LoggerInterface> batch_logger;

        public:

            KindLogger(std::unique_ptr<LoggerInterface> instant_logger,
                       std::unique_ptr<LoggerInterface> batch_logger) noexcept: instant_logger(std::move(instant_logger)),
                                                                                batch_logger(std::move(batch_logger)){} 


            void critical(const char * what) noexcept{

                this->instant_logger->log(what, "critical");
            }

            void error(const char * what) noexcept{

                this->instant_logger->log(what, "error");
            }

            void error_fast(const char * what) noexcept{

                this->batch_logger->log(what, "error");
            }

            void error_optional(const char * what) noexcept{

                try{
                    this->instant_logger->log(what, "error");
                } catch (...){
                    (void) what;
                }
            }

            void error_fast_optional(const char * what) noexcept{

                try{
                    this->batch_logger->log(what, "error");
                } catch (...){
                    (void) what;
                }
            }

            void journal(const char * what) noexcept{

                this->instant_logger->log(what, "journal");
            }

            void journal_fast(const char * what) noexcept{

                this->batch_logger->log(what, "journal");
            }

            void journal_optional(const char * what) noexcept{

                try{
                    this->instant_logger->log(what, "journal");
                } catch (...){
                    (void) what;
                }
            }

            void journal_fast_optional(const char * what) noexcept{

                try{
                    this->batch_logger->log(what, "journal");
                } catch (...){
                    (void) what;
                }
            }

            void flush() noexcept{
                
                this->instant_logger->flush();
                this->batch_logger->flush();
            }

            void flush_optional() noexcept{

                try{
                    this->instant_logger->flush();
                    this->batch_logger->flush();
                } catch (...){
                    // (void) flush;
                }
            }
    };

    struct Factory{

        static auto spawn_instant_logger() -> std::unique_ptr<LoggerInterface>{

            return std::make_unique<InstantLogger>();
        } 

        static auto spawn_batch_logger(size_t capacity) -> std::unique_ptr<LoggerInterface>{

            const size_t MIN_CAPACITY   = 1u;
            const size_t MAX_CAPACITY   = size_t{1} << 20;

            if (std::clamp(capacity, MIN_CAPACITY, MAX_CAPACITY) != capacity){
                dg::network_exception::throw_exception(dg::network_exception::INVALID_ARGUMENT);
            }

            dg::vector<dg::network_postgres_db::model::SystemLog> syslog_vec{};
            std::unique_ptr<std::atomic_flag> lck = std::make_unique<std::atomic_flag>();
            lck->clear();
            syslog_vec.reserve(capacity);

            return std::make_unique<BatchLogger>(std::move(syslog_vec), std::move(lck));
        }

        static auto spawn_concurrent_logger(dg::vector<std::unique_ptr<LoggerInterface>> logger_vec) -> std::unique_ptr<LoggerInterface>{

            const size_t MIN_LOGGER_VEC_SZ  = 1u;
            const size_t MAX_LOGGER_VEC_SZ  = size_t{1} << 10;

            if (std::clamp(static_cast<size_t>(logger_vec.size()), MIN_LOGGER_VEC_SZ, MAX_LOGGER_VEC_SZ) != logger_vec.size()){
                dg::network_exception::throw_exception(dg::network_exception::INVALID_ARGUMENT);
            }

            if (!stdx::is_pow2(logger_vec.size())){
                dg::network_exception::throw_exception(dg::network_exception::INVALID_ARGUMENT);
            }

            return std::make_unique<ConcurrentLogger>(std::move(logger_vec));
        } 

        static auto spawn_kind_logger(size_t fastlog_capacity, size_t concurrency_sz) -> std::unique_ptr<KindLoggerInterface>{

            dg::vector<std::unique_ptr<LoggerInterface>> batch_logger_vec = {};

            for (size_t i = 0u; i < concurrency_sz; ++i){
                batch_logger_vec.push_back(spawn_batch_logger(fastlog_capacity));
            }

            std::unique_ptr<LoggerInterface> concurrent_instant_logger  = spawn_instant_logger();
            std::unique_ptr<LoggerInterface> concurrent_batch_logger    = spawn_concurrent_logger(std::move(batch_logger_vec));

            return std::make_unique<KindLogger>(std::move(concurrent_instant_logger), std::move(concurrent_batch_logger));
        }
    };
}

namespace dg::network_log{

    constexpr size_t FAST_LOG_CAPACITY  = size_t{1} << 8;
    constexpr size_t CONCURRENCY_SZ     = size_t{1} << 8;

    inline std::unique_ptr<implementation::KindLoggerInterface> logger{};

    void init(){

        logger = implementation::Factory::spawn_kind_logger(FAST_LOG_CAPACITY, CONCURRENCY_SZ);
    } 

    void deinit() noexcept{
        
        logger = nullptr;
    }

    void critical(const char * what) noexcept{

        logger->critical(what);
    }

    void error(const char * what) noexcept{

        logger->error(what);
    }

    void error_fast(const char * what) noexcept{

        logger->error_fast(what);
    }
    
    void error_optional(const char * what) noexcept{

        logger->error_optional(what);
    }

    void error_fast_optional(const char * what) noexcept{

        logger->error_fast_optional(what);
    }

    void journal(const char * what) noexcept{

        logger->journal(what);
    }

    void journal_fast(const char * what) noexcept{

        logger->journal_fast(what);
    }

    void journal_optional(const char * what) noexcept{

        logger->journal_optional(what);
    }

    void journal_fast_optional(const char * what) noexcept{

        logger->journal_fast_optional(what);
    }

    void flush() noexcept{

        logger->flush();
    }

    void flush_optional() noexcept{

        logger->flush_optional();
    }
}

namespace dg::network_log_stackdump{

    auto add_stack_trace(const char * what) -> dg::string{

        // dg::string stacktrace_str  = std::stacktrace::current().to_string(); //TODOs:
        dg::string stacktrace_str     = {};
        dg::string what_str           = dg::string(what); 
        dg::string rs                 = {};

        std::format_to(std::back_inserter(rs), "<stackdump_begin>\n{}\n<stackdump_end>\n{}", stacktrace_str, what_str);

        return rs;
    } 

    void critical(const char * what) noexcept{

        auto new_what = add_stack_trace(what);
        network_log::critical(new_what.c_str());
    }

    void critical() noexcept{

        critical("");
    }
    
    void error(const char * what) noexcept{

        auto new_what = add_stack_trace(what);
        network_log::error(new_what.c_str());
    }

    void error() noexcept{

        error("");
    }

    void error_fast(const char * what) noexcept{

        auto new_what = add_stack_trace(what);
        network_log::error_fast(new_what.c_str());
    }
    
    void error_fast() noexcept{

        error_fast("");
    }

    void error_optional(const char * what) noexcept{

        auto new_what = add_stack_trace(what);
        network_log::error_optional(new_what.c_str());
    }

    void error_optional() noexcept{

        error_optional("");
    }

    void error_fast_optional(const char * what) noexcept{

        auto new_what = add_stack_trace(what);
        network_log::error_fast_optional(new_what.c_str());
    }

    void error_fast_optional() noexcept{

        error_fast_optional("");
    }

    void journal(const char * what) noexcept{

        auto new_what = add_stack_trace(what);
        network_log::journal(new_what.c_str());
    }

    void journal() noexcept{

        journal("");
    }

    void journal_fast(const char * what) noexcept{

        auto new_what = add_stack_trace(what);
        network_log::journal_fast(new_what.c_str());
    }

    void journal_fast() noexcept{

        journal_fast("");
    }

    void journal_optional(const char * what) noexcept{

        auto new_what = add_stack_trace(what);
        network_log::journal_optional(new_what.c_str());
    }

    void journal_optional() noexcept{

        journal_optional("");
    }

    void journal_fast_optional(const char * what) noexcept{

        auto new_what = add_stack_trace(what);
        network_log::journal_fast_optional(new_what.c_str());
    }

    void journal_fast_optional() noexcept{

        network_log::journal_fast_optional("");
    }

    void flush() noexcept{

        network_log::flush();
    }

    void flush_optional() noexcept{

        network_log::flush_optional();
    }
}

namespace dg::network_log_scope{

    auto critical_terminate() noexcept{

        static int i    = 0;
        auto destructor = [=](int *) noexcept{
            
            try{
                std::rethrow_exception(std::current_exception());
                network_log_stackdump::critical();
            } catch (std::exception& e){
                network_log_stackdump::critical(e.what()); //
                std::terminate();
            }
        };

        return std::unique_ptr<int, decltype(destructor)>(&i, destructor);
    } 

    auto flush() noexcept{

        static int i    = 0;
        auto destructor = [=](int *) noexcept{
            network_log::flush();
        };

        return std::unique_ptr<int, decltype(destructor)>(&i, destructor);
    }

    auto flush_optional() noexcept{

        static int i    = 0;
        auto destructor = [=](int *) noexcept{
            network_log::flush_optional();
        };

        return std::unique_ptr<int, decltype(destructor)>(&i, destructor);
    }
}

#endif