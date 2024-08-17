#ifndef __NETWORK_LOGGER_H__
#define __NETWORK_LOGGER_H__

#include <string>
#include "network_concurrency.h" 
#include <filesystem> 
#include <fstream>
#include <utility>
#include <algorithm>
#include <format>
#include <memory>
#include <array>
#include <string_view>

namespace dg::network_log{

    using err_t = int;
    static inline constexpr bool HAS_ENFORCE_WRITE = false;

    template <class ID>
    struct ConcurrentLogger{

        private:

            static inline std::filesystem::path logger_path{};
            static inline std::filesystem::path logger_ext;

        public:

            void init(std::filesystem::path arg_logger_path, std::filesystem::path arg_logger_ext) noexcept{

                logger_path = arg_logger_path;
                logger_ext  = arg_logger_ext;
            } 

            void log(const char * buf, size_t buf_sz){

                size_t idx = dg::network_concurrency::to_thread_idx(std::this_thread::get_id());
                std::filesystem::path outpath = logger_path / (std::to_string(idx));
                outpath.replace_extension(logger_ext);
                std::ofstream outfile{outpath, std::ios::app};
                outfile.write(buf, buf_sz);
            }

            void log(const std::string& msg){
                
                log(msg.data(), msg.size());
            }
    };

    template <class ID>
    struct AggregatedLogger{

        private:

            using base = ConcurrentLogger<AggregatedLogger<ID>>;

            static inline std::array<std::string, dg::network_concurrency::THREAD_COUNT> log_table{};
            static inline size_t unit_capacity{}; 

        public:

            void init(std::filesystem::path arg_logger_path, std::filesystem::path arg_logger_ext, size_t arg_unit_capacity) noexcept{

                base::init(std::move(arg_logger_path), std::move(arg_logger_ext));
                unit_capacity = arg_unit_capacity;

                for (size_t i = 0; i < log_table.size(); ++i){
                    log_table[i].reserve(arg_unit_capacity);
                }
            }

            void log(const char * buf, size_t buf_sz){
                
                if (buf_sz > unit_capacity){
                    std::abort();
                }

                size_t idx          = dg::network_concurrency::this_thread_idx();
                std::string& cur    = log_table[idx]; 

                if (cur.size() + buf_sz > unit_capacity){
                    base::log(cur);
                    cur.clear();
                }
                
                std::copy(buf, buf + buf_sz, std::back_inserter(cur));
            }

            void log(const std::string& msg){

                log(msg.data(), msg.size());
            }

            void flush(){

                size_t idx = dg::network_concurrency::this_thread_idx();
                base::log(log_table[idx]);
                log_table[idx].clear();
            }
    };

    using concurrent_logger = ConcurrentLogger<std::integral_constant<size_t, 0>>;
    using aggregated_logger = std::conditional_t<HAS_ENFORCE_WRITE,
                                                 concurrent_logger, 
                                                 AggregatedLogger<std::integral_constant<size_t, 1>>>; 

    struct MsgFactory{
        
        static inline auto make_log_msg(const char * header, const char * err_msg) -> std::string{
            
            return std::format("({}) {} @ {}\n", std::string(header), std::string(err_msg), timestamp_pretty());
        }

        static inline auto make_log_msg(const char * header, const char * err_msg, err_t err_code) -> std::string{

            return std::format("({}) {} {} @ {}\n", std::string(header), std::string(err_msg), err_code, timestamp_pretty());
        }
    };

    void init(std::filesystem::path logger_path, std::filesystem::path logger_ext, size_t fastlog_buf_sz) noexcept{

        concurrent_logger::init(logger_path, logger_ext);
        aggregated_logger::init(logger_path, logger_ext, fastlog_buf_sz);
    } 

    void critical_error(const char * what, err_t err_code) noexcept{

        concurrent_logger::log(MsgFactory::make_log_msg("critical", what, err_code));
    }

    void critical_error(const char * what) noexcept{

        concurrent_logger::log(MsgFactory::make_log_msg("critical", what));
    }

    void error(const char * what, err_t err_code) noexcept{

        concurrent_logger::log(MsgFactory::make_log_msg("critical", what, err_code));
    }

    void error(const char * what) noexcept{

        concurrent_logger::log(MsgFactory::make_log_msg("error", what));
    }

    void error_fast(const char * what, err_t err_code) noexcept{

        aggregated_logger::log(MsgFactory::make_log_msg("critical", what, err_code));
    }

    void error_fast(const char * what) noexcept{

        aggregated_logger::log(MsgFactory::make_log_msg("error", what));
    }
    
    void journal(const char * what) noexcept{

        concurrent_logger::log(MsgFactory::make_log_msg("journal", what));
    }

    void journal_fast(const char * what) noexcept{

        aggregated_logger::log(MsgFactory::make_log_msg("journal", what));
    }

    void journal_optional(const char * what) noexcept{

        try{
            concurrent_logger::log(MsgFactory::make_log_msg("journal", what));
        } catch (...){
            return;
        }
    }

    void journal_optional_fast(const char * what) noexcept{

        try{
            aggregated_logger::log(MsgFactory::make_log_msg("journal", what));
        } catch (...){
            return;
        }
    }

    void flush() noexcept{

        aggregated_logger::flush();
    }

    void flush_optional() noexcept{

        try{
            aggregated_logger::flush();
        } catch (...){
            return;
        }
    }
}

namespace dg::network_log_stackdump{

    void critical_error(const char * what, err_t err_code) noexcept{

    }

    void critical_error(const char * what) noexcept{

    }

    void critical_error() noexcept{

    }
    
    void error(const char * what, err_t err_code) noexcept{

    }

    void error(const char * what) noexcept{

    }

    void error() noexcept{

    }

    void error_fast(const char * what, err_t err_code) noexcept{

    }

    void error_fast(const char * what) noexcept{

    }
    
    void error_fast() noexcept{

    }
    
    void journal(const char * what) noexcept{

    }

    void journal() noexcept{

    }

    void journal_fast(const char * what) noexcept{

    }

    void journal_fast() noexcept{

    }

    void journal_optional(const char * what) noexcept{

    }

    void journal_optional() noexcept{

    }

    void journal_optional_fast(const char * what) noexcept{

    }

    void journal_optional_fast() noexcept{

    }
}

namespace dg::network_log_scope{

    using network_log::err_t; 

    //forced to be static string (either quoted or static storage) - otherwise invalid ptr dereference 

    inline auto critical_error_terminate() noexcept{

        static int i    = 0;
        auto destructor = [=](int *) noexcept{
            
            try{
                std::rethrow_exception(std::current_exception());
                network_log_stackdump::critical_error();
            } catch (std::exception& e){
                network_log_stackdump::critical_error(e.what());
                std::terminate();
            }
        };

        return std::unique_ptr<int, decltype(destructor)>(&i, destructor);
    } 

    inline auto flush() noexcept{

        static int i    = 0;
        auto destructor = [=](int *) noexcept{
            network_log::flush();
        };

        return std::unique_ptr<int, decltype(destructor)>(&i, destructor);
    }

    inline auto flush_optional() noexcept{

        static int i    = 0;
        auto destructor = [=](int *) noexcept{
            network_log::flush_optional();
        };

        return std::unique_ptr<int, decltype(destructor)>(&i, destructor);
    }
}

#endif