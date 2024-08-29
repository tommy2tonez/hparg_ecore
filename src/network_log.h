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

namespace dg::network_log_implementation{

    using logger_option_t = uint8_t; 

    enum logger_option_t: logger_option_t{
        HAS_WRITE               = 0b0001,
        HAS_FAST_WRITE          = 0b0010,
        HAS_OPTIONAL            = 0b0100,
        HAS_CONCURRENT_LOGGER   = 0b1000
    };

    template <class T>
    struct LoggerInterface{

        static void log(const char * buf, size_t sz){

            T::log(buf, sz);
        }

        static void flush(){

            T::flush();
        }
    };

    template <class ID>
    struct MtxLogger: LoggerInterface<MtxLogger<ID>>{

        private:

            static inline std::filesystem::path logger_path{};
            static inline std::mutex mtx{};

        public:

            static void init(std::filesystem::path arg_path) noexcept{

                logger_path = std::move(arg_path);
            }

            static void log(const char * buf, size_t buf_sz){

                auto lck_grd    = std::lock_guard<std::mutex>(mtx);
                auto writer     = std::ofstream(logger_path, std::ios::app);
                writer.write(buf, buf_sz);
            }

            static void flush(){

                (void) flush;
            }
    };

    template <class ID>
    struct ConcurrentLogger: LoggerInterface<ConcurrentLogger<ID>>{

        private:

            static inline std::vector<std::filesystem::path> path_table{};

        public:

            static void init(std::vector<std::filesystem::path> arg_path_table) noexcept{
                
                path_table = std::move(arg_path_table);
            } 

            static void log(const char * buf, size_t buf_sz){

                const auto& cur_path    = path_table[dg::network_concurrency::this_thread_idx()];
                auto writer             = std::ofstream(cur_path, std::ios::app);
                writer.write(buf, buf_sz);
            }

            static void flush(){
                
                (void) flush;
            }
    };

    template <class ID, class Logger>
    struct AggregatedLogger{};

    template <class ID, class T>
    struct AggregatedLogger<ID, LoggerInterface<T>>: LoggerInterface<AggregatedLogger<ID, LoggerInterface<T>>>{

        private:

            using base = LoggerInterface<T>;
            static inline std::vector<std::string> log_table{};

        public:
 
            static void init(std::vector<std::string> arg_log_table) noexcept{
                
                log_table = std::move(arg_log_table);
            }

            static void log(const char * buf, size_t buf_sz){

                size_t idx          = dg::network_concurrency::this_thread_idx();
                std::string& cur    = log_table[idx]; 

                if (buf_sz > cur.capacity()){
                    base::log(buf, buf_sz);
                    return;
                }

                if (cur.size() + buf_sz > cur.capacity()){
                    base::log(cur.data(), cur.size());
                    cur.clear();
                }
                
                std::copy(buf, buf + buf_sz, std::back_inserter(cur));
            }

            static void flush(){

                size_t idx = dg::network_concurrency::this_thread_idx();
                base::log(log_table[idx].data(), log_table[idx].size());
                log_table[idx].clear();
                base::flush();
            }
    };

    template <class ID>
    struct VoidLogger: LoggerInterface<VoidLogger<ID>>{
        
        static void log(const char * buf, size_t sz){

            (void) buf;
        }

        static void flush(){

            (void) flush;
        }
    } 

    template <class ID, class SlowLogger, class FastLogger, class HasOptional>
    struct TaxoLogger{}; 

    template <class ID, class T, class T1, bool HAS_OPTIONAL>
    struct TaxoLogger<ID, LoggerInterface<T>, LoggerInterface<T1>, std::integral_constant<bool, HAS_OPTIONAL>>{

        using slow_logger = LoggerInterface<T>;
        using fast_logger = LoggerInterface<T1>; 

        static auto make_log(const char * err, const char * what) -> std::string{

            std::string fmt                     = "[{}]\n{}\n{} unix_ns\n------------\n";
            std::string err_str                 = std::string(err); 
            std::string what_str                = std::string(what);
            std::chrono::nanoseconds unix_ts    = dg::utility::unix_timestamp(); 
            std::string unix_ts_str             = std::to_string(static_cast<size_t>(unix_ts.count()))

            return std::format(fmt, err_str, what_str, unix_ts_str);
        }

        static void critical(const char * what) noexcept{

            std::string msg = make_log("critical", what);
            slow_logger::log(msg.data(), msg.size());
        }

        static void error(const char * what) noexcept{

            std::string msg = make_log("error", what);
            slow_logger::log(msg.data(), msg.size());
        }

        static void error_fast(const char * what) noexcept{

            std::string msg = make_log("error", what);
            fast_logger::log(msg.data(), msg.size());
        }

        static void error_optional(const char * what) noexcept{

            if constexpr(HAS_OPTIONAL){
                try{
                    std::string msg = make_log("error", what);
                    slow_logger::log(msg.data(), msg.size());
                } catch (...){
                    (void) what;
                }
            } else{
                error(what);
            }
        }

        static void error_optional_fast(const char * what) noexcept{

            if constexpr(HAS_OPTIONAL){
                try{
                    std::string msg = make_log("error", what);
                    fast_logger::log(msg.data(), msg.size());
                } catch (...){
                    (void) what;
                }
            } else{
                error_fast(what);
            }
        }

        static void journal(const char * what) noexcept{

            std::string msg = make_log("journal", what);
            slow_logger::log(msg.data(), msg.size());
        }

        static void journal_fast(const char * what) noexcept{

            std::string msg = make_log("journal", what);
            fast_logger::log(msg.data(), msg.size());
        }

        static void journal_optional(const char * what) noexcept{

            if constexpr(HAS_OPTIONAL){
                try{
                    std::string msg = make_log("journal", what);
                    slow_logger::log(msg.data(), msg.size());
                } catch (...){
                    (void) what;
                }
            } else{
                journal(what);
            }
        }

        static void journal_optional_fast(const char * what) noexcept{

            if constexpr(HAS_OPTIONAL){
                try{
                    std::string msg = make_log("journal", what);
                    fast_logger::log(msg.data(), msg.size());
                } catch (...){
                    (void) what;
                }
            } else{
                journal_fast(what);
            }
        }

        static void flush() noexcept{

            slow_logger::flush();
            fast_logger::flush();
        }

        static void flush_optional() noexcept{

            if constexpr(HAS_OPTIONAL){
                try{
                    slow_logger::flush();
                    fast_logger::flush();
                } catch (...){
                    (void) flush;
                }
            } else{
                flush();
            }
        }
    };

    template <class ID, logger_option_t LOGGER_OPTION_VALUE>
    auto make_taxo_logger(std::vector<std::filesystem::path> logger_dir, 
                          const ID,
                          const std::integral_constant<logger_option_t, LOGGER_OPTION_VALUE>, 
                          const char * EXTENSION = "log",
                          const size_t FASTBUF_CAPACITY = 1024){

        constexpr bool HAS_OPTIONAL_VALUE           = (LOGGER_OPTION_VALUE & HAS_OPTIONAL) != 0u;
        constexpr bool HAS_WRITE_VALUE              = (LOGGER_OPTION_VALUE & HAS_WRITE) != 0u;
        constexpr bool HAS_CONCURRENT_LOGGER_VALUE  = (LOGGER_OPTION_VALUE & HAS_CONCURRENT_LOGGER) != 0u;
        constexpr bool HAS_FAST_WRITE_VALUE         = (LOGGER_OPTION_VALUE & HAS_FAST_WRITE) != 0u;

        auto slow_logger_initializer = [&]{
            if constexpr(HAS_WRITE_VALUE){
                if constexpr(HAS_CONCURRENT_LOGGER_VALUE){
                    ConcurrentLogger<ID>::init(logger_dir, EXTENSION);
                    return typename ConcurrentLogger<ID>::interface_t{};
                } else{
                    MtxLogger<ID>::init(logger_dir / EXTENSION); //
                    return typename MtxLogger<ID>::interface_t{};
                }
            } else{
                return typename VoidLogger<ID>::interface_t{};
            }
        }; 

        auto fast_logger_initializer = [&]{
            if constexpr(HAS_FAST_WRITE_VALUE){
                AggregatedLogger<ID, slow_logger>::init(FASTBUF_CAPACITY);
                return typename AggregatedLogger<ID, slow_logger>::interface_t{};
            } else{
                return slow_logger{};
            }
        };
        
        auto slow_logger    = slow_logger_initializer();
        auto fast_logger    = fast_logger_initializer();

        return TaxoLogger<ID, decltype(slow_logger), decltype(fast_logger), std::integral_constant<bool, HAS_OPTIONAL_VALUE>>{};
    } 
}

namespace dg::network_log{

    static inline constexpr network_log_implementation::logger_option_t LOGGER_OPTION_VALUE = 0u; 
    using logger = decltype(network_log_implementation::make_taxo_logger({}, std::integral_constant<network_log_implementation::logger_option_t, LOGGER_OPTION_VALUE>{})); 

    void init(std::vector<std::filesystem::path> logger_path){

        network_log_implementation::make_taxo_logger(logger_path, std::integral_constant<network_log_implementation::logger_option_t, LOGGER_OPTION_VALUE>{});
    } 

    void critical(const char * what) noexcept{

        logger::critical(what);
    }

    void error(const char * what) noexcept{

        logger::error(what);
    }

    void error_fast(const char * what) noexcept{

        logger::error_fast(what);
    }
    
    void error_optional(const char * what) noexcept{

        logger::error_optional(what);
    }

    void error_optional_fast(const char * what) noexcept{

        logger::error_optional_fast(what);
    }

    void journal(const char * what) noexcept{

        logger::journal(what);
    }

    void journal_fast(const char * what) noexcept{

        logger::journal_fast(what);
    }

    void journal_optional(const char * what) noexcept{

        logger::journal_optional(what);
    }

    void journal_optional_fast(const char * what) noexcept{

        logger::journal_optional_fast(what);
    }

    void flush() noexcept{

        logger::flush();
    }

    void flush_optional() noexcept{

        logger::flush_optional();
    }
}

namespace dg::network_log_stackdump{

    auto add_stack_trace(const char * what) -> std::string{

        std::string fmt             = "<stackdump_begin>\n{}\n<stackdump_end>\n{}"; 
        std::string stacktrace_str  = std::stacktrace::current().to_string();
        std::string what_str        = std::string(what); 

        return std::format(fmt, stacktrace_str, what_str);
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

        auto new_what = dg::functional::invoke_nothrow_or_empty(add_stack_trace, what);
        network_log::error_optional(new_what);
    }

    void error_optional() noexcept{

        error_optional("");
    }

    void error_optional_fast(const char * what) noexcept{

        auto new_what = dg::functional::invoke_nothrow_or_empty(add_stack_trace, what);
        network_log::error_optional_fast(new_what);
    }

    void error_optional_fast() noexcept{

        error_optional_fast("");
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

        auto new_what = dg::functional::invoke_nothrow_or_empty(add_stack_trace, what);
        network_log::journal_optional(new_what.c_str());
    }

    void journal_optional() noexcept{

        journal_optional("");
    }

    void journal_optional_fast(const char * what) noexcept{

        auto new_what = dg::functional::invoke_nothrow_or_empty(add_stack_trace, what);
        network_log::journal_optional_fast(new_what.c_str());
    }

    void journal_optional_fast() noexcept{

        network_log::journal_optional_fast("");
    }

    void flush() noexcept{

        network_log::flush();
    }

    void flush_optional() noexcept{

        network_log::flush_optional();
    }
}

namespace dg::network_log_scope{

    using network_log::err_t; 

    auto critical_terminate() noexcept{

        static int i    = 0;
        auto destructor = [=](int *) noexcept{
            
            try{
                std::rethrow_exception(std::current_exception());
                network_log_stackdump::critical_error();
            } catch (std::exception& e){
                network_log_stackdump::critical_error(e.what()); //
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