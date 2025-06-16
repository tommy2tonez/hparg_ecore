#ifndef __DG_NETWORK_MEMPRESS_COLLECTOR_INSTANCE_H__
#define __DG_NETWORK_MEMPRESS_COLLECTOR_INSTANCE_H__

#include "network_mempress_dispatch_warehouse_instance.h"
#include "network_mempress_instance.h"
#include "network_mempress_collector_impl1.h"
#include <variant>
#include "network_std_container.h"
#include "network_concurrency.h"
#include "network_concurrency_x.h"
#include "network_producer_consumer.h"
#include "network_memcommit_model.h"
#include <vector>

namespace dg::network_mempress_collector_instance{

    struct InfRetryConfig{
        std::chrono::nanoseconds dur;
        size_t retry_sz;
        bool has_inf_cyclic_loop;
        bool has_expbackoff; 
    };

    struct TryCollectorConfig{
        std::optional<std::vector<uma_ptr_t>> collect_region;
        std::optional<InfRetryConfig> infretry_machine_config;
        uint64_t warehouse_connector_ingestion_sz;
        uint64_t warehouse_connector_digestion_sz;
        uint64_t mempress_collect_cap;
        uint32_t scan_frequency;
    };

    struct CompetitiveTryCollectorConfig{
        std::optional<std::vector<uma_ptr_t>> collect_region;
        std::optional<InfRetryConfig> infretry_machine_config;
        uint64_t warehouse_connector_ingestion_sz;
        uint64_t warehouse_connector_digestion_sz;
        uint64_t mempress_collect_cap;
        uint32_t scan_frequency;
    };

    struct ClockCollectorConfig{
        std::optional<std::vector<uma_ptr_t>> collect_region;
        std::optional<InfRetryConfig> infretry_machine_config;
        std::vector<std::chrono::nanoseconds> update_interval_table;
        uint64_t ops_clock_resolution;
        uint64_t warehouse_connector_ingestion_sz;
        uint64_t warehouse_connector_digestion_sz;
        uint64_t mempress_collect_cap;
        uint32_t scan_frequency;
    };

    struct ClockTryCollectorConfig{
        std::optional<std::vector<uma_ptr_t>> collect_region;
        std::optional<InfRetryConfig> infretry_machine_config;
        std::vector<std::chrono::nanoseconds> update_interval_table;
        uint64_t ops_clock_resolution;
        uint64_t warehouse_connector_ingestion_sz;
        uint64_t warehouse_connector_digestion_sz;
        uint64_t mempress_collect_cap;
        uint32_t scan_frequency;
    };

    struct ClockCompetitiveTryCollectorConfig{
        std::optional<std::vector<uma_ptr_t>> collect_region;
        std::optional<InfRetryConfig> infretry_machine_config;
        std::vector<std::chrono::nanoseconds> update_interval_table;
        uint64_t ops_clock_resolution;
        uint64_t warehouse_connector_ingestion_sz;
        uint64_t warehouse_connector_digestion_sz;
        uint64_t mempress_collect_cap;
        uint32_t scan_frequency;
    };

    struct ConfigFactory{

        static auto spawn_try_collector(const TryCollectorConfig& config) -> dg::network_concurrency::daemon_raii_handle_t{

            using event_t = dg::network_memcommit_factory::virtual_emmory_event_t;

            std::unique_ptr<RangePressInterface> range_press = {};

            if (!config.collect_region.has_value()){
                range_press = dg::network_mempress_collector::Factory::spawn_range_press(dg::network_mempress_instance::get_instance());
            } else{
                range_press = dg::network_mempress_collector::Factory::spawn_range_press(dg::network_mempress_instance::get_instance(), config.collect_region.value());
            }

            std::unique_ptr<dg::network_producer_consumer::ConsumerInterface<event_t>> connector = {};
            
            if (!config.infretry_machine_config.has_value()){
                connector = dg::network_mempress_collector::Factory::spawn_warehouse_connector(dg::network_mempress_dispatch_warehouse_instance::get_instance(),
                                                                                               conifg.warehouse_connector_ingestion_sz);
            } else{
                connector = dg::network_mempress_collector::Factory::spawn_warehouse_connector(dg::network_mempress_dispatch_warehouse_instance::get_instance(),
                                                                                               dg::network_concurrency_infretry_x::get_normal_infretry_machine(config.infretry_machine_config.dur,
                                                                                                                                                               config.infretry_machine_config.retry_sz,
                                                                                                                                                               config.infretry_machine_config.has_inf_cyclic_loop,
                                                                                                                                                               config.infretry_machine_config.has_expbackoff),
                                                                                               config.warehouse_connector_ingestion_sz);
            }

            std::unique_ptr<dg::network_concurrency::WorkerInterface> worker = dg::network_mempress_collector::Factory::spawn_try_collector(std::move(range_press),
                                                                                                                                            std::move(connector),
                                                                                                                                            config.mempress_collect_cap,
                                                                                                                                            config.warehouse_connector_digestion_sz,
                                                                                                                                            config.scan_frequency);
            
            return dg::network_exception_handler::throw_nolog(dg::network_concurrency::daemon_saferegister(dg::network_concurrency::COMPUTING_DAEMON, std::move(worker)));
        }

        static auto spawn_competitive_try_collector(const CompetitiveTryCollectorConfig& config) -> dg::network_concurrency::daemon_raii_handle_t{

            using event_t = dg::network_memcommit_factory::virtual_emmory_event_t;

            std::unique_ptr<RangePressInterface> range_press = {};

            if (!config.collect_region.has_value()){
                range_press = dg::network_mempress_collector::Factory::spawn_range_press(dg::network_mempress_instance::get_instance());
            } else{
                range_press = dg::network_mempress_collector::Factory::spawn_range_press(dg::network_mempress_instance::get_instance(), config.collect_region.value());
            }

            std::unique_ptr<dg::network_producer_consumer::ConsumerInterface<event_t>> connector = {};

            if (!config.infretry_machine_config.has_value()){
                connector = dg::network_mempress_collector::Factory::spawn_warehouse_connector(dg::network_mempress_dispatch_warehouse_instance::get_instance(),
                                                                                               conifg.warehouse_connector_ingestion_sz);
            } else{
                connector = dg::network_mempress_collector::Factory::spawn_warehouse_connector(dg::network_mempress_dispatch_warehouse_instance::get_instance(),
                                                                                               dg::network_concurrency_infretry_x::get_normal_infretry_machine(config.infretry_machine_config.dur,
                                                                                                                                                               config.infretry_machine_config.retry_sz,
                                                                                                                                                               config.infretry_machine_config.has_inf_cyclic_loop,
                                                                                                                                                               config.infretry_machine_config.has_expbackoff),
                                                                                               config.warehouse_connector_ingestion_sz);
            }

            std::unique_ptr<dg::network_concurrency::WorkerInterface> worker = dg::network_mempress_collector::Factory::spawn_competitive_try_collector(std::move(range_press),
                                                                                                                                                        std::move(connector),
                                                                                                                                                        config.mempress_collect_cap,
                                                                                                                                                        config.warehouse_connector_digestion_sz,
                                                                                                                                                        config.scan_frequency);

            return dg::network_exception_handler::throw_nolog(dg::network_concurrency::daemon_saferegister(dg::network_concurrency::COMPUTING_DAEMON, std::move(worker)));
        }

        static auto spawn_clock_collector(const ClockCollectorConfig& config) -> dg::network_concurrency::daemon_raii_handle_t{

            using event_t = dg::network_memcommit_factory::virtual_emmory_event_t;

            std::unique_ptr<RangePressInterface> range_press = {};

            if (!config.collect_region.has_value()){
                range_press = dg::network_mempress_collector::Factory::spawn_range_press(dg::network_mempress_instance::get_instance());
            } else{
                range_press = dg::network_mempress_collector::Factory::spawn_range_press(dg::network_mempress_instance::get_instance(), config.collect_region.value());
            }

            std::unique_ptr<dg::network_producer_consumer::ConsumerInterface<event_t>> connector = {};
            
            if (!config.infretry_machine_config.has_value()){
                connector = dg::network_mempress_collector::Factory::spawn_warehouse_connector(dg::network_mempress_dispatch_warehouse_instance::get_instance(),
                                                                                               conifg.warehouse_connector_ingestion_sz);
            } else{
                connector = dg::network_mempress_collector::Factory::spawn_warehouse_connector(dg::network_mempress_dispatch_warehouse_instance::get_instance(),
                                                                                               dg::network_concurrency_infretry_x::get_normal_infretry_machine(config.infretry_machine_config.dur,
                                                                                                                                                               config.infretry_machine_config.retry_sz,
                                                                                                                                                               config.infretry_machine_config.has_inf_cyclic_loop,
                                                                                                                                                               config.infretry_machine_config.has_expbackoff),
                                                                                               config.warehouse_connector_ingestion_sz);
            }

            std::unique_ptr<dg::network_concurrency::WorkerInterface> worker = dg::network_mempress_collector::Factory::spawn_clock_collector(std::move(range_press),
                                                                                                                                              std::move(connector),
                                                                                                                                              config.update_interval_table,
                                                                                                                                              config.ops_clock_resolution,
                                                                                                                                              config.mempress_collect_cap,
                                                                                                                                              config.warehouse_connector_digestion_sz,
                                                                                                                                              config.scan_frequency);
            
            return dg::network_exception_handler::throw_nolog(dg::network_concurrency::daemon_saferegister(dg::network_concurrency::COMPUTING_DAEMON, std::move(worker)));
        }

        static auto spawn_clock_try_collector(const ClockTryCollectorConfig& config) -> dg::network_concurrency::daemon_raii_handle_t{

            using event_t = dg::network_memcommit_factory::virtual_emmory_event_t;

            std::unique_ptr<RangePressInterface> range_press = {};

            if (!config.collect_region.has_value()){
                range_press = dg::network_mempress_collector::Factory::spawn_range_press(dg::network_mempress_instance::get_instance());
            } else{
                range_press = dg::network_mempress_collector::Factory::spawn_range_press(dg::network_mempress_instance::get_instance(), config.collect_region.value());
            }

            std::unique_ptr<dg::network_producer_consumer::ConsumerInterface<event_t>> connector = {};
            
            if (!config.infretry_machine_config.has_value()){
                connector = dg::network_mempress_collector::Factory::spawn_warehouse_connector(dg::network_mempress_dispatch_warehouse_instance::get_instance(),
                                                                                               conifg.warehouse_connector_ingestion_sz);
            } else{
                connector = dg::network_mempress_collector::Factory::spawn_warehouse_connector(dg::network_mempress_dispatch_warehouse_instance::get_instance(),
                                                                                               dg::network_concurrency_infretry_x::get_normal_infretry_machine(config.infretry_machine_config.dur,
                                                                                                                                                               config.infretry_machine_config.retry_sz,
                                                                                                                                                               config.infretry_machine_config.has_inf_cyclic_loop,
                                                                                                                                                               config.infretry_machine_config.has_expbackoff),
                                                                                               config.warehouse_connector_ingestion_sz);
            }

            std::unique_ptr<dg::network_concurrency::WorkerInterface> worker = dg::network_mempress_collector::Factory::spawn_clock_try_collector(std::move(range_press),
                                                                                                                                                  std::move(connector),
                                                                                                                                                  config.update_interval_table,
                                                                                                                                                  config.ops_clock_resolution,
                                                                                                                                                  config.mempress_collect_cap,
                                                                                                                                                  config.warehouse_connector_digestion_sz,
                                                                                                                                                  config.scan_frequency);
            
            return dg::network_exception_handler::throw_nolog(dg::network_concurrency::daemon_saferegister(dg::network_concurrency::COMPUTING_DAEMON, std::move(worker)));
        }

        static auto spawn_clock_competitive_try_collector(const ClockCompetitiveTryCollectorConfig& config) -> dg::network_concurrency::daemon_raii_handle_t{

            using event_t = dg::network_memcommit_factory::virtual_emmory_event_t;

            std::unique_ptr<RangePressInterface> range_press = {};

            if (!config.collect_region.has_value()){
                range_press = dg::network_mempress_collector::Factory::spawn_range_press(dg::network_mempress_instance::get_instance());
            } else{
                range_press = dg::network_mempress_collector::Factory::spawn_range_press(dg::network_mempress_instance::get_instance(), config.collect_region.value());
            }

            std::unique_ptr<dg::network_producer_consumer::ConsumerInterface<event_t>> connector = {};
            
            if (!config.infretry_machine_config.has_value()){
                connector = dg::network_mempress_collector::Factory::spawn_warehouse_connector(dg::network_mempress_dispatch_warehouse_instance::get_instance(),
                                                                                               conifg.warehouse_connector_ingestion_sz);
            } else{
                connector = dg::network_mempress_collector::Factory::spawn_warehouse_connector(dg::network_mempress_dispatch_warehouse_instance::get_instance(),
                                                                                               dg::network_concurrency_infretry_x::get_normal_infretry_machine(config.infretry_machine_config.dur,
                                                                                                                                                               config.infretry_machine_config.retry_sz,
                                                                                                                                                               config.infretry_machine_config.has_inf_cyclic_loop,
                                                                                                                                                               config.infretry_machine_config.has_expbackoff),
                                                                                               config.warehouse_connector_ingestion_sz);
            }

            std::unique_ptr<dg::network_concurrency::WorkerInterface> worker = dg::network_mempress_collector::Factory::spawn_clock_competitive_try_collector(std::move(range_press),
                                                                                                                                                              std::move(connector),
                                                                                                                                                              config.update_interval_table,
                                                                                                                                                              config.ops_clock_resolution,
                                                                                                                                                              config.mempress_collect_cap,
                                                                                                                                                              config.warehouse_connector_digestion_sz,
                                                                                                                                                              config.scan_frequency);

            return dg::network_exception_handler::throw_nolog(dg::network_concurrency::daemon_saferegister(dg::network_concurrency::COMPUTING_DAEMON, std::move(worker)));
        }
    };

    struct Config{
        std::vector<std::variant<TryCollectorConfig, CompetitiveTryCollectorConfig, ClockCollectorConfig, ClockTryCollectorConfig, ClockCompetitiveTryCollectorConfig>> collector_config_vec;
    };

    struct MempressCollectorSignature{};

    using collector_singleton = stdx::singleton<MempressCollectorSignature, std::vector<dg::network_concurrency::daemon_raii_handle_t>>;

    void init(const Config& config){

        std::vector<dg::network_concurrency::daemon_raii_handle_t resource = {}; 

        for (const auto& collector_config: config.collector_config_vec){
            if (std::holds_alternative<TryCollectorConfig>(collector_config)){
                resource.push_back(ConfigFactory::spawn_try_collector(std::get<TryCollectorConfig>(collector_config)));
            } else if (std::holds_alternative<CompetitiveTryCollectorConfig>(collector_config)){
                resource.push_back(ConfigFactory::spawn_competitive_try_collector(std::get<CompetitiveTryCollectorConfig>(collector_config)));
            } else if (std::holds_alternative<ClockCollectorConfig>(collector_config)){
                resource.push_back(ConfigFactory::spawn_clock_collector(std::get<ClockCollectorConfig>(collector_config)));
            } else if (std::holds_alternative<ClockTryCollectorConfig>(collector_config)){
                resource.push_back(ConfigFactory::spawn_clock_try_collector(std::get<ClockTryCollectorConfig>(collector_config)));
            } else if (std::holds_alternative<ClockCompetitiveTryCollectorConfig>(collector_config)){
                resource.push_back(ConfigFactory::spawn_clock_competitive_try_collector(std::get<ClockCompetitiveTryCollectorConfig>(collector_config)));
            } else{
                dg::network_exception::throw_exception(dg::network_exception::INVALID_ARGUMENT);
            }
        }

        collector_singleton::get() = std::move(resource);
    }

    void deinit() noexcept{

        collector_singleton::get() = {};
    }
} 

#endif