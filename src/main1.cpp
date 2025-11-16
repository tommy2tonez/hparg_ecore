#define DEBUG_MODE_FLAG true
#define STRONG_MEMORY_ORDERING_FLAG true

#include <iostream>
#include "network_bytecode.h"
#include "network_concurrency.h"
#include <string>
#include <thread>
#include <mutex>
#include "network_kernel_mailbox_impl1.h"
#include "network_kernel_mailbox_impl1_x.h"
#include "network_kernelmap_x.h"

static inline std::mutex print_mtx;

void print(const std::string& msg)
{
    std::lock_guard<std::mutex> lck_grd(print_mtx);
    std::cout << msg << std::endl;
}

// class Worker: public virtual dg::network_concurrency::WorkerInterface
// {
//     private:

//         size_t id;

//     public:

//         Worker(size_t id): id(id){}

//         bool run_one_epoch() noexcept
//         {
//             print(std::to_string(id) + " <worker> has been here");
//             std::this_thread::sleep_for(std::chrono::seconds(1));            
//             return false;
//         }
// };

class IPSiever: public virtual dg::network_kernel_mailbox_impl1::external_interface::IPSieverInterface{

    public:

        auto thru(dg::network_kernel_mailbox_impl1::model::Address) noexcept -> std::expected<bool, exception_t>{

            return true;
        }
};

class HelloWorldWorker : public virtual dg::network_concurrency::WorkerInterface
{
    private:

        dg::network_kernel_mailbox_impl1::core::MailboxInterface * sock;
    
    public:

        HelloWorldWorker(dg::network_kernel_mailbox_impl1::core::MailboxInterface * sock) noexcept: sock(sock){}

        bool run_one_epoch() noexcept
        {
            // std::cout << "Hello World Worker begins" << std::endl;

            dg::network_kernel_mailbox_impl1::model::Address self_addr = {};
            self_addr.ip = dg::network_kernel_mailbox_impl1::model::IP{.ip = dg::network_kernel_mailbox_impl1::utility::ipv4_std_formatted_str_to_compact("127.0.0.1").value()};
            self_addr.port = 5000;

            for (size_t i = 0u; i < 10; ++i){
                dg::network_kernel_mailbox_impl1::model::MailBoxArgument arg = {};

                arg.to = self_addr;
                arg.content = "Hello World!";
                arg.content_sz = 12;

                exception_t err;

                sock->send(&arg, 1u, &err);
            }

            // std::cout << "Hello World Worker sent" << std::endl;

            while (true){
                char recv_buf[12];
                void * recv_buf_void = recv_buf;
                size_t cap = 12u;
                size_t sz = 0u; 
                size_t recv_buf_sz;
                
                sock->recv(&recv_buf_void, &cap, &sz, recv_buf_sz, 1u);

                if (recv_buf_sz != 0u){
                    std::cout << std::string_view(recv_buf, sz) << "<recv>" << std::endl;
                }

                print("hello_world_worker sleeping...");
                std::this_thread::sleep_for(std::chrono::seconds(1));
            }
        }
};

class RecvWorker: public virtual dg::network_concurrency::WorkerInterface
{

    private:

        dg::network_kernel_mailbox_impl1::core::MailboxInterface * sock;

    public:

        RecvWorker(dg::network_kernel_mailbox_impl1::core::MailboxInterface * sock) noexcept: sock(sock){}

        bool run_one_epoch() noexcept
        {
            size_t total_vec_sz = 0u;

            const size_t RECV_SZ    = 1 << 10;
            const size_t BUF_CAP    = 1 << 10;  

            dg::vector<void *> recv_arr(RECV_SZ);
            dg::vector<size_t> cap_arr(RECV_SZ);
            dg::vector<size_t> sz_arr(RECV_SZ);

            for (size_t i = 0u; i < RECV_SZ; ++i)
            {
                recv_arr[i] = std::malloc(BUF_CAP);
                cap_arr[i]  = BUF_CAP;
                sz_arr[i]   = 0u; 
            }

            size_t actual_recv_sz; 

            while (true)
            {
                {
                    sock->recv(recv_arr.data(), cap_arr.data(), sz_arr.data(), actual_recv_sz, RECV_SZ);

                    if (actual_recv_sz != 0u)
                    {
                        total_vec_sz += actual_recv_sz;
                        std::cout << "<recv>" << total_vec_sz << "<total_vec_sz>" << std::endl;
                        std::cout << "<stamp>" << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::utc_clock::now().time_since_epoch()) << std::endl;
                    } else{
                        std::cout << "<recv_worker_beating>" << std::endl;
                    }
                }
            }
        }
};

class SendWorker: public virtual dg::network_concurrency::WorkerInterface
{
    private:

        dg::network_kernel_mailbox_impl1::core::MailboxInterface * sock;

    public:

        SendWorker(dg::network_kernel_mailbox_impl1::core::MailboxInterface * sock) noexcept: sock(sock){}

        bool run_one_epoch() noexcept
        {

            std::cout << "<begin0>" << std::endl;

            dg::vector<dg::network_kernel_mailbox_impl1::model::MailBoxArgument> str_vec    = make_str_vec(size_t{1} << 7, size_t{1} << 17, 5000);

            dg::vector<exception_t> err_vec(str_vec.size());

            std::cout << "<begin>" << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::utc_clock::now().time_since_epoch()) << std::endl;
            
            while (true)
            {
                size_t max_sz = size_t{1} << 17;
                size_t iterable_sz = str_vec.size() / max_sz + (str_vec.size() % max_sz != 0u); 

                for (size_t i = 0u; i < iterable_sz; ++i)
                {
                    size_t first = i * max_sz;
                    size_t last = std::min((i + 1) * max_sz, str_vec.size());

                    sock->send(std::next(str_vec.data(), first), (last - first), err_vec.data());

                    std::this_thread::sleep_for(std::chrono::seconds(10));
                }
            }

            std::cout << "<sent>" << static_cast<size_t>(err_vec[0]) << std::endl;

            std::cout << "<sent>" << std::count(err_vec.begin(), err_vec.end(), dg::network_exception::SUCCESS) << std::endl;

            while (true)
            {
                std::this_thread::sleep_for(std::chrono::seconds(10));
            }

            return true;
        }
    
    private:
        
        auto make_str(size_t packet_sz) -> char *
        {
            char * rs = static_cast<char *>(std::malloc(packet_sz));
            std::generate(rs, std::next(rs, packet_sz), dg::network_randomizer::randomize_int<char>);

            return rs;
        }

        auto make_str_vec(size_t packet_sz, size_t packet_count, uint16_t port) -> dg::vector<dg::network_kernel_mailbox_impl1::model::MailBoxArgument>
        {
            dg::network_kernel_mailbox_impl1::model::Address self_addr = {};
            self_addr.ip = dg::network_kernel_mailbox_impl1::model::IP{.ip = dg::network_kernel_mailbox_impl1::utility::ipv4_std_formatted_str_to_compact("127.0.0.1").value()};
            self_addr.port = port;

            dg::vector<dg::network_kernel_mailbox_impl1::model::MailBoxArgument> rs{};

            for (size_t i = 0u; i < packet_count; ++i)
            {
                dg::network_kernel_mailbox_impl1::model::MailBoxArgument arg = {};

                arg.to          = self_addr;
                arg.content     = make_str(packet_sz);
                arg.content_sz  = packet_sz;

                if (i % 10000 == 0u)
                {
                    std::cout << i << "<>" << std::endl;
                }
                rs.push_back(std::move(arg));
            }

            return rs;
        }
};

int main()
{
    {
        dg::network_concurrency::init(dg::network_concurrency::Config{
            .computing_cpu_usage = 0.1,
            .io_cpu_usage = 0.9,
            .transportation_cpu_usage = 0.1,
            .heartbeat_cpu_usage = 0.1,
            .high_parallel_hyperthread_per_core = 1,
            .high_compute_hyperthread_per_core = 20,
            .uniform_affine_group = std::vector<int>{1, 2, 3, 4, 5, 6, 7, 8}
        });

        {
            dg::network_stack_allocation::init(10000, size_t{1} << 25);

            auto [retry_device_up, destructor] = dg::network_concurrency_infretry_x::get_infretry_machine(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::milliseconds(1000))); 
            std::shared_ptr<dg::network_concurrency_infretry_x::ExecutorInterface> retry_device = std::move(retry_device_up);

            std::cout << "making socket ..." << std::endl;

            dg::network_kernel_mailbox_impl1::allocation::init({.total_mempiece_count = 1 << 18,
                                                                .mempiece_sz = 1 << 12,
                                                                .affined_refill_sz = 1 << 8,
                                                                .affined_mem_vec_capacity = 1 << 8,
                                                                .affined_free_vec_capacity = 1 << 8});
            
            dg::network_kernel_mailbox_impl1_flash_streamx::init_memory(
                                                            {.total_mempiece_count = 1 << 20,
                                                             .mempiece_sz = 1 << 12,
                                                             .affined_refill_sz = 1 << 8,
                                                             .affined_mem_vec_capacity = 1 << 8,
                                                             .affined_free_vec_capacity = 1 << 8}, 
                                                            {
                                                             .total_mempiece_count = 1 << 20,
                                                             .mempiece_sz = 1 << 12,
                                                             .affined_refill_sz = 1 << 8,
                                                             .affined_mem_vec_capacity = 1 << 8,
                                                             .affined_free_vec_capacity = 1 << 8
                                                            },
                                                            {.total_mempiece_count = 1 << 12,
                                                             .mempiece_sz = 1 << 12,
                                                             .affined_refill_sz = 1 << 8,
                                                             .affined_mem_vec_capacity = 1 << 8,
                                                             .affined_free_vec_capacity = 1 << 8
                                                            });

            auto sock = dg::network_kernel_mailbox_impl1::spawn(dg::network_kernel_mailbox_impl1::Config{
                .num_kernel_inbound_worker = 16,
                .num_process_inbound_worker = 16,
                .num_outbound_worker = 16,
                .num_kernel_rescue_worker = 1,
                .num_retry_worker = 1,

                .inbound_socket_concurrency_sz = 16,
                .outbound_socket_concurrency_sz = 16,
                .sin_fam = AF_INET,
                .comm = SOCK_DGRAM,
                .protocol = 0,
                .host_ip = {.ip = dg::network_kernel_mailbox_impl1::utility::ipv4_std_formatted_str_to_compact("127.0.0.1").value()},
                .host_port_inbound = 5000,
                .host_port_outbound = 5001,

                .is_void_retransmission_controller = false,
                .retransmission_delay = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::seconds(1)),
                .retransmission_concurrency_sz = 1,
                .retransmission_queue_cap = 1 << 16,
                .retransmission_user_queue_cap = 1 << 14,
                .retransmission_packet_cap = 10,
                .retransmission_idhashset_cap = 1 << 25,
                .retransmission_ticking_clock_resolution = 1 << 10,
                .retransmission_has_react_pattern = false,
                .retransmission_react_sz = 1 << 8,
                .retransmission_react_queue_cap = 1 << 10,
                .retransmission_react_time = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::milliseconds(10)),
                .retransmission_has_exhaustion_control = true,

                .inbound_buffer_concurrency_sz = 32,
                .inbound_buffer_container_cap = 1 << 14,
                .inbound_buffer_has_react_pattern = true,
                .inbound_buffer_react_sz = 1 << 10,
                .inbound_buffer_react_queue_cap = 1 << 12,
                .inbound_buffer_react_time = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::milliseconds(10)),
                .inbound_buffer_has_fair_redistribution = true,
                .inbound_buffer_fair_distribution_queue_cap = 1 << 14,
                .inbound_buffer_fair_waiting_queue_cap = 1 << 10,
                .inbound_buffer_fair_leftover_queue_cap = 1 << 10,
                .inbound_buffer_fair_unit_sz = 1 << 12,
                .inbound_buffer_has_exhaustion_control = true,

                .inbound_packet_concurrency_sz = 32,
                .inbound_packet_container_cap = 1 << 14,
                .inbound_packet_has_react_pattern = true,
                .inbound_packet_react_sz = 1 << 10,
                .inbound_packet_react_queue_cap = 1 << 12,
                .inbound_packet_react_time = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::milliseconds(10)),
                .inbound_packet_has_fair_redistribution = true,
                .inbound_packet_fair_packet_queue_cap = 1 << 14,
                .inbound_packet_fair_waiting_queue_cap = 1 << 10,
                .inbound_packet_fair_leftover_queue_cap = 1 << 10,
                .inbound_packet_fair_unit_sz = 1 << 12,
                .inbound_packet_has_exhaustion_control = true, 

                .inbound_idhashset_concurrency_sz = 32,
                .inbound_idhashset_cap = 1 << 14,

                .worker_inbound_buffer_fair_container_fr_warehouse_get_cap = 1 << 12,
                .worker_inbound_buffer_fair_container_to_warehouse_push_cap = 1 << 12,
                .worker_inbound_buffer_fair_container_busy_threshold = 0u,

                .worker_inbound_fair_packet_fr_warehouse_get_cap = 1 << 12,
                .worker_inbound_fair_packet_to_warehouse_push_cap = 1 << 12,
                .worker_inbound_fair_packet_busy_threshold = 0u,

                .worker_inbound_buffer_accumulation_sz = 1 << 12,
                .worker_inbound_packet_consumption_cap = 1 << 12,
                .worker_inbound_packet_busy_threshold_sz = 1,
                .worker_rescue_packet_sz_per_transmit = 1 << 6,
                .worker_kernel_rescue_dispatch_threshold = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::seconds(100)),
                .worker_retransmission_consumption_cap = 1000,
                .worker_retransmission_busy_threshold_sz = 1,
                .worker_outbound_packet_consumption_cap = 10,
                .worker_outbound_packet_busy_threshold_sz = 1,

                .mailbox_inbound_cap = size_t{1} << 20,
                .mailbox_outbound_cap = size_t{1} << 20,
                .traffic_reset_duration = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::seconds(1)),

                .outbound_transmit_frequency = uint32_t{1} << 18,

                .outbound_container_request_packet_container_cap = 1 << 20,
                .outbound_container_ack_packet_container_cap = 1 << 20,
                .outbound_container_krescue_packet_container_cap = 1 << 20,
                .outbound_container_waiting_queue_capacity = 1 << 10,
                .outbound_container_leftover_queue_capacity = 1 << 10,
                .outbound_container_unit_sz = 1 << 10,
                .outbound_container_has_exhaustion_control = false,

                .inbound_tc_has_borderline_per_inbound_worker = true,
                .inbound_tc_peraddr_cap = uint32_t{1} << 20,
                .inbound_tc_global_cap = uint32_t{1} << 20,
                .inbound_tc_addrmap_cap = uint32_t{1} << 20,
                .inbound_tc_side_cap = uint32_t{1} << 20,

                .outbound_tc_has_borderline_per_outbound_worker = true,
                .outbound_tc_border_line_sz = uint32_t{1} << 20,
                .outbound_tc_peraddr_cap = uint32_t{1} << 20,
                .outbound_tc_global_cap = uint32_t{1} << 20,
                .outbound_tc_addrmap_cap = uint32_t{1} << 20,
                .outbound_tc_side_cap = uint32_t{1} << 20,
                .natip_controller = dg::network_kernel_mailbox_impl1::get_default_natip_controller(std::make_unique<IPSiever>(), std::make_unique<IPSiever>(), 1024, 1024),
                // .busy_retriever = nullptr,
                .retry_device = retry_device
            });

            // auto sock2 = dg::network_kernel_mailbox_impl1_flash_streamx::spawn(dg::network_kernel_mailbox_impl1_flash_streamx::Config{
            //     .factory_addr = {.ip = dg::network_kernel_mailbox_impl1::utility::ipv4_std_formatted_str_to_compact("127.0.0.1").value(),
            //                      .port = 5001},
            //     .packetizer_segment_bsz = size_t{1} << 8,
            //     .packetizer_max_bsz = size_t{1} << 20,
            //     .packetizer_has_integrity_transmit = true,

            //     .gate_controller_ato_component_sz = 16,
            //     .gate_controller_ato_map_capacity = size_t{1} << 20,
            //     .gate_controller_ato_dur = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::seconds(100)),
            //     .gate_controller_ato_keyvalue_feed_cap = size_t{1} << 10,

            //     .gate_controller_blklst_component_sz = 16,
            //     .gate_controller_blklst_bloomfilter_cap = size_t{1} << 24,
            //     .gate_controller_blklst_bloomfilter_rehash_sz = 4,
            //     .gate_controller_blklst_bloomfilter_reliability_decay_factor = 8,
            //     .gate_controller_blklst_keyvalue_feed_cap = size_t{1} << 10,

            //     .latency_controller_component_sz = 16,
            //     .latency_controller_queue_cap = size_t{1} << 20,
            //     .latency_controller_unique_id_cap = size_t{1} << 20,
            //     .latency_controller_expiry_period = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::seconds(100)),
            //     .latency_controller_keyvalue_feed_cap = size_t{1} << 10,
            //     .latency_controller_has_exhaustion_control = true,

            //     .packet_assembler_component_sz = 16,
            //     .packet_assembler_map_cap = size_t{1} << 20,
            //     .packet_assembler_global_segment_cap = size_t{1} << 20,
            //     .packet_assembler_max_segment_per_stream = size_t{1} << 11,
            //     .packet_assembler_keyvalue_feed_cap = size_t{1} << 10,
            //     .packet_assembler_has_exhaustion_control = true,

            //     .inbound_container_component_sz = 16,
            //     .inbound_container_cap = size_t{1} << 20,
            //     .inbound_container_has_exhaustion_control = true,
            //     .inbound_container_has_react_pattern = true,
            //     .inbound_container_react_sz = size_t{1} << 10,
            //     .inbound_container_subscriber_cap = size_t{1} << 10,
            //     .inbound_container_react_latency = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::milliseconds(100)),
            //     .inbound_container_has_redistributor = true,
            //     .inbound_container_redistributor_distribution_queue_sz = size_t{1} << 20,
            //     .inbound_container_redistributor_waiting_queue_sz = size_t{1} << 20,
            //     .inbound_container_redistributor_concurrent_sz = size_t{1} << 20,
            //     .inbound_container_redistributor_unit_sz = size_t{1} << 6,

            //     .expiry_worker_count = 1,
            //     .expiry_worker_packet_assembler_vectorization_sz = size_t{1} << 10,
            //     .expiry_worker_consume_sz = size_t{1} << 10,
            //     .expiry_worker_busy_consume_sz = 1,

            //     .inbound_worker_count = 1,
            //     .inbound_worker_packet_assembler_vectorization_sz = size_t{1} << 10,
            //     .inbound_worker_inbound_gate_vectorization_sz = size_t{1} << 10,
            //     .inbound_worker_blacklist_gate_vectorization_sz = size_t{1} << 10,
            //     .inbound_worker_latency_controller_vectorization_sz = size_t{1} << 10,
            //     .inbound_worker_inbound_container_vectorization_sz = size_t{1} << 10,
            //     .inbound_worker_consume_sz = size_t{1} << 10,
            //     .inbound_worker_busy_consume_sz = 1,
            //     .inbound_redistributor_worker_suck_cap = size_t{1} << 10,
            //     .inbound_redistributor_worker_push_cap = size_t{1} << 10,
            //     .inbound_redistributor_worker_busy_threshold = 1,

            //     .mailbox_transmission_vectorization_sz = size_t{1} << 10,

            //     .outbound_rule = dg::network_kernel_mailbox_impl1_flash_streamx::get_empty_outbound_rule(),
            //     .infretry_device = retry_device,
            //     .base = std::move(sock)
            // });

            std::cout << "made socket ..." << std::endl;

            auto recv_worker = std::make_unique<RecvWorker>(sock.get());
            auto send_worker = std::make_unique<SendWorker>(sock.get());
            // auto hello_world_worker = std::make_unique<HelloWorldWorker>(sock.get());

            dg::network_concurrency::daemon_register(dg::network_concurrency::COMPUTING_DAEMON, std::move(send_worker));
            dg::network_concurrency::daemon_register(dg::network_concurrency::COMPUTING_DAEMON, std::move(recv_worker));
            // dg::network_concurrency::daemon_register(dg::network_concurrency::IO_DAEMON, std::move(hello_world_worker));

            while (true){
                print("sleeping...");
                std::this_thread::sleep_for(std::chrono::seconds(1));
            }
        }

        dg::network_concurrency::deinit();
    }

    for (size_t i = 0u; i < 10; ++i)
    {
        print("sleeping...");
        std::this_thread::sleep_for(std::chrono::seconds(1));
    }

    // dg::network_bytecode::run({}, {}, {}, {});
}