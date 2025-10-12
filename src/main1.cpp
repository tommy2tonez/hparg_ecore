#define DEBUG_MODE_FLAG true
#define STRONG_MEMORY_ORDERING_FLAG true

#include "network_bytecode.h"
#include "network_concurrency.h"
#include <string>
#include <iostream>
#include <thread>
#include <mutex>
#include "network_kernel_mailbox_impl1.h"

static inline std::mutex print_mtx;

void print(const std::string& msg)
{
    std::lock_guard<std::mutex> lck_grd(print_mtx);
    std::cout << msg << std::endl;
}

class Worker: public virtual dg::network_concurrency::WorkerInterface
{
    private:

        size_t id;

    public:

        Worker(size_t id): id(id){}

        bool run_one_epoch() noexcept
        {
            print(std::to_string(id) + " <worker> has been here");
            std::this_thread::sleep_for(std::chrono::seconds(1));            
            return false;
        }
};

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
            dg::network_kernel_mailbox_impl1::model::Address self_addr = {};
            self_addr.ip = dg::network_kernel_mailbox_impl1::model::IP{.ip = dg::network_kernel_mailbox_impl1::utility::ipv4_std_formatted_str_to_compact("127.0.0.1").value()};
            self_addr.port = 5000;

            for (size_t i = 0u; i < 10; ++i){
                dg::network_kernel_mailbox_impl1::model::MailBoxArgument arg = {};
                arg.to = self_addr;
                arg.content = "Hello World! " + std::to_string(i);
                exception_t err;

                sock->send(std::make_move_iterator(&arg), 1u, &err);
            }

            while (true){
                dg::string recv_buf;
                size_t recv_buf_sz;
                
                sock->recv(&recv_buf, recv_buf_sz, 1u);

                if (recv_buf_sz != 0u){
                    std::cout << recv_buf << "<recv>" << std::endl;
                }

                print("hello_world_worker sleeping...");
                std::this_thread::sleep_for(std::chrono::seconds(1));
            }
        }
};

int main()
{
    {
        dg::network_concurrency::init(dg::network_concurrency::Config{
            .computing_cpu_usage = 0.1,
            .io_cpu_usage = 0.1,
            .transportation_cpu_usage = 0.1,
            .heartbeat_cpu_usage = 0.1,
            .high_parallel_hyperthread_per_core = 20,
            .high_compute_hyperthread_per_core = 20,
            .uniform_affine_group = std::vector<int>{1, 2, 3, 4}
        });

        {
            dg::network_stack_allocation::init(10000, size_t{1} << 25);

            auto [retry_device, destructor] = dg::network_concurrency_infretry_x::get_infretry_machine(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::milliseconds(10))); 

            std::cout << "making socket ..." << std::endl;

            auto sock = dg::network_kernel_mailbox_impl1::spawn(dg::network_kernel_mailbox_impl1::Config{
                .num_kernel_inbound_worker = 1,
                .num_process_inbound_worker = 1,
                .num_outbound_worker = 1,
                .num_kernel_rescue_worker = 1,
                .num_retry_worker = 1,

                .socket_concurrency_sz = 1,
                .sin_fam = AF_INET,
                .comm = SOCK_DGRAM,
                .protocol = 0,
                .host_ip = {},
                .host_port = 5000,
                .has_exhaustion_control = false,

                .retransmission_delay = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::seconds(1)),
                .retransmission_concurrency_sz = 1,
                .retransmission_queue_cap = 1024,
                .retransmission_packet_cap = 10,
                .retransmission_idhashset_cap = 1024,
                .retransmission_has_react_pattern = false,
                .retransmission_react_sz = 1,
                .retransmission_react_queue_cap = 1024,
                .retransmission_react_time = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::milliseconds(10)),

                .inbound_buffer_concurrency_sz = 1,
                .inbound_buffer_container_cap = 1024,
                .inbound_buffer_has_react_pattern = true,
                .inbound_buffer_react_sz = 1,
                .inbound_buffer_react_queue_cap = 1024,
                .inbound_buffer_react_time = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::milliseconds(10)),

                .inbound_packet_concurrency_sz = 1,
                .inbound_packet_container_cap = 1024,
                .inbound_packet_has_react_pattern = true,
                .inbound_packet_react_sz = 1,
                .inbound_packet_react_queue_cap = 1024,
                .inbound_packet_react_time = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::seconds(1)),

                .inbound_idhashset_concurrency_sz = 1,
                .inbound_idhashset_cap = 1024,

                .worker_inbound_buffer_accumulation_sz = 1,
                .worker_inbound_packet_consumption_cap = 1000,
                .worker_inbound_packet_busy_threshold_sz = 1,
                .worker_rescue_packet_sz_per_transmit = 100,
                .worker_kernel_rescue_dispatch_threshold = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::seconds(1)),
                .worker_retransmission_consumption_cap = 1000,
                .worker_retransmission_busy_threshold_sz = 1,
                .worker_outbound_packet_consumption_cap = 1000,
                .worker_outbound_packet_busy_threshold_sz = 1,

                .mailbox_inbound_cap = 1000,
                .mailbox_outbound_cap = 1000,
                .traffic_reset_duration = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::seconds(1)),

                .outbound_packet_concurrency_sz = 1,
                .outbound_ack_packet_container_cap = 1024,
                .outbound_request_packet_container_cap = 1024,
                .outbound_krescue_packet_container_cap = 1024,
                .outbound_transmit_frequency = uint32_t{1} << 20,
                .outbound_packet_has_react_pattern = true,
                .outbound_packet_react_sz = 1,
                .outbound_packet_react_queue_cap = 1024,
                .outbound_packet_react_time = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::milliseconds(10)),

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
                .retry_device = std::move(retry_device)
            });

            std::cout << "made socket ..." << std::endl;

            auto helloworld_worker = std::make_unique<HelloWorldWorker>(sock.get());
            dg::network_concurrency::daemon_register(dg::network_concurrency::IO_DAEMON, std::move(helloworld_worker));

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