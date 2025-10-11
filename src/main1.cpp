#define DEBUG_MODE_FLAG true
#define STRONG_MEMORY_ORDERING_FLAG true

#include "network_bytecode.h"
#include "network_concurrency.h"
#include <string>
#include <iostream>
#include <thread>
#include <mutex>

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

int main()
{
    {
        dg::network_concurrency::init(dg::network_concurrency::Config{
            .computing_cpu_usage = 0.1,
            .io_cpu_usage = 0.1,
            .transportation_cpu_usage = 0.1,
            .heartbeat_cpu_usage = 0.1,
            .high_parallel_hyperthread_per_core = 10,
            .high_compute_hyperthread_per_core = 10,
            .uniform_affine_group = std::vector<int>{1, 2, 3, 4}
        });

        auto worker1 = std::make_unique<Worker>(1);
        auto worker2 = std::make_unique<Worker>(2);
        auto worker3 = std::make_unique<Worker>(3);

        size_t worker1_value = dg::network_concurrency::daemon_register(dg::network_concurrency::IO_DAEMON, std::move(worker1)).value();
        size_t worker2_value = dg::network_concurrency::daemon_register(dg::network_concurrency::IO_DAEMON, std::move(worker2)).value();
        size_t worker3_value = dg::network_concurrency::daemon_register(dg::network_concurrency::IO_DAEMON, std::move(worker3)).value();

        for (size_t i = 0u; i < 10; ++i)
        {
            print("sleeping...");
            std::this_thread::sleep_for(std::chrono::seconds(1));
        }

        print("stopping 1");
        dg::network_concurrency::daemon_deregister(worker1_value);

        for (size_t i = 0u; i < 10; ++i)
        {
            print("sleeping...");
            std::this_thread::sleep_for(std::chrono::seconds(1));
        }

        print("stopping 2");
        dg::network_concurrency::daemon_deregister(worker2_value);

        for (size_t i = 0u; i < 10; ++i)
        {
            print("sleeping...");
            std::this_thread::sleep_for(std::chrono::seconds(1));
        }

        print("stopping 3");
        dg::network_concurrency::daemon_deregister(worker3_value);

        for (size_t i = 0u; i < 10; ++i)
        {
            print("sleeping...");
            std::this_thread::sleep_for(std::chrono::seconds(1));
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