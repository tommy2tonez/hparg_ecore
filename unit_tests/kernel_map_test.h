#ifndef __KERNEL_MAP_TEST_H__
#define __KERNEL_MAP_TEST_H__

#include <algorithm>
#include <utility>
#include <random>
#include "../src/network_kernelmap_x_impl1.h"
#include "../src/network_kernelmap_x.h"
#include "../src/network_pointer.h"
#include "../src/network_fileio_unified_x.h"
#include <chrono>
#include <iostream>

namespace kernel_map_test
{
    using fsys_ptr_t = uintptr_t;

    class DiskIODevice: public virtual dg::network_kernelmap_x_impl1::interface::KernelDiskIODeviceInterface
    {
        public:

            auto read_binary(const std::filesystem::path& src, void * dst, size_t bsz) noexcept -> exception_t
            {
                std::cout << "reading binary > " << bsz << std::endl;
                auto rs = dg::network_fileio_chksum_x::dg_read_binary(src.c_str(), dst, bsz);
                std::cout << "read binary" << std::endl;

                return rs;
            }

            auto write_binary(const std::filesystem::path& dst, void * src, size_t bsz) noexcept -> exception_t
            {
                std::cout << "writing binary > " << bsz << std::endl;
                auto rs = dg::network_fileio_chksum_x::dg_write_binary(dst.c_str(), src, bsz);
                std::cout << "wrote binary" << std::endl;
            }
    };

    constexpr auto ceil(size_t arg, size_t width) -> size_t 
    {
        if (arg == 0u)
        {
            return arg;
        }

        return (((arg - 1u) / width) + 1u) * width;
    }

    auto split_segments(fsys_ptr_t first, fsys_ptr_t last,
                        size_t memregion_sz) -> std::vector<std::pair<fsys_ptr_t, size_t>>
    {
        uintptr_t first_uptr            = dg::pointer_cast<uintptr_t>(first);
        uintptr_t last_uptr             = dg::pointer_cast<uintptr_t>(last);

        uintptr_t upround_first_uptr    = ceil(first_uptr, memregion_sz);

        std::vector<std::pair<fsys_ptr_t, size_t>> result{};

        if (upround_first_uptr != first_uptr)
        {
            result.push_back({dg::pointer_cast<fsys_ptr_t>(first_uptr), upround_first_uptr - first_uptr});
        }

        size_t sz       = last_uptr - upround_first_uptr;
        size_t slot_sz  = sz / memregion_sz + static_cast<size_t>(sz % memregion_sz != 0u);

        for (size_t i = 0u; i < slot_sz; ++i)
        {
            uintptr_t round_first   = upround_first_uptr + i * memregion_sz;
            uintptr_t round_last    = std::min(last_uptr, static_cast<uintptr_t>(round_first + memregion_sz));

            result.push_back({dg::pointer_cast<fsys_ptr_t>(round_first), round_last - round_first});
        }

        return result;
    }

    void full_scan_flush_memory(fsys_ptr_t first, fsys_ptr_t last)
    {
        std::cout << "flushing memory" << std::endl;

        size_t sum              = 0u;
        uintptr_t first_uptr    = dg::pointer_cast<uintptr_t>(first);
        uintptr_t last_uptr     = dg::pointer_cast<uintptr_t>(last);
        size_t sz               = last_uptr - first_uptr; 

        for (size_t i = 0u; i < sz; ++i)
        {
            uintptr_t offset    = first_uptr + i;
            fsys_ptr_t ptr_addr = dg::pointer_cast<fsys_ptr_t>(offset);
            auto map_resource   = dg::network_exception_handler::nothrow_log(dg::network_kernelmap_x::map_safe(ptr_addr));
            void * ptr          = dg::network_kernelmap_x::get_host_ptr(map_resource);

            uint8_t u8_value;
            std::memcpy(&u8_value, ptr, sizeof(uint8_t));

            sum                 += u8_value;
        }

        std::cout << "full scan performed > " << sum << std::endl;
    } 

    void write_memory(fsys_ptr_t first, fsys_ptr_t last,
                      size_t memregion_sz,
                      const char * buf)
    {
        std::vector<std::pair<fsys_ptr_t, size_t>> segment_vec = split_segments(first, last, memregion_sz);
 
        std::cout << "writing memory, memregion_sz > " << memregion_sz << " segment_vec_sz > " << segment_vec.size() << std::endl;

        for (const auto& segment: segment_vec)
        {
            size_t offset       = dg::pointer_cast<uintptr_t>(segment.first) - dg::pointer_cast<uintptr_t>(first);
            const char * src    = std::next(buf, offset); 
            auto map_resource   = dg::network_exception_handler::nothrow_log(dg::network_kernelmap_x::map_safe(segment.first));
            void * map_ptr      = dg::network_kernelmap_x::get_host_ptr(map_resource);

            std::memcpy(map_ptr, src, segment.second);
        }
    }

    auto read_memory(fsys_ptr_t first, fsys_ptr_t last,
                     size_t memregion_sz,
                     char * buf)
    {
        std::cout << "reading memory" << std::endl;

        std::vector<std::pair<fsys_ptr_t, size_t>> segment_vec  = split_segments(first, last, memregion_sz);

        for (const auto& segment: segment_vec)
        {
            size_t offset       = dg::pointer_cast<uintptr_t>(segment.first) - dg::pointer_cast<uintptr_t>(first);
            char * dst          = std::next(buf, offset);
            auto map_resource   = dg::network_exception_handler::nothrow_log(dg::network_kernelmap_x::map_safe(segment.first));
            void * map_ptr      = dg::network_kernelmap_x::get_host_ptr(map_resource);

            std::memcpy(dst, map_ptr, segment.second);
        }
    }

    auto randomize_memory(size_t bsz) -> std::string
    {
        static auto random_device = std::bind(std::uniform_int_distribution<char>{}, std::mt19937_64{static_cast<uint32_t>(std::chrono::high_resolution_clock::now().time_since_epoch().count())});

        std::string result(bsz, 0);
        std::generate(result.begin(), result.end(), std::ref(random_device));

        return result;
    }

    void test_random_memory(fsys_ptr_t first, fsys_ptr_t last, size_t memregion_sz,
                            size_t mempiece_range,
                            size_t test_sz)
    {
        std::cout << "testing random memory" << std::endl;

        constexpr size_t READ_RANDOM_MEMORY     = 0u;
        constexpr size_t WRITE_RANDOM_MEMORY    = 1u; 
        constexpr size_t MEMORY_FLAG_SZ         = 2u; 

        size_t memory_sz    = dg::pointer_cast<uintptr_t>(last) - dg::pointer_cast<uintptr_t>(first);

        std::cout << "memory sz > " << memory_sz << std::endl;
 
        std::vector<char> mem_vec(memory_sz, 0);
        write_memory(first, last, memregion_sz, mem_vec.data());

        auto random_device  = std::bind(std::uniform_int_distribution<size_t>{}, std::mt19937_64{static_cast<uint32_t>(std::chrono::high_resolution_clock::now().time_since_epoch().count())});

        for (size_t i = 0u; i < test_sz; ++i)
        {
            size_t memory_flag      = random_device() % MEMORY_FLAG_SZ; 

            switch (memory_flag)
            {
                case WRITE_RANDOM_MEMORY:
                {
                    size_t memory_offset    = random_device() % memory_sz;
                    size_t max_sz           = memory_sz - memory_offset; 
                    size_t mempiece_sz      = std::min(random_device() % mempiece_range, max_sz);

                    fsys_ptr_t memory_first = dg::pointer_cast<fsys_ptr_t>(dg::pointer_cast<uintptr_t>(first) + memory_offset);
                    fsys_ptr_t memory_last  = dg::pointer_cast<fsys_ptr_t>(dg::pointer_cast<uintptr_t>(memory_first) + mempiece_sz);

                    std::string memory_data = randomize_memory(mempiece_sz);

                    std::memcpy(std::next(mem_vec.data(), memory_offset), memory_data.data(), memory_data.size());
                    write_memory(memory_first, memory_last, memregion_sz, memory_data.data());

                    break;
                }
                case READ_RANDOM_MEMORY:
                {
                    size_t memory_offset    = random_device() % memory_sz;
                    size_t max_sz           = memory_sz - memory_offset;
                    size_t mempiece_sz      = std::min(random_device() % mempiece_range, max_sz);

                    fsys_ptr_t memory_first = dg::pointer_cast<fsys_ptr_t>(dg::pointer_cast<uintptr_t>(first) + memory_offset);
                    fsys_ptr_t memory_last  = dg::pointer_cast<fsys_ptr_t>(dg::pointer_cast<uintptr_t>(memory_first) + mempiece_sz);

                    std::string memory_data = std::string(mempiece_sz, 0);

                    read_memory(memory_first, memory_last, memregion_sz, memory_data.data());

                    if (std::equal(std::next(mem_vec.data(), memory_offset), std::next(mem_vec.data(), memory_offset + mempiece_sz),
                                   memory_data.data(), std::next(memory_data.data(), memory_data.size())) == false)
                    {
                        std::cout << "failed random memory test" << std::endl;
                        std::abort();
                    }

                    break;
                }
                default:
                {
                    std::unreachable();
                }
            }
        }
    } 

    auto randomize_hex() -> char
    {
        static auto randomizer = std::bind(std::uniform_int_distribution<size_t>{}, std::mt19937_64{static_cast<uint32_t>(std::chrono::high_resolution_clock::now().time_since_epoch().count())}); 
        size_t idx = randomizer() % 16u;

        if (idx < 10)
        {
            return '0' + idx;
        }

        return 'a' + (idx - 10);
    }

    auto randomize_hex(size_t str_sz) -> std::string
    {
        std::string rs(str_sz, 0);
        std::generate(rs.begin(), rs.end(), []{return randomize_hex();});

        return rs;
    } 

    auto get_random_data_file_whose_folder(std::filesystem::path fsys_dir) -> std::filesystem::path
    {
        const size_t FILE_NAME_SZ   = 8u;
        std::string filename        = randomize_hex(FILE_NAME_SZ);

        return (fsys_dir / filename).replace_extension(".data");
    } 

    auto create_random_data_file_whose_folder(std::filesystem::path fsys_dir,
                                              size_t fsz) -> std::filesystem::path
    {
        std::filesystem::path major_path    = get_random_data_file_whose_folder(fsys_dir);
        std::filesystem::path file_path     = get_random_data_file_whose_folder(fsys_dir);
        std::filesystem::path file_path2    = get_random_data_file_whose_folder(fsys_dir);

        dg::network_exception_handler::throw_nolog(dg::network_fileio_chksum_x::dg_create_cbinary(major_path.c_str(), fsz));

        return major_path;
    } 

    auto get_random_first_pointer(size_t memregion_sz) -> fsys_ptr_t
    {
        const size_t FIRST_PTR_OFFSET_RANGE = 16u;
        static auto randomizer  = std::bind(std::uniform_int_distribution<size_t>{}, std::mt19937_64{static_cast<uint32_t>(std::chrono::high_resolution_clock::now().time_since_epoch().count())}); 

        size_t slot_offset      = randomizer() % FIRST_PTR_OFFSET_RANGE + 1u;
        size_t addr_offset      = slot_offset * memregion_sz;

        return dg::pointer_cast<fsys_ptr_t>(addr_offset);
    }

    auto next_pointer(fsys_ptr_t ptr, size_t offset) -> fsys_ptr_t
    {
        return dg::pointer_cast<fsys_ptr_t>(dg::pointer_cast<uintptr_t>(ptr) + offset);
    }

    auto discretize_region(fsys_ptr_t first, fsys_ptr_t last, size_t memregion_sz) -> std::vector<fsys_ptr_t>
    {
        std::vector<fsys_ptr_t> rs_vec{};

        while (first != last)
        {
            rs_vec.push_back(first);
            first = next_pointer(first, memregion_sz);
        }

        return rs_vec;
    }

    auto init_fsys(std::filesystem::path fsys_dir,
                   size_t memregion_sz,
                   size_t memregion_count,
                   double ram_to_disk_ratio,
                   size_t distribution_factor) -> std::pair<fsys_ptr_t, fsys_ptr_t>
    {
        try
        {
            bool rs                             = std::filesystem::create_directories(fsys_dir);

            if (!rs)
            {
                dg::network_exception::throw_exception(dg::network_exception::INVALID_ARGUMENT);
            }

            fsys_ptr_t first_ptr                = get_random_first_pointer(memregion_sz);
            fsys_ptr_t last_ptr                 = next_pointer(first_ptr, memregion_sz * memregion_count);

            std::cout << "first > " << first_ptr << " last > " << last_ptr << " memregion_sz > " << memregion_sz << std::endl;
 
            std::vector<fsys_ptr_t> region_vec  = discretize_region(first_ptr, last_ptr, memregion_sz);

            std::unordered_map<fsys_ptr_t, std::filesystem::path> virtual_mem_map{}; 

            for (fsys_ptr_t region: region_vec)
            {
                virtual_mem_map[region] = create_random_data_file_whose_folder(fsys_dir, memregion_sz);
            }
            
            std::cout << "created random files" << std::endl;

            dg::network_kernelmap_x::init(virtual_mem_map,
                                          std::make_unique<DiskIODevice>(),
                                          memregion_sz,
                                          ram_to_disk_ratio,
                                          distribution_factor);
            
            std::cout << "inited successfully" << std::endl;

            return {first_ptr, last_ptr};
        }
        catch (...)
        {
            std::filesystem::remove_all(fsys_dir);
            throw;
        }
    }

    void deinit_fsys(std::filesystem::path fsys_dir)
    {
        dg::network_kernelmap_x::deinit();
        std::filesystem::remove_all(fsys_dir);
    }

    void test_unit(std::filesystem::path fsys_dir,
                   size_t memregion_sz,
                   size_t memregion_count,
                   double ram_to_disk_ratio,
                   size_t distribution_factor)
    {
        const size_t FULL_FLUSH_CHANCE              = size_t{1} << 10;
        const size_t RANDOM_MEMORY_TEST_SZ_RANGE    = size_t{1} << 10;
        const size_t MEMPIECE_RANGE                 = memregion_sz * 10;
        const size_t TEST_SZ                        = size_t{1} << 8;

        auto sub_dir        = fsys_dir / randomize_hex(8u);
        auto random_device  = std::bind(std::uniform_int_distribution<size_t>{}, std::mt19937_64{static_cast<uint32_t>(std::chrono::high_resolution_clock::now().time_since_epoch().count())});
        auto [first, last]  = init_fsys(sub_dir, memregion_sz, memregion_count, ram_to_disk_ratio, distribution_factor);

        for (size_t i = 0u; i < TEST_SZ; ++i)
        {
            size_t value = random_device() % FULL_FLUSH_CHANCE;
            
            if (value == 0u)
            {
                full_scan_flush_memory(first, last);
            }
            else
            {
                test_random_memory(first, last, memregion_sz,
                                   MEMPIECE_RANGE,
                                   random_device() % RANDOM_MEMORY_TEST_SZ_RANGE);
            }
        }

        deinit_fsys(sub_dir);
    }

    auto get_random_memregion_size() -> size_t
    {
        const size_t MEMREGION_POW2_RANGE   = 4u;
        static auto randomizer              = std::bind(std::uniform_int_distribution<size_t>{}, std::mt19937_64{static_cast<uint32_t>(std::chrono::high_resolution_clock::now().time_since_epoch().count())}); 

        size_t random_pow2 = randomizer() % MEMREGION_POW2_RANGE; 

        return size_t{1} << 12;
    }

    auto get_random_memregion_count() -> size_t
    {
        const size_t MEMREGION_COUNT_RANGE  = size_t{1} << 10;
        static auto randomizer              = std::bind(std::uniform_int_distribution<size_t>{}, std::mt19937_64{static_cast<uint32_t>(std::chrono::high_resolution_clock::now().time_since_epoch().count())}); 

        return randomizer() % MEMREGION_COUNT_RANGE;
    }

    auto get_random_ram_to_disk_ratio() -> double
    {
        const size_t PERCENTAGE_RANGE       = size_t{1} << 10;
        static auto randomizer              = std::bind(std::uniform_int_distribution<size_t>{}, std::mt19937_64{static_cast<uint32_t>(std::chrono::high_resolution_clock::now().time_since_epoch().count())}); 

        return static_cast<double>(randomizer() % PERCENTAGE_RANGE) / static_cast<double>(PERCENTAGE_RANGE);
    }

    auto get_random_distribution_factor() -> size_t
    {
        const size_t DISTRIBUTION_FACTOR_RANGE  = size_t{1} << 10;
        static auto randomizer                  = std::bind(std::uniform_int_distribution<size_t>{}, std::mt19937_64{static_cast<uint32_t>(std::chrono::high_resolution_clock::now().time_since_epoch().count())}); 

        return (randomizer() % DISTRIBUTION_FACTOR_RANGE) + 1u;
    }

    void kernel_map_base_test(std::filesystem::path test_dir)
    {
        std::cout << "<initializing_kernel_map_base_test>" << std::endl; 

        const size_t TEST_SZ    = size_t{1} << 10;
        const size_t CHECK_SZ   = size_t{1} << 6; 

        for (size_t i = 0u; i < TEST_SZ; ++i)
        {
            test_unit(test_dir,
                      get_random_memregion_size(),
                      get_random_memregion_count(),
                      get_random_ram_to_disk_ratio(),
                      get_random_distribution_factor());

            if (i % CHECK_SZ == 0u)
            {
                std::cout << "testing > " << i << "/" << TEST_SZ << std::endl; 
            }
        }

        std::cout << "<kernel_map_base_test_completed>" << std::endl;
    }

    void run()
    {
        std::string folder_path = std::filesystem::temp_directory_path() / randomize_hex(8u);

        bool rs = std::filesystem::create_directories(folder_path);

        if (!rs)
        {
            throw std::runtime_error("unable to create directory");
        }

        try
        {
            kernel_map_base_test(folder_path);
        }
        catch (...)
        {
            std::filesystem::remove_all(folder_path);
            throw;
        }

        std::filesystem::remove_all(folder_path);
    }
} 

#endif