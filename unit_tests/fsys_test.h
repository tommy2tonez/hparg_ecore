#include "../src/network_fileio.h"
#include "iostream"
#include <random>
#include "../src/stdx.h"
#include <filesystem>
#include <unordered_map>
#include <vector>
#include <filesystem>
#include <algorithm>
#include <utility>
#include <random>
#include <functional>
#include <memory>
#include <type_traits>
#include "../src/network_fileio_chksum_x.h"
#include "../src/network_fileio_unified_x.h"

namespace fileio_test{

    //we dont have a lot to write
    //this is probably the best of filesystems that we could come up with
    //we focused on three things
    //the OS support, OS fault and OS fault reduction

    //OS support            == fileio_base
    //OS fault              == fileio_chksum
    //OS fault_reduction    == fileio_unified

    //we dont take in engineered coordinated attacks, there could be thousands of ways to do so
    //we take in statistical chances, reduce the statistical chances, bring it to RAM fault rate 
    //we are doing 10MB/read, store 8GB of cache on the temporal map, and we hope for the best

    //alright, this test should be under 3000 lines of code
    //we'll write the tests today

    template <class T, std::enable_if_t<std::is_unsigned_v<T>, bool> = true>
    constexpr auto is_pow2(T val) noexcept -> bool{

        return val != 0u && (val & (val - 1)) == 0u;
    }

    template <class T, std::enable_if_t<std::is_unsigned_v<T>, bool> = true>
    constexpr auto ulog2(T val) noexcept -> T{

        return static_cast<T>(sizeof(T) * CHAR_BIT - 1u) - static_cast<T>(std::countl_zero(val));
    }

    template <class T, std::enable_if_t<std::is_unsigned_v<T>, bool> = true>
    static constexpr auto ceil2(T val) noexcept -> T{

        if (val < 2u) [[unlikely]]{
            return 1u;
        } else [[likely]]{
            T uplog_value = ulog2(static_cast<T>(val - 1u)) + 1u;
            return T{1u} << uplog_value;
        }
    }

    auto make_buf(size_t alignment, size_t buf_sz) -> std::shared_ptr<char[]>{

        void * buf = std::aligned_alloc(alignment, buf_sz);

        if (buf == nullptr){
            throw std::bad_alloc();
        }

        auto destructor = [](char * ptr) noexcept{
            std::free(ptr);
        };

        return std::unique_ptr<char[], decltype(destructor)>(static_cast<char *>(buf), destructor);
    } 

    auto randomize_buf(size_t alignment, size_t buf_sz) -> std::shared_ptr<char[]>{

        auto randomizer = std::bind(std::uniform_int_distribution<char>{}, std::mt19937{static_cast<size_t>(std::chrono::high_resolution_clock::now().time_since_epoch().count())});
        auto buf        = make_buf(alignment, buf_sz);
        std::generate(buf.get(), std::next(buf.get(), buf_sz), randomizer);

        return buf;
    }

    auto randomize_str(size_t buf_sz) -> std::string{

        auto randomizer = std::bind(std::uniform_int_distribution<char>{}, std::mt19937{static_cast<size_t>(std::chrono::high_resolution_clock::now().time_since_epoch().count())});
        auto buf        = std::string(buf_sz, ' ');
        std::generate(buf.begin(), std::next(buf.begin(), buf_sz), randomizer);

        return buf;
    }

    auto randomize_filename(size_t file_sz) -> std::string{

        auto randomizer = std::bind(std::uniform_int_distribution<char>('a', 'z'), std::mt19937{static_cast<size_t>(std::chrono::high_resolution_clock::now().time_since_epoch().count())});
        auto buf        = std::string(file_sz, ' ');
        std::generate(buf.begin(), std::next(buf.begin(), file_sz), randomizer);

        return buf;
    } 

    auto floor2(size_t val) -> size_t{

        return size_t{1} << ulog2(val);
    }

    auto align(char * buf, uintptr_t alignment_sz) noexcept -> char *{

        uintptr_t fwd_sz            = alignment_sz - 1u;
        uintptr_t neg_sz            = ~fwd_sz;
        uintptr_t buf_arithmetic    = reinterpret_cast<uintptr_t>(buf);
        uintptr_t fwd_buf           = (buf_arithmetic + fwd_sz) & neg_sz; 

        return reinterpret_cast<char *>(fwd_buf);        
    }

    //alright, I got headache for this
    auto get_replicated_path(std::filesystem::path p, size_t path_sz) -> std::vector<std::string>{

        std::vector<std::string> rs{};

        for (size_t i = 0u; i < path_sz; ++i){
            auto new_fpath  = p;
            auto ext        = new_fpath.extension();

            rs.push_back(new_fpath.replace_extension("").replace_filename(new_fpath.filename().native() + static_cast<char>(char('0') + char(i))).replace_extension(ext).native());    
        }

        return rs;
    } 

    auto create_tmp_file(std::filesystem::path p){

        const size_t FILE_SZ = 16u;

        return p / (randomize_filename(FILE_SZ) + ".data");
    }

    void fileio_base_test(std::filesystem::path test_dir){

        //G
        //dont be funny, this virtual construct is for me to write this thing EVERYDAY, I dont mind, it might take me years
        //probably 1 year, 2 years, 4 years, 8 years, 16 years
        //the only thing that makes us feel alive is differences, why should I ever want to disrupt that

        std::cout << "<initializing_fileio_base_test>" << std::endl;

        std::unordered_map<std::filesystem::path, std::string> filedata_map{};
        std::vector<std::filesystem::path> past_file_vec{};
        std::vector<std::filesystem::path> current_file_vec{};

        const size_t CLEAR_INTERVAL_SZ                              = size_t{1} << 12;
        const size_t OPERATION_SZ                                   = size_t{1} << 20;
        const size_t TENPERCENT_TEST_SZ                             = OPERATION_SZ / 10u;
        const size_t NEW_BINARY_RANGE                               = 16384u; 
        const size_t POW2_RANGE                                     = 16u;

        const uint8_t OPS_CODE_CHECK_FSZ_EXIST                      = 0u;
        const uint8_t OPS_CODE_CHECK_CURRENT_FILE_EXISTS            = 1u;
        const uint8_t OPS_CODE_CHECK_DELETED_FILE_EXISTS            = 2u;
        const uint8_t OPS_CODE_CREATE_BINARY_DUP_FILE               = 3u;
        const uint8_t OPS_CODE_CREATE_BINARY_NEW_FILE               = 4u;

        const uint8_t OPS_CODE_READ_BINARY_DIRECT_VALID             = 5u;
        const uint8_t OPS_CODE_READ_BINARY_DIRECT_UNALIGNED_FSZ     = 6u;
        const uint8_t OPS_CODE_READ_BINARY_DIRECT_UNALIGNED_DST     = 7u;
        const uint8_t OPS_CODE_READ_BINARY_DIRECT_NOEXIST           = 8u;

        const uint8_t OPS_CODE_REMOVE_FILE_BINARY_EXIST             = 9u;
        const uint8_t OPS_CODE_REMOVE_FILE_BINARY_NOEXIST           = 10u;

        const uint8_t OPS_CODE_READ_BINARY_INDIRECT_VALID           = 11u;
        const uint8_t OPS_CODE_READ_BINARY_INDIRECT_NOEXIST         = 12u;
        const uint8_t OPS_CODE_READ_BINARY_INDIRECT_UNALIGNED       = 13u;

        const uint8_t OPS_CODE_READ_BINARY_VALID                    = 14u;
        const uint8_t OPS_CODE_READ_BINARY_NOEXIST                  = 15u;
        const uint8_t OPS_CODE_READ_BINARY_UNALIGNED                = 16u;
        const uint8_t OPS_CODE_READ_BINARY_ALIGNED                  = 17u;

        const uint8_t OPS_CODE_WRITE_BINARY_DIRECT_VALID            = 18u;
        const uint8_t OPS_CODE_WRITE_BINARY_DIRECT_UNALIGNED_FSZ    = 19u;
        const uint8_t OPS_CODE_WRITE_BINARY_DIRECT_UNALIGNED_DST    = 20u;
        const uint8_t OPS_CODE_WRITE_BINARY_DIRECT_NOEXIST          = 21u;

        const uint8_t OPS_CODE_WRITE_BINARY_INDIRECT_VALID          = 22u;
        const uint8_t OPS_CODE_WRITE_BINARY_INDIRECT_NOEXIST        = 23u;
        const uint8_t OPS_CODE_WRITE_BINARY_INDIRECT_UNALIGNED      = 24u;

        const uint8_t OPS_CODE_WRITE_BINARY_VALID                   = 25u;
        const uint8_t OPS_CODE_WRITE_BINARY_NOEXIST                 = 26u;
        const uint8_t OPS_CODE_WRITE_BINARY_UNALIGNED               = 27u;
        const uint8_t OPS_CODE_WRITE_BINARY_ALIGNED                 = 28u;

        const uint8_t OPS_CODE_CREATE_CBINARY_DUP_FILE              = 29u;
        const uint8_t OPS_CODE_CREATE_CBINARY_NEW_FILE              = 30u;

        const uint8_t OPS_CODE_CMP                                  = 31u;
        const uint8_t OPS_CODE_CHECK_FSZ_NOEXIST                    = 32u;

        auto ops_code_randomizer                                    = std::bind(std::uniform_int_distribution<uint8_t>(0u, 32u), std::mt19937{static_cast<size_t>(std::chrono::high_resolution_clock::now().time_since_epoch().count())});
        auto naive_sz_randomizer                                    = std::bind(std::uniform_int_distribution<size_t>{}, std::mt19937{static_cast<size_t>(std::chrono::high_resolution_clock::now().time_since_epoch().count())});

        auto sz_randomizer = [&]{
            size_t coin_flip = naive_sz_randomizer() % 2;

            if (coin_flip){
                return size_t{1} << (naive_sz_randomizer() % POW2_RANGE);
            }  else{
                return naive_sz_randomizer();
            }
        };

        for (size_t i = 0u; i < OPERATION_SZ; ++i){
            uint8_t ops_code = ops_code_randomizer();

            switch (ops_code){
                case OPS_CODE_CHECK_FSZ_EXIST:
                {
                    if (current_file_vec.empty()){
                        break;
                    }

                    size_t idx                              = sz_randomizer() % current_file_vec.size();
                    std::filesystem::path fp                = current_file_vec[idx];
                    std::expected<size_t, exception_t> fsz  = dg::network_fileio::dg_file_size(fp.c_str());

                    if (!fsz.has_value()){
                        std::cout << "mayday, check_fsz unexpected dg_file_size error" << std::endl; 
                        std::abort();
                    }

                    if (fsz.value() != filedata_map.find(fp)->second.size()){
                        std::cout << "mayday, check_fsz mismatched file_size" << "<>" << fsz.value() << "/" << filedata_map.find(fp)->second.size() << std::endl;
                        std::abort();
                    }

                    break;
                }
                case OPS_CODE_CHECK_CURRENT_FILE_EXISTS:
                {
                    if (current_file_vec.empty()){
                        break;
                    }

                    size_t idx                              = sz_randomizer() % current_file_vec.size();
                    std::filesystem::path fp                = current_file_vec[idx];
                    std::expected<bool, exception_t> status = dg::network_fileio::dg_file_exists(fp.c_str());

                    if (!status.has_value()){
                        std::cout << "mayday, check_current_file_exists unexpected dg_file_exists error" << std::endl;
                        std::abort();
                    }

                    if (!status.value()){
                        std::cout << "mayday, check_current_file_exists file not found" << std::endl;
                        std::abort();
                    }

                    break;
                }
                case OPS_CODE_CHECK_DELETED_FILE_EXISTS:
                {
                    if (past_file_vec.empty()){
                        break;
                    }

                    size_t idx                              = sz_randomizer() % past_file_vec.size();
                    std::filesystem::path fp                = past_file_vec[idx];
                    std::expected<bool, exception_t> status = dg::network_fileio::dg_file_exists(fp.c_str());

                    if (!status.has_value()){
                        std::cout << "mayday, check_deleted_file_exists unexpected dg_file_exists error" << std::endl;
                        std::abort();
                    }

                    if (status.value()){
                        std::cout << "mayday, check_deleted_file_exists file found" << std::endl;
                        std::abort();
                    }

                    break;
                }
                case OPS_CODE_CREATE_BINARY_DUP_FILE:
                {
                    if (current_file_vec.empty()){
                        break;
                    }

                    size_t idx                  = sz_randomizer() % current_file_vec.size();
                    std::filesystem::path fp    = current_file_vec[idx];
                    size_t fsz                  = sz_randomizer() % NEW_BINARY_RANGE;
                    exception_t err             = dg::network_fileio::dg_create_binary(fp.c_str(), fsz);

                    if (err != dg::network_exception::RUNTIME_FILEIO_ERROR){
                        std::cout << "mayday, create_binary_dup_file yields unexpected error code" << std::endl;
                        std::abort();
                    }

                    break;
                }
                case OPS_CODE_CREATE_BINARY_NEW_FILE:
                {
                    std::filesystem::path fp = create_tmp_file(test_dir);

                    if (std::find(current_file_vec.begin(), current_file_vec.end(), fp) != current_file_vec.end()){
                        break;
                    }

                    if (std::find(past_file_vec.begin(), past_file_vec.end(), fp) != past_file_vec.end()){
                        break;
                    }

                    size_t fsz      = sz_randomizer() % NEW_BINARY_RANGE;
                    exception_t err = dg::network_fileio::dg_create_binary(fp.c_str(), fsz);

                    if (dg::network_exception::is_failed(err)){
                        std::cout << "mayday, create_binary_new_file unexpected dg_create_binary error" << std::endl;
                        std::abort();
                    }

                    filedata_map[fp]    = std::string(fsz, 0);
                    exception_t rerr    = dg::network_fileio::dg_read_binary_indirect(fp.c_str(), filedata_map[fp].data(), fsz);

                    if (dg::network_exception::is_failed(err)){
                        std::cout << "mayday, create_binary_new_file unhandled_exception" << std::endl;
                        std::abort();
                    }

                    current_file_vec.push_back(fp);
                    break;
                }
                case OPS_CODE_READ_BINARY_DIRECT_VALID:
                {
                    if (current_file_vec.empty()){
                        break;
                    }

                    size_t idx                  = sz_randomizer() % current_file_vec.size();
                    std::filesystem::path fp    = current_file_vec[idx];
                    size_t fsz                  = filedata_map[fp].size();

                    if (!is_pow2(fsz) || fsz < dg::network_fileio::DG_LEAST_DIRECTIO_BLK_SZ){
                        break;
                    }

                    char * ptr                  = static_cast<char *>(std::aligned_alloc(dg::network_fileio::DG_LEAST_DIRECTIO_BLK_SZ, fsz));

                    if (!ptr){
                        std::cout << "mayday, read_binary_direct_valid unexpected aligned_alloc error" << std::endl;
                        std::abort();
                    }

                    exception_t err = dg::network_fileio::dg_read_binary_direct(fp.c_str(), ptr, fsz);

                    if (dg::network_exception::is_failed(err)){
                        std::cout << "mayday, read_binary_direct_valid unexpected dg_read_binary_direct error" << std::endl;
                        std::abort();
                    }

                    if (std::memcmp(filedata_map[fp].data(), ptr, fsz) != 0){
                        std::cout << fsz << "<current_fsz>" << dg::network_fileio::dg_file_size(fp.c_str()).value() << "<expecting_fsz>" << std::endl;
                        std::cout << "mayday, read_binary_direct_valid not equal" << std::endl;
                        std::abort();
                    }

                    std::free(ptr);
                    break;
                }
                case OPS_CODE_READ_BINARY_DIRECT_UNALIGNED_FSZ:
                {
                    if (current_file_vec.empty()){
                        break;
                    }

                    size_t idx                  = sz_randomizer() % current_file_vec.size();
                    std::filesystem::path fp    = current_file_vec[idx];
                    size_t fsz                  = filedata_map[fp].size();

                    if (fsz % dg::network_fileio::DG_LEAST_DIRECTIO_BLK_SZ == 0u){
                        break;
                    }

                    char * ptr                  = static_cast<char *>(std::aligned_alloc(dg::network_fileio::DG_LEAST_DIRECTIO_BLK_SZ, fsz));

                    if (!ptr){
                        std::cout << "mayday, read_binary_direct_unaligned_fsz unexpected aligned_alloc error" << std::endl;
                        std::abort();
                    }

                    exception_t err = dg::network_fileio::dg_read_binary_direct(fp.c_str(), ptr, fsz);

                    if (err != dg::network_exception::BAD_ALIGNMENT){
                        std::cout << "mayday, read_binary_direct_unaligned_fsz unexpected error" << "<>" << static_cast<size_t>(err) << "<err>" << "<>" << fsz << "<fsz>" << std::endl;
                        std::abort();
                    }

                    std::free(ptr);
                    break;
                }
                case OPS_CODE_READ_BINARY_DIRECT_UNALIGNED_DST:
                {
                    if (current_file_vec.empty()){
                        break;
                    }

                    size_t idx                  = sz_randomizer() % current_file_vec.size();
                    std::filesystem::path fp    = current_file_vec[idx];
                    size_t fsz                  = filedata_map[fp].size();

                    if (!is_pow2(fsz) || fsz < dg::network_fileio::DG_LEAST_DIRECTIO_BLK_SZ){
                        break;
                    }

                    char * ptr                  = static_cast<char *>(std::malloc(fsz + dg::network_fileio::DG_LEAST_DIRECTIO_BLK_SZ));

                    if (!ptr){
                        std::cout << "mayday, read_binary_direct_unaligned_dst unexpected malloc error" << std::endl;
                        std::abort();
                    }

                    char * unaligned_ptr        = std::next(align(ptr, dg::network_fileio::DG_LEAST_DIRECTIO_BLK_SZ));
                    exception_t err             = dg::network_fileio::dg_read_binary_direct(fp.c_str(), unaligned_ptr, fsz);

                    if (err != dg::network_exception::BAD_ALIGNMENT){
                        std::cout << "mayday, read_binary_direct_unaligned_dst unexpected dg_read_binary_direct error" << std::endl;
                        std::abort();
                    }

                    std::free(ptr);
                    break;
                }
                case OPS_CODE_READ_BINARY_DIRECT_NOEXIST:
                {
                    if (past_file_vec.empty()){
                        break;
                    }

                    size_t idx                  = sz_randomizer() % past_file_vec.size();
                    std::filesystem::path fp    = past_file_vec[idx];
                    auto buf                    = std::make_unique<char[]>(NEW_BINARY_RANGE);
                    exception_t err             = dg::network_fileio::dg_read_binary_direct(fp.c_str(), buf.get(), NEW_BINARY_RANGE);

                    if (dg::network_exception::is_success(err)){
                        std::cout << "mayday, read_binary_direct_noexist unexpected success" << std::endl;
                        std::abort();
                    }

                    break;
                }
                case OPS_CODE_REMOVE_FILE_BINARY_EXIST:
                {
                    if (current_file_vec.empty()){
                        break;
                    }

                    size_t idx                  = sz_randomizer() % current_file_vec.size();
                    std::filesystem::path fp    = current_file_vec[idx];
                    exception_t err             = dg::network_fileio::dg_remove(fp.c_str());

                    if (dg::network_exception::is_failed(err)){
                        std::cout << "mayday, remove_file_binary_exist unexpected error" << std::endl;
                        std::abort();
                    }

                    filedata_map.erase(fp);
                    past_file_vec.push_back(fp);
                    current_file_vec.erase(std::find(current_file_vec.begin(), current_file_vec.end(), fp));

                    break;
                }
                case OPS_CODE_REMOVE_FILE_BINARY_NOEXIST:
                {
                    if (past_file_vec.empty()){
                        break;
                    }

                    size_t idx                  = sz_randomizer() % past_file_vec.size();
                    std::filesystem::path fp    = past_file_vec[idx];
                    exception_t err             = dg::network_fileio::dg_remove(fp.c_str());

                    if (dg::network_exception::is_success(err)){
                        std::cout << "mayday, remove_file_binary_noexist unexpected success" << std::endl;
                    }

                    break;
                }
                case OPS_CODE_READ_BINARY_INDIRECT_VALID:
                {
                    break;
                }
                case OPS_CODE_READ_BINARY_INDIRECT_NOEXIST:
                {
                    if (past_file_vec.empty()){
                        break;
                    }

                    size_t idx                  = sz_randomizer() % past_file_vec.size();
                    std::filesystem::path fp    = past_file_vec[idx];
                    auto buf                    = make_buf(1u, NEW_BINARY_RANGE);
                    exception_t err             = dg::network_fileio::dg_read_binary_indirect(fp.c_str(), buf.get(), NEW_BINARY_RANGE);

                    if (dg::network_exception::is_success(err)){
                        std::cout << "mayday, read_binary_noexist unexpected success" << std::endl;
                        std::abort();
                    }
                    
                    break;
                }
                case OPS_CODE_READ_BINARY_INDIRECT_UNALIGNED:
                {
                    if (current_file_vec.empty()){
                        break;
                    }

                    size_t idx                                  = sz_randomizer() % current_file_vec.size();
                    std::filesystem::path fp                    = current_file_vec[idx];
                    size_t fsz                                  = filedata_map[fp].size();

                    if (is_pow2(fsz)){
                        break;
                    }

                    std::expected<size_t, exception_t> real_fsz = dg::network_fileio::dg_file_size(fp.c_str());

                    if (!real_fsz.has_value()){
                        std::cout << "mayday, read_binary_indirect_unaligned unexpected file_size error" << std::endl;
                        std::abort();
                    }

                    if (fsz != real_fsz.value()){
                        std::cout << "mayday, read_binary_indirect_unaligned size mismatched" << std::endl;
                        std::abort();
                    }

                    auto buf        = make_buf(1u, fsz);
                    exception_t err = dg::network_fileio::dg_read_binary_indirect(fp.c_str(), buf.get(), fsz);

                    if (dg::network_exception::is_failed(err)){
                        std::cout << "mayday, read_binary_unaligned unexpected dg_read_binary error" << std::endl;
                        std::abort();
                    }

                    if (std::memcmp(buf.get(), filedata_map[fp].data(), fsz) != 0){
                        std::cout << "mayday, read_binary_unaligned failed cmp" << std::endl;
                        std::abort();
                    }

                    break;
                }
                case OPS_CODE_READ_BINARY_VALID:
                {
                    break;
                }
                case OPS_CODE_READ_BINARY_NOEXIST:
                {
                    if (past_file_vec.empty()){
                        break;
                    }

                    size_t idx                  = sz_randomizer() % past_file_vec.size();
                    std::filesystem::path fp    = past_file_vec[idx];
                    auto buf                    = make_buf(1u, NEW_BINARY_RANGE);
                    exception_t err             = dg::network_fileio::dg_read_binary(fp.c_str(), buf.get(), NEW_BINARY_RANGE);

                    if (dg::network_exception::is_success(err)){
                        std::cout << "mayday, read_binary_noexist unexpected success" << std::endl;
                        std::abort();
                    }
                    
                    break;
                }
                case OPS_CODE_READ_BINARY_UNALIGNED:
                {
                    if (current_file_vec.empty()){
                        break;
                    }

                    size_t idx                                  = sz_randomizer() % current_file_vec.size();
                    std::filesystem::path fp                    = current_file_vec[idx];
                    size_t fsz                                  = filedata_map[fp].size();

                    if (is_pow2(fsz)){
                        break;
                    }

                    std::expected<size_t, exception_t> real_fsz = dg::network_fileio::dg_file_size(fp.c_str());

                    if (!real_fsz.has_value()){
                        std::cout << "mayday, read_binary_unaligned unexpected file_size error" << std::endl;
                        std::abort();
                    }

                    if (fsz != real_fsz.value()){
                        std::cout << "mayday, read_binary_unaligned size mismatched" << "<>" << fsz << "/" << real_fsz.value() <<  std::endl;
                        std::abort();
                    }

                    auto buf        = make_buf(1u, fsz);
                    exception_t err = dg::network_fileio::dg_read_binary(fp.c_str(), buf.get(), fsz);

                    if (dg::network_exception::is_failed(err)){
                        std::cout << "mayday, read_binary_unaligned unexpected dg_read_binary error" << std::endl;
                        std::abort();
                    }

                    if (std::memcmp(buf.get(), filedata_map[fp].data(), fsz) != 0){
                        std::cout << "mayday, read_binary_unaligned failed cmp" << std::endl;
                        std::abort();
                    }

                    break;
                }
                case OPS_CODE_READ_BINARY_ALIGNED:
                {
                    if (current_file_vec.empty()){
                        break;
                    }

                    size_t idx                                  = sz_randomizer() % current_file_vec.size();
                    std::filesystem::path fp                    = current_file_vec[idx];
                    size_t fsz                                  = filedata_map[fp].size();

                    if (!is_pow2(fsz) || fsz < dg::network_fileio::DG_LEAST_DIRECTIO_BLK_SZ){
                        break;
                    }

                    std::expected<size_t, exception_t> real_fsz = dg::network_fileio::dg_file_size(fp.c_str());

                    if (!real_fsz.has_value()){
                        std::cout << "mayday, read_binary_aligned unexpected file_size error" << std::endl;
                        std::abort();
                    }

                    if (fsz != real_fsz.value()){
                        std::cout << "mayday, read_binary_aligned size mismatched" << std::endl;
                        std::abort();
                    }

                    auto buf            = make_buf(dg::network_fileio::DG_LEAST_DIRECTIO_BLK_SZ, fsz);
                    exception_t err     = dg::network_fileio::dg_read_binary(fp.c_str(), buf.get(), fsz);

                    if (dg::network_exception::is_failed(err)){
                        std::cout << "mayday, read_binary_aligned unexpected dg_read_binary error" << std::endl;
                        std::abort();
                    }

                    if (std::memcmp(buf.get(), filedata_map[fp].data(), fsz) != 0){
                        std::cout << "mayday, read_binary_aligned content mismatched" << std::endl;
                        std::abort();
                    }

                    break;
                }
                case OPS_CODE_WRITE_BINARY_DIRECT_VALID:
                {
                    if (current_file_vec.empty()){
                        break;
                    }

                    size_t idx                  = sz_randomizer() % current_file_vec.size();
                    std::filesystem::path fp    = current_file_vec[idx];
                    size_t fsz                  = floor2(std::max(size_t{dg::network_fileio::DG_LEAST_DIRECTIO_BLK_SZ}, sz_randomizer() % NEW_BINARY_RANGE));
                    auto random_buf             = randomize_buf(dg::network_fileio::DG_LEAST_DIRECTIO_BLK_SZ, fsz);
                    exception_t err             = dg::network_fileio::dg_write_binary_direct(fp.c_str(), random_buf.get(), fsz);

                    if (dg::network_exception::is_failed(err)){
                        std::cout << "mayday, write_binary_direct unexpected error" << std::endl;
                        std::abort();
                    }

                    filedata_map[fp].resize(fsz);
                    std::memcpy(filedata_map[fp].data(), random_buf.get(), fsz);

                    break;
                }
                case OPS_CODE_WRITE_BINARY_DIRECT_UNALIGNED_FSZ:
                {
                    if (current_file_vec.empty()){
                        break;
                    }

                    size_t idx                  = sz_randomizer() % current_file_vec.size();
                    std::filesystem::path fp    = current_file_vec[idx];
                    size_t fsz                  = sz_randomizer() % NEW_BINARY_RANGE;

                    if (fsz % dg::network_fileio::DG_LEAST_DIRECTIO_BLK_SZ == 0u){
                        break;
                    }

                    auto random_buf             = randomize_buf(dg::network_fileio::DG_LEAST_DIRECTIO_BLK_SZ, fsz);
                    exception_t err             = dg::network_fileio::dg_write_binary_direct(fp.c_str(), random_buf.get(), fsz);

                    if (dg::network_exception::is_success(err)){
                        std::cout << "mayday, write_binary_direct_unaligned_fsz unexpected error" << std::endl;
                        std::abort();
                    }

                    break;
                }
                case OPS_CODE_WRITE_BINARY_DIRECT_UNALIGNED_DST:
                {
                    if (current_file_vec.empty()){
                        break;
                    }

                    size_t idx                  = sz_randomizer() % current_file_vec.size();
                    std::filesystem::path fp    = current_file_vec[idx];
                    size_t fsz                  = floor2(std::max(size_t{1u}, sz_randomizer() % NEW_BINARY_RANGE));
                    char * random_buf           = static_cast<char *>(std::malloc(fsz + dg::network_fileio::DG_LEAST_DIRECTIO_BLK_SZ));

                    if (random_buf == nullptr){
                        std::cout << "mayday, write_binary_direct_unaligned_dst unexpected error" << std::endl;
                    }

                    char * unaligned_buf        = std::next(align(random_buf, dg::network_fileio::DG_LEAST_DIRECTIO_BLK_SZ));
                    exception_t err             = dg::network_fileio::dg_write_binary_direct(fp.c_str(), unaligned_buf, fsz);

                    if (dg::network_exception::is_success(err)){
                        std::cout << "mayday, write_binary_direct_unaligned_dst unexpected error" << std::endl;
                        std::abort();
                    }

                    std::free(random_buf);                    
                    break;
                }
                case OPS_CODE_WRITE_BINARY_DIRECT_NOEXIST:
                {
                    if (past_file_vec.empty()){
                        break;
                    }

                    size_t idx                  = sz_randomizer() % past_file_vec.size();
                    std::filesystem::path fp    = past_file_vec[idx];
                    size_t fsz                  = floor2(std::max(size_t{1u}, sz_randomizer() % NEW_BINARY_RANGE));
                    auto random_buf             = randomize_buf(dg::network_fileio::DG_LEAST_DIRECTIO_BLK_SZ, fsz);
                    exception_t err             = dg::network_fileio::dg_write_binary_direct(fp.c_str(), random_buf.get(), fsz);

                    if (dg::network_exception::is_success(err)){
                        std::cout << "mayday, write_binary_direct_noexist unexpected error" << std::endl;
                        std::abort();
                    }

                    break;
                }
                case OPS_CODE_WRITE_BINARY_INDIRECT_VALID:
                {
                    break;
                }
                case OPS_CODE_WRITE_BINARY_INDIRECT_NOEXIST:
                {
                    if (past_file_vec.empty()){
                        break;
                    }

                    size_t idx                  = sz_randomizer() % past_file_vec.size();
                    std::filesystem::path fp    = past_file_vec[idx];
                    size_t fsz                  = sz_randomizer() % NEW_BINARY_RANGE;
                    auto random_buf             = randomize_buf(1u, fsz);
                    exception_t err             = dg::network_fileio::dg_write_binary_indirect(fp.c_str(), random_buf.get(), fsz);

                    if (dg::network_exception::is_success(err)){
                        std::cout << "mayday, write_binary_indirect_noexist unexpected error" << std::endl;
                        std::abort();
                    }

                    break;
                }
                case OPS_CODE_WRITE_BINARY_INDIRECT_UNALIGNED:
                {
                    if (current_file_vec.empty()){
                        break;
                    }

                    size_t idx                  = sz_randomizer() % current_file_vec.size();
                    std::filesystem::path fp    = current_file_vec[idx];
                    size_t fsz                  = sz_randomizer() % NEW_BINARY_RANGE;
                    auto random_buf             = randomize_buf(1u, fsz);
                    exception_t err             = dg::network_fileio::dg_write_binary_indirect(fp.c_str(), random_buf.get(), fsz);

                    if (dg::network_exception::is_failed(err)){
                        std::cout << "mayday, write_binary_unaligned unexpected error" << std::endl;
                        std::abort();
                    }

                    filedata_map[fp].resize(fsz);
                    std::memcpy(filedata_map[fp].data(), random_buf.get(), fsz);

                    break;
                }
                case OPS_CODE_WRITE_BINARY_VALID:
                {
                    break;
                }
                case OPS_CODE_WRITE_BINARY_NOEXIST:
                {
                    if (past_file_vec.empty()){
                        break;
                    }

                    size_t idx                  = sz_randomizer() % past_file_vec.size();
                    std::filesystem::path fp    = past_file_vec[idx];
                    size_t fsz                  = sz_randomizer() % NEW_BINARY_RANGE;
                    auto random_buf             = randomize_buf(1u, fsz);
                    exception_t err             = dg::network_fileio::dg_write_binary(fp.c_str(), random_buf.get(), fsz);

                    if (dg::network_exception::is_success(err)){
                        std::cout << "mayday, write_binary_noexist unexpected error" << std::endl;
                        std::abort();
                    }

                    break;
                }
                case OPS_CODE_WRITE_BINARY_UNALIGNED:
                {
                    if (current_file_vec.empty()){
                        break;
                    }

                    size_t idx                  = sz_randomizer() % current_file_vec.size();
                    std::filesystem::path fp    = current_file_vec[idx];
                    size_t fsz                  = sz_randomizer() % NEW_BINARY_RANGE;
                    auto random_buf             = randomize_buf(1u, fsz);
                    exception_t err             = dg::network_fileio::dg_write_binary(fp.c_str(), random_buf.get(), fsz);

                    if (dg::network_exception::is_failed(err)){
                        std::cout << "mayday, write_binary_unaligned unexpected error" << std::endl;
                        std::abort();
                    }

                    filedata_map[fp].resize(fsz);
                    std::memcpy(filedata_map[fp].data(), random_buf.get(), fsz);

                    break;
                }
                case OPS_CODE_WRITE_BINARY_ALIGNED:
                {
                    if (current_file_vec.empty()){
                        break;
                    }

                    size_t idx                  = sz_randomizer() % current_file_vec.size();
                    std::filesystem::path fp    = current_file_vec[idx];
                    size_t fsz                  = floor2(std::max(size_t{dg::network_fileio::DG_LEAST_DIRECTIO_BLK_SZ}, sz_randomizer() % NEW_BINARY_RANGE));
                    auto random_buf             = randomize_buf(dg::network_fileio::DG_LEAST_DIRECTIO_BLK_SZ, fsz);
                    exception_t err             = dg::network_fileio::dg_write_binary(fp.c_str(), random_buf.get(), fsz);

                    if (dg::network_exception::is_failed(err)){
                        std::cout << "mayday, write_binary_aligned unexpected error" << std::endl;
                        std::abort();
                    }

                    filedata_map[fp].resize(fsz);
                    std::memcpy(filedata_map[fp].data(), random_buf.get(), fsz);

                    break;
                }
                case OPS_CODE_CREATE_CBINARY_DUP_FILE:
                {
                    if (current_file_vec.empty()){
                        break;
                    }

                    size_t idx  = sz_randomizer() % current_file_vec.size();
                    std::filesystem::path fp = current_file_vec[idx];
                    size_t fsz  = sz_randomizer() % NEW_BINARY_RANGE; 

                    exception_t err = dg::network_fileio::dg_create_cbinary(fp.c_str(), fsz);

                    if (dg::network_exception::is_success(err)){
                        std::cout << "mayday, cbinary_cmp_file unexpected error" << std::endl;
                        std::abort();
                    }

                    break;
                }
                case OPS_CODE_CREATE_CBINARY_NEW_FILE:
                {
                    break;
                }
                case OPS_CODE_CMP:
                {
                    for (const std::filesystem::path& fp: current_file_vec){
                        size_t fsz = filedata_map[fp].size();
                        std::expected<size_t, exception_t> real_fsz = dg::network_fileio::dg_file_size(fp.c_str());

                        if (!real_fsz.has_value()){
                            std::cout << "mayday, cmp failed, bad file size get " << std::endl;
                            std::abort();
                        } 

                        if (fsz != real_fsz.value()){
                            std::cout << "mayday, cmp failed, bad file size cmp" << std::endl;
                            std::abort();
                        }

                        auto buf = std::make_unique<char[]>(fsz);
                        exception_t err = dg::network_fileio::dg_read_binary_indirect(fp.c_str(), buf.get(), fsz);

                        if (dg::network_exception::is_failed(err)){
                            std::cout << "mayday, cmp failed, bad file read" << std::endl;
                            std::abort();
                        }

                        if (std::memcmp(buf.get(), filedata_map[fp].data(), fsz) != 0){
                            std::cout << "mayday, cmp failed, bad file content cmp" << std::endl;
                            std::abort();
                        }
                    }

                    break;
                }
                case OPS_CODE_CHECK_FSZ_NOEXIST:
                {
                    if (past_file_vec.empty()){
                        break;
                    }

                    size_t idx = sz_randomizer() % past_file_vec.size();
                    std::filesystem::path fp = past_file_vec[idx];
                    std::expected<size_t, exception_t> fsz = dg::network_fileio::dg_file_size(fp.c_str());

                    if (fsz.has_value()){
                        std::cout << "mayday, check_fsz_noexist exists" << std::endl; 
                        std::abort();
                    }

                    break;
                }
                default:
                {
                    break;
                }
            }

            if (i % CLEAR_INTERVAL_SZ == 0u){
                for (const auto& fp: current_file_vec){
                    exception_t err = dg::network_fileio::dg_remove(fp.c_str());

                    if (dg::network_exception::is_failed(err)){
                        std::cout << "mayday, clear file" << std::endl;
                        std::abort();
                    }
                }

                filedata_map.clear();
                past_file_vec.clear();
                current_file_vec.clear();
            }


            if (i % TENPERCENT_TEST_SZ == 0u){
                std::cout << "progress >> " << i << "/" << OPERATION_SZ << std::endl;
            }
        }

        for (const auto& fp: current_file_vec){
            exception_t err = dg::network_fileio::dg_remove(fp.c_str());

            if (dg::network_exception::is_failed(err)){
                std::cout << "mayday, clear file" << std::endl;
                std::abort();
            }
        }

        std::cout << "<fileio_base_test_completed>" << std::endl;
    }

    void filo_base_leak_test(){

        //overriding posix functions, to check file leaks        
    }

    void fileio_chksum_test(std::filesystem::path test_dir){

        std::cout << "<initializing_fileio_chksum_test>" << std::endl;

        const size_t NEW_BINARY_RANGE                                       = (size_t{1} << 8) + 1u;
        const size_t NEW_BLK_RANGE                                          = ((size_t{1} << 20) / dg::network_fileio_chksum_x::DG_LEAST_DIRECTIO_BLK_SZ) + 1u;  

        const uint8_t OPS_CODE_FILE_SIZE_EXPECT_EQUAL                       = 0u;
        const uint8_t OPS_CODE_WRITE_DIRECT_READ_LESS_EXPECT_ERROR          = 1u;
        const uint8_t OPS_CODE_WRITE_DIRECT_READ_NOT_LESS_EXPECT_SUCCESS    = 2u;
        const uint8_t OPS_CODE_WRITE_INDIRECT_READ_LESS_EXPECT_ERROR        = 3u;
        const uint8_t OPS_CODE_WRITE_INDIRECT_READ_NOT_LESS_EXPECT_SUCCESS  = 4u;
        const uint8_t OPS_CODE_FILE_EXISTS_CURRENT_EXPECT_SUCCESS           = 5u;
        const uint8_t OPS_CODE_FILE_EXISTS_REMOVED_EXPECT_ERROR             = 6u;

        const size_t TEST_SZ                                                = size_t{1} << 13;
        const size_t TENPERCENT_TEST_SZ                                     = TEST_SZ / 10u;
        auto ops_code_gen                                                   = std::bind(std::uniform_int_distribution<uint8_t>(0u, 6u), std::mt19937{static_cast<uint32_t>(std::chrono::high_resolution_clock::now().time_since_epoch().count())});
        auto sz_gen                                                         = std::bind(std::uniform_int_distribution<size_t>(0u, NEW_BINARY_RANGE), std::mt19937{static_cast<uint32_t>(std::chrono::high_resolution_clock::now().time_since_epoch().count())});
        auto blk_sz_gen                                                     = std::bind(std::uniform_int_distribution<size_t>(0u, NEW_BLK_RANGE), std::mt19937{static_cast<uint32_t>(std::chrono::high_resolution_clock::now().time_since_epoch().count())}); 

        for (size_t i = 0u; i < TEST_SZ; ++i){
            uint8_t ops_code = ops_code_gen();

            switch (ops_code){
                case OPS_CODE_FILE_SIZE_EXPECT_EQUAL:
                {
                    std::filesystem::path fp    = create_tmp_file(test_dir);
                    size_t fsz                  = sz_gen(); 
                    std::string f_buf           = randomize_str(fsz);
                    exception_t cbin_err        = dg::network_fileio_chksum_x::dg_create_cbinary(fp.c_str(), fsz + 1u);
                    
                    if (dg::network_exception::is_failed(cbin_err)){
                        std::cout << "mayday, file_size_check unexpected dg_create_cbinary error" << std::endl;
                        std::abort();
                    }

                    std::expected<size_t, exception_t> fsz_chk1 = dg::network_fileio_chksum_x::dg_file_size(fp.c_str());

                    if (!fsz_chk1.has_value()){
                        std::cout << "mayday, file_size_check unexpected dg_file_size error" << std::endl;
                        std::abort();
                    } 

                    if (fsz_chk1.value() != fsz + 1u){
                        std::cout << "mayday, file_size_check compared not equal" << std::endl;
                        std::abort();
                    }

                    exception_t write_bin_err   = dg::network_fileio_chksum_x::dg_write_binary_indirect(fp.c_str(), f_buf.data(), f_buf.size());

                    if (dg::network_exception::is_failed(write_bin_err)){
                        std::cout << "mayday, file_size_check dg_write_binary_indirect unexpected error" << std::endl;
                    }

                    std::expected<size_t, exception_t> fsz_chk2 = dg::network_fileio_chksum_x::dg_file_size(fp.c_str());

                    if (!fsz_chk2.has_value()){
                        std::cout << "mayday, file_size_check unexpected dg_file_size error" << std::endl;
                        std::abort();
                    }

                    if (fsz_chk2.value() != fsz){
                        std::cout << "mayday, file_size_check compared not equal" << std::endl;
                        std::abort();
                    }

                    exception_t rm_err = dg::network_fileio_chksum_x::dg_remove(fp.c_str());

                    if (dg::network_exception::is_failed(rm_err)){
                        std::cout << "mayday, file_size_check unexpected remove error" << std::endl;
                        std::abort(); 
                    }

                    break;
                }
                case OPS_CODE_WRITE_INDIRECT_READ_LESS_EXPECT_ERROR:
                {
                    std::filesystem::path fp    = create_tmp_file(test_dir);
                    size_t fsz                  = sz_gen() + 1u;
                    std::string f_buf           = randomize_str(fsz);
                    exception_t cbin_err        = dg::network_fileio_chksum_x::dg_create_cbinary(fp.c_str(), fsz);

                    if (dg::network_exception::is_failed(cbin_err)){
                        std::cout << "mayday, write_indirect_read_less unexpected dg_create_cbinary error" << std::endl;
                        std::abort();
                    }

                    exception_t write_bin_err   = dg::network_fileio_chksum_x::dg_write_binary_indirect(fp.c_str(), f_buf.data(), f_buf.size());

                    if (dg::network_exception::is_failed(write_bin_err)){
                        std::cout << "mayday, write_indirect_read_less unexpected write_binary_indirect error" << std::endl;
                        std::abort();
                    }

                    std::string in_buf          = randomize_str(fsz - 1u);
                    exception_t read_bin_err    = dg::network_fileio_chksum_x::dg_read_binary_indirect(fp.c_str(), in_buf.data(), in_buf.size());

                    if (read_bin_err != dg::network_exception::RUNTIME_FILEIO_ERROR){
                        std::cout << "mayday, write_indirect_read_less unexpected dg_read_binary_indirect err" << std::endl;
                        std::abort();
                    }

                    exception_t rm_err = dg::network_fileio_chksum_x::dg_remove(fp.c_str());

                    if (dg::network_exception::is_failed(rm_err)){
                        std::cout << "mayday, write_indirect_read_less unexpected remove error" << std::endl;
                        std::abort();
                    }

                    break;
                }
                case OPS_CODE_WRITE_INDIRECT_READ_NOT_LESS_EXPECT_SUCCESS:
                {
                    std::filesystem::path fp    = create_tmp_file(test_dir);
                    size_t fsz                  = sz_gen();
                    std::string f_buf           = randomize_str(fsz);
                    exception_t cbin_err        = dg::network_fileio_chksum_x::dg_create_cbinary(fp.c_str(), fsz);

                    if (dg::network_exception::is_failed(cbin_err)){
                        std::cout << "mayday, write_indirect_read_not_less unexpected dg_create_cbinary error" << std::endl;
                        std::abort();
                    }

                    exception_t write_bin_err   = dg::network_fileio_chksum_x::dg_write_binary_indirect(fp.c_str(), f_buf.data(), f_buf.size());

                    if (dg::network_exception::is_failed(write_bin_err)){
                        std::cout << "mayday, write_indirect_read_not_less unexpected write_binary_indirect error" << std::endl;
                        std::abort();
                    }

                    std::string in_buf          = randomize_str(fsz + sz_gen());
                    exception_t read_bin_err    = dg::network_fileio_chksum_x::dg_read_binary_indirect(fp.c_str(), in_buf.data(), in_buf.size());

                    if (dg::network_exception::is_failed(read_bin_err)){
                        std::cout << "mayday, write_indirect_read_not_less unexpected dg_read_binary_indirect err" << std::endl;
                        std::abort();
                    }

                    if (std::memcmp(f_buf.data(), in_buf.data(), f_buf.size()) != 0){
                        std::cout << "mayday, write_indirect_read_not_less compared not equal" << std::endl;
                        std::abort();
                    }

                    exception_t rm_err = dg::network_fileio_chksum_x::dg_remove(fp.c_str());

                    if (dg::network_exception::is_failed(rm_err)){
                        std::cout << "mayday, write_indirect_read_less unexpected remove error" << std::endl;
                        std::abort();
                    }

                    break;
                }
                case OPS_CODE_WRITE_DIRECT_READ_LESS_EXPECT_ERROR:
                {
                    std::filesystem::path fp    = create_tmp_file(test_dir);
                    size_t blk_fsz              = blk_sz_gen() + 1u;
                    size_t fsz                  = blk_fsz * dg::network_fileio_chksum_x::DG_LEAST_DIRECTIO_BLK_SZ;
                    auto f_buf                  = randomize_buf(dg::network_fileio_chksum_x::DG_LEAST_DIRECTIO_BLK_SZ, fsz);
                    exception_t cbin_err        = dg::network_fileio_chksum_x::dg_create_cbinary(fp.c_str(), fsz);

                    if (dg::network_exception::is_failed(cbin_err)){
                        std::cout << "mayday, write_indirect_read_less unexpected dg_create_cbinary error" << std::endl;
                        std::abort();
                    }

                    exception_t write_bin_err   = dg::network_fileio_chksum_x::dg_write_binary_direct(fp.c_str(), f_buf.get(), fsz);

                    if (dg::network_exception::is_failed(write_bin_err)){
                        std::cout << "mayday, write_indirect_read_less unexpected write_binary_indirect error" << std::endl;
                        std::abort();
                    }

                    size_t bad_fsz              = (blk_fsz - 1u) * dg::network_fileio_chksum_x::DG_LEAST_DIRECTIO_BLK_SZ;
                    auto bad_f_buf              = randomize_buf(dg::network_fileio_chksum_x::DG_LEAST_DIRECTIO_BLK_SZ, bad_fsz);
                    exception_t read_bin_err    = dg::network_fileio_chksum_x::dg_read_binary_direct(fp.c_str(), bad_f_buf.get(), bad_fsz);

                    if (read_bin_err != dg::network_exception::RUNTIME_FILEIO_ERROR){
                        std::cout << "mayday, write_indirect_read_less unexpected dg_read_binary_indirect err" << std::endl;
                        std::abort();
                    }

                    exception_t rm_err = dg::network_fileio_chksum_x::dg_remove(fp.c_str());

                    if (dg::network_exception::is_failed(rm_err)){
                        std::cout << "mayday, write_indirect_read_less unexpected remove error" << std::endl;
                        std::abort();
                    }

                    break;                
                }
                case OPS_CODE_WRITE_DIRECT_READ_NOT_LESS_EXPECT_SUCCESS:
                {
                    std::filesystem::path fp    = create_tmp_file(test_dir);
                    size_t blk_fsz              = blk_sz_gen() + 1u;
                    size_t fsz                  = blk_fsz * dg::network_fileio_chksum_x::DG_LEAST_DIRECTIO_BLK_SZ;
                    auto f_buf                  = randomize_buf(dg::network_fileio_chksum_x::DG_LEAST_DIRECTIO_BLK_SZ, fsz);
                    exception_t cbin_err        = dg::network_fileio_chksum_x::dg_create_cbinary(fp.c_str(), fsz);

                    if (dg::network_exception::is_failed(cbin_err)){
                        std::cout << "mayday, write_indirect_read_not_less unexpected dg_create_cbinary error" << std::endl;
                        std::abort();
                    }

                    exception_t write_bin_err   = dg::network_fileio_chksum_x::dg_read_binary_direct(fp.c_str(), f_buf.get(), fsz);

                    if (dg::network_exception::is_failed(write_bin_err)){
                        std::cout << "mayday, write_indirect_read_not_less unexpected write_binary_indirect error" << std::endl;
                        std::abort();
                    }

                    size_t good_blk_fsz         = blk_fsz + blk_sz_gen();
                    size_t good_fsz             = good_blk_fsz * dg::network_fileio_chksum_x::DG_LEAST_DIRECTIO_BLK_SZ;
                    auto in_buf                 = randomize_buf(dg::network_fileio_chksum_x::DG_LEAST_DIRECTIO_BLK_SZ, good_fsz);
                    exception_t read_bin_err    = dg::network_fileio_chksum_x::dg_read_binary_direct(fp.c_str(), in_buf.get(), good_fsz);

                    if (dg::network_exception::is_failed(read_bin_err)){
                        std::cout << "mayday, write_indirect_read_not_less unexpected dg_read_binary_indirect err" << std::endl;
                        std::abort();
                    }

                    if (std::memcmp(f_buf.get(), in_buf.get(), fsz) != 0){
                        std::cout << "mayday, write_indirect_read_not_less compared not equal" << std::endl;
                        std::abort();
                    }

                    exception_t rm_err = dg::network_fileio_chksum_x::dg_remove(fp.c_str());

                    if (dg::network_exception::is_failed(rm_err)){
                        std::cout << "mayday, write_indirect_read_less unexpected remove error" << std::endl;
                        std::abort();
                    }

                    break;
                }
                case OPS_CODE_FILE_EXISTS_CURRENT_EXPECT_SUCCESS:
                {
                    std::filesystem::path fp    = create_tmp_file(test_dir);
                    size_t fsz                  = sz_gen(); 
                    std::string f_buf           = randomize_str(fsz);
                    exception_t cbin_err        = dg::network_fileio_chksum_x::dg_create_cbinary(fp.c_str(), fsz + 1u);
                    
                    if (dg::network_exception::is_failed(cbin_err)){
                        std::cout << "mayday, file_exists_current unexpected dg_create_cbinary error" << std::endl;
                        std::abort();
                    }

                    std::expected<bool, exception_t> status = dg::network_fileio_chksum_x::dg_file_exists(fp.c_str());

                    if (!status.has_value()){
                        std::cout << "mayday, file_exists_current dg_file_exists unexpected error" << std::endl;
                        std::abort();
                    }

                    if (!status.value()){
                        std::cout << "mayday, file_exists_current dg_file_exists file not found" << std::endl;
                        std::abort();
                    }

                    exception_t rm_err = dg::network_fileio_chksum_x::dg_remove(fp.c_str());

                    if (dg::network_exception::is_failed(rm_err)){
                        std::cout << "mayday, file_exists_current unexpected remove error" << std::endl;
                        std::abort(); 
                    }

                    break;
                }
                case OPS_CODE_FILE_EXISTS_REMOVED_EXPECT_ERROR:
                {
                    std::filesystem::path fp    = create_tmp_file(test_dir);
                    size_t fsz                  = sz_gen(); 
                    std::string f_buf           = randomize_str(fsz);
                    exception_t cbin_err        = dg::network_fileio_chksum_x::dg_create_cbinary(fp.c_str(), fsz + 1u);
                    
                    if (dg::network_exception::is_failed(cbin_err)){
                        std::cout << "mayday, file_exists_removed unexpected dg_create_cbinary error" << std::endl;
                        std::abort();
                    }

                    exception_t rm_err = dg::network_fileio_chksum_x::dg_remove(fp.c_str());

                    if (dg::network_exception::is_failed(rm_err)){
                        std::cout << "mayday, file_exists_removed unexpected remove error" << std::endl;
                        std::abort(); 
                    }

                    std::expected<bool, exception_t> status = dg::network_fileio_chksum_x::dg_file_exists(fp.c_str());

                    if (!status.has_value()){
                        std::cout << "mayday, file_exists_removed dg_file_exists unexpected error" << std::endl;
                        std::abort();
                    }

                    if (status.value()){
                        std::cout << "mayday, file_exists_removed dg_file_exists file not found" << std::endl;
                        std::abort();
                    }

                    break;                
                }
                default:
                {
                    break;
                }
            }              

            if (i % TENPERCENT_TEST_SZ == 0u){
                std::cout << "progress >> " << i << "/" << TEST_SZ << std::endl;
            }
        }

        std::cout << "<fileio_chksum_test_completed>" << std::endl;
    } 

    void fileio_unified_test(std::filesystem::path test_dir){

        std::cout << "<initializing_fileio_unified_test>" << std::endl;

        const size_t NEW_BINARY_RANGE                                       = (size_t{1} << 8) + 1u;
        const size_t NEW_BLK_RANGE                                          = ((size_t{1} << 20) / dg::network_fileio_unified_x::DG_LEAST_DIRECTIO_BLK_SZ) + 1u;  

        const uint8_t OPS_CODE_FILE_SIZE_EXPECT_EQUAL                       = 0u;
        const uint8_t OPS_CODE_WRITE_DIRECT_READ_LESS_EXPECT_ERROR          = 1u;
        const uint8_t OPS_CODE_WRITE_DIRECT_READ_NOT_LESS_EXPECT_SUCCESS    = 2u;
        const uint8_t OPS_CODE_WRITE_INDIRECT_READ_LESS_EXPECT_ERROR        = 3u;
        const uint8_t OPS_CODE_WRITE_INDIRECT_READ_NOT_LESS_EXPECT_SUCCESS  = 4u;
        const uint8_t OPS_CODE_WRITE_EXPECT_SUCCESS                         = 5u;
        const uint8_t OPS_CODE_FILE_EXISTS_CURRENT_EXPECT_SUCCESS           = 6u;
        const uint8_t OPS_CODE_FILE_EXISTS_REMOVED_EXPECT_ERROR             = 7u;

        const size_t TEST_SZ                                                = size_t{1} << 13;
        const size_t TENPERCENT_TEST_SZ                                     = TEST_SZ / 10u;
        auto ops_code_gen                                                   = std::bind(std::uniform_int_distribution<uint8_t>(0u, 6u), std::mt19937{static_cast<uint32_t>(std::chrono::high_resolution_clock::now().time_since_epoch().count())});
        auto sz_gen                                                         = std::bind(std::uniform_int_distribution<size_t>(0u, NEW_BINARY_RANGE), std::mt19937{static_cast<uint32_t>(std::chrono::high_resolution_clock::now().time_since_epoch().count())});
        auto blk_sz_gen                                                     = std::bind(std::uniform_int_distribution<size_t>(0u, NEW_BLK_RANGE), std::mt19937{static_cast<uint32_t>(std::chrono::high_resolution_clock::now().time_since_epoch().count())}); 

        for (size_t i = 0u; i < TEST_SZ; ++i){
            uint8_t ops_code = ops_code_gen();

            switch (ops_code){
                case OPS_CODE_FILE_SIZE_EXPECT_EQUAL:
                {
                    std::filesystem::path fp    = create_tmp_file(test_dir);
                    size_t fsz                  = sz_gen(); 
                    std::string f_buf           = randomize_str(fsz);
                    exception_t cbin_err        = dg::network_fileio_unified_x::dg_create_cbinary(fp.c_str(), get_replicated_path(fp, 3), fsz + 1u);
                    
                    if (dg::network_exception::is_failed(cbin_err)){
                        std::cout << "mayday, file_size_check unexpected dg_create_cbinary error" << std::endl;
                        std::abort();
                    }

                    std::expected<size_t, exception_t> fsz_chk1 = dg::network_fileio_unified_x::dg_file_size(fp.c_str());

                    if (!fsz_chk1.has_value()){
                        std::cout << "mayday, file_size_check unexpected dg_file_size error" << std::endl;
                        std::abort();
                    } 

                    if (fsz_chk1.value() != fsz + 1u){
                        std::cout << "mayday, file_size_check compared not equal" << std::endl;
                        std::abort();
                    }

                    exception_t write_bin_err   = dg::network_fileio_unified_x::dg_write_binary_indirect(fp.c_str(), f_buf.data(), f_buf.size());

                    if (dg::network_exception::is_failed(write_bin_err)){
                        std::cout << "mayday, file_size_check dg_write_binary_indirect unexpected error" << std::endl;
                    }

                    std::expected<size_t, exception_t> fsz_chk2 = dg::network_fileio_unified_x::dg_file_size(fp.c_str());

                    if (!fsz_chk2.has_value()){
                        std::cout << "mayday, file_size_check unexpected dg_file_size error" << std::endl;
                        std::abort();
                    }

                    if (fsz_chk2.value() != fsz){
                        std::cout << "mayday, file_size_check compared not equal" << std::endl;
                        std::abort();
                    }

                    exception_t rm_err = dg::network_fileio_unified_x::dg_remove(fp.c_str());

                    if (dg::network_exception::is_failed(rm_err)){
                        std::cout << "mayday, file_size_check unexpected remove error" << std::endl;
                        std::abort(); 
                    }

                    break;
                }
                case OPS_CODE_WRITE_INDIRECT_READ_LESS_EXPECT_ERROR:
                {
                    std::filesystem::path fp    = create_tmp_file(test_dir);
                    size_t fsz                  = sz_gen() + 1u;
                    std::string f_buf           = randomize_str(fsz);
                    exception_t cbin_err        = dg::network_fileio_unified_x::dg_create_cbinary(fp.c_str(), get_replicated_path(fp, 3), fsz);

                    if (dg::network_exception::is_failed(cbin_err)){
                        std::cout << "mayday, write_indirect_read_less unexpected dg_create_cbinary error" << std::endl;
                        std::abort();
                    }

                    exception_t write_bin_err   = dg::network_fileio_unified_x::dg_write_binary_indirect(fp.c_str(), f_buf.data(), f_buf.size());

                    if (dg::network_exception::is_failed(write_bin_err)){
                        std::cout << "mayday, write_indirect_read_less unexpected write_binary_indirect error" << std::endl;
                        std::abort();
                    }

                    std::string in_buf          = randomize_str(fsz - 1u);
                    exception_t read_bin_err    = dg::network_fileio_unified_x::dg_read_binary_indirect(fp.c_str(), in_buf.data(), in_buf.size());

                    if (read_bin_err != dg::network_exception::RUNTIME_FILEIO_ERROR){
                        std::cout << "mayday, write_indirect_read_less unexpected dg_read_binary_indirect err" << std::endl;
                        std::abort();
                    }

                    exception_t rm_err = dg::network_fileio_unified_x::dg_remove(fp.c_str());

                    if (dg::network_exception::is_failed(rm_err)){
                        std::cout << "mayday, write_indirect_read_less unexpected remove error" << std::endl;
                        std::abort();
                    }

                    break;
                }
                case OPS_CODE_WRITE_INDIRECT_READ_NOT_LESS_EXPECT_SUCCESS:
                {
                    std::filesystem::path fp    = create_tmp_file(test_dir);
                    size_t fsz                  = sz_gen();
                    std::string f_buf           = randomize_str(fsz);
                    exception_t cbin_err        = dg::network_fileio_unified_x::dg_create_cbinary(fp.c_str(), get_replicated_path(fp, 3), fsz);

                    if (dg::network_exception::is_failed(cbin_err)){
                        std::cout << "mayday, write_indirect_read_not_less unexpected dg_create_cbinary error" << std::endl;
                        std::abort();
                    }

                    exception_t write_bin_err   = dg::network_fileio_unified_x::dg_write_binary_indirect(fp.c_str(), f_buf.data(), f_buf.size());

                    if (dg::network_exception::is_failed(write_bin_err)){
                        std::cout << "mayday, write_indirect_read_not_less unexpected write_binary_indirect error" << std::endl;
                        std::abort();
                    }

                    std::string in_buf          = randomize_str(fsz + sz_gen());
                    exception_t read_bin_err    = dg::network_fileio_unified_x::dg_read_binary_indirect(fp.c_str(), in_buf.data(), in_buf.size());

                    if (dg::network_exception::is_failed(read_bin_err)){
                        std::cout << "mayday, write_indirect_read_not_less unexpected dg_read_binary_indirect err" << std::endl;
                        std::abort();
                    }

                    if (std::memcmp(f_buf.data(), in_buf.data(), f_buf.size()) != 0){
                        std::cout << "mayday, write_indirect_read_not_less compared not equal" << std::endl;
                        std::abort();
                    }

                    exception_t rm_err = dg::network_fileio_unified_x::dg_remove(fp.c_str());

                    if (dg::network_exception::is_failed(rm_err)){
                        std::cout << "mayday, write_indirect_read_less unexpected remove error" << std::endl;
                        std::abort();
                    }

                    break;
                }
                case OPS_CODE_WRITE_DIRECT_READ_LESS_EXPECT_ERROR:
                {
                    std::filesystem::path fp    = create_tmp_file(test_dir);
                    size_t blk_fsz              = blk_sz_gen() + 1u;
                    size_t fsz                  = blk_fsz * dg::network_fileio_unified_x::DG_LEAST_DIRECTIO_BLK_SZ;
                    auto f_buf                  = randomize_buf(dg::network_fileio_unified_x::DG_LEAST_DIRECTIO_BLK_SZ, fsz);
                    exception_t cbin_err        = dg::network_fileio_unified_x::dg_create_cbinary(fp.c_str(), get_replicated_path(fp, 3), fsz);

                    if (dg::network_exception::is_failed(cbin_err)){
                        std::cout << "mayday, write_indirect_read_less unexpected dg_create_cbinary error" << std::endl;
                        std::abort();
                    }

                    exception_t write_bin_err   = dg::network_fileio_unified_x::dg_write_binary_direct(fp.c_str(), f_buf.get(), fsz);

                    if (dg::network_exception::is_failed(write_bin_err)){
                        std::cout << "mayday, write_indirect_read_less unexpected write_binary_indirect error" << std::endl;
                        std::abort();
                    }

                    size_t bad_fsz              = (blk_fsz - 1u) * dg::network_fileio_unified_x::DG_LEAST_DIRECTIO_BLK_SZ;
                    auto bad_f_buf              = randomize_buf(dg::network_fileio_unified_x::DG_LEAST_DIRECTIO_BLK_SZ, bad_fsz);
                    exception_t read_bin_err    = dg::network_fileio_unified_x::dg_read_binary_direct(fp.c_str(), bad_f_buf.get(), bad_fsz);

                    if (read_bin_err != dg::network_exception::RUNTIME_FILEIO_ERROR){
                        std::cout << "mayday, write_indirect_read_less unexpected dg_read_binary_indirect err" << std::endl;
                        std::abort();
                    }

                    exception_t rm_err = dg::network_fileio_unified_x::dg_remove(fp.c_str());

                    if (dg::network_exception::is_failed(rm_err)){
                        std::cout << "mayday, write_indirect_read_less unexpected remove error" << std::endl;
                        std::abort();
                    }

                    break;                
                }
                case OPS_CODE_WRITE_DIRECT_READ_NOT_LESS_EXPECT_SUCCESS:
                {
                    std::filesystem::path fp    = create_tmp_file(test_dir);
                    size_t blk_fsz              = blk_sz_gen() + 1u;
                    size_t fsz                  = blk_fsz * dg::network_fileio_unified_x::DG_LEAST_DIRECTIO_BLK_SZ;
                    auto f_buf                  = randomize_buf(dg::network_fileio_unified_x::DG_LEAST_DIRECTIO_BLK_SZ, fsz);
                    exception_t cbin_err        = dg::network_fileio_unified_x::dg_create_cbinary(fp.c_str(), get_replicated_path(fp, 3), fsz);

                    if (dg::network_exception::is_failed(cbin_err)){
                        std::cout << "mayday, write_indirect_read_not_less unexpected dg_create_cbinary error" << std::endl;
                        std::abort();
                    }

                    exception_t write_bin_err   = dg::network_fileio_unified_x::dg_read_binary_direct(fp.c_str(), f_buf.get(), fsz);

                    if (dg::network_exception::is_failed(write_bin_err)){
                        std::cout << "mayday, write_indirect_read_not_less unexpected write_binary_indirect error" << std::endl;
                        std::abort();
                    }

                    size_t good_blk_fsz         = blk_fsz + blk_sz_gen();
                    size_t good_fsz             = good_blk_fsz * dg::network_fileio_unified_x::DG_LEAST_DIRECTIO_BLK_SZ;
                    auto in_buf                 = randomize_buf(dg::network_fileio_unified_x::DG_LEAST_DIRECTIO_BLK_SZ, good_fsz);
                    exception_t read_bin_err    = dg::network_fileio_unified_x::dg_read_binary_direct(fp.c_str(), in_buf.get(), good_fsz);

                    if (dg::network_exception::is_failed(read_bin_err)){
                        std::cout << "mayday, write_indirect_read_not_less unexpected dg_read_binary_indirect err" << std::endl;
                        std::abort();
                    }

                    if (std::memcmp(f_buf.get(), in_buf.get(), fsz) != 0){
                        std::cout << "mayday, write_indirect_read_not_less compared not equal" << std::endl;
                        std::abort();
                    }

                    exception_t rm_err = dg::network_fileio_unified_x::dg_remove(fp.c_str());

                    if (dg::network_exception::is_failed(rm_err)){
                        std::cout << "mayday, write_indirect_read_less unexpected remove error" << std::endl;
                        std::abort();
                    }

                    break;
                }
                case OPS_CODE_WRITE_EXPECT_SUCCESS:
                {
                    std::filesystem::path fp    = create_tmp_file(test_dir);
                    size_t fsz                  = [&]{
                        if (sz_gen() % 2 == 0){
                            return sz_gen();
                        } else{
                            return blk_sz_gen() * dg::network_fileio_unified_x::DG_LEAST_DIRECTIO_BLK_SZ;
                        }
                    }();
                    std::string f_buf           = randomize_str(fsz);
                    exception_t cbin_err        = dg::network_fileio_unified_x::dg_create_cbinary(fp.c_str(), get_replicated_path(fp, 3), fsz);

                    if (dg::network_exception::is_failed(cbin_err)){
                        std::cout << "mayday, write_expect_success unexpected dg_create_cbinary error" << std::endl;
                        std::abort();
                    }

                    exception_t write_bin_err   = dg::network_fileio_unified_x::dg_write_binary(fp.c_str(), f_buf.data(), f_buf.size());

                    if (dg::network_exception::is_failed(write_bin_err)){
                        std::cout << "mayday, write_expect_success unexpected write_binary_indirect error" << std::endl;
                        std::abort();
                    }

                    std::string in_buf          = randomize_str(fsz + sz_gen());
                    exception_t read_bin_err    = dg::network_fileio_unified_x::dg_read_binary(fp.c_str(), in_buf.data(), in_buf.size());

                    if (dg::network_exception::is_failed(read_bin_err)){
                        std::cout << "mayday, write_expect_success unexpected dg_read_binary_indirect err" << std::endl;
                        std::abort();
                    }

                    if (std::memcmp(f_buf.data(), in_buf.data(), f_buf.size()) != 0){
                        std::cout << "mayday, write_expect_success compared not equal" << std::endl;
                        std::abort();
                    }

                    exception_t rm_err = dg::network_fileio_unified_x::dg_remove(fp.c_str());

                    if (dg::network_exception::is_failed(rm_err)){
                        std::cout << "mayday, write_indirect_read_less unexpected remove error" << std::endl;
                        std::abort();
                    }

                    break;
                }
                case OPS_CODE_FILE_EXISTS_CURRENT_EXPECT_SUCCESS:
                {
                    std::filesystem::path fp    = create_tmp_file(test_dir);
                    size_t fsz                  = sz_gen(); 
                    std::string f_buf           = randomize_str(fsz);
                    exception_t cbin_err        = dg::network_fileio_unified_x::dg_create_cbinary(fp.c_str(), get_replicated_path(fp, 3), fsz + 1u);
                    
                    if (dg::network_exception::is_failed(cbin_err)){
                        std::cout << "mayday, file_exists_current unexpected dg_create_cbinary error" << std::endl;
                        std::abort();
                    }

                    std::expected<bool, exception_t> status = dg::network_fileio_unified_x::dg_file_exists(fp.c_str());

                    if (!status.has_value()){
                        std::cout << "mayday, file_exists_current dg_file_exists unexpected error" << std::endl;
                        std::abort();
                    }

                    if (!status.value()){
                        std::cout << "mayday, file_exists_current dg_file_exists file not found" << std::endl;
                        std::abort();
                    }

                    exception_t rm_err = dg::network_fileio_unified_x::dg_remove(fp.c_str());

                    if (dg::network_exception::is_failed(rm_err)){
                        std::cout << "mayday, file_exists_current unexpected remove error" << std::endl;
                        std::abort(); 
                    }

                    break;
                }
                case OPS_CODE_FILE_EXISTS_REMOVED_EXPECT_ERROR:
                {
                    std::filesystem::path fp    = create_tmp_file(test_dir);
                    size_t fsz                  = sz_gen(); 
                    std::string f_buf           = randomize_str(fsz);
                    exception_t cbin_err        = dg::network_fileio_unified_x::dg_create_cbinary(fp.c_str(), get_replicated_path(fp, 3), fsz + 1u);
                    
                    if (dg::network_exception::is_failed(cbin_err)){
                        std::cout << "mayday, file_exists_removed unexpected dg_create_cbinary error" << std::endl;
                        std::abort();
                    }

                    exception_t rm_err = dg::network_fileio_unified_x::dg_remove(fp.c_str());

                    if (dg::network_exception::is_failed(rm_err)){
                        std::cout << "mayday, file_exists_removed unexpected remove error" << std::endl;
                        std::abort(); 
                    }

                    std::expected<bool, exception_t> status = dg::network_fileio_unified_x::dg_file_exists(fp.c_str());

                    if (!status.has_value()){
                        std::cout << "mayday, file_exists_removed dg_file_exists unexpected error" << std::endl;
                        std::abort();
                    }

                    if (status.value()){
                        std::cout << "mayday, file_exists_removed dg_file_exists file not found" << std::endl;
                        std::abort();
                    }

                    break;                
                }
                default:
                {
                    break;
                }
            }              

            if (i % TENPERCENT_TEST_SZ == 0u){
                std::cout << "progress >> " << i << "/" << TEST_SZ << std::endl;
            }
        }

        std::cout << "<fileio_unified_test_completed>" << std::endl;
    }

    void run(){

        std::string folder_path = "/home/tommy2tonez/dg_projects/dg_polyobjects/unit_tests/fsys_test_folder";
        std::filesystem::create_directory(folder_path);
        fileio_base_test(folder_path);
        fileio_chksum_test(folder_path);
        fileio_unified_test(folder_path);
        std::filesystem::remove(folder_path);
    }
}