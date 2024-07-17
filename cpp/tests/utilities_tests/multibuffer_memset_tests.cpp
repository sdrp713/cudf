#include <cudf_test/base_fixture.hpp>
#include <cudf_test/cudf_gtest.hpp>
#include <cudf_test/type_lists.hpp>
#include <src/io/utilities/multibuffer_memset.hpp>
#include <thrust/iterator/transform_iterator.h>
#include <type_traits>
#include <cudf/io/parquet.hpp>
#include <rmm/device_uvector.hpp>

template <typename T>
struct MultiBufferTestIntegral : public cudf::test::BaseFixture {};

TEST(MultiBufferTestIntegral, BasicTest)
{
    std::vector<long> BUF_SIZES{2};
    long NUM_BUFS = BUF_SIZES.size();

    // Device init
    std::vector<cudf::device_span<uint64_t>> bufs;
    auto stream = cudf::get_default_stream();
    auto _mr = rmm::mr::get_current_device_resource();

    std::vector<rmm::device_uvector<uint64_t>> cont;
    for (int i = 0; i < NUM_BUFS; i++) {
        cont.push_back(rmm::device_uvector<uint64_t>(BUF_SIZES[i], stream, _mr));
    }

    for (int i = 0; i < NUM_BUFS; i++) {
        bufs.push_back(cudf::device_span<uint64_t>(cont[i]));
    }
    multibuffer_memset(bufs, 0xFFFF, stream, _mr);

    // Compare
    for (int i = 0; i < NUM_BUFS; i++) {
        std::vector<uint64_t> temp(BUF_SIZES[i]);
        cudf::host_span<uint64_t> host(temp);
        CUDF_CUDA_TRY(cudaMemcpy(host.data(), bufs[i].data(), BUF_SIZES[i], cudaMemcpyDefault));
        for (int j = 0; j < BUF_SIZES[i]; j++) {
            fprintf(stderr, "idx: %u \n", j);
           EXPECT_EQ(host[j], 0xFFFF);
        }
    }

}


// TEST(MultiBufferTestIntegral, BasicTest2)
// {
//     std::vector<long> BUF_SIZES{131073};
//     long NUM_BUFS = BUF_SIZES.size();

//     // Device init
//     std::vector<cudf::device_span<uint8_t>> bufs;
//     auto stream = cudf::get_default_stream();
//     auto _mr = rmm::mr::get_current_device_resource();
//     for (int i = 0; i < NUM_BUFS; i++) {
//         rmm::device_uvector<uint8_t> temp(BUF_SIZES[i], stream, _mr);
//         bufs.push_back(cudf::device_span<uint8_t>(temp));
//     }
//     multibuffer_memset(bufs, 0, stream, _mr);

//     // Compare
//     for (int i = 0; i < NUM_BUFS; i++) {
//         std::vector<uint8_t> temp(BUF_SIZES[i]);
//         cudf::host_span<uint8_t> host(temp);
//         CUDF_CUDA_TRY(cudaMemcpy(host.data(), bufs[i].data(), BUF_SIZES[i], cudaMemcpyDefault));
//         for (int j = 0; j < BUF_SIZES[i]; j++) {
//             EXPECT_EQ(host[j], 0);
//         }
//     }
// }

// TEST(MultiBufferTestIntegral, BasicTest3)
// {
//     std::vector<long> BUF_SIZES{100, 200};
//     long NUM_BUFS = BUF_SIZES.size();

//     // Device init
//     std::vector<cudf::device_span<uint8_t>> bufs;
//     auto stream = cudf::get_default_stream();
//     auto _mr = rmm::mr::get_current_device_resource();
//     for (int i = 0; i < NUM_BUFS; i++) {
//         rmm::device_uvector<uint8_t> temp(BUF_SIZES[i], stream, _mr);
//         bufs.push_back(cudf::device_span<uint8_t>(temp));
//     }
//     multibuffer_memset(bufs, 0, stream, _mr);

//     // Compare
//     for (int i = 0; i < NUM_BUFS; i++) {
//         std::vector<uint8_t> temp(BUF_SIZES[i]);
//         cudf::host_span<uint8_t> host(temp);
//         CUDF_CUDA_TRY(cudaMemcpy(host.data(), bufs[i].data(), BUF_SIZES[i], cudaMemcpyDefault));
//         for (int j = 0; j < BUF_SIZES[i]; j++) {
//             EXPECT_EQ(host[j], 0);
//         }
//     }
// }

// TEST(MultiBufferTestIntegral, BasicTest4)
// {
//     std::vector<long> BUF_SIZES{131073, 200, 160000, 300000, 500000, 600, 131700, 800};
//     long NUM_BUFS = BUF_SIZES.size();

//     // Device init
//     std::vector<cudf::device_span<uint8_t>> bufs;
//     auto stream = cudf::get_default_stream();
//     auto _mr = rmm::mr::get_current_device_resource();
//     for (int i = 0; i < NUM_BUFS; i++) {
//         rmm::device_uvector<uint8_t> temp(BUF_SIZES[i], stream, _mr);
//         bufs.push_back(cudf::device_span<uint8_t>(temp));
//     }
//     multibuffer_memset(bufs, 0, stream, _mr);

//     // Compare
//     for (int i = 0; i < NUM_BUFS; i++) {
//         std::vector<uint8_t> temp(BUF_SIZES[i]);
//         cudf::host_span<uint8_t> host(temp);
//         CUDF_CUDA_TRY(cudaMemcpy(host.data(), bufs[i].data(), BUF_SIZES[i], cudaMemcpyDefault));
//         for (int j = 0; j < BUF_SIZES[i]; j++) {
//             EXPECT_EQ(host[j], 0);
//         }
//     }
// }