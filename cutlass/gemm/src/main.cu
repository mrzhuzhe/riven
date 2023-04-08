#include <iostream>
#include <cutlass/cutlass.h>
#include <cutlass/numeric_types.h>
#include <cutlass/core_io.h>

#include <cutlass/gemm/device/gemm.h>
#include <cutlass/util/host_tensor.h>


#include "cutlass/epilogue/collective/default_epilogue.hpp"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"

#include "cutlass/util/packed_stride.hpp"

#include <cute/layout.hpp>
#include <cute/tensor.hpp>
//  #include <cute/util/debug.hpp>

template <class Shape, class Stride>
void print2D(cute::Layout<Shape,Stride> const& layout)
{
  for (int m = 0; m < cute::size<0>(layout); ++m){
    for (int n = 0; n < cute::size<1>(layout); ++n){
      printf("%3d", layout(m,n));
    }
    printf("\n");
  }
}

int main() {

  cutlass::half_t x = 2.25_hf;

  std::cout << x << std::endl;

  auto layout_8s = cute::make_layout(cute::Int<8>{});
  auto layout_8d = cute::make_layout(8);
  printf("%d %d \n", layout_8s(7), layout_8d(2));


  auto layout_2sx4s = cute::make_layout(cute::make_shape(cute::Int<2>{}, cute::Int<4>{}));
  auto layout_2sx4d = cute::make_layout(cute::make_shape(cute::Int<2>{}, 4));
  auto layout_2x4 = cute::make_layout(cute::make_shape(2, cute::make_shape(2, 2)),
  cute::make_stride(4, cute::make_stride(1, 2))); 

  printf("\n");
  print2D(layout_2sx4s);
  printf("\n");
  print2D(layout_2sx4d);
  printf("\n");
  print2D(layout_2x4);
  printf("\n");

  cute::Layout layout = cute::make_layout(cute::make_shape (cute::_2{}, cute::_3{}),
  cute::make_stride(cute::_3{}, cute::_1{}));

  cute::print_layout(layout);
  for (int i = 0; i < cute::size(layout); ++i){
    cute::print(layout(i));
    cute::print(", ");
  }
  cute::print("\n");
  cute::print(layout(1, 1));
  cute::print("\n");

  cute::Layout layout2 = cute::Layout<cute::Shape <cute::Shape <cute::_4, cute::_3>, cute::_1>,
    cute::Stride<cute::Stride<cute::_3, cute::_1>, cute::_0>>{};  
  print2D(layout2);

  cute::Layout flat_layout = cute::flatten(layout2);
  //cute::print_layout(flat_layout);

  cute::Layout tile = cute::Layout<cute::Shape <cute::_2, cute::_2>, cute::Stride<cute::_1, cute::_2>>{};
  cute::Layout matrix_of_tiles = cute::Layout<cute::Shape <cute::_3, cute::_4>, cute::Stride<cute::_4, cute::_1>>{};

  cute::print_layout(cute::blocked_product(tile, matrix_of_tiles));

  cute::print_layout(cute::logical_product(tile, matrix_of_tiles));

  cute::print_layout(cute::raked_product(tile, matrix_of_tiles));

  //cute::Tensor gmem_8s = make_tensor(make_gmem_ptr(A), Int<8>{});

  cute::Tensor rmem_4x8_col = cute::make_tensor<float>(cute::make_shape(cute::Int<4>{},cute::Int<8>{}));

  // quick start part
  using Gemm = cutlass::gemm::device::Gemm<
    cutlass::half_t,  //  A
    cutlass::layout::ColumnMajor,
    cutlass::half_t,  //  B
    cutlass::layout::ColumnMajor,
    cutlass::half_t,  // out
    cutlass::layout::ColumnMajor,
    float,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80
  >;

  Gemm gemm_op;
  cutlass::Status status;

  int M = 512;
  int N = 256;
  int K = 128;

  float alpha = 1.25f;
  float beta = -1.25f;

  cutlass::HostTensor<cutlass::half_t, cutlass::layout::ColumnMajor> A({M, K});
  cutlass::HostTensor<cutlass::half_t, cutlass::layout::ColumnMajor> B({K, N});
  cutlass::HostTensor<cutlass::half_t, cutlass::layout::ColumnMajor> C({M, N});

  cutlass::half_t const *ptrA = A.device_data();
  cutlass::half_t const *ptrB = B.device_data();
  cutlass::half_t const *ptrC = C.device_data();
  cutlass::half_t *ptrD = C.device_data();

  int lda = A.device_ref().stride(0);
  int ldb = B.device_ref().stride(0);
  int ldc = C.device_ref().stride(0);
  int ldd = C.device_ref().stride(0);
  
  /*
  status = gemm_op({
    {M, N, K},
    {ptrA, lda},
    {ptrB, ldb},
    {ptrC, ldc},
    {ptrD, ldd},
    {alpha, beta},
  });
  */
  printf("\n\n");

  status = gemm_op({
    {M, N, K},
    A.device_ref(),
    B.device_ref(),
    C.device_ref(),
    C.device_ref(),
    {alpha, beta}
  });

  if (status != cutlass::Status::kSuccess){
    return -1;
  }

  using ElementA = cutlass::half_t;
  using LayoutA = cutlass::layout::RowMajor;
  const int AlignmentA = 128 / cutlass::sizeof_bits<ElementA>::value;

  using ElementB = cutlass::half_t;
  using LayoutB = cutlass::layout::ColumnMajor;
  const int AlignmentB = 128 / cutlass::sizeof_bits<ElementB>::value;

  using ElementC = cutlass::half_t;
  using LayoutC = cutlass::layout::ColumnMajor;

  using ElementAccumulator = float;
  //using ArchTag = cutlass::arch::Sm80;
  using ArchTag = cutlass::arch::Sm90;  // not for 80

  using OperatorClass = cutlass::arch::OpClassTensorOp;
  using TilesShape = cute::Shape<cute::_128, cute::_128, cute::_64>;
  using ClusterShape = cute::Shape<cute::_1, cute::_2, cute::_1>;
  using StageCountType = cutlass::gemm::collective::StageCountAuto;

  using KernelSchedule = cutlass::gemm::collective::KernelScheduleAuto;

  
  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
  ArchTag, OperatorClass, 
  ElementA, LayoutA, AlignmentA,
  ElementB, LayoutB, AlignmentB,
  ElementAccumulator,
  TilesShape, ClusterShape,
  StageCountType,
  KernelSchedule
  >::CollectiveOp;

  using CollectiveEpilogue = cutlass::epilogue::collective::DefaultEpilogue<
  cutlass::gemm::TagToStrideC_t<LayoutC>,
  cutlass::gemm::TagToStrideC_t<LayoutC>,
  cutlass::epilogue::thread::LinearCombination<ElementC, 1, ElementAccumulator, ElementAccumulator>
  >;

  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
  cute::Shape<int, int, int>,
  CollectiveMainloop,
  CollectiveEpilogue
  >;

  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
  cute::Shape<int, int, int>,
  CollectiveMainloop,
  CollectiveEpilogue
  >;

  using Gemm2 = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

  Gemm2 gemm_op2;
  cutlass::Status status2;

  cutlass::DeviceAllocation<typename Gemm2::ElementA> block_A;
  cutlass::DeviceAllocation<typename Gemm2::ElementB> block_B;
  cutlass::DeviceAllocation<typename Gemm2::ElementC> block_C;
  cutlass::DeviceAllocation<typename Gemm2::EpilogueOutputOp::ElementOutput> block_D;


  using StrideA = typename Gemm2::GemmKernel::StrideA;
  using StrideB = typename Gemm2::GemmKernel::StrideB;
  using StrideC = typename Gemm2::GemmKernel::StrideC;
  using StrideD = typename Gemm2::GemmKernel::StrideD;

  StrideA stride_A;
  StrideB stride_B;
  StrideC stride_C;
  StrideD stride_D;

  stride_A = make_cute_packed_stride(StrideA{}, cute::make_shape(M, K, cute::Int<1>{}));
  stride_B = make_cute_packed_stride(StrideB{}, cute::make_shape(N, K, cute::Int<1>{}));
  stride_C = make_cute_packed_stride(StrideC{}, cute::make_shape(M, N, cute::Int<1>{}));
  stride_D = make_cute_packed_stride(StrideD{}, cute::make_shape(M, N, cute::Int<1>{}));
  
  block_A.reset(M * K);
  block_B.reset(K * N);
  block_C.reset(M * N);
  block_D.reset(M * N);

  /*
  status = gemm_op2({
    cutlass::gemm::GemmUniversalMode::kGemm,
    {M, N, K},
    block_A.get(),
    stride_A,
    block_B.get(),
    stride_B,
    {block_C.get(), stride_C, block_D.get(), stride_D, {alpha, beta}}
  });

  if (status2 != cutlass::Status::kSuccess){
    return -1;
  }
  */
  

  return 0;
}