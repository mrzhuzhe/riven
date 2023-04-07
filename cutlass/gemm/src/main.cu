#include <iostream>
#include <cutlass/cutlass.h>
#include <cutlass/numeric_types.h>
#include <cutlass/core_io.h>

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

  return 0;
}