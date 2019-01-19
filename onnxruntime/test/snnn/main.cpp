#include <iostream>
#if defined(__GNUC__)
#pragma GCC diagnostic push
#if __GNUC__ >= 6
#pragma GCC diagnostic ignored "-Wignored-attributes"
#endif
#pragma GCC diagnostic ignored "-Wunused-parameter"
#endif

#include "unsupported/Eigen/CXX11/Tensor"
#include "unsupported/Eigen/CXX11/ThreadPool"

#if defined(__GNUC__) && __GNUC__ >= 6
#pragma GCC diagnostic pop
#endif

using namespace Eigen;

int main(){
  Tensor<float, 2> a(30, 40);
  int v = 0;
  for(int j=0;j!=40;++j) {
    for (int i = 0; i != 30; ++i) {
       a(i,j) = ++v;
    }
  }

  Tensor<float, 2> b(30, 40);
  for(int j=0;j!=40;++j) {
    for (int i = 0; i != 30; ++i) {
      b(i,j) = ++v;
    }
  }
  Tensor<float, 2> c(30, 40);
  int thread_count = 4;
  ThreadPool tp(thread_count);
  Eigen::ThreadPoolDevice my_device(&tp, thread_count);
  c.device(my_device) = a + b;
  std::cout<<  c << std::endl;
  return 0;
}
