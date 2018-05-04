#include "core/framework/allocator.h"
#include "core/framework/allocatormgr.h"
#include <stdlib.h>
#include <sstream>

namespace Lotus {
REGISTER_DEVICE_ALLOCATOR(
    Cpu,
    [] { return std::make_unique<CPUAllocator>(); },
    std::numeric_limits<size_t>::max())  //TODO: set correct cpu memory limit?

void* CPUAllocator::Alloc(size_t size) {
  if (size <= 0)
    return nullptr;
  //todo: we should pin the memory in some case
  void* p = malloc(size);
  return p;
}

void CPUAllocator::Free(void* p) {
  //todo: unpin the memory
  free(p);
}

const AllocatorInfo& CPUAllocator::Info() const {
  static AllocatorInfo cpuAllocatorInfo(CPU, AllocatorType::kDeviceAllocator);
  return cpuAllocatorInfo;
}

}  // namespace Lotus
