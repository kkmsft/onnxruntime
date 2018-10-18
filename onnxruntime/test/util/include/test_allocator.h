// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/session/allocator.h"
#include <atomic>
#include <stdexcept>
#include "core/framework/allocator_info.h"

#define ALLOCATOR_MACRO_COPY(TypeName) TypeName(const TypeName&) = delete

#define ALLOCATOR_MACRO_ASSIGNMENT(TypeName) TypeName& operator=(const TypeName&) = delete

#define ALLOCATOR_MACRO_COPY_AND_ASSIGNMENT(TypeName) \
  ALLOCATOR_MACRO_COPY(TypeName);                     \
  ALLOCATOR_MACRO_ASSIGNMENT(TypeName)

ONNXRUNTIME_ALLOCATOR_IMPL_BEGIN(MockedONNXRuntimeAllocator)
private:
std::atomic<size_t> memory_inuse;
ONNXRuntimeAllocatorInfo* cpuAllocatorInfo;

public:
ALLOCATOR_MACRO_COPY_AND_ASSIGNMENT(MockedONNXRuntimeAllocator);
ONNXRuntimeAllocatorInteface** Upcast() {
  return const_cast<ONNXRuntimeAllocatorInteface**>(&vtable_);
}
MockedONNXRuntimeAllocator() : memory_inuse(0) {
  ONNXRUNTIME_TRHOW_ON_ERROR(ONNXRuntimeCreateAllocatorInfo("Cpu", ONNXRuntimeDeviceAllocator, 0, ONNXRuntimeMemTypeDefault, &cpuAllocatorInfo));
}
~MockedONNXRuntimeAllocator() {
  ReleaseONNXRuntimeAllocatorInfo(cpuAllocatorInfo);
}
void* Alloc(size_t size) {
  constexpr size_t extra_len = sizeof(size_t);
  memory_inuse.fetch_add(size += extra_len);
  void* p = ::malloc(size);
  *(size_t*)p = size;
  return (char*)p + extra_len;
}
void Free(void* p) {
  constexpr size_t extra_len = sizeof(size_t);
  if (!p) return;
  p = (char*)p - extra_len;
  size_t len = *(size_t*)p;
  memory_inuse.fetch_sub(len);
  return ::free(p);
}
const ONNXRuntimeAllocatorInfo* Info() {
  return cpuAllocatorInfo;
}

void LeakCheck() {
  if (memory_inuse.load())
    throw std::runtime_error("memory leak!!!");
}

//The method returns the new reference count.
uint32_t AddRef() {
  return 0;
}
uint32_t Release() {
  return 0;
}
ONNXRUNTIME_ALLOCATOR_IMPL_END

//By default, Visual Studio applies internal linkage to constexpr variables even if the extern keyword is used.
constexpr ONNXRuntimeAllocatorInteface MockedONNXRuntimeAllocator::table_;
