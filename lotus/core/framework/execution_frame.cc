#include "core/framework/execution_frame.h"
#include <sstream>
#include "core/framework/op_kernel.h"
#include "core/framework/session_state.h"
#include "core/framework/mem_pattern_planner.h"

namespace Lotus {

ExecutionFrame::ExecutionFrame(const std::unordered_map<std::string, MLValue>& feeds,
                               const std::vector<std::string>& output_names,
                               const std::vector<MLValue>& fetches,
                               const Lotus::SessionState& session_state)
    : session_state_(session_state), mem_patterns_(nullptr), planner_(nullptr) {
  Init(session_state.GetGraph(), feeds, output_names, fetches);
  InitArenas();

  // If the session enable memory pattern optimization
  // and we have execution plan generated, try to setup
  // memory pattern optimizaiton.
  if (session_state.GetEnableMemoryPattern() &&
      session_state.GetExecutionPlan()) {
    std::vector<TensorShape> input_shapes;
    bool all_tensors = true;
    for (auto it = feeds.begin(); it != feeds.end(); it++) {
      if (!(it->second.IsTensor())) {
        all_tensors = false;
        break;
      }
      auto& tensor = it->second.Get<Tensor>();
      input_shapes.push_back(tensor.Shape());
    }
    // if there is some traditional ml value type in inputs
    // disable the memory pattern optimization.
    if (all_tensors) {
      mem_patterns_ = session_state.GetMemoryPatternGroup(input_shapes);
      // if no existing patterns, generate one in this executionframe
      if (!mem_patterns_) {
        planner_ = std::make_unique<MLValuePatternPlanner>(session_state);
      } else {
        // pre-allocate the big chunk requested in memory pattern.
        // all the internal kernel's input/output tensors will be allocated on these buffer.
        for (int i = 0; i < mem_patterns_->locations.size(); i++) {
          LOTUS_ENFORCE(buffers_.find(mem_patterns_->locations[i]) == buffers_.end());
          ArenaPtr alloc = GetArena(mem_patterns_->locations[i]);
          //use Reserve to reserve a big chunk. This chunk could be unload when session closed.
          void* buffer = alloc->Reserve(mem_patterns_->patterns[i].peak_size());
          buffers_[mem_patterns_->locations[i]] = BufferUniquePtr(buffer, alloc);
        }
      }
    }
  }
}

Status ExecutionFrame::AllocateMLValueTensorSelfOwnBuffer(int mlvalue_index,
                                                          const MLDataType element_type,
                                                          const AllocatorInfo& location,
                                                          const TensorShape& shape) {
  LOTUS_ENFORCE(mlvalue_index >= 0 && mlvalue_index < all_values_.size());
  return AllocateMLValueTensorSelfOwnBufferHelper(mlvalue_index, element_type, location, shape);
}

Status ExecutionFrame::AllocateMLValueTensorSelfOwnBufferHelper(int mlvalue_index,
                                                                const MLDataType element_type,
                                                                const AllocatorInfo& location,
                                                                const TensorShape& shape) {
  auto p_mlvalue = &all_values_[mlvalue_index];
  if (p_mlvalue->IsAllocated()) {
    return Status::OK();
  }
  auto alloc = GetArena(location);
  auto size = element_type->Size() * shape.Size();

  // if we have pre-calcuated memory pattern, and the mlvalue is not output mlvalue
  // try to alloacted on pre-allocated big chunk.
  if (mem_patterns_ &&
      std::find(output_indices_.begin(), output_indices_.end(), mlvalue_index) == output_indices_.end()) {
    auto pattern = mem_patterns_->GetPatterns(location);
    if (pattern) {
      auto block = pattern->GetBlock(mlvalue_index);
      // if block not found, fall back to default behavior
      if (block) {
        auto it = buffers_.find(location);
        // if the block is not correct, log messsage then fall back to default behavior
        if (it != buffers_.end() && block->size_ == size) {
          void* buffer = it->second.get();
          AllocateTensorWithPreAllocateBufferHelper(p_mlvalue,
                                                    static_cast<void*>(static_cast<char*>(buffer) + block->offset_),
                                                    element_type,
                                                    location,
                                                    shape);
          return Status::OK();
        } else if (block->size_ != size) {
          LOGS_DEFAULT(WARNING) << "For mlvalue with index: " << mlvalue_index << ", block in memory pattern size is: "
                                << block->size_ << " but the actually size is: " << size << ", fall back to default allocation behavior";
        } else if (it == buffers_.end()) {
          LOGS_DEFAULT(WARNING) << "For mlvalue with index: " << mlvalue_index << ", block not found in target loation. "
                                                                                  " fall back to default allocation behavior";
        }
      }
    }
  }
  //no memory pattern, or the pattern is not correct.
  void* buffer = alloc->Reserve(size);
  std::unique_ptr<Tensor> p_tensor = std::make_unique<Tensor>(element_type,
                                                              shape,
                                                              buffer,
                                                              location,
                                                              alloc);
  p_mlvalue->Init(p_tensor.release(),
                  DataTypeImpl::GetType<Tensor>(),
                  DataTypeImpl::GetType<Tensor>()->GetDeleteFunc());
  // trace the memory allocation.
  // don't trace the memory allocation on string tensors, as it need
  // placement new, we don't suppport it in memory pattern optimizaiton.
  if (element_type != DataTypeImpl::GetType<std::string>())
    TraceAllocate(mlvalue_index, size);

  return Status::OK();
}

void ExecutionFrame::TraceAllocate(int mlvalue_idx, size_t size) {
  // don't trace the output tensors.
  if (planner_ &&
      std::find(output_indices_.begin(), output_indices_.end(), mlvalue_idx) == output_indices_.end()) {
    planner_->TraceAllocation(mlvalue_idx, size);
  }
}

Status ExecutionFrame::AllocateTensorWithSelfOwnBuffer(const int index,
                                                       const MLDataType element_type,
                                                       const AllocatorInfo& location,
                                                       const TensorShape& shape) {
  LOTUS_ENFORCE(index >= 0 && index < node_values_.size());
  return AllocateMLValueTensorSelfOwnBufferHelper(node_values_[index], element_type, location, shape);
}

Status ExecutionFrame::AllocateMLValueTensorPreAllocateBuffer(int mlvalue_index_to_allocate,
                                                              int mlvalue_index_reuse,
                                                              const MLDataType element_type,
                                                              const AllocatorInfo& location,
                                                              const TensorShape& shape) {
  LOTUS_ENFORCE(mlvalue_index_to_allocate >= 0 && mlvalue_index_to_allocate < all_values_.size());
  MLValue* p_mlvalue = &all_values_[mlvalue_index_to_allocate];

  LOTUS_ENFORCE(mlvalue_index_reuse >= 0 && mlvalue_index_reuse < all_values_.size());
  MLValue* p_mlvalue_reuse = &all_values_[mlvalue_index_reuse];

  Tensor* reuse_tensor = p_mlvalue_reuse->GetMutable<Tensor>();
  void* reuse_buffer = reuse_tensor->GetRaw();

  return AllocateTensorWithPreAllocateBufferHelper(p_mlvalue, reuse_buffer, element_type, location, shape);
}

Status ExecutionFrame::AllocateTensorWithPreAllocateBufferHelper(MLValue* p_mlvalue,
                                                                 void* pBuffer,
                                                                 const MLDataType element_type,
                                                                 const AllocatorInfo& location,
                                                                 const TensorShape& shape) {
  if (p_mlvalue->IsAllocated()) {
    return Common::Status::OK();
  }
  std::unique_ptr<Tensor> p_tensor = std::make_unique<Tensor>(element_type,
                                                              shape,
                                                              pBuffer,
                                                              location);
  p_mlvalue->Init(p_tensor.release(),
                  DataTypeImpl::GetType<Tensor>(),
                  DataTypeImpl::GetType<Tensor>()->GetDeleteFunc());

  return Common::Status::OK();
}

Status ExecutionFrame::AllocateTensorWithPreAllocateBuffer(const int offset,
                                                           void* pBuffer,
                                                           const MLDataType element_type,
                                                           const AllocatorInfo& location,
                                                           const TensorShape& shape) {
  LOTUS_ENFORCE(offset >= 0 && offset < node_values_.size());
  auto value = &all_values_[node_values_[offset]];
  return AllocateTensorWithPreAllocateBufferHelper(value, pBuffer, element_type, location, shape);
}

void ExecutionFrame::Release(const int offset) {
  LOTUS_ENFORCE(offset >= 0 && offset < node_offsets_.size());
  all_values_[node_values_[offset]] = MLValue();
  TraceFree(node_values_[offset]);
}

Common::Status AllocateTraditionalMLValue(MLValue* p_mlvalue,
                                          const NonTensorTypeBase* type,
                                          const MLValueAllocationParameters& parameters) {
  // right now we don't need any parameter for ml value creation,
  // keep it in api for extensibility
  UNUSED_PARAMETER(parameters);
  auto creator = type->GetCreateFunc();
  p_mlvalue->Init(creator(),
                  type,
                  type->GetDeleteFunc());
  return Status::OK();
}

// This method is not thread safe!
Common::Status ExecutionFrame::AllocateAsPerAllocationPlan(int mlvalue_index,
                                                           const MLValueAllocationParameters& parameters) {
  if (mlvalue_index < 0 || mlvalue_index >= all_values_.size())
    return Status(LOTUS, INVALID_ARGUMENT,
                  "Tried to allocated with invalid mlvalue index: " + std::to_string(mlvalue_index));
  const SequentialExecutionPlan* p_seq_exec_plan = session_state_.GetExecutionPlan();
  const auto& alloc_plan = p_seq_exec_plan->allocation_plan;
  LOTUS_ENFORCE(mlvalue_index >= 0 && mlvalue_index < alloc_plan.size());
  const auto& per_alloc_plan = alloc_plan[mlvalue_index];

  // TODO: both alloc_info and ml_data_type will be supplied by the allocation
  // plan later. This is a hack for now.
  auto alloc_info = per_alloc_plan.location;
  auto ml_type = per_alloc_plan.value_type;
  if (!ml_type->IsTensorType()) {
    return AllocateTraditionalMLValue(&all_values_[mlvalue_index],
                                      static_cast<const NonTensorTypeBase*>(ml_type),
                                      parameters);
  }

  // tensors
  auto ml_data_type = static_cast<const TensorTypeBase*>(ml_type)->GetElementType();

  AllocKind alloc_kind = per_alloc_plan.alloc_kind;
  switch (alloc_kind) {
    case AllocKind::kAllocate: {
      LOTUS_RETURN_IF_ERROR(AllocateMLValueTensorSelfOwnBuffer(mlvalue_index,
                                                               ml_data_type,
                                                               alloc_info,
                                                               parameters.tensor_shape));
      break;
    }
    case AllocKind::kReuse: {
      int reuse_mlvalue_index = per_alloc_plan.reused_buffer;
      LOTUS_RETURN_IF_ERROR(AllocateMLValueTensorPreAllocateBuffer(mlvalue_index,
                                                                   reuse_mlvalue_index,
                                                                   ml_data_type,
                                                                   alloc_info,
                                                                   parameters.tensor_shape));
      break;
    }
    default: {
      std::ostringstream ostr;
      ostr << "Invalid allocation kind: " << static_cast<std::underlying_type<AllocKind>::type>(alloc_kind);
      return Common::Status(Common::LOTUS, Common::FAIL, ostr.str());
    }
  }

  return Common::Status::OK();
}

void ExecutionFrame::Init(const LotusIR::Graph* graph,
                          const std::unordered_map<string, MLValue>& feeds,
                          const std::vector<string>& output_names,
                          const std::vector<MLValue>& fetches) {
  LOTUS_ENFORCE(graph);

  // 1. resize the node_offsets and all_value_ vector
  // We need to use the max index rather than number of nodes as we use Node.Index()
  // when inserting into node_offsets_
  auto max_node_index = graph->MaxNodeIndex();
  node_offsets_.resize(max_node_index);

  all_values_.resize(session_state_.GetMaxMLValueIdx() + 1);

  // 2. handle the weights.
  for (const auto& entry : session_state_.GetInitializedTensors()) {
    auto mlvalue_index = entry.first;
    all_values_[mlvalue_index] = entry.second;  // this copy should be cheap
  }

  // 3. handle feed in values
  for (auto it = feeds.begin(); it != feeds.end(); it++) {
    int mlvalue_idx;
    Common::Status status = session_state_.GetMLValueIdx(it->first, &mlvalue_idx);
    LOTUS_ENFORCE(status.IsOK());
    // we are sharing the underline tensor/object for MLValue
    all_values_[mlvalue_idx] = it->second;
  }

  // 4. Handle non-empty output vector
  // setup output_indices_, we dont' want to generate mem plan on output tensors.
  for (const auto& oname : output_names) {
    int mlvalue_idx;
    Common::Status status = session_state_.GetMLValueIdx(oname, &mlvalue_idx);
    LOTUS_ENFORCE(status.IsOK());
    output_indices_.push_back(mlvalue_idx);
  }

  if (!fetches.empty()) {
    // should've already verified this much before when Run() starts
    LOTUS_ENFORCE(output_names.size() == fetches.size(),
                  "output_names vector size: " + std::to_string(output_names.size()) +
                      " does not match that of fetches vector: " + std::to_string(fetches.size()));

    auto idx = 0;
    for (const auto& oname : output_names) {
      int mlvalue_idx;
      Common::Status status = session_state_.GetMLValueIdx(oname, &mlvalue_idx);
      LOTUS_ENFORCE(status.IsOK());
      all_values_[mlvalue_idx] = fetches.at(idx++);
      output_indices_.push_back(mlvalue_idx);
    }
  }

  // 5. set node args
  for (auto& node : graph->Nodes()) {
    LOTUS_ENFORCE(node.Index() < node_offsets_.size());
    node_offsets_[node.Index()] = static_cast<int>(node_values_.size());

    for (auto input_def : node.InputDefs()) {
      SetupNodeArg(input_def);
    }

    for (auto output_def : node.OutputDefs()) {
      SetupNodeArg(output_def);
    }
  }
}

void ExecutionFrame::SetupNodeArg(const LotusIR::NodeArg* arg) {
  LOTUS_ENFORCE(arg);
  auto& name = arg->Name();
  int index;
  Common::Status status = session_state_.GetMLValueIdx(name, &index);
  LOTUS_ENFORCE(status.IsOK());
  node_values_.push_back(index);
}

void ExecutionFrame::TraceFree(int mlvalue_idx) {
  // don't trace free on output tensors.
  if (planner_ &&
      std::find(output_indices_.begin(), output_indices_.end(), mlvalue_idx) == output_indices_.end()) {
    const SequentialExecutionPlan* p_seq_exec_plan = session_state_.GetExecutionPlan();
    const auto& alloc_plan = p_seq_exec_plan->allocation_plan;
    const auto& per_alloc_plan = alloc_plan.at(mlvalue_idx);

    // only trace tensors
    auto alloc_info = per_alloc_plan.location;
    auto ml_type = per_alloc_plan.value_type;
    if (ml_type->IsTensorType()) {
      // tensors
      auto ml_data_type = static_cast<const TensorTypeBase*>(ml_type)->GetElementType();
      // don't trace string tensors
      if (ml_data_type != DataTypeImpl::GetType<std::string>())
        planner_->TraceFree(mlvalue_idx);
    }
  }
}

// generate memory pattern based on the tracing of memory allocation/free in current execution
// return error if the planner is not setup.
Status ExecutionFrame::GeneratePatterns(MemoryPatternGroup* out) const {
  if (!planner_) {
    return Status(LOTUS, FAIL, "Memory pattern planner is not enabled on this execution framework.");
  }

  return planner_->GeneratePatterns(out);
}

void ExecutionFrame::InitArenas() {
  auto& alloc_mgr = AllocatorManager::Instance();

  // always have CPU arena allocator in execution frame
  std::set<AllocatorInfo> allocators_in_use = {alloc_mgr.GetArena(CPU).Info()};

  // The session may not have execution plan in tests
  auto p_exec_plan = session_state_.GetExecutionPlan();
  if (p_exec_plan)
    for (const auto& alloc_plan : p_exec_plan->allocation_plan)
      allocators_in_use.insert(alloc_plan.location);

  for (const auto& info : allocators_in_use) {
    if (info.type == AllocatorType::kArenaAllocator)
      arenas_.push_back(&alloc_mgr.GetArena(info.name, info.id));
  }
}

}  // namespace Lotus
