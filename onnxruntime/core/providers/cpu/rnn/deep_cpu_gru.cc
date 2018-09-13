#include "core/providers/cpu/rnn/deep_cpu_gru.h"

#include <algorithm>
#include <future>
#include <stdexcept>

#include "core/common/logging/logging.h"
#include "core/framework/allocator.h"
#include "core/framework/tensor.h"

/*
ONNX_OPERATOR_SCHEMA(GRU)
    .SetDoc(R"DOC(
Computes an one-layer GRU. This operator is usually supported via some custom
implementation such as CuDNN.

Notations:

`X` - input tensor
`z` - update gate
`r` - reset gate
`h` - hidden gate
`t` - time step (t-1 means previous time step)
`W[zrh]` - W parameter weight matrix for update, reset, and hidden gates
`R[zrh]` - R recurrence weight matrix for update, reset, and hidden gates
`Wb[zrh]` - W bias vectors for update, reset, and hidden gates
`Rb[zrh]` - R bias vectors for update, reset, and hidden gates
`WB[zrh]` - W parameter weight matrix for backward update, reset, and hidden gates
`RB[zrh]` - R recurrence weight matrix for backward update, reset, and hidden gates
`WBb[zrh]` - W bias vectors for backward update, reset, and hidden gates
`RBb[zrh]` - R bias vectors for backward update, reset, and hidden gates
`H` - Hidden state
`num_directions` - 2 if direction == bidirectional else 1

Activation functions:

  Relu(x)                - max(0, x)
  Tanh(x)                - (1 - e^{-2x})/(1 + e^{-2x})
  Sigmoid(x)             - 1/(1 + e^{-x})

  (NOTE: Below are optional)
  Affine(x)              - alpha*x + beta
  LeakyRelu(x)           - x if x >= 0 else alpha * x
  ThresholdedRelu(x)     - x if x >= alpha else 0
  ScaledTanh(x)          - alpha*Tanh(beta*x)
  HardSigmoid(x)         - min(max(alpha*x + beta, 0), 1)
  Elu(x)                 - x if x >= 0 else alpha*(e^x - 1)
  Softsign(x)            - x/(1 + |x|)
  Softplus(x)            - log(1 + e^x)

Equations (Default: f=Sigmoid, g=Tanh):
  - zt = f(Xt*(Wz^T) + Ht-1*Rz + Wbz + Rbz)
  - rt = f(Xt*(Wr^T) + Ht-1*Rr + Wbr + Rbr)
  - ht = g(Xt*(Wh^T) + (rt (.) Ht-1)*Rh + Rbh + Wbh) # default, when linear_before_reset = 0
  - ht = g(Xt*(Wh^T) + (rt (.) (Ht-1*Rh + Rbh) + Wbh) # when linear_before_reset != 0
  - Ht = (1 - zt) (.) ht + zt (.) Ht-1
)DOC")
    .SinceVersion(3)
    .Attr("direction", "Specify if the RNN is forward, reverse, or bidirectional. "
                "Must be one of forward (default), reverse, or bidirectional.",
                AttributeProto::STRING,
                std::string("forward"))
    .Attr("hidden_size", "Number of neurons in the hidden layer", AttributeProto::INT, OPTIONAL)
    .Attr("activations", "A list of 2 (or 4 if bidirectional) activation functions "
                "for update, reset, and hidden gates. The activation functions must be one "
                "of the activation functions specified above. Optional: See the equations "
                "for default if not specified.",
                AttributeProto::STRINGS,
                OPTIONAL)
    .Attr("activation_alpha",
                "Optional scaling values used by some activation functions. The values "
                "are consumed in the order of activation functions, for example (f, g, h) "
                "in LSTM.",
                AttributeProto::FLOATS,
                OPTIONAL)
    .Attr("activation_beta",
                "Optional scaling values used by some activation functions. The values "
                "are consumed in the order of activation functions, for example (f, g, h) "
                "in LSTM.",
                AttributeProto::FLOATS,
                OPTIONAL)
    .Attr("output_sequence",
                "The sequence output for the hidden is optional if 0. Default 0.",
                AttributeProto::INT,
                static_cast<int64_t>(0));
    .Attr("clip", "Cell clip threshold. Clipping bounds the elements of a tensor "
                "in the range of [-threshold, +threshold] and is applied to the input "
                "of activations. No clip if not specified.", AttributeProto::FLOAT, OPTIONAL)
    .Attr("linear_before_reset", "When computing the output of the hidden gate, "
                "apply the linear transformation before multiplying by the output of the "
                "reset gate.",
                AttributeProto::INT,
                static_cast<int64_t>(0))
    .Input(0, "X",
                "The input sequences packed (and potentially padded) into one 3-D "
                "tensor with the shape of `[seq_length, batch_size, input_size]`.", "T")
    .Input(1, "W",
                "The weight tensor for the gates. Concatenation of `W[zrh]` and `WB[zrh]` "
                "(if bidirectional) along dimension 0. This tensor has shape "
                "`[num_directions, 3*hidden_size, input_size]`.", "T")
    .Input(2, "R",
                "The recurrence weight tensor. Concatenation of `R[zrh]` and `RB[zrh]` "
                "(if bidirectional) along dimension 0. This tensor has shape "
                "`[num_directions, 3*hidden_size, hidden_size]`.", "T")
    .Input(3, "B",
                "The bias tensor for the gates. Concatenation of `[Wb[zrh], Rb[zrh]]` and "
                "`[WBb[zrh], RBb[zrh]]` (if bidirectional) along dimension 0. This tensor "
                "has shape `[num_directions, 6*hidden_size]`. Optional: If not specified "
                "- assumed to be 0", "T",
                OpSchema::Optional)        
    .Input(4, "sequence_lens",
                "Optional tensor specifying lengths of the sequences in a batch. "
                "If not specified - assumed all sequences in the batch to have "
                "length `seq_length`. It has shape `[batch_size]`.", "T1",
                OpSchema::Optional)
    .Input(5, "initial_h",
                "Optional initial value of the hidden. If not specified - assumed "
                "to be 0. It has shape `[num_directions, batch_size, hidden_size]`.",
                "  T", OpSchema::Optional)
    .Output(0, "Y",
                "A tensor that concats all the intermediate output values of the hidden. "
                "It has shape `[seq_length, num_directions, batch_size, hidden_size]`. ",
                "T", OpSchema::Optional);
    .Output(1, "Y_h",
                "The last output value of the hidden. It has shape "
                "`[num_directions, batch_size, hidden_size]`.", "T", OpSchema::Optional);
    .TypeConstraint("T", { "tensor(float16)", "tensor(float)", "tensor(double)" },
                "Constrain input and output types to float tensors.")
    .TypeConstraint("T1", { "tensor(int32)" }, "Constrain seq_lens to integer tensor.");
*/

namespace onnxruntime {

ONNX_CPU_OPERATOR_KERNEL(
    GRU,
    7,
    KernelDefBuilder().TypeConstraint("T", {DataTypeImpl::GetTensorType<float>(),
                                            DataTypeImpl::GetTensorType<double>()})
        .TypeConstraint("T1", DataTypeImpl::GetTensorType<int32_t>()),
    DeepCpuGruOp);

using namespace Rnn::detail;

// internal helper code
namespace detail {

template <typename T>
static void TransposeCopy(gsl::span<const T> src, size_t src_offset,
                          const int src_nrow, const int src_ncol,
                          gsl::span<T> dst);

/// The class represents DeepCPU implementation of a gated recurrent unit (GRU) operator.
/// For details, refer to http://aka.ms/dl-optimization/.
template <typename T>
class UniDirectionalGru {
 public:
  UniDirectionalGru(AllocatorPtr allocator,
                    const Logging::Logger& logger,
                    const int seq_length,
                    const int batch_size,
                    const int input_size,
                    const int hidden_size,
                    const bool linear_before_reset,
                    Direction direction,
                    const gsl::span<const T>& input_weights,
                    const gsl::span<const T>& recurrent_weights,
                    const gsl::span<const T>& bias,
                    const gsl::span<const T>& initial_hidden_state,
                    const ActivationFuncs::Entry& activation_func_f,
                    const ActivationFuncs::Entry& activation_func_g,
                    const float clip,
                    TaskThreadPool& ttp);

  void Compute(const gsl::span<const T>& inputs,
               const gsl::span<const int>& sequence_lengths,
               const int num_directions,
               gsl::span<T>& outputs,
               gsl::span<T>& final_hidden_state);

  ~UniDirectionalGru() = default;

 private:
  AllocatorPtr allocator_;
  const Logging::Logger& logger_;
  TaskThreadPool& ttp_;

  int seq_length_;
  int batch_size_;
  int input_size_;
  int hidden_size_;
  bool linear_before_reset_;

  const float clip_;

  Direction direction_;
  bool use_bias_;
  bool batch_parallel_;

  int input_num_threads_ = -1;
  int hidden_num_threads_ = -1;

  IAllocatorUniquePtr<T> input_weightsZRH_ptr_, recurrent_weightsZR_ptr_, recurrent_weightsH_ptr_;
  IAllocatorUniquePtr<T> outputZRH_ptr_;

  gsl::span<T> input_weightsZRH_, recurrent_weightsZR_, recurrent_weightsH_;
  gsl::span<T> outputZRH_;

  IAllocatorUniquePtr<T> cur_h_ptr_;
  IAllocatorUniquePtr<T> batched_hidden0_ptr_;
  IAllocatorUniquePtr<int> sequence_lengths_ptr_;
  gsl::span<T> cur_h_;
  gsl::span<T> batched_hidden0_;
  gsl::span<int> sequence_lengths_;

  // Wb[zr] and Rb[zr] can always be added together upfront, and repeated to match the batch size for
  // faster GEMM calculations, so these two members are all the
  // Wb[z] + Rb[z] values added together, repeated batch_size_ times
  IAllocatorUniquePtr<T> batched_bias_WRz_ptr_, batched_bias_WRr_ptr_;
  gsl::span<T> batched_bias_WRz_, batched_bias_WRr_;

  // Wbh and Rbh can only be combined upfront if linear_before_reset_ is false
  IAllocatorUniquePtr<T> batched_bias_WRh_ptr_;
  gsl::span<T> batched_bias_WRh_;

  // if linear_before_reset_ is true, we need to setup Wbh and Rbh separately
  IAllocatorUniquePtr<T> batched_bias_Wh_ptr_, batched_bias_Rh_ptr_;
  gsl::span<T> batched_bias_Wh_, batched_bias_Rh_;

  IAllocatorUniquePtr<T> linear_output_ptr_;
  gsl::span<T> linear_output_;

  IAllocatorUniquePtr<T> inputs_reverse_ptr_;
  IAllocatorUniquePtr<T> outputs_reverse_ptr_;
  gsl::span<T> inputs_reverse_;
  gsl::span<T> outputs_reverse_;

  deepcpu::ClipWithBiasFuncPtr clip_with_bias_ptr_ = nullptr;

  float zr_alpha_ = 0.f, zr_beta_ = 0.f;
  float h_alpha_ = 0.f, h_beta_ = 0.f;

  deepcpu::GruResetGateFuncPtr reset_gate_ = nullptr;
  deepcpu::ActivationFuncPtr update_gate_ = nullptr;
  deepcpu::GruOutputGateFuncPtr output_gate_ = nullptr;

  void AllocateBuffers();
  void SetNumThreads();
};
}  // namespace detail

// #define DUMP_MATRIXES to provide lots of diagnostic output
#if defined(DUMP_MATRIXES)
#define DumpMatrix(...) ::onnxruntime::Rnn::detail::DumpMatrixImpl(__VA_ARGS__)
#else
#define DumpMatrix(...) ((void)0)
#endif

Status DeepCpuGruOp::Compute(OpKernelContext* context) const {
  const Tensor& X = *context->Input<Tensor>(0);  // inputs. [seq_length, batch_size, input_size]

  Status status;
  // auto& logger = context->Logger();

  auto data_type = X.DataType();
  if (data_type == DataTypeImpl::GetType<float>())
    status = ComputeImpl<float>(*context);
  else if (data_type == DataTypeImpl::GetType<double>()) {
    /* Need to update all the helpers to support double...
    status = ComputeImpl<double>(*context); */
    LOTUS_NOT_IMPLEMENTED("GRU operator does not support double yet");
  } else
    LOTUS_THROW("Invalid data type for GRU operator of ", data_type);

  return status;
}

template <typename T>
Status DeepCpuGruOp::ComputeImpl(OpKernelContext& context) const {
  auto& logger = context.Logger();

  const Tensor& X = *context.Input<Tensor>(0);  // inputs. [seq_length, batch_size, input_size]
  const Tensor& W = *context.Input<Tensor>(1);  // weights. [num_directions, 3*hidden_size, input_size]
  const Tensor& R = *context.Input<Tensor>(2);  // recurrence weights. [num_directions, 3*hidden_size, hidden_size]

  // optional
  const Tensor* B = OptionalInput(context, 3);              // bias. [num_directions, 6*hidden_size]
  const Tensor* sequence_lens = OptionalInput(context, 4);  // [batch_size]
  const Tensor* initial_h = OptionalInput(context, 5);      // initial hidden. [num_directions, batch_size, hidden_size]

  auto& X_shape = X.Shape();

  int seq_length = gsl::narrow<int>(X_shape[0]);
  int batch_size = gsl::narrow<int>(X_shape[1]);
  int input_size = gsl::narrow<int>(X_shape[2]);

  auto status = ValidateCommonRnnInputs(X, W, R, B, 3, sequence_lens, initial_h, num_directions_, hidden_size_);
  LOTUS_RETURN_IF_ERROR(status);

  // GRU outputs are optional but must be in the same order
  std::vector<int64_t> Y_dims{seq_length, num_directions_, batch_size, hidden_size_};
  Tensor* Y = context.Output(/*index*/ 0, Y_dims);  // TODO: Adjust for however optional outputs gets implemented

  std::vector<int64_t> Y_h_dims{num_directions_, batch_size, hidden_size_};
  Tensor* Y_h = context.Output(/*index*/ 1, Y_h_dims);

  AllocatorPtr alloc;
  status = context.GetTempSpaceAllocator(&alloc);
  LOTUS_RETURN_IF_ERROR(status);
  gsl::span<const T> input_weights = W.DataAsSpan<T>();
  gsl::span<const T> recurrent_weights = R.DataAsSpan<T>();
  gsl::span<const T> bias = B != nullptr ? B->DataAsSpan<T>() : gsl::span<const T>();

  // spans for first direction
  const size_t input_weights_size_per_direction = 3 * hidden_size_ * input_size;
  const size_t recurrent_weights_size_per_direction = 3 * hidden_size_ * hidden_size_;
  const size_t bias_size_per_direction = 6 * hidden_size_;

  gsl::span<const T> input_weights_1 = input_weights.subspan(0, input_weights_size_per_direction);
  gsl::span<const T> recurrent_weights_1 = recurrent_weights.subspan(0, recurrent_weights_size_per_direction);
  gsl::span<const T> bias_1 = bias.empty() ? bias : bias.subspan(0, bias_size_per_direction);

  gsl::span<const T> input = X.DataAsSpan<T>();
  gsl::span<const int> sequence_lens_span = sequence_lens != nullptr ? sequence_lens->DataAsSpan<int>()
                                                                     : gsl::span<const int>();

  const size_t initial_hidden_size_per_direction = batch_size * hidden_size_;
  gsl::span<const T> initial_hidden = initial_h != nullptr ? initial_h->DataAsSpan<T>() : gsl::span<const T>();
  gsl::span<const T> initial_hidden_1 = initial_hidden.empty()
                                            ? initial_hidden
                                            : initial_hidden.subspan(0, initial_hidden_size_per_direction);

  // output shape is [seq_length, num_directions, batch_size, hidden_size]
  // so it's not a case of all the output for one direction being first.
  // due to that we can only easily check that the end of the output for each direction is valid.
  const size_t output_size = Y != nullptr ? Y->Shape().Size() : 0;
  const size_t per_direction_offset = batch_size * hidden_size_;
  gsl::span<T> output = Y != nullptr ? Y->MutableDataAsSpan<T>() : gsl::span<T>();
  gsl::span<T> output_1 = output.empty()
                              ? output
                              : output.subspan(0, output_size - (num_directions_ - 1) * per_direction_offset);

  // UniDirectionalGru needs somewhere to write output, so even if we aren't returning Y_h
  // we provide an appropriately sized buffer for that purpose.
  const size_t hidden_output_size_per_direction = batch_size * hidden_size_;
  IAllocatorUniquePtr<T> local_hidden_output;
  gsl::span<T> hidden_output =
      Y_h ? Y_h->MutableDataAsSpan<T>()
          : Allocate<T>(alloc, hidden_output_size_per_direction * num_directions_, local_hidden_output);

  gsl::span<T> hidden_output_1 = hidden_output.subspan(0, hidden_output_size_per_direction);

  std::unique_ptr<detail::UniDirectionalGru<T>> fw;
  std::unique_ptr<detail::UniDirectionalGru<T>> bw;

  if (direction_ == Direction::kBidirectional) {
    // spans for second direction
    gsl::span<const T> input_weights_2 = input_weights.subspan(input_weights_size_per_direction,
                                                               input_weights_size_per_direction);
    gsl::span<const T> recurrent_weights_2 = recurrent_weights.subspan(recurrent_weights_size_per_direction,
                                                                       recurrent_weights_size_per_direction);
    gsl::span<const T> bias_2 = bias.empty() ? bias : bias.subspan(bias_size_per_direction, bias_size_per_direction);

    gsl::span<const T> initial_hidden_2 = initial_hidden.empty()
                                              ? initial_hidden
                                              : initial_hidden.subspan(initial_hidden_size_per_direction,
                                                                       initial_hidden_size_per_direction);
    gsl::span<T> output_2 = output.empty()
                                ? output
                                : output.subspan(per_direction_offset, output_size - per_direction_offset);

    gsl::span<T> hidden_output_2 = hidden_output.subspan(hidden_output_size_per_direction,
                                                         hidden_output_size_per_direction);

    fw = std::make_unique<detail::UniDirectionalGru<T>>(
        alloc, logger,
        seq_length, batch_size, input_size, hidden_size_, linear_before_reset_, Direction::kForward,
        input_weights_1, recurrent_weights_1, bias_1, initial_hidden_1,
        activation_funcs_.Entries()[0],
        activation_funcs_.Entries()[1],
        clip_, ttp_);

    bw = std::make_unique<detail::UniDirectionalGru<T>>(
        alloc, logger,
        seq_length, batch_size, input_size, hidden_size_, linear_before_reset_, Direction::kReverse,
        input_weights_2, recurrent_weights_2, bias_2, initial_hidden_2,
        activation_funcs_.Entries()[2],
        activation_funcs_.Entries()[3],
        clip_, ttp_);

    // TODO: Investigate running two calls to Compute in parallel
    // TODO: Or alternatively if we split out the buffers into a separate class we could re-use them for both
    // forward and reverse calculations if they're done sequentially.
    fw->Compute(input, sequence_lens_span, num_directions_, output_1, hidden_output_1);
    bw->Compute(input, sequence_lens_span, num_directions_, output_2, hidden_output_2);

  } else {
    fw = std::make_unique<detail::UniDirectionalGru<T>>(
        alloc, logger,
        seq_length, batch_size, input_size, hidden_size_, linear_before_reset_, direction_,
        input_weights_1, recurrent_weights_1, bias_1, initial_hidden_1,
        activation_funcs_.Entries()[0],
        activation_funcs_.Entries()[1],
        clip_, ttp_);

    fw->Compute(input, sequence_lens_span, num_directions_, output_1, hidden_output_1);
  }

  if (!output.empty())
    DumpMatrix("Y", output.data(), seq_length * num_directions_ * batch_size, hidden_size_);

  DumpMatrix("Y_h", hidden_output.data(), num_directions_ * batch_size, hidden_size_);

  return Status::OK();
}

//
// Implementation of internal helper code
namespace detail {

template <typename T>
static void TransposeCopy(gsl::span<const T> src, size_t src_offset,
                          const int src_nrow, const int src_ncol,
                          gsl::span<T> dst) {
  const int dst_nrow = src_ncol;
  const int dst_ncol = src_nrow;
  auto out_offset = 0;
  for (int i = 0; i < dst_nrow; i++) {
    auto in_offset = src_offset + i;
    for (int j = 0; j < dst_ncol; j++) {
      // dst[i * dst_ncol + j] = src[src_offset + j * src_ncol + i];
      dst[out_offset + j] = src[in_offset];
      in_offset += src_ncol;
    }

    out_offset += dst_ncol;
  }
}

template <typename T>
UniDirectionalGru<T>::UniDirectionalGru(AllocatorPtr allocator,
                                        const Logging::Logger& logger,
                                        const int seq_length,
                                        const int batch_size,
                                        const int input_size,
                                        const int hidden_size,
                                        const bool linear_before_reset,
                                        Direction direction,
                                        const gsl::span<const T>& input_weights,
                                        const gsl::span<const T>& recurrent_weights,
                                        const gsl::span<const T>& bias,
                                        const gsl::span<const T>& initial_hidden_state,
                                        const ActivationFuncs::Entry& activation_func_f,
                                        const ActivationFuncs::Entry& activation_func_g,
                                        const float clip,
                                        TaskThreadPool& ttp)
    : allocator_(allocator),
      logger_(logger),
      seq_length_(seq_length),
      batch_size_(batch_size),
      input_size_(input_size),
      hidden_size_(hidden_size),
      linear_before_reset_(linear_before_reset),
      direction_(direction),
      use_bias_(!bias.empty()),
      clip_(clip),
      ttp_(ttp) {
  //
  clip_with_bias_ptr_ = use_bias_ ? deepcpu::clip_add_bias : deepcpu::clip_ignore_bias;

  // setup activation function pointers and alpha/beta values to use with them
  reset_gate_ = deepcpu::GruResetGateFuncByName(activation_func_f.name);
  update_gate_ = deepcpu::ActivationFuncByName(activation_func_f.name);
  output_gate_ = deepcpu::GruOutputGateFuncByName(activation_func_g.name);

  zr_alpha_ = activation_func_f.alpha;
  zr_beta_ = activation_func_f.beta;
  h_alpha_ = activation_func_g.alpha;
  h_beta_ = activation_func_g.beta;

  DumpMatrix("input_weights", input_weights.data(), 3 * hidden_size_, input_size_);
  DumpMatrix("recurrent_weights", recurrent_weights.data(), 3 * hidden_size_, hidden_size_);

  input_weightsZRH_ = Allocate(allocator_, 3 * input_size_ * hidden_size_, input_weightsZRH_ptr_);
  recurrent_weightsZR_ = Allocate(allocator_, 2 * hidden_size_ * hidden_size_, recurrent_weightsZR_ptr_);
  recurrent_weightsH_ = Allocate(allocator_, hidden_size_ * hidden_size_, recurrent_weightsH_ptr_);

  TransposeCopy(input_weights, 0, 3 * hidden_size_, input_size_, input_weightsZRH_);
  TransposeCopy(recurrent_weights, 0, 2 * hidden_size_, hidden_size_, recurrent_weightsZR_);
  TransposeCopy(recurrent_weights, 2 * hidden_size_ * hidden_size_, hidden_size_, hidden_size_, recurrent_weightsH_);

  DumpMatrix("input_weightsZRH_", input_weightsZRH_.data(), input_size_, 3 * hidden_size_);
  DumpMatrix("recurrent_weightsZR_[Z]", recurrent_weightsZR_.data(), hidden_size_, hidden_size_, 0, 2 * hidden_size_);
  DumpMatrix("recurrent_weightsZR_[R]", recurrent_weightsZR_.data(), hidden_size_, hidden_size_, hidden_size_, 2 * hidden_size_);
  DumpMatrix("recurrent_weightsH_", recurrent_weightsH_.data(), hidden_size_, hidden_size_);

  SetNumThreads();
  AllocateBuffers();

  if (use_bias_) {
    auto bias_Wz = bias.subspan(0 * hidden_size_, hidden_size_);
    auto bias_Wr = bias.subspan(1 * hidden_size_, hidden_size_);
    auto bias_Wo = bias.subspan(2 * hidden_size_, hidden_size_);
    auto bias_Rz = bias.subspan(3 * hidden_size_, hidden_size_);
    auto bias_Rr = bias.subspan(4 * hidden_size_, hidden_size_);
    auto bias_Ro = bias.subspan(5 * hidden_size_, hidden_size_);

    // add Wb[zr] and Rb[zr] and replicate so we have batch_size_ copies of the result
    auto combine_and_replicate = [&](gsl::span<const T>& bias_w,
                                     gsl::span<const T>& bias_r,
                                     gsl::span<T>& output) {
      // add once
      for (int i = 0; i < hidden_size_; ++i) {
        output[i] = bias_w[i] + bias_r[i];
      }

      // replicate what we just wrote to the start of the output span so we have batch_size_ copies
      auto values = output.cbegin();
      IGNORE_RETURN_VALUE(RepeatVectorToConstructArray(values, values + hidden_size_,
                                                       output.begin() + hidden_size_,  // skip the first batch
                                                       batch_size_ - 1));              // and replicate batch size - 1 times
    };

    // we can always combine the z and r weights
    combine_and_replicate(bias_Wz, bias_Rz, batched_bias_WRz_);
    combine_and_replicate(bias_Wr, bias_Rr, batched_bias_WRr_);

    // how we treat the h weight depends on whether linear_before_reset_ is set
    if (linear_before_reset_) {
      // need to replicate Wb[o] and Rb[o] separately
      IGNORE_RETURN_VALUE(RepeatVectorToConstructArray(bias_Wo.cbegin(), bias_Wo.cend(), batched_bias_Wh_.begin(), batch_size_));
      IGNORE_RETURN_VALUE(RepeatVectorToConstructArray(bias_Ro.cbegin(), bias_Ro.cend(), batched_bias_Rh_.begin(), batch_size_));
    } else {
      combine_and_replicate(bias_Wo, bias_Ro, batched_bias_WRh_);
    }
  }

  if (!initial_hidden_state.empty()) {
    std::copy(initial_hidden_state.cbegin(), initial_hidden_state.cend(), batched_hidden0_.begin());
  }
}

template <typename T>
void UniDirectionalGru<T>::Compute(const gsl::span<const T>& inputs_arg,
                                   const gsl::span<const int>& sequence_lengths_arg,
                                   const int num_directions,
                                   gsl::span<T>& outputs,
                                   gsl::span<T>& final_hidden_state) {
  using span_T_const_iter = typename gsl::span<T>::const_iterator;
  using span_T_iter = typename gsl::span<T>::iterator;

  // copy inputs_arg as we may change it to point to inputs_reverse_
  gsl::span<const T> inputs = inputs_arg;
  gsl::span<const int> sequence_lengths = sequence_lengths_arg;

  // if sequence lengths weren't provided, use internal array and init all to seq_length
  if (sequence_lengths.empty()) {
    sequence_lengths_ = Allocate(allocator_, batch_size_, sequence_lengths_ptr_, true, seq_length_);
    sequence_lengths = sequence_lengths_;
  }

  DumpMatrix("Inputs", inputs.data(), seq_length_ * batch_size_, input_size_);

  gsl::span<T> original_outputs = outputs;
  const bool output_sequence = !outputs.empty();

  if (direction_ == kReverse) {
    ReverseSequence(inputs, inputs_reverse_, sequence_lengths, seq_length_, batch_size_, input_size_, 1);
    // DumpMatrix("Reversed inputs", inputs_reverse_.data(), seq_length_ * batch_size_, input_size_);

    inputs = inputs_reverse_;

    if (output_sequence) {
      outputs = outputs_reverse_;
    }
  }

  // Calculate the max and min length
  int32_t max_sequence_length = *std::max_element(sequence_lengths.cbegin(), sequence_lengths.cend());

  // macro 'min' coming from windows headers breaks 'std::min'
#undef min
  int32_t min_sequence_length = std::min(seq_length_, *std::min_element(sequence_lengths.cbegin(),
                                                                        sequence_lengths.cend()));

  const int hidden_size_x2 = 2 * hidden_size_;
  const int hidden_size_x3 = 3 * hidden_size_;
  const int total_rows = max_sequence_length * batch_size_;

  float alpha = 1.0f;
  float beta = 0.0f;  // zero out outputZRH_ when calling ComputeGemm.

#if defined(HAVE_PARALLELIZED_GEMM)
  // apply weights to all the inputs
  ComputeGemm(total_rows, hidden_size_x3, input_size_, alpha,
              inputs.cbegin(), inputs.cend(),
              input_size_,
              input_weightsZRH_.cbegin(), input_weightsZRH_.cend(),
              hidden_size_x3, beta,
              outputZRH_.begin(), outputZRH_.end(),
              hidden_size_x3);
#else
  int fused_input_rows = total_rows / input_num_threads_;
  if (total_rows % input_num_threads_ != 0)
    fused_input_rows++;

  // lambda to apply weights to all the inputs
  auto input_gemm = [&](int row) {
    int local_fused_input_rows = fused_input_rows;

    // last row may go beyond the end if rows didn't divide equally by input_num_threads_,
    // so limit to remaining rows.
    if ((row + fused_input_rows) > total_rows)
      local_fused_input_rows = total_rows - row;

    ComputeGemm(local_fused_input_rows, hidden_size_x3, input_size_, alpha,
                inputs.cbegin() + row * input_size_, inputs.cend(),
                input_size_,
                input_weightsZRH_.cbegin(), input_weightsZRH_.cend(),
                hidden_size_x3, beta,
                outputZRH_.begin() + row * hidden_size_x3, outputZRH_.end(),
                hidden_size_x3);
  };

  // Xt * W[zrh] for all rows are written to outputZRH_
  ExecuteLambdaInParallel("Applying weights to inputs", input_gemm, total_rows, fused_input_rows,
                          ttp_, logger_);
#endif

  DumpMatrix("inputs with weights applied", outputZRH_.data(), seq_length_ * batch_size_ * 3, hidden_size_);

  // set to 1 so the weighted inputs in outputZRH_ are added to the result in the next call to ComputeGemm
  beta = 1.0f;

  int fused_hidden_rows = batch_size_ / hidden_num_threads_;
  if (batch_size_ % hidden_num_threads_ != 0)
    fused_hidden_rows++;

  // output shape is [seq_length, num_directions, batch_size, hidden_size]
  // if we are doing 2 directions and this is the forward pass we're writing to the real output so
  // need to include num_directions in the step length.
  // we do not need to do that if there are two directions and we're doing the backwards pass as we
  // are writing to a temporary buffer (as outputs == outputs_reverse_) which is later copied
  // to the real output by ReverseSequence. this later copy includes num_directions in the step length.
  int output_step_length = batch_size_ * hidden_size_;
  if (direction_ == kForward && num_directions == 2)
    output_step_length = 2 * batch_size_ * hidden_size_;

  // convenience end iterators we use in the loops below to detect any bounds issues
  span_T_const_iter batched_bias_WRz_local_end = batched_bias_WRz_.cend();
  span_T_const_iter batched_bias_WRr_local_end = batched_bias_WRr_.cend();
  span_T_const_iter batched_bias_Wh_local_end = batched_bias_Wh_.cend();
  span_T_const_iter batched_bias_Rh_local_end = batched_bias_Rh_.cend();
  span_T_const_iter batched_bias_WRh_local_end = batched_bias_WRh_.cend();

  if (batch_parallel_) {
    // lambda executed by TaskThreadPool
    auto hidden_gemm_and_activations = [&](const int row) {
      //handling boundaries
      int local_fused_hidden_rows = fused_hidden_rows;
      if ((row + fused_hidden_rows) > batch_size_)
        local_fused_hidden_rows = batch_size_ - row;

      size_t out_added_offset;
      span_T_const_iter prev_Ht = batched_hidden0_.cbegin() + row * hidden_size_;  // Ht-1
      span_T_const_iter prev_Ht_end = batched_hidden0_.cend();
      span_T_iter cur_h_local = cur_h_.begin() + row * hidden_size_;
      span_T_iter cur_h_local_end = cur_h_.end();
      span_T_iter linear_output_local;
      span_T_iter linear_output_local_end;

      span_T_const_iter batched_bias_WRz_local;
      span_T_const_iter batched_bias_WRr_local;
      span_T_const_iter batched_bias_WRh_local;
      span_T_const_iter batched_bias_Wh_local;
      span_T_const_iter batched_bias_Rh_local;

      if (use_bias_) {
        batched_bias_WRz_local = batched_bias_WRz_.cbegin() + row * hidden_size_;
        batched_bias_WRr_local = batched_bias_WRr_.cbegin() + row * hidden_size_;

        if (linear_before_reset_) {
          batched_bias_Wh_local = batched_bias_Wh_.cbegin() + row * hidden_size_;
          batched_bias_Rh_local = batched_bias_Rh_.cbegin() + row * hidden_size_;
          linear_output_local = linear_output_.begin() + row * hidden_size_;
          linear_output_local_end = linear_output_.end();
        } else {
          batched_bias_WRh_local = batched_bias_WRh_.cbegin() + row * hidden_size_;
        }
      }

      for (int step = 0; step < max_sequence_length; step++) {
        const std::string row_str = " [row=" + std::to_string(row) + ",seqno=" + std::to_string(step) + "]";

        DumpMatrix("Ht-1" + row_str, &*prev_Ht, local_fused_hidden_rows, hidden_size_);

        out_added_offset = (step * batch_size_ + row) * hidden_size_x3;

        // calculate Ht-1*R[zr], and add to the weighted inputs that are in outputZRH_
        ComputeGemm(local_fused_hidden_rows, hidden_size_x2, hidden_size_, alpha,
                    prev_Ht, prev_Ht_end,
                    hidden_size_,
                    recurrent_weightsZR_.cbegin(), recurrent_weightsZR_.cend(),
                    hidden_size_x2, beta,
                    outputZRH_.begin() + out_added_offset, outputZRH_.end(),
                    hidden_size_x3);

        DumpMatrix("Xt*(W[zr]^T) + Ht-1 * R[zr]" + row_str,
                   outputZRH_.data() + out_added_offset, local_fused_hidden_rows, hidden_size_x2, 0, hidden_size_x3);

        if (linear_before_reset_) {
          // copy Rbh to linear output
          std::copy(batched_bias_Rh_local, batched_bias_Rh_local + local_fused_hidden_rows * hidden_size_,
                    linear_output_local);

          // compute Ht-1 * (Rh^T) + Rbh
          ComputeGemm(local_fused_hidden_rows, hidden_size_, hidden_size_, alpha,
                      prev_Ht, prev_Ht_end,  // Ht-1
                      hidden_size_,
                      recurrent_weightsH_.cbegin(), recurrent_weightsH_.cend(),  // Rh^T
                      hidden_size_, beta,
                      linear_output_local, linear_output_.end(),  // pre: Rbh, post:output
                      hidden_size_);

          DumpMatrix("Ht-1 * (Rh^T) + Rbh " + row_str, &*linear_output_local, batch_size_, hidden_size_);
        }

        // 1st Set Of Activations
        for (int r = 0; r < local_fused_hidden_rows; r++) {
          const T* p_bias_r = use_bias_ ? SafeRawConstPointer<T>(batched_bias_WRr_local + r * hidden_size_,
                                                                 batched_bias_WRr_local_end, hidden_size_)
                                        : nullptr;

          // initialize p_rt with input to calculate rt. outputZRH_ has Xt*(Wr^T) + Ht-1*(Rr^T).
          T* p_rt = SafeRawPointer(outputZRH_, out_added_offset + r * hidden_size_x3 + hidden_size_, hidden_size_);

          // add the bias and clip. post: p_rt == Xt*(Wr^T) + Ht-1*(Rr^T) + Wbr + Rbr
          clip_with_bias_ptr_(clip_, p_bias_r, p_rt, hidden_size_);

          if (linear_before_reset_) {
            // p_linear_output = Ht-1 * (Rh^T) + Rbh
            T* p_linear_output = SafeRawPointer<T>(linear_output_local + r * hidden_size_,
                                                   linear_output_local_end, hidden_size_);
            T* p_cur_h = SafeRawPointer<T>(cur_h_local + r * hidden_size_, cur_h_local_end, hidden_size_);

            // calculate rt in-place [p_rt = f(p_rt)]
            // calculate rt (.) (Ht-1 * (Rh^T) + Rbh) using p_linear_output. write to p_cur_h
            reset_gate_(p_linear_output, p_rt, p_cur_h, hidden_size_, zr_alpha_, zr_beta_);

          } else {
            const T* p_prev_Ht = SafeRawConstPointer<T>(prev_Ht + r * hidden_size_, prev_Ht_end, hidden_size_);
            T* p_cur_h = SafeRawPointer<T>(cur_h_local + r * hidden_size_, cur_h_local_end, hidden_size_);

            // calculate rt in-place [p_rt = f(p_rt)]
            // calculate rt (.) Ht-1 using p_prev_Ht, and write to p_cur_h
            reset_gate_(p_prev_Ht, p_rt, p_cur_h, hidden_size_, zr_alpha_, zr_beta_);
          }
        }

        std::string label = linear_before_reset_ ? "rt (.) (Ht-1 * (Rh^T) + Rbh)" : "rt (.) Ht-1";
        DumpMatrix(label + row_str, &*cur_h_local, local_fused_hidden_rows, hidden_size_);

        if (linear_before_reset_) {
          // input contains rt (.) (Ht-1*(Rh^T) + Rbh)
          auto input = cur_h_local;
          // out_H currently contains Xt*(W[zrh]^T).
          auto out_H = outputZRH_.begin() + out_added_offset;

          for (int r = 0; r < local_fused_hidden_rows; r++) {
            // skip over the inputs with Z and R weights
            out_H += hidden_size_x2;
            for (int h = 0; h < hidden_size_; ++h) {
              *out_H += *input;
              ++out_H;
              ++input;
            }
          }
        } else {
          label += " * Rh^T";
          ComputeGemm(local_fused_hidden_rows, hidden_size_, hidden_size_, alpha,
                      cur_h_local, cur_h_local_end,
                      hidden_size_,
                      recurrent_weightsH_.cbegin(), recurrent_weightsH_.cend(),
                      hidden_size_, beta,
                      outputZRH_.begin() + out_added_offset + hidden_size_x2, outputZRH_.end(),
                      hidden_size_x3);
        }

        DumpMatrix("Xt*(Wh^T) + (" + label + ")" + row_str,
                   outputZRH_.data() + out_added_offset, local_fused_hidden_rows, hidden_size_,
                   hidden_size_x2, hidden_size_x3);

        // 2nd Set of Activations
        span_T_iter output;
        span_T_iter output_end;
        if (output_sequence) {
          output = outputs.begin() + step * output_step_length + row * hidden_size_;
          output_end = outputs.end();

        } else {
          output = final_hidden_state.begin() + row * hidden_size_;
          output_end = final_hidden_state.end();
        }

        for (int r = 0; r < local_fused_hidden_rows; r++) {
          if (step >= min_sequence_length && step >= sequence_lengths[row + r]) {
            if (output_sequence) {
              auto fill_output = output + r * hidden_size_;
              std::fill(fill_output, fill_output + hidden_size_, T{});
            }

            continue;
          }

          const T* p_bias_z = use_bias_ ? SafeRawConstPointer<T>(batched_bias_WRz_local, batched_bias_WRz_local_end,
                                                                 hidden_size_)
                                        : nullptr;

          // initialize p_zt with Xt*(Wz^T) + Ht-1*(Rz^T), which is most of the input to calculate zt:
          T* p_zt = SafeRawPointer<T>(outputZRH_, out_added_offset + r * hidden_size_x3, hidden_size_);

          // using p_zt, add bias and clip in-place
          clip_with_bias_ptr_(clip_, p_bias_z, p_zt, hidden_size_);

          // calculate zt in-place. p_zt = f(p_zt)
          update_gate_(p_zt, hidden_size_, zr_alpha_, zr_beta_);

          DumpMatrix("zt[" + std::to_string(r) + "]" + row_str, p_zt, 1, hidden_size_);

          const T* p_bias_h = nullptr;
          if (use_bias_) {
            if (linear_before_reset_) {
              // Wbh
              p_bias_h = SafeRawConstPointer<T>(batched_bias_Wh_local + r * hidden_size_,
                                                batched_bias_Wh_local_end, hidden_size_);

            } else {
              // Wbh + Wrh
              p_bias_h = SafeRawConstPointer<T>(batched_bias_WRh_local + r * hidden_size_,
                                                batched_bias_WRh_local_end, hidden_size_);
            }
          }

          // setup p_ht with input to calculate ht
          // p_ht = Xt*(Wh^T) + (rt (.) Ht-1 * Rh^T)          #  linear_before_reset_ == false
          //      = Xt*(Wh^T) + (rt (.) (Ht-1*(Rh^T) + Rbh))  #  linear_before_reset_ == true
          T* p_ht = SafeRawPointer<T>(outputZRH_, out_added_offset + r * hidden_size_x3 + hidden_size_x2, hidden_size_);

          // add Wbh [and Wrh] and clip
          clip_with_bias_ptr_(clip_, p_bias_h, p_ht, hidden_size_);  // post: p_ht = input to g() for calculating ht

          DumpMatrix("ht input [" + std::to_string(r) + "]" + row_str, p_ht, 1, hidden_size_);

          const T* p_prev_Ht = SafeRawConstPointer<T>(prev_Ht + r * hidden_size_, prev_Ht_end, hidden_size_);
          T* p_Ht = SafeRawPointer<T>(output + r * hidden_size_, output_end, hidden_size_);

          // calculate ht = g(p_ht) and write in-place to p_ht
          // calculate Ht = (1 - zt) (.) ht + zt (.) Ht-1 and write to p_Ht
          output_gate_(p_ht, p_zt, p_prev_Ht, p_Ht, hidden_size_, h_alpha_, h_beta_);
        }

        DumpMatrix("output" + row_str, &*output, 1, hidden_size_);

        prev_Ht = output;
        prev_Ht_end = output_end;
      }
    };

    ExecuteLambdaInParallel("Processing batch", hidden_gemm_and_activations, batch_size_, fused_hidden_rows,
                            ttp_, logger_);
  } else {
    size_t out_added_offset;

    span_T_const_iter prev_Ht = batched_hidden0_.cbegin();  // Ht-1
    span_T_const_iter prev_Ht_end = batched_hidden0_.cend();
    span_T_iter cur_h_local = cur_h_.begin();
    span_T_iter cur_h_local_end = cur_h_.end();

    span_T_const_iter batched_bias_WRz_local;
    span_T_const_iter batched_bias_WRr_local;
    span_T_const_iter batched_bias_WRh_local;
    span_T_const_iter batched_bias_Wh_local;
    span_T_const_iter batched_bias_Rh_local;

    if (use_bias_) {
      batched_bias_WRz_local = batched_bias_WRz_.cbegin();
      batched_bias_WRr_local = batched_bias_WRr_.cbegin();

      if (linear_before_reset_) {
        batched_bias_Wh_local = batched_bias_Wh_.cbegin();
        batched_bias_Rh_local = batched_bias_Rh_.cbegin();
      } else {
        batched_bias_WRh_local = batched_bias_WRh_.cbegin();
      }
    }

    // for each item in sequence run all calculations
    for (int step = 0; step < max_sequence_length; step++) {
      const std::string seqno_str = " [seqno=" + std::to_string(step) + "]";

      DumpMatrix("Ht-1" + seqno_str, &*prev_Ht, batch_size_, hidden_size_);

      out_added_offset = (step * batch_size_) * hidden_size_x3;

#if defined(HAVE_PARALLELIZED_GEMM)
      // calculate Ht-1*R[zr], and add to the weighted inputs that are in outputZRH_
      // Ht-1 * R[zr] + Xt*(W[zr]^T)
      ComputeGemm(batch_size_, hidden_size_x2, hidden_size_, alpha,
                  prev_Ht, prev_Ht_end,
                  hidden_size_,
                  recurrent_weightsZR_.cbegin(), recurrent_weightsZR_.cend(),
                  hidden_size_x2, beta,
                  outputZRH_.begin() + out_added_offset, outputZRH_.end(),
                  hidden_size_x3);
#else
      auto hidden_gemm_compute1 = [&](int thread_id) {
        int local_cols = hidden_size_x2 / hidden_num_threads_;
        int start_col = thread_id * local_cols;
        int compute_cols = local_cols;

        if (thread_id == hidden_num_threads_ - 1)
          compute_cols = hidden_size_x2 - thread_id * local_cols;

        // calculate Ht-1*R[zr], and add to the weighted inputs that are in outputZRH_
        // Ht-1 * R[zr] + Xt*(W[zr]^T)
        ComputeGemm(batch_size_, compute_cols, hidden_size_, alpha,
                    prev_Ht, prev_Ht_end,
                    hidden_size_,
                    recurrent_weightsZR_.cbegin() + start_col, recurrent_weightsZR_.cend(),
                    hidden_size_x2, beta,
                    outputZRH_.begin() + out_added_offset + start_col, outputZRH_.end(),
                    hidden_size_x3);
      };

      ExecuteLambdaInParallel("Calculating Ht-1*R[zr] + Xt*(W[zr]^T)" + seqno_str,
                              hidden_gemm_compute1, hidden_num_threads_, 1, ttp_, logger_);
#endif

      DumpMatrix("Ht-1 * R[zr] + Xt*(W[zr]^T)" + seqno_str,
                 outputZRH_.data() + out_added_offset, batch_size_, hidden_size_x2, 0, hidden_size_x3);

      if (linear_before_reset_) {
        // copy Rbh to linear output
        std::copy(batched_bias_Rh_local, batched_bias_Rh_local_end, linear_output_.begin());

#if defined(HAVE_PARALLELIZED_GEMM)
        // compute Ht-1 * (Rh^T) + Rbh
        ComputeGemm(batch_size_, hidden_size_, hidden_size_, alpha,
                    prev_Ht, prev_Ht_end,  // Ht-1
                    hidden_size_,
                    recurrent_weightsH_.cbegin(), recurrent_weightsH_.cend(),  // Rh^T
                    hidden_size_, beta,
                    linear_output_.begin(), linear_output_.end(),  // pre: Rbh, post:output
                    hidden_size_);
#else
        // Ht-1 * (Rh^T) + Rbh
        auto linear_gemm = [&](int thread_id) {
          int local_cols = hidden_size_ / hidden_num_threads_;
          int start_col = thread_id * local_cols;
          int compute_cols = local_cols;

          if (thread_id == hidden_num_threads_ - 1)
            compute_cols = hidden_size_ - thread_id * local_cols;

          ComputeGemm(batch_size_, compute_cols, hidden_size_, alpha,
                      prev_Ht, prev_Ht_end,  // Ht-1
                      hidden_size_,
                      recurrent_weightsH_.cbegin() + start_col, recurrent_weightsH_.cend(),  // Rh^T
                      hidden_size_, beta,
                      linear_output_.begin() + start_col, linear_output_.end(),  // pre: Rbh, post:output
                      hidden_size_);
        };

        ExecuteLambdaInParallel("Calculating Ht-1 * (Rh^T) + Rbh" + seqno_str,
                                linear_gemm, hidden_num_threads_, 1, ttp_, logger_);
#endif
        DumpMatrix("Ht-1 * (Rh^T) + Rbh " + seqno_str, linear_output_.data(), batch_size_, hidden_size_);
      }

      // 1st Set Of Activations
      for (int r = 0; r < batch_size_; r++) {
        const T* p_bias_r = use_bias_ ? SafeRawConstPointer<T>(batched_bias_WRr_local + r * hidden_size_,
                                                               batched_bias_WRr_local_end, hidden_size_)
                                      : nullptr;

        // initialize p_rt with input to calculate rt. outputZRH_ has Xt*(Wr^T) + Ht-1*(Rr^T).
        T* p_rt = SafeRawPointer(outputZRH_, out_added_offset + r * hidden_size_x3 + hidden_size_, hidden_size_);

        // add the bias and clip. post: p_rt == Xt*(Wr^T) + Ht-1*(Rr^T) + Wbr + Rbr
        clip_with_bias_ptr_(clip_, p_bias_r, p_rt, hidden_size_);

        if (linear_before_reset_) {
          // p_linear_output = Ht-1 * (Rh^T) + Rbh
          T* p_linear_output = SafeRawPointer<T>(linear_output_, r * hidden_size_, hidden_size_);
          T* p_cur_h = SafeRawPointer<T>(cur_h_local + r * hidden_size_, cur_h_local_end, hidden_size_);

          // calculate rt in-place [p_rt = f(p_rt)]
          // calculate rt (.) (Ht-1 * (Rh^T) + Rbh) using p_linear_output. write to p_cur_h
          reset_gate_(p_linear_output, p_rt, p_cur_h, hidden_size_, zr_alpha_, zr_beta_);

        } else {
          const T* p_prev_Ht = SafeRawConstPointer<T>(prev_Ht + r * hidden_size_, prev_Ht_end, hidden_size_);
          T* p_cur_h = SafeRawPointer<T>(cur_h_local + r * hidden_size_, cur_h_local_end, hidden_size_);

          // calculate rt in-place [p_rt = f(p_rt)]
          // calculate rt (.) Ht-1 using p_prev_Ht, and write to p_cur_h
          reset_gate_(p_prev_Ht, p_rt, p_cur_h, hidden_size_, zr_alpha_, zr_beta_);
        }
      }

      std::string label = linear_before_reset_ ? "rt (.) (Ht-1 * (Rh^T) + Rbh)" : "rt (.) Ht-1";
      DumpMatrix(label + seqno_str, &*cur_h_local, batch_size_, hidden_size_);

      if (linear_before_reset_) {
        // input contains rt (.) (Ht-1*(Rh^T) + Rbh)
        auto input = cur_h_local;
        // out_H currently contains Xt*(W[zrh]^T).
        auto out_H = outputZRH_.begin() + out_added_offset;

        for (int r = 0; r < batch_size_; r++) {
          // skip over the inputs with Z and R weights
          out_H += hidden_size_x2;
          for (int h = 0; h < hidden_size_; ++h) {
            *out_H += *input;
            ++out_H;
            ++input;
          }
        }
      } else {
        label += " * Rh^T";

#if defined(HAVE_PARALLELIZED_GEMM)
        // out_H currently contains Xt*(Wh^T).
        auto out_H = outputZRH_.begin() + out_added_offset + hidden_size_x2;

        // Calculate Xt*(Wh^T) + rt (.) Ht-1 * Rh
        ComputeGemm(batch_size_, hidden_size_, hidden_size_, alpha,
                    cur_h_local, cur_h_local_end,  // rt (.) Ht-1
                    hidden_size_,
                    recurrent_weightsH_.cbegin(), recurrent_weightsH_.cend(),  // Rh^T
                    hidden_size_, beta,
                    out_H, outputZRH_.end(),
                    hidden_size_x3);
#else
        auto hidden_gemm_compute2 = [&](int thread_id) {
          int local_cols = hidden_size_ / hidden_num_threads_;
          int start_col = thread_id * local_cols;
          int compute_cols = local_cols;

          if (thread_id == hidden_num_threads_ - 1)
            compute_cols = hidden_size_ - thread_id * local_cols;

          // out_H currently contains Xt*(Wh^T).
          auto out_H = outputZRH_.begin() + out_added_offset + hidden_size_x2 + start_col;

          ComputeGemm(batch_size_, compute_cols, hidden_size_, alpha,
                      cur_h_local, cur_h_local_end,  // rt (.) Ht-1
                      hidden_size_,
                      recurrent_weightsH_.cbegin() + start_col, recurrent_weightsH_.cend(),  // Rh^T
                      hidden_size_, beta,
                      out_H, outputZRH_.end(),  // pre:Xt*(Wh^T), post: output
                      hidden_size_x3);
        };

        ExecuteLambdaInParallel("Calculating Xt*(Wh^T) + " + label + seqno_str,
                                hidden_gemm_compute2, hidden_num_threads_, 1, ttp_, logger_);
#endif
      }

      DumpMatrix("Xt*(Wh^T) + (" + label + ")" + seqno_str, outputZRH_.data() + out_added_offset,
                 batch_size_, hidden_size_, hidden_size_x2, hidden_size_x3);

      //2nd Set of Activations
      span_T_iter output;
      span_T_iter output_end;
      if (output_sequence) {
        output = outputs.begin() + step * output_step_length;
        output_end = outputs.end();

      } else {
        output = final_hidden_state.begin();
        output_end = final_hidden_state.end();
      }

      for (int r = 0; r < batch_size_; r++) {
        if (step >= min_sequence_length && step >= sequence_lengths[r]) {
          if (output_sequence) {
            auto fill_output = output + r * hidden_size_;
            std::fill(fill_output, fill_output + hidden_size_, T{});
          }

          continue;
        }

        const T* p_bias_z = use_bias_ ? SafeRawConstPointer<T>(batched_bias_WRz_local,
                                                               batched_bias_WRz_local_end, hidden_size_)
                                      : nullptr;

        // initialize p_zt with Xt*(Wz^T) + Ht-1*(Rz^T), which is most of the input to calculate zt:
        T* p_zt = SafeRawPointer<T>(outputZRH_, out_added_offset + r * hidden_size_x3, hidden_size_);

        // using p_zt, add bias and clip in-place
        clip_with_bias_ptr_(clip_, p_bias_z, p_zt, hidden_size_);

        // calculate zt in-place. p_zt = f(p_zt)
        update_gate_(p_zt, hidden_size_, zr_alpha_, zr_beta_);

        DumpMatrix("zt[" + std::to_string(r) + "]" + seqno_str, p_zt, 1, hidden_size_);

        const T* p_bias_h = nullptr;
        if (use_bias_) {
          if (linear_before_reset_) {
            // Wbh
            p_bias_h = SafeRawConstPointer<T>(batched_bias_Wh_local + r * hidden_size_,
                                              batched_bias_Wh_local_end, hidden_size_);

          } else {
            // Wbh + Wrh
            p_bias_h = SafeRawConstPointer<T>(batched_bias_WRh_local + r * hidden_size_,
                                              batched_bias_WRh_local_end, hidden_size_);
          }
        }

        // setup p_ht with input to calculate ht
        // p_ht = Xt*(Wh^T) + (rt (.) Ht-1 * Rh^T)          #  linear_before_reset_ == false
        //      = Xt*(Wh^T) + (rt (.) (Ht-1*(Rh^T) + Rbh))  #  linear_before_reset_ == true
        T* p_ht = SafeRawPointer<T>(outputZRH_, out_added_offset + r * hidden_size_x3 + hidden_size_x2, hidden_size_);

        // add Wbh [and Wrh] and clip
        clip_with_bias_ptr_(clip_, p_bias_h, p_ht, hidden_size_);  // post: p_ht == input to g() for calculating ht

        DumpMatrix("ht input [" + std::to_string(r) + "]" + seqno_str, p_ht, 1, hidden_size_);

        const T* p_prev_Ht = SafeRawConstPointer<T>(prev_Ht + r * hidden_size_, prev_Ht_end, hidden_size_);
        T* p_Ht = SafeRawPointer<T>(output + r * hidden_size_, output_end, hidden_size_);

        // calculate ht = g(p_ht) and write in-place to p_ht
        // calculate Ht = (1 - zt) (.) ht + zt (.) Ht-1 and write to p_Ht
        output_gate_(p_ht, p_zt, p_prev_Ht, p_Ht, hidden_size_, h_alpha_, h_beta_);  // calculate ht and Ht
      }

      DumpMatrix("output" + seqno_str, &*output, batch_size_, hidden_size_);

      prev_Ht = output;
      prev_Ht_end = output_end;
    }
  }

  if (output_sequence) {
    // copy last output to final_hidden_state
    for (int i = 0; i < batch_size_; i++) {
      const int seq_len = sequence_lengths[i];
      span_T_const_iter src = outputs.cbegin() + (seq_len - 1) * output_step_length + i * hidden_size_;
      span_T_iter dest = final_hidden_state.begin() + i * hidden_size_;
      std::copy(src, src + hidden_size_, dest);
    }

    if (direction_ == kReverse) {
      ReverseSequence<T>(outputs, original_outputs,
                         sequence_lengths, seq_length_,
                         batch_size_, hidden_size_, num_directions);
    }
  }
}

template <typename T>
void UniDirectionalGru<T>::AllocateBuffers() {
  cur_h_ = Allocate(allocator_, hidden_size_ * batch_size_, cur_h_ptr_);
  batched_hidden0_ = Allocate(allocator_, batch_size_ * hidden_size_, batched_hidden0_ptr_, true);

  if (use_bias_) {
    batched_bias_WRz_ = Allocate(allocator_, batch_size_ * hidden_size_, batched_bias_WRz_ptr_);
    batched_bias_WRr_ = Allocate(allocator_, batch_size_ * hidden_size_, batched_bias_WRr_ptr_);

    if (linear_before_reset_) {
      batched_bias_Wh_ = Allocate(allocator_, batch_size_ * hidden_size_, batched_bias_Wh_ptr_);
      batched_bias_Rh_ = Allocate(allocator_, batch_size_ * hidden_size_, batched_bias_Rh_ptr_);
      linear_output_ = Allocate(allocator_, batch_size_ * hidden_size_, linear_output_ptr_);
    } else {
      batched_bias_WRh_ = Allocate(allocator_, batch_size_ * hidden_size_, batched_bias_WRh_ptr_);
    }
  }

  auto batch_times_seq_length = batch_size_ * seq_length_;

  outputZRH_ = Allocate(allocator_, hidden_size_ * 3 * batch_times_seq_length, outputZRH_ptr_, true);

  if (direction_ == kReverse) {
    inputs_reverse_ = Allocate(allocator_, batch_times_seq_length * input_size_, inputs_reverse_ptr_);
    outputs_reverse_ = Allocate(allocator_, batch_times_seq_length * hidden_size_, outputs_reverse_ptr_);
  }
}

//Threads set based on performance profiling on Surface Book and a Xeon
template <typename T>
void UniDirectionalGru<T>::SetNumThreads() {
  int threads = std::thread::hardware_concurrency() - 1;

  if (threads < 1)
    threads = 1;

  int imt = threads;
  if (imt > 8 && hidden_size_ <= 128)
    imt = 8;
  if (imt > 16 && hidden_size_ <= 256)
    imt = 16;
  if (imt > 24)
    imt = 24;

  // total number of operations in the call to ComputeGemm to apply the weights to the inputs
  auto work = seq_length_ * batch_size_ * hidden_size_ * 3 * input_size_;
  const double kMinTaskSize = 50000;  // this value is copied from Eigen. simplistic and could be refined.

  input_num_threads_ = std::max<int>(1, std::min<int>(imt, static_cast<int>(work / kMinTaskSize)));

  VLOGS(logger_, 1) << "Input Threads : " << input_num_threads_;

  int hmt = threads;
  batch_parallel_ = false;

  // for readability of the below logic
  const auto num_rows = batch_size_;
  const auto num_columns = hidden_size_;

  // parallelize by partitioning the batch rows
  if (num_rows > 4 ||
      (num_rows >= 2 && num_columns <= 256) ||
      (num_rows >= 3 && num_columns <= 512)) {
    hidden_num_threads_ = hmt;
    batch_parallel_ = true;
  } else {  // otherwise parallelize by partitioning the columns.
    if (hmt > 2 && num_columns <= 128)
      hmt = 2;
    if (hmt > 3 && num_columns <= 256)
      hmt = 3;
    if (hmt > 8 && num_columns <= 1024)
      hmt = 8;
    if (hmt > 24)
      hmt = 24;

    hidden_num_threads_ = hmt;
  }

  VLOGS(logger_, 1) << "Hidden Threads : " << hidden_num_threads_;

  LOTUS_ENFORCE(input_num_threads_ >= 1);
  LOTUS_ENFORCE(hidden_num_threads_ >= 1);
}
}  // namespace detail
}  // namespace onnxruntime