#include "ffi_handler.h"

#include <cstdint>
#include <string>

#include "abi_layout.h"
#include "launch.h"
#include "xla/ffi/api/ffi.h"

namespace ffi = xla::ffi;
using namespace beamz::cuda::abi;

namespace {

constexpr float kEps0 = 8.8541878128e-12f;
constexpr float kMu0 = 1.25663706212e-6f;
void SetBoundaryCode(BeamzLaunch* launch, int32_t code) {
  launch->metallic_edges = code & 0x3f;
  launch->uniform_cpml_thickness = code >> 8;
}

BeamzLaunch InitializeLaunch(int32_t abi_version, int32_t cuda_flags,
                             int32_t phase, int32_t nterms, int32_t metric_kind,
                             float dt, float resolution,
                             int32_t boundary_code) {
  BeamzLaunch launch{};
  launch.abi_version = abi_version;
  launch.cuda_flags = cuda_flags;
  launch.phase = phase;
  launch.nterms = nterms;
  launch.metric_kind = metric_kind;
  launch.dt = dt;
  launch.resolution = resolution;
  launch.inv_resolution = 1.0f / resolution;
  launch.dt_over_eps = dt / kEps0;
  launch.dt_over_mu = dt / kMu0;
  SetBoundaryCode(&launch, boundary_code);
  return launch;
}

struct GraphLaunches {
  BeamzLaunch h;
  BeamzLaunch e;
};

GraphLaunches InitializeGraphLaunches(
    int32_t abi_version, int32_t cuda_flags, float dt, float resolution,
    int32_t boundary_code, int32_t metric_kind, bool cpml_enabled,
    const BeamzBuffer* inputs, const BeamzBuffer* outputs) {
  const int32_t nterms = cpml_enabled ? kCpmlTermCount : 0;
  GraphLaunches launches{
      InitializeLaunch(abi_version, cuda_flags, 0, nterms, metric_kind, dt,
                       resolution, boundary_code),
      InitializeLaunch(abi_version, cuda_flags, 1, nterms, metric_kind, dt,
                       resolution, boundary_code),
  };
  for (int component = 0; component < 3; ++component) {
    launches.h.inputs[component] = inputs[component];
    launches.h.inputs[3 + component] = inputs[3 + component];
    launches.h.outputs[component] = outputs[component];
    launches.e.inputs[component] = inputs[3 + component];
    // The E phase consumes the just-updated, aliased magnetic outputs.
    launches.e.inputs[3 + component] = outputs[component];
    launches.e.outputs[component] = outputs[3 + component];
  }
  if (cpml_enabled) {
    for (int index = 0; index < kCpmlPhaseInputCount; ++index) {
      launches.h.inputs[kFieldCount + index] = inputs[kFieldCount + index];
      launches.e.inputs[kFieldCount + index] =
          inputs[kCpmlEInputOffset + index];
    }
    for (int term = 0; term < kCpmlTermCount; ++term) {
      launches.h.outputs[3 + term] = outputs[kFieldCount + term];
      launches.e.outputs[3 + term] =
          outputs[kFieldCount + kCpmlTermCount + term];
    }
    for (int axis = 0; axis < 3; ++axis) {
      launches.h.metrics[axis] = inputs[kCpmlHMetricOffset + axis];
      launches.e.metrics[axis] = inputs[kCpmlEMetricOffset + axis];
    }
  } else {
    for (int material = 0; material < kFieldCount; ++material) {
      launches.h.inputs[kFieldCount + material] =
          inputs[kFieldCount + material];
      launches.e.inputs[kFieldCount + material] =
          inputs[2 * kFieldCount + material];
    }
    for (int axis = 0; axis < 3; ++axis) {
      launches.h.metrics[axis] = inputs[3 * kFieldCount + axis];
      launches.e.metrics[axis] = inputs[3 * kFieldCount + 3 + axis];
    }
  }
  return launches;
}

struct TemporalCpmlLaunches {
  BeamzLaunch h_ab;
  BeamzLaunch e_ab;
  BeamzLaunch h_ba;
  BeamzLaunch e_ba;
};

BeamzProgramLaunch InPlaceProgram(
    const GraphLaunches& launches, int32_t nsteps,
    const BeamzSourceGroupLaunch* source_groups = nullptr,
    int32_t source_group_count = 0,
    const BeamzDftGroupLaunch* monitors = nullptr,
    int32_t graph_cache_capacity = 32) {
  BeamzProgramLaunch program{};
  program.h_ab = launches.h;
  program.e_ab = launches.e;
  program.source_groups = source_groups;
  program.source_group_count = source_group_count;
  program.monitors = monitors;
  program.field_bank_count = 1;
  program.nsteps = nsteps;
  program.graph_cache_capacity = graph_cache_capacity;
  return program;
}

BeamzProgramLaunch TemporalProgram(
    const TemporalCpmlLaunches& launches, int32_t nsteps,
    const BeamzSourceGroupLaunch* source_groups = nullptr,
    int32_t source_group_count = 0,
    const BeamzDftGroupLaunch* monitors = nullptr,
    int32_t graph_cache_capacity = 32) {
  BeamzProgramLaunch program{};
  program.h_ab = launches.h_ab;
  program.e_ab = launches.e_ab;
  program.h_ba = launches.h_ba;
  program.e_ba = launches.e_ba;
  program.source_groups = source_groups;
  program.source_group_count = source_group_count;
  program.monitors = monitors;
  program.field_bank_count = 2;
  program.nsteps = nsteps;
  program.graph_cache_capacity = graph_cache_capacity;
  return program;
}

TemporalCpmlLaunches InitializeTemporalCpmlLaunches(
    int32_t abi_version, int32_t cuda_flags, float dt, float resolution,
    int32_t boundary_code, int32_t metric_kind, const BeamzBuffer* inputs,
    const BeamzBuffer* outputs) {
  auto initialize = [&](int32_t phase) {
    return InitializeLaunch(abi_version, cuda_flags, phase, kCpmlTermCount,
                            metric_kind, dt, resolution, boundary_code);
  };
  TemporalCpmlLaunches launches{
      initialize(0), initialize(1), initialize(0), initialize(1)};
  for (int component = 0; component < 3; ++component) {
    launches.h_ab.inputs[component] = outputs[component];
    launches.h_ab.inputs[3 + component] = outputs[3 + component];
    launches.h_ab.outputs[component] = outputs[kFieldCount + component];
    launches.e_ab.inputs[component] = outputs[3 + component];
    launches.e_ab.inputs[3 + component] = outputs[kFieldCount + component];
    launches.e_ab.outputs[component] = outputs[kFieldCount + 3 + component];

    launches.h_ba.inputs[component] = outputs[kFieldCount + component];
    launches.h_ba.inputs[3 + component] =
        outputs[kFieldCount + 3 + component];
    launches.h_ba.outputs[component] = outputs[component];
    launches.e_ba.inputs[component] = outputs[kFieldCount + 3 + component];
    launches.e_ba.inputs[3 + component] = outputs[component];
    launches.e_ba.outputs[component] = outputs[3 + component];
  }
  for (int index = 0; index < kCpmlPhaseInputCount; ++index) {
    launches.h_ab.inputs[kFieldCount + index] =
        launches.h_ba.inputs[kFieldCount + index] = inputs[kFieldCount + index];
    launches.e_ab.inputs[kFieldCount + index] =
        launches.e_ba.inputs[kFieldCount + index] =
            inputs[kCpmlEInputOffset + index];
  }
  for (int term = 0; term < kCpmlTermCount; ++term) {
    launches.h_ab.outputs[3 + term] =
        outputs[kTemporalHPsiWorkspaceOffset + term];
    launches.h_ba.inputs[kCpmlHPsiInputOffset + term] =
        outputs[kTemporalHPsiWorkspaceOffset + term];
    launches.h_ba.outputs[3 + term] = outputs[kTemporalHPsiOutputOffset + term];
    launches.e_ab.outputs[3 + term] =
        outputs[kTemporalEPsiWorkspaceOffset + term];
    launches.e_ba.inputs[kCpmlHPsiInputOffset + term] =
        outputs[kTemporalEPsiWorkspaceOffset + term];
    launches.e_ba.outputs[3 + term] = outputs[kTemporalEPsiOutputOffset + term];
  }
  for (int axis = 0; axis < 3; ++axis) {
    launches.h_ab.metrics[axis] = launches.h_ba.metrics[axis] =
        inputs[kCpmlHMetricOffset + axis];
    launches.e_ab.metrics[axis] = launches.e_ba.metrics[axis] =
        inputs[kCpmlEMetricOffset + axis];
  }
  return launches;
}

void InitializeSourceGroups(
    BeamzSourceGroupLaunch (&groups)[kSourceGroupCount],
    const BeamzBuffer* inputs, size_t offset, const BeamzBuffer& current_step,
    int32_t coincident_mask) {
  for (int32_t index = 0; index < kSourceGroupCount; ++index) {
    const size_t group_offset = offset + kSourceGroupBufferCount * index;
    groups[index].coefficients =
        inputs[group_offset + kSourceGroupCoefficientsInput];
    groups[index].waveforms =
        inputs[group_offset + kSourceGroupWaveformsInput];
    groups[index].starts = inputs[group_offset + kSourceGroupStartsInput];
    groups[index].current_step = current_step;
    groups[index].timing = index / 3;
    groups[index].component = index % 3;
    groups[index].coincident = (coincident_mask & (1 << index)) != 0;
  }
}

BeamzDftGroupLaunch InitializeDftGroups(
    const BeamzBuffer* inputs, size_t offset, const BeamzBuffer& dft_re,
    const BeamzBuffer& dft_im, const BeamzBuffer& dft_weight,
    int32_t monitor_count) {
  BeamzDftGroupLaunch monitors{};
  monitors.indices = inputs[offset + kMonitorIndicesInput];
  monitors.weights = inputs[offset + kMonitorWeightsInput];
  monitors.frequencies = inputs[offset + kMonitorFrequenciesInput];
  monitors.component_masks = inputs[offset + kMonitorComponentMasksInput];
  monitors.counts = inputs[offset + kMonitorCountsInput];
  monitors.codes = inputs[offset + kMonitorCodesInput];
  monitors.windows = inputs[offset + kMonitorWindowsInput];
  monitors.dft_re = dft_re;
  monitors.dft_im = dft_im;
  monitors.dft_weight = dft_weight;
  monitors.time = inputs[offset + kMonitorTimeInput];
  monitors.current_step = inputs[offset + kMonitorCurrentStepInput];
  monitors.monitor_count = monitor_count;
  return monitors;
}

ffi::Error DecodeBuffer(const ffi::AnyBuffer& value, BeamzBuffer* output) {
  if (value.element_type() != ffi::DataType::F32 &&
      value.element_type() != ffi::DataType::S32 &&
      value.element_type() != ffi::DataType::BF16) {
    return ffi::Error::InvalidArgument(
        "BeamZ CUDA accepts f32, bf16, and s32 buffers");
  }
  const auto dims = value.dimensions();
  if (dims.size() > 4) {
    return ffi::Error::InvalidArgument(
        "BeamZ CUDA accepts buffers of rank <= 4");
  }
  output->data = value.untyped_data();
  output->rank = static_cast<int32_t>(dims.size());
  output->element_type = value.element_type() == ffi::DataType::F32
                             ? kBeamzF32
                         : value.element_type() == ffi::DataType::BF16
                             ? kBeamzBF16
                             : kBeamzS32;
  output->dims[0] = output->dims[1] = output->dims[2] = output->dims[3] = 1;
  for (size_t index = 0; index < dims.size(); ++index) {
    output->dims[index] = dims[index];
  }
  return ffi::Error::Success();
}

template <size_t N>
ffi::Error DecodeArgs(const ffi::RemainingArgs& args,
                      BeamzBuffer (&buffers)[N], size_t count = N) {
  if (count > N) {
    return ffi::Error::InvalidArgument("too many BeamZ CUDA input buffers");
  }
  for (size_t index = 0; index < count; ++index) {
    auto decoded = args.get<ffi::AnyBuffer>(index);
    if (!decoded) return decoded.error();
    if (auto error = DecodeBuffer(*decoded, &buffers[index]); error.failure()) {
      return error;
    }
  }
  return ffi::Error::Success();
}

template <size_t N>
ffi::Error DecodeRets(const ffi::RemainingRets& rets,
                      BeamzBuffer (&buffers)[N], size_t count = N) {
  if (count > N) {
    return ffi::Error::InvalidArgument("too many BeamZ CUDA output buffers");
  }
  for (size_t index = 0; index < count; ++index) {
    auto decoded = rets.get<ffi::AnyBuffer>(index);
    if (!decoded) return decoded.error();
    if (auto error = DecodeBuffer(**decoded, &buffers[index]); error.failure()) {
      return error;
    }
  }
  return ffi::Error::Success();
}

using Launcher = int (*)(void*, const BeamzLaunch&);

ffi::Error Dispatch(Launcher launcher, void* stream, ffi::RemainingArgs args,
                    ffi::RemainingRets rets, int32_t abi_version,
                    int32_t cuda_flags, int32_t phase, int32_t nterms,
                    float dt, float resolution, int32_t boundary_code,
                    int32_t metric_kind) {
  if (abi_version != kAbiVersion) {
    return ffi::Error::InvalidArgument("beamz_cuda ABI version mismatch");
  }
  if (phase < 0 || phase > 1 || (nterms != 0 && nterms != 6) ||
      metric_kind < 0 || metric_kind > 2) {
    return ffi::Error::InvalidArgument(
        "invalid BeamZ CUDA phase or CPML term count");
  }
  const size_t payload_count = 13 + 4 * static_cast<size_t>(nterms);
  const size_t output_count = 3 + static_cast<size_t>(nterms);
  BeamzLaunch launch = InitializeLaunch(abi_version, cuda_flags, phase, nterms,
                                        metric_kind, dt, resolution,
                                        boundary_code);
  if (auto error = DecodeArgs(args, launch.inputs, payload_count);
      error.failure()) return error;
  for (size_t axis = 0; axis < 3; ++axis) {
    auto decoded = args.get<ffi::AnyBuffer>(payload_count + axis);
    if (!decoded) return decoded.error();
    if (auto error = DecodeBuffer(*decoded, &launch.metrics[axis]);
        error.failure()) {
      return error;
    }
  }
  if (auto error = DecodeRets(rets, launch.outputs, output_count);
      error.failure()) return error;
  const int error = launcher(stream, launch);
  return error == 0 ? ffi::Error::Success()
                    : ffi::Error::Internal("BeamZ CUDA kernel launch failed: " +
                                           std::to_string(error));
}

ffi::Error StreamedHandler(void* stream, ffi::RemainingArgs args,
                           ffi::RemainingRets rets, int32_t abi_version,
                           int32_t cuda_flags, int32_t phase, int32_t nterms,
                           float dt, float resolution, int32_t boundary_code,
                           int32_t metric_kind) {
  return Dispatch(BeamzLaunchStreamed, stream, args, rets, abi_version,
                  cuda_flags, phase, nterms, dt, resolution, boundary_code,
                  metric_kind);
}

ffi::Error StreamedStepsHandler(void* stream, ffi::RemainingArgs args,
                                ffi::RemainingRets rets, int32_t abi_version,
                                int32_t cuda_flags, int32_t nsteps, float dt,
                                float resolution, int32_t boundary_code,
                                int32_t metric_kind,
                                int32_t graph_cache_capacity) {
  if (abi_version != kAbiVersion) {
    return ffi::Error::InvalidArgument("beamz_cuda ABI version mismatch");
  }
  if (nsteps < 1) {
    return ffi::Error::InvalidArgument("BeamZ CUDA step count must be positive");
  }
  if (metric_kind < 0 || metric_kind > 2) {
    return ffi::Error::InvalidArgument("invalid BeamZ CUDA metric kind");
  }
  BeamzBuffer inputs[kYeeGraphInputCount]{};
  BeamzBuffer outputs[kYeeGraphOutputCount]{};
  if (auto error = DecodeArgs(args, inputs); error.failure()) return error;
  if (auto error = DecodeRets(rets, outputs); error.failure()) return error;

  GraphLaunches launches = InitializeGraphLaunches(
      abi_version, cuda_flags, dt, resolution, boundary_code, metric_kind,
      false, inputs, outputs);
  const int error = BeamzLaunchProgram(
      stream, InPlaceProgram(launches, nsteps, nullptr, 0, nullptr,
                              graph_cache_capacity));
  return error == 0 ? ffi::Error::Success()
                    : ffi::Error::Internal(
                          "BeamZ CUDA multi-step launch failed: " +
                          std::to_string(error));
}

// Out-of-place temporal updates need a second, XLA-owned field set.  Exposing
// that workspace as aliased results makes device writes visible to XLA instead
// of mutating buffers which the typed FFI contract considers read-only.
ffi::Error TemporalStepsHandler(void* stream, ffi::RemainingArgs args,
                                ffi::RemainingRets rets, int32_t abi_version,
                                int32_t cuda_flags, int32_t nsteps, float dt,
                                float resolution, int32_t boundary_code,
                                int32_t metric_kind,
                                int32_t graph_cache_capacity) {
  if (abi_version != kAbiVersion) {
    return ffi::Error::InvalidArgument("beamz_cuda ABI version mismatch");
  }
  if (nsteps < 1 || metric_kind < 0 || metric_kind > 2) {
    return ffi::Error::InvalidArgument("invalid BeamZ CUDA temporal attributes");
  }
  constexpr size_t kInputCount = kYeeGraphInputCount + kFieldCount;
  constexpr size_t kOutputCount = 2 * kFieldCount;
  BeamzBuffer inputs[kInputCount]{};
  BeamzBuffer outputs[kOutputCount]{};
  if (auto error = DecodeArgs(args, inputs); error.failure()) return error;
  if (auto error = DecodeRets(rets, outputs); error.failure()) return error;

  auto initialize = [&](int32_t phase) {
    return InitializeLaunch(abi_version, cuda_flags, phase, 0,
                            metric_kind, dt, resolution, boundary_code);
  };
  BeamzLaunch h_ab = initialize(0);
  BeamzLaunch e_ab = initialize(1);
  BeamzLaunch h_ba = initialize(0);
  BeamzLaunch e_ba = initialize(1);
  for (int component = 0; component < 3; ++component) {
    h_ab.inputs[component] = inputs[component];
    h_ab.inputs[3 + component] = inputs[3 + component];
    h_ab.outputs[component] = outputs[kFieldCount + component];
    e_ab.inputs[component] = inputs[3 + component];
    e_ab.inputs[3 + component] = outputs[kFieldCount + component];
    e_ab.outputs[component] = outputs[kFieldCount + 3 + component];

    h_ba.inputs[component] = outputs[kFieldCount + component];
    h_ba.inputs[3 + component] = outputs[kFieldCount + 3 + component];
    h_ba.outputs[component] = outputs[component];
    e_ba.inputs[component] = outputs[kFieldCount + 3 + component];
    e_ba.inputs[3 + component] = outputs[component];
    e_ba.outputs[component] = outputs[3 + component];
  }
  for (int material = 0; material < kFieldCount; ++material) {
    h_ab.inputs[kFieldCount + material] =
        inputs[2 * kFieldCount + material];
    h_ba.inputs[kFieldCount + material] =
        inputs[2 * kFieldCount + material];
    e_ab.inputs[kFieldCount + material] =
        inputs[3 * kFieldCount + material];
    e_ba.inputs[kFieldCount + material] =
        inputs[3 * kFieldCount + material];
  }
  for (int axis = 0; axis < 3; ++axis) {
    h_ab.metrics[axis] = h_ba.metrics[axis] =
        inputs[4 * kFieldCount + axis];
    e_ab.metrics[axis] = e_ba.metrics[axis] =
        inputs[4 * kFieldCount + 3 + axis];
  }
  BeamzProgramLaunch program{};
  program.h_ab = h_ab;
  program.e_ab = e_ab;
  program.h_ba = h_ba;
  program.e_ba = e_ba;
  program.field_bank_count = 2;
  program.nsteps = nsteps;
  program.graph_cache_capacity = graph_cache_capacity;
  const int error = BeamzLaunchProgram(stream, program);
  return error == 0 ? ffi::Error::Success()
                    : ffi::Error::Internal(
                          "BeamZ CUDA temporal workspace launch failed: " +
                          std::to_string(error));
}

ffi::Error StreamedCpmlStepsHandler(
    void* stream, ffi::RemainingArgs args, ffi::RemainingRets rets,
    int32_t abi_version, int32_t cuda_flags, int32_t nsteps, float dt,
    float resolution, int32_t boundary_code, int32_t metric_kind,
    int32_t graph_cache_capacity) {
  if (abi_version != kAbiVersion) {
    return ffi::Error::InvalidArgument("beamz_cuda ABI version mismatch");
  }
  if (nsteps < 1 || metric_kind < 0 || metric_kind > 2) {
    return ffi::Error::InvalidArgument("BeamZ CUDA step count must be positive");
  }
  BeamzBuffer inputs[kCpmlGraphInputCount]{};
  BeamzBuffer outputs[kCpmlGraphOutputCount]{};
  if (auto error = DecodeArgs(args, inputs); error.failure()) return error;
  if (auto error = DecodeRets(rets, outputs); error.failure()) return error;

  GraphLaunches launches = InitializeGraphLaunches(
      abi_version, cuda_flags, dt, resolution, boundary_code, metric_kind,
      true, inputs, outputs);
  const int error = BeamzLaunchProgram(
      stream, InPlaceProgram(launches, nsteps, nullptr, 0, nullptr,
                              graph_cache_capacity));
  return error == 0 ? ffi::Error::Success()
                    : ffi::Error::Internal(
                          "BeamZ CUDA multi-step CPML launch failed: " +
                          std::to_string(error));
}

ffi::Error StreamedSourceGroupsCpmlStepsHandler(
    void* stream, ffi::RemainingArgs args, ffi::RemainingRets rets,
    int32_t abi_version, int32_t cuda_flags, int32_t nsteps, float dt,
    float resolution, int32_t boundary_code, int32_t metric_kind,
    int32_t cpml_enabled, int32_t coincident_source_group_mask,
    int32_t graph_cache_capacity) {
  if (abi_version != kAbiVersion || nsteps < 1 ||
      metric_kind < 0 || metric_kind > 2 || cpml_enabled < 0 ||
      cpml_enabled > 1 || coincident_source_group_mask < 0 ||
      coincident_source_group_mask >= (1 << kSourceGroupCount)) {
    return ffi::Error::InvalidArgument(
        "invalid BeamZ CUDA source-group graph attributes");
  }
  constexpr size_t kSourceInputCount =
      kSourceGroupBufferCount * kSourceGroupCount + 1;
  constexpr size_t kInputCapacity = kCpmlGraphInputCount + kSourceInputCount;
  BeamzBuffer inputs[kInputCapacity]{};
  BeamzBuffer outputs[kCpmlGraphOutputCount]{};
  const size_t graph_input_count =
      cpml_enabled ? kCpmlGraphInputCount : kYeeGraphInputCount;
  const size_t graph_output_count =
      cpml_enabled ? kCpmlGraphOutputCount : kYeeGraphOutputCount;
  if (auto error = DecodeArgs(args, inputs,
                              graph_input_count + kSourceInputCount);
      error.failure()) return error;
  if (auto error = DecodeRets(rets, outputs, graph_output_count);
      error.failure()) return error;

  GraphLaunches launches = InitializeGraphLaunches(
      abi_version, cuda_flags, dt, resolution, boundary_code, metric_kind,
      cpml_enabled != 0, inputs, outputs);

  BeamzSourceGroupLaunch groups[kSourceGroupCount]{};
  const BeamzBuffer& current_step =
      inputs[graph_input_count +
             kSourceGroupBufferCount * kSourceGroupCount];
  InitializeSourceGroups(groups, inputs, graph_input_count, current_step,
                         coincident_source_group_mask);
  const int error = BeamzLaunchProgram(
      stream, InPlaceProgram(launches, nsteps, groups, kSourceGroupCount,
                             nullptr, graph_cache_capacity));
  return error == 0
             ? ffi::Error::Success()
             : ffi::Error::Internal(
                   "BeamZ CUDA source-group CPML graph launch failed: " +
                   std::to_string(error));
}

// A second XLA-owned field bank lets the CUDA implementation freeze every
// timestep's inputs.  The native scheduler can then fuse the CPML-free core
// without racing another block that still needs the old magnetic halo.
ffi::Error TemporalSourceGroupsCpmlStepsHandler(
    void* stream, ffi::RemainingArgs args, ffi::RemainingRets rets,
    int32_t abi_version, int32_t cuda_flags, int32_t nsteps, float dt,
    float resolution, int32_t boundary_code, int32_t metric_kind,
    int32_t coincident_source_group_mask, int32_t graph_cache_capacity) {
  constexpr size_t kGraphInputCount = kCpmlGraphInputCount;
  constexpr size_t kWorkspaceInputCount = 3 * kFieldCount;
  constexpr size_t kSourceInputCount =
      kSourceGroupBufferCount * kSourceGroupCount + 1;
  constexpr size_t kInputCount =
      kGraphInputCount + kWorkspaceInputCount + kSourceInputCount;
  constexpr size_t kOutputCount =
      2 * kFieldCount + 4 * kCpmlTermCount;
  if (abi_version != kAbiVersion || nsteps < 1 ||
      metric_kind < 0 || metric_kind > 2 ||
      coincident_source_group_mask < 0 ||
      coincident_source_group_mask >= (1 << kSourceGroupCount)) {
    return ffi::Error::InvalidArgument(
        "invalid BeamZ CUDA temporal CPML source-group attributes");
  }
  BeamzBuffer inputs[kInputCount]{};
  BeamzBuffer outputs[kOutputCount]{};
  if (auto error = DecodeArgs(args, inputs); error.failure()) return error;
  if (auto error = DecodeRets(rets, outputs); error.failure()) return error;

  TemporalCpmlLaunches launches = InitializeTemporalCpmlLaunches(
      abi_version, cuda_flags, dt, resolution, boundary_code, metric_kind,
      inputs, outputs);

  BeamzSourceGroupLaunch groups[kSourceGroupCount]{};
  constexpr size_t kSourceOffset = kGraphInputCount + kWorkspaceInputCount;
  const BeamzBuffer& current_step =
      inputs[kSourceOffset + kSourceGroupBufferCount * kSourceGroupCount];
  InitializeSourceGroups(groups, inputs, kSourceOffset, current_step,
                         coincident_source_group_mask);
  const int error = BeamzLaunchProgram(
      stream, TemporalProgram(launches, nsteps, groups, kSourceGroupCount,
                              nullptr, graph_cache_capacity));
  return error == 0
             ? ffi::Error::Success()
             : ffi::Error::Internal(
                   "BeamZ CUDA temporal source-group CPML launch failed: " +
                   std::to_string(error));
}

ffi::Error TemporalProgramCpmlStepsHandler(
    void* stream, ffi::RemainingArgs args, ffi::RemainingRets rets,
    int32_t abi_version, int32_t cuda_flags, int32_t nsteps, float dt,
    float resolution, int32_t boundary_code, int32_t metric_kind,
    int32_t monitor_count, int32_t coincident_source_group_mask,
    int32_t graph_cache_capacity) {
  constexpr size_t kGraphInputCount = kCpmlGraphInputCount;
  constexpr size_t kWorkspaceInputCount = 3 * kFieldCount;
  constexpr size_t kSourceInputCount =
      kSourceGroupBufferCount * kSourceGroupCount;
  constexpr size_t kInputCount = kGraphInputCount + kWorkspaceInputCount +
                                 kSourceInputCount + kMonitorInputCount;
  constexpr size_t kStateOutputCount =
      2 * kFieldCount + 4 * kCpmlTermCount;
  constexpr size_t kOutputCount = kStateOutputCount + 3;
  if (abi_version != kAbiVersion || nsteps < 1 ||
      metric_kind < 0 || metric_kind > 2 || monitor_count < 1 ||
      coincident_source_group_mask < 0 ||
      coincident_source_group_mask >= (1 << kSourceGroupCount)) {
    return ffi::Error::InvalidArgument(
        "invalid BeamZ CUDA temporal CPML program attributes");
  }
  BeamzBuffer inputs[kInputCount]{};
  BeamzBuffer outputs[kOutputCount]{};
  if (auto error = DecodeArgs(args, inputs); error.failure()) return error;
  if (auto error = DecodeRets(rets, outputs); error.failure()) return error;

  TemporalCpmlLaunches launches = InitializeTemporalCpmlLaunches(
      abi_version, cuda_flags, dt, resolution, boundary_code, metric_kind,
      inputs, outputs);

  constexpr size_t kSourceOffset = kGraphInputCount + kWorkspaceInputCount;
  constexpr size_t kMonitorOffset = kSourceOffset + kSourceInputCount;
  const BeamzBuffer& current_step =
      inputs[kMonitorOffset + kMonitorCurrentStepInput];
  BeamzSourceGroupLaunch groups[kSourceGroupCount]{};
  InitializeSourceGroups(groups, inputs, kSourceOffset, current_step,
                         coincident_source_group_mask);
  BeamzDftGroupLaunch monitors = InitializeDftGroups(
      inputs, kMonitorOffset, outputs[kStateOutputCount],
      outputs[kStateOutputCount + 1], outputs[kStateOutputCount + 2],
      monitor_count);

  const int error = BeamzLaunchProgram(
      stream, TemporalProgram(launches, nsteps, groups, kSourceGroupCount,
                              &monitors, graph_cache_capacity));
  return error == 0
             ? ffi::Error::Success()
             : ffi::Error::Internal(
                   "BeamZ CUDA temporal CPML program launch failed: " +
                   std::to_string(error));
}

ffi::Error StreamedProgramCpmlStepsHandler(
    void* stream, ffi::RemainingArgs args, ffi::RemainingRets rets,
    int32_t abi_version, int32_t cuda_flags, int32_t nsteps, float dt,
    float resolution, int32_t boundary_code, int32_t metric_kind,
    int32_t cpml_enabled, int32_t monitor_count,
    int32_t coincident_source_group_mask, int32_t graph_cache_capacity) {
  if (abi_version != kAbiVersion || nsteps < 1 ||
      metric_kind < 0 || metric_kind > 2 || cpml_enabled < 0 ||
      cpml_enabled > 1 || monitor_count < 1 ||
      coincident_source_group_mask < 0 ||
      coincident_source_group_mask >= (1 << kSourceGroupCount)) {
    return ffi::Error::InvalidArgument(
        "invalid BeamZ CUDA program graph attributes");
  }
  constexpr size_t kSourceInputCount =
      kSourceGroupBufferCount * kSourceGroupCount;
  constexpr size_t kInputCapacity =
      kCpmlGraphInputCount + kSourceInputCount + kMonitorInputCount;
  constexpr size_t kOutputCapacity = kCpmlGraphOutputCount + 3;
  BeamzBuffer inputs[kInputCapacity]{};
  BeamzBuffer outputs[kOutputCapacity]{};
  const size_t graph_input_count =
      cpml_enabled ? kCpmlGraphInputCount : kYeeGraphInputCount;
  const size_t graph_output_count =
      cpml_enabled ? kCpmlGraphOutputCount : kYeeGraphOutputCount;
  if (auto error = DecodeArgs(args, inputs,
                              graph_input_count + kSourceInputCount +
                                  kMonitorInputCount);
      error.failure()) return error;
  if (auto error = DecodeRets(rets, outputs, graph_output_count + 3);
      error.failure()) return error;

  GraphLaunches launches = InitializeGraphLaunches(
      abi_version, cuda_flags, dt, resolution, boundary_code, metric_kind,
      cpml_enabled != 0, inputs, outputs);

  const size_t monitor_start = graph_input_count + kSourceInputCount;
  const BeamzBuffer& current_step =
      inputs[monitor_start + kMonitorCurrentStepInput];
  BeamzSourceGroupLaunch groups[kSourceGroupCount]{};
  InitializeSourceGroups(groups, inputs, graph_input_count, current_step,
                         coincident_source_group_mask);
  BeamzDftGroupLaunch monitors = InitializeDftGroups(
      inputs, monitor_start, inputs[monitor_start + kMonitorDftReInput],
      inputs[monitor_start + kMonitorDftImInput],
      inputs[monitor_start + kMonitorDftWeightInput], monitor_count);

  const int error = BeamzLaunchProgram(
      stream, InPlaceProgram(launches, nsteps, groups, kSourceGroupCount,
                             &monitors, graph_cache_capacity));
  return error == 0
             ? ffi::Error::Success()
             : ffi::Error::Internal(
                   "BeamZ CUDA program graph launch failed: " +
                   std::to_string(error));
}

ffi::Error ProgramHandler(
    void* stream, ffi::RemainingArgs args, ffi::RemainingRets rets,
    int32_t abi_version, int32_t cuda_flags, int32_t nsteps, float dt,
    float resolution, int32_t boundary_code, int32_t metric_kind,
    int32_t program_layout, int32_t cpml_enabled, int32_t monitor_count,
    int32_t coincident_source_group_mask, int32_t graph_cache_capacity) {
  if (graph_cache_capacity < 0 || graph_cache_capacity > 4096) {
    return ffi::Error::InvalidArgument(
        "BeamZ CUDA graph-cache capacity must be from 0 to 4096");
  }
  switch (program_layout) {
    case kProgramLayoutYeeInPlace:
      return StreamedStepsHandler(stream, args, rets, abi_version, cuda_flags,
                                  nsteps, dt, resolution, boundary_code,
                                  metric_kind, graph_cache_capacity);
    case kProgramLayoutYeeTemporal:
      return TemporalStepsHandler(stream, args, rets, abi_version, cuda_flags,
                                  nsteps, dt, resolution, boundary_code,
                                  metric_kind, graph_cache_capacity);
    case kProgramLayoutCpmlInPlace:
      return StreamedCpmlStepsHandler(stream, args, rets, abi_version,
                                      cuda_flags, nsteps, dt, resolution,
                                      boundary_code, metric_kind,
                                      graph_cache_capacity);
    case kProgramLayoutSourceInPlace:
      return StreamedSourceGroupsCpmlStepsHandler(
          stream, args, rets, abi_version, cuda_flags, nsteps, dt, resolution,
          boundary_code, metric_kind, cpml_enabled,
          coincident_source_group_mask, graph_cache_capacity);
    case kProgramLayoutSourceTemporalCpml:
      return TemporalSourceGroupsCpmlStepsHandler(
          stream, args, rets, abi_version, cuda_flags, nsteps, dt, resolution,
          boundary_code, metric_kind, coincident_source_group_mask,
          graph_cache_capacity);
    case kProgramLayoutMonitorInPlace:
      return StreamedProgramCpmlStepsHandler(
          stream, args, rets, abi_version, cuda_flags, nsteps, dt, resolution,
          boundary_code, metric_kind, cpml_enabled, monitor_count,
          coincident_source_group_mask, graph_cache_capacity);
    case kProgramLayoutMonitorTemporalCpml:
      return TemporalProgramCpmlStepsHandler(
          stream, args, rets, abi_version, cuda_flags, nsteps, dt, resolution,
          boundary_code, metric_kind, monitor_count,
          coincident_source_group_mask, graph_cache_capacity);
    default:
      return ffi::Error::InvalidArgument(
          "unknown BeamZ CUDA program buffer layout");
  }
}

ffi::Error HopperHandler(void* stream, ffi::RemainingArgs args,
                         ffi::RemainingRets rets, int32_t abi_version,
                         int32_t cuda_flags, int32_t phase, int32_t nterms,
                         float dt, float resolution, int32_t boundary_code,
                         int32_t metric_kind) {
  return Dispatch(BeamzLaunchHopper, stream, args, rets, abi_version,
                  cuda_flags, phase, nterms, dt, resolution, boundary_code,
                  metric_kind);
}

}  // namespace

XLA_FFI_DEFINE_HANDLER_SYMBOL(beamz_cuda_streamed, StreamedHandler,
                              ffi::Ffi::Bind()
                                  .Ctx<ffi::PlatformStream<void*>>()
                                  .RemainingArgs()
                                  .RemainingRets()
                                  .Attr<int32_t>("abi_version")
                                  .Attr<int32_t>("cuda_flags")
                                  .Attr<int32_t>("phase")
                                  .Attr<int32_t>("nterms")
                                  .Attr<float>("dt")
                                  .Attr<float>("resolution")
                                  .Attr<int32_t>("boundary_code")
                                  .Attr<int32_t>("metric_kind"));

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    beamz_cuda_program, ProgramHandler,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<void*>>()
        .RemainingArgs()
        .RemainingRets()
        .Attr<int32_t>("abi_version")
        .Attr<int32_t>("cuda_flags")
        .Attr<int32_t>("nsteps")
        .Attr<float>("dt")
        .Attr<float>("resolution")
        .Attr<int32_t>("boundary_code")
        .Attr<int32_t>("metric_kind")
        .Attr<int32_t>("program_layout")
        .Attr<int32_t>("cpml_enabled")
        .Attr<int32_t>("monitor_count")
        .Attr<int32_t>("coincident_source_group_mask")
        .Attr<int32_t>("graph_cache_capacity"));
XLA_FFI_DEFINE_HANDLER_SYMBOL(beamz_cuda_hopper, HopperHandler,
                              ffi::Ffi::Bind()
                                  .Ctx<ffi::PlatformStream<void*>>()
                                  .RemainingArgs()
                                  .RemainingRets()
                                  .Attr<int32_t>("abi_version")
                                  .Attr<int32_t>("cuda_flags")
                                  .Attr<int32_t>("phase")
                                  .Attr<int32_t>("nterms")
                                  .Attr<float>("dt")
                                  .Attr<float>("resolution")
                                  .Attr<int32_t>("boundary_code")
                                  .Attr<int32_t>("metric_kind"));
