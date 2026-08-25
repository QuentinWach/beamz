#include <cuda_runtime_api.h>

#include <cstdint>
#include <limits>

#include "abi_layout.h"
#include "graph.h"
#include "kernels.h"
#include "launch.h"

namespace {

using namespace beamz::cuda::abi;

bool FlagEnabled(const BeamzLaunch& launch, int32_t flag) {
  return (launch.cuda_flags & flag) != 0;
}

bool ScheduleFlagEnabled(const BeamzProgramLaunch& program, int32_t flag) {
  return (program.schedule_flags & flag) != 0;
}

bool FitsIntOffsets(const BeamzBuffer& value) {
  if (value.rank < 0 || value.rank > 4) return false;
  int64_t elements = 1;
  for (int axis = 0; axis < value.rank; ++axis) {
    if (value.dims[axis] < 0 ||
        value.dims[axis] > std::numeric_limits<int>::max()) {
      return false;
    }
    if (value.dims[axis] != 0 &&
        elements > std::numeric_limits<int>::max() / value.dims[axis]) {
      return false;
    }
    elements *= value.dims[axis];
    if (elements > std::numeric_limits<int>::max()) return false;
  }
  return true;
}

bool HasType(const BeamzBuffer& value, BeamzElementType type) {
  return value.element_type == type;
}

cudaError_t ValidateSourceGroups(const BeamzSourceGroupLaunch* groups,
                                 int32_t count) {
  if (groups == nullptr) return count == 0 ? cudaSuccess : cudaErrorInvalidValue;
  if (count != kSourceGroupCount) return cudaErrorInvalidValue;
  for (int32_t index = 0; index < count; ++index) {
    const BeamzSourceGroupLaunch& group = groups[index];
    if (group.component < 0 || group.component > 2 || group.timing < 0 ||
        group.timing > 2 || group.coefficients.rank != 4 ||
        group.waveforms.rank != 2 || group.starts.rank != 2 ||
        group.current_step.rank != 0 || group.coincident < 0 ||
        group.coincident > 1 || group.disjoint < 0 || group.disjoint > 1 ||
        (group.coincident != 0 && group.disjoint != 0) ||
        !HasType(group.coefficients, kBeamzF32) ||
        !HasType(group.waveforms, kBeamzF32) ||
        !HasType(group.starts, kBeamzS32) ||
        !HasType(group.current_step, kBeamzS32) ||
        group.coefficients.dims[0] != group.waveforms.dims[0] ||
        group.coefficients.dims[0] != group.starts.dims[0] ||
        group.starts.dims[1] != 3 || group.waveforms.dims[1] < 1 ||
        group.coefficients.dims[1] < 1 || group.coefficients.dims[2] < 1 ||
        group.coefficients.dims[3] < 1 || group.current_step.data == nullptr ||
        (group.coefficients.dims[0] > 0 &&
         (group.coefficients.data == nullptr || group.waveforms.data == nullptr ||
          group.starts.data == nullptr)) ||
        !FitsIntOffsets(group.coefficients) ||
        !FitsIntOffsets(group.waveforms) || !FitsIntOffsets(group.starts) ||
        !FitsIntOffsets(group.current_step)) {
      return cudaErrorInvalidValue;
    }
  }
  return cudaSuccess;
}

bool HasPackedLosslessMaterial(const BeamzLaunch& launch) {
  if (launch.phase != 1) return false;
  for (int component = 0; component < 3; ++component) {
    if (launch.inputs[6 + component].rank != 1 ||
        launch.inputs[9 + component].rank != 1 ||
        launch.inputs[9 + component].element_type != kBeamzS32) {
      return false;
    }
  }
  return true;
}

bool UniformCpml(const BeamzProgramLaunch& program) {
  if (program.h_ab.nterms != kCpmlTermCount ||
      program.h_ab.uniform_cpml_thickness <= 0 ||
      program.e_ab.uniform_cpml_thickness !=
          program.h_ab.uniform_cpml_thickness) {
    return false;
  }
  if (program.field_bank_count == 2 &&
      (program.h_ba.uniform_cpml_thickness !=
           program.h_ab.uniform_cpml_thickness ||
       program.e_ba.uniform_cpml_thickness !=
           program.h_ab.uniform_cpml_thickness)) {
    return false;
  }
  return true;
}

bool CombinedCpmlCoreSupported(const BeamzProgramLaunch& program) {
  if (!UniformCpml(program) || program.h_ab.metric_kind != 0 ||
      program.e_ab.metric_kind != 0 ||
      !HasPackedLosslessMaterial(program.e_ab)) {
    return false;
  }
  if (program.field_bank_count == 2 &&
      (program.h_ba.metric_kind != 0 || program.e_ba.metric_kind != 0 ||
       !HasPackedLosslessMaterial(program.e_ba))) {
    return false;
  }
  const BeamzLaunch* h_launches[] = {
      &program.h_ab, program.field_bank_count == 2 ? &program.h_ba : nullptr};
  for (const BeamzLaunch* h_launch : h_launches) {
    if (h_launch == nullptr) continue;
    for (int material = 0; material < 6; ++material) {
      if (h_launch->inputs[6 + material].rank != 0) return false;
    }
  }
  const int64_t thickness = program.h_ab.uniform_cpml_thickness;
  for (int axis = 0; axis < 3; ++axis) {
    int64_t extent = program.h_ab.outputs[0].dims[axis];
    for (int component = 1; component < 3; ++component) {
      if (program.h_ab.outputs[component].dims[axis] < extent) {
        extent = program.h_ab.outputs[component].dims[axis];
      }
    }
    if (extent <= 2 * thickness) return false;
  }
  return true;
}

cudaError_t ValidateSchedulePlan(const BeamzProgramLaunch& program) {
  constexpr int32_t kKnownFlags =
      kNativeScheduleCpml | kNativeScheduleTemporal |
      kNativeScheduleUniformCpml | kNativeSchedulePackedMaterial |
      kNativeScheduleCombinedCpmlCore | kNativeScheduleSources |
      kNativeScheduleMonitors | kNativeScheduleGraphCache;
  if (program.schedule_flags < 0 ||
      (program.schedule_flags & ~kKnownFlags) != 0) {
    return cudaErrorInvalidValue;
  }
  const bool cpml = program.h_ab.nterms == kCpmlTermCount;
  const bool temporal = program.field_bank_count == 2;
  const bool sources = program.source_group_count != 0;
  const bool monitors = program.monitors != nullptr;
  const bool packed_material =
      HasPackedLosslessMaterial(program.e_ab) &&
      (program.field_bank_count == 1 ||
       HasPackedLosslessMaterial(program.e_ba));
  const bool graph_cache = FlagEnabled(program.h_ab, kBeamzGraphCache);
  if (ScheduleFlagEnabled(program, kNativeScheduleCpml) != cpml ||
      ScheduleFlagEnabled(program, kNativeScheduleTemporal) != temporal ||
      ScheduleFlagEnabled(program, kNativeScheduleUniformCpml) !=
          UniformCpml(program) ||
      ScheduleFlagEnabled(program, kNativeSchedulePackedMaterial) !=
          packed_material ||
      ScheduleFlagEnabled(program, kNativeScheduleSources) != sources ||
      ScheduleFlagEnabled(program, kNativeScheduleMonitors) != monitors ||
      ScheduleFlagEnabled(program, kNativeScheduleGraphCache) != graph_cache ||
      (ScheduleFlagEnabled(program, kNativeScheduleCombinedCpmlCore) &&
       !CombinedCpmlCoreSupported(program))) {
    return cudaErrorInvalidValue;
  }
  return cudaSuccess;
}

cudaError_t ValidateMonitors(const BeamzDftGroupLaunch* value) {
  if (value == nullptr) return cudaSuccess;
  const BeamzDftGroupLaunch& monitors = *value;
  if (monitors.monitor_count < 1 || monitors.indices.rank != 4 ||
      monitors.weights.rank != 4 || monitors.frequencies.rank != 2 ||
      monitors.component_masks.rank != 2 || monitors.counts.rank != 2 ||
      monitors.codes.rank != 2 || monitors.windows.rank != 2 ||
      monitors.dft_re.rank != 1 || monitors.dft_im.rank != 1 ||
      monitors.dft_weight.rank != 1 || monitors.time.rank != 0 ||
      monitors.current_step.rank != 0 || monitors.phase_sin.rank != 2 ||
      monitors.phase_cos.rank != 2 || monitors.phase_window.rank != 2 ||
      !HasType(monitors.indices, kBeamzS32) ||
      !HasType(monitors.weights, kBeamzF32) ||
      !HasType(monitors.frequencies, kBeamzF32) ||
      !HasType(monitors.component_masks, kBeamzF32) ||
      !HasType(monitors.counts, kBeamzS32) ||
      !HasType(monitors.codes, kBeamzS32) ||
      !HasType(monitors.windows, kBeamzF32) ||
      !HasType(monitors.dft_re, kBeamzF32) ||
      !HasType(monitors.dft_im, kBeamzF32) ||
      !HasType(monitors.dft_weight, kBeamzF32) ||
      !HasType(monitors.phase_sin, kBeamzF32) ||
      !HasType(monitors.phase_cos, kBeamzF32) ||
      !HasType(monitors.phase_window, kBeamzF32) ||
      !HasType(monitors.time, kBeamzF32) ||
      !HasType(monitors.current_step, kBeamzS32) ||
      monitors.indices.dims[0] < monitors.monitor_count ||
      monitors.indices.dims[1] != 6 ||
      monitors.weights.dims[0] != monitors.indices.dims[0] ||
      monitors.weights.dims[1] != monitors.indices.dims[1] ||
      monitors.weights.dims[2] != monitors.indices.dims[2] ||
      monitors.weights.dims[3] != monitors.indices.dims[3] ||
      monitors.frequencies.dims[0] < monitors.monitor_count ||
      monitors.component_masks.dims[0] < monitors.monitor_count ||
      monitors.component_masks.dims[1] != 6 ||
      monitors.counts.dims[0] < monitors.monitor_count ||
      monitors.counts.dims[1] != 5 ||
      monitors.codes.dims[0] < monitors.monitor_count ||
      monitors.codes.dims[1] != 2 ||
      monitors.windows.dims[0] < monitors.monitor_count ||
      monitors.windows.dims[1] != 3 || monitors.dft_re.dims[0] < 1 ||
      monitors.dft_im.dims[0] != monitors.dft_re.dims[0] ||
      monitors.dft_weight.dims[0] < 1 || monitors.indices.dims[2] < 1 ||
      monitors.indices.dims[3] < 1 || monitors.frequencies.dims[1] < 1 ||
      monitors.phase_sin.dims[0] < monitors.monitor_count ||
      monitors.phase_sin.dims[1] != monitors.frequencies.dims[1] ||
      monitors.phase_cos.dims[0] != monitors.phase_sin.dims[0] ||
      monitors.phase_cos.dims[1] != monitors.phase_sin.dims[1] ||
      monitors.phase_window.dims[0] != monitors.phase_sin.dims[0] ||
      monitors.phase_window.dims[1] != monitors.phase_sin.dims[1] ||
      monitors.indices.data == nullptr ||
      monitors.weights.data == nullptr || monitors.frequencies.data == nullptr ||
      monitors.component_masks.data == nullptr || monitors.counts.data == nullptr ||
      monitors.codes.data == nullptr || monitors.windows.data == nullptr ||
      monitors.dft_re.data == nullptr || monitors.dft_im.data == nullptr ||
      monitors.dft_weight.data == nullptr || monitors.time.data == nullptr ||
      monitors.current_step.data == nullptr || monitors.phase_sin.data == nullptr ||
      monitors.phase_cos.data == nullptr ||
      monitors.phase_window.data == nullptr) {
    return cudaErrorInvalidValue;
  }
  const BeamzBuffer buffers[] = {
      monitors.indices,         monitors.weights, monitors.frequencies,
      monitors.component_masks, monitors.counts,  monitors.codes,
      monitors.windows,         monitors.dft_re,   monitors.dft_im,
      monitors.dft_weight,      monitors.phase_sin, monitors.phase_cos,
      monitors.phase_window,    monitors.time,     monitors.current_step};
  for (const BeamzBuffer& buffer : buffers) {
    if (!FitsIntOffsets(buffer)) return cudaErrorInvalidValue;
  }
  return cudaSuccess;
}

cudaError_t ValidateProgram(const BeamzProgramLaunch& program) {
  if (program.nsteps < 1 || program.graph_cache_capacity < 0 ||
      (program.field_bank_count != 1 && program.field_bank_count != 2) ||
      program.h_ab.phase != 0 || program.e_ab.phase != 1 ||
      program.h_ab.nterms != program.e_ab.nterms) {
    return cudaErrorInvalidValue;
  }
  if (cudaError_t error = BeamzValidatePhase(program.h_ab);
      error != cudaSuccess) {
    return error;
  }
  if (cudaError_t error = BeamzValidatePhase(program.e_ab);
      error != cudaSuccess) {
    return error;
  }
  if (program.field_bank_count == 2) {
    if (program.h_ba.phase != 0 || program.e_ba.phase != 1 ||
        program.h_ba.nterms != program.h_ab.nterms ||
        program.e_ba.nterms != program.e_ab.nterms) {
      return cudaErrorInvalidValue;
    }
    if (cudaError_t error = BeamzValidatePhase(program.h_ba);
        error != cudaSuccess) {
      return error;
    }
    if (cudaError_t error = BeamzValidatePhase(program.e_ba);
        error != cudaSuccess) {
      return error;
    }
    if (program.h_ab.nterms == 0) {
      if (program.nsteps < 4 || program.source_group_count != 0 ||
          program.monitors != nullptr ||
          program.e_ab.metric_kind != program.h_ab.metric_kind ||
          program.h_ba.metric_kind != program.h_ab.metric_kind ||
          program.e_ba.metric_kind != program.h_ab.metric_kind ||
          program.h_ab.metallic_edges != 63 ||
          program.e_ab.metallic_edges != 63 ||
          program.h_ba.metallic_edges != 63 ||
          program.e_ba.metallic_edges != 63) {
        return cudaErrorInvalidValue;
      }
      for (int material = 0; material < 6; ++material) {
        const int h_rank = program.h_ab.inputs[6 + material].rank;
        const int e_rank = program.e_ab.inputs[6 + material].rank;
        if ((h_rank != 0 && h_rank != 3) ||
            (e_rank != 0 && e_rank != 3)) {
          return cudaErrorInvalidValue;
        }
      }
    } else if (program.h_ab.nterms != kCpmlTermCount ||
               program.source_group_count != kSourceGroupCount) {
      return cudaErrorInvalidValue;
    }
  }
  if (cudaError_t error = ValidateSourceGroups(program.source_groups,
                                                program.source_group_count);
      error != cudaSuccess) {
    return error;
  }
  if (cudaError_t error = ValidateMonitors(program.monitors);
      error != cudaSuccess) {
    return error;
  }
  return ValidateSchedulePlan(program);
}

int LaunchInPlaceProgram(void* raw_stream, const BeamzProgramLaunch& program) {
  const BeamzLaunch& h_launch = program.h_ab;
  const BeamzLaunch& e_launch = program.e_ab;
  const BeamzSourceGroupLaunch* source_groups = program.source_groups;
  const int32_t source_group_count = program.source_group_count;
  const BeamzDftGroupLaunch* monitors = program.monitors;
  auto stream = reinterpret_cast<cudaStream_t>(raw_stream);
  const std::string graph_key = BeamzGraphKey("in-place", raw_stream, program);
  const bool cache_enabled = FlagEnabled(h_launch, kBeamzGraphCache);
  auto enqueue = [&]() {
    cudaError_t error = cudaSuccess;
    const bool split_cpml =
        ScheduleFlagEnabled(program, kNativeScheduleCombinedCpmlCore);
    auto enqueue_sources = [&](int timing, int32_t step) {
      for (int32_t index = 0; index < source_group_count; ++index) {
        const BeamzSourceGroupLaunch& group = source_groups[index];
        if (group.timing != timing || group.coefficients.dims[0] == 0) continue;
        const BeamzLaunch& target_launch = timing == 1 ? h_launch : e_launch;
        error = BeamzEnqueueSourceGroup(
            stream, h_launch, target_launch.outputs[group.component], group,
            step);
        if (error != cudaSuccess) return;
      }
    };
    for (int32_t step = 0; step < program.nsteps; ++step) {
      if (source_groups != nullptr) {
        enqueue_sources(0, step);
        if (error != cudaSuccess) break;
      }
      if (split_cpml) {
        // H and E remain separate launches: graph outputs alias their inputs, so
        // cross-block H-to-E fusion would race while reading the old H halo.
        error = BeamzEnqueueCpmlPhase(stream, h_launch);
        if (error != cudaSuccess) break;
        if (source_groups != nullptr) {
          enqueue_sources(1, step);
          if (error != cudaSuccess) break;
        }
        error = BeamzEnqueueCpmlPhase(stream, e_launch);
      } else {
        error = BeamzEnqueuePhase(raw_stream, h_launch);
        if (error != cudaSuccess) break;
        if (source_groups != nullptr) {
          enqueue_sources(1, step);
          if (error != cudaSuccess) break;
        }
        error = BeamzEnqueuePhase(raw_stream, e_launch);
      }
      if (error != cudaSuccess) break;
      if (source_groups != nullptr) {
        enqueue_sources(2, step);
        if (error != cudaSuccess) break;
      }
      if (monitors != nullptr) {
        error = BeamzEnqueueDftGroups(stream, h_launch, e_launch, *monitors,
                                      step);
        if (error != cudaSuccess) break;
      }
    }
    return error;
  };
  return BeamzLaunchGraph(stream, graph_key, cache_enabled,
                          program.graph_cache_capacity, enqueue);
}

int LaunchTemporalYeeProgram(void* raw_stream,
                             const BeamzProgramLaunch& program) {
  const BeamzLaunch& h_ab = program.h_ab;
  const BeamzLaunch& e_ab = program.e_ab;
  const BeamzLaunch& h_ba = program.h_ba;
  const BeamzLaunch& e_ba = program.e_ba;
  auto stream = reinterpret_cast<cudaStream_t>(raw_stream);

  BeamzLaunch h_tail = h_ab;
  BeamzLaunch e_tail = e_ab;
  for (int component = 0; component < 3; ++component) {
    h_tail.outputs[component] = h_ba.outputs[component];
    e_tail.inputs[3 + component] = h_tail.outputs[component];
    e_tail.outputs[component] = e_ba.outputs[component];
  }

  const std::string graph_key =
      BeamzGraphKey("temporal-yee", raw_stream, program);
  const bool cache_enabled = FlagEnabled(h_ab, kBeamzGraphCache);
  auto enqueue = [&]() {
    cudaError_t error = cudaSuccess;
    for (int32_t pair = 0; pair < program.nsteps / 2; ++pair) {
      error = BeamzEnqueueFusedFullStep(stream, h_ab, e_ab);
      if (error != cudaSuccess) return error;
      error = BeamzEnqueueFusedFullStep(stream, h_ba, e_ba);
      if (error != cudaSuccess) return error;
    }
    for (int32_t step = 2 * (program.nsteps / 2); step < program.nsteps;
         ++step) {
      error = BeamzEnqueuePhase(raw_stream, h_tail);
      if (error != cudaSuccess) return error;
      error = BeamzEnqueuePhase(raw_stream, e_tail);
      if (error != cudaSuccess) return error;
    }
    return error;
  };
  return BeamzLaunchGraph(stream, graph_key, cache_enabled,
                          program.graph_cache_capacity, enqueue);
}

int LaunchTemporalCpmlProgram(void* raw_stream,
                              const BeamzProgramLaunch& program) {
  const BeamzLaunch& h_ab = program.h_ab;
  const BeamzLaunch& e_ab = program.e_ab;
  const BeamzLaunch& h_ba = program.h_ba;
  const BeamzLaunch& e_ba = program.e_ba;
  const BeamzSourceGroupLaunch* source_groups = program.source_groups;
  const int32_t source_group_count = program.source_group_count;
  const BeamzDftGroupLaunch* monitors = program.monitors;
  auto stream = reinterpret_cast<cudaStream_t>(raw_stream);
  const std::string graph_key =
      BeamzGraphKey("temporal-cpml", raw_stream, program);
  const bool cache_enabled = FlagEnabled(h_ab, kBeamzGraphCache);

  auto enqueue = [&]() {
    cudaError_t error = cudaSuccess;
    auto enqueue_sources = [&](const BeamzLaunch& h_launch,
                               const BeamzLaunch& e_launch, int timing,
                               int32_t step) {
      for (int32_t index = 0; index < source_group_count; ++index) {
        const BeamzSourceGroupLaunch& group = source_groups[index];
        if (group.timing != timing || group.coefficients.dims[0] == 0) continue;
        const BeamzBuffer& target =
            timing == 0 ? e_launch.inputs[group.component]
                        : (timing == 1 ? h_launch.outputs[group.component]
                                       : e_launch.outputs[group.component]);
        error = BeamzEnqueueSourceGroup(stream, h_launch, target, group, step);
        if (error != cudaSuccess) return;
      }
    };
    for (int32_t step = 0; step < program.nsteps; ++step) {
      const bool ab = (step & 1) == 0;
      const BeamzLaunch& h_launch = ab ? h_ab : h_ba;
      const BeamzLaunch& e_launch = ab ? e_ab : e_ba;
      enqueue_sources(h_launch, e_launch, 0, step);
      if (error != cudaSuccess) return error;
      if (ScheduleFlagEnabled(program, kNativeScheduleCombinedCpmlCore)) {
        error = BeamzEnqueueCpmlPhase(stream, h_launch);
        if (error != cudaSuccess) return error;
        enqueue_sources(h_launch, e_launch, 1, step);
        if (error != cudaSuccess) return error;
        error = BeamzEnqueueCpmlPhase(stream, e_launch);
      } else {
        error = BeamzEnqueuePhase(raw_stream, h_launch);
        if (error != cudaSuccess) return error;
        enqueue_sources(h_launch, e_launch, 1, step);
        if (error != cudaSuccess) return error;
        error = BeamzEnqueuePhase(raw_stream, e_launch);
      }
      if (error != cudaSuccess) return error;
      enqueue_sources(h_launch, e_launch, 2, step);
      if (error != cudaSuccess) return error;
      if (monitors != nullptr) {
        error = BeamzEnqueueDftGroups(stream, h_launch, e_launch, *monitors,
                                      step);
        if (error != cudaSuccess) return error;
      }
    }
    return error;
  };
  return BeamzLaunchGraph(stream, graph_key, cache_enabled,
                          program.graph_cache_capacity, enqueue);
}

}  // namespace

int BeamzLaunchProgram(void* raw_stream, const BeamzProgramLaunch& program) {
  if (cudaError_t error = ValidateProgram(program); error != cudaSuccess) {
    return static_cast<int>(error);
  }
  if (program.field_bank_count == 1) {
    return LaunchInPlaceProgram(raw_stream, program);
  }
  if (program.h_ab.nterms == 0 && program.source_group_count == 0 &&
      program.monitors == nullptr) {
    return LaunchTemporalYeeProgram(raw_stream, program);
  }
  if (program.h_ab.nterms == 6 && program.source_group_count == 9) {
    return LaunchTemporalCpmlProgram(raw_stream, program);
  }
  return cudaErrorInvalidValue;
}
