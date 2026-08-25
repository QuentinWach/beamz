#include <cuda_runtime_api.h>

#include <cstdint>

#include "kernels.h"
#include "launch.h"

namespace {

constexpr int kTileX = 32;
constexpr int kTileY = 4;
constexpr int kTileZ = 2;

dim3 SourceThreads(int64_t x_extent) {
  if (x_extent >= 32) {
    return dim3(kTileX, kTileY, kTileZ);
  }
  if (x_extent <= 1) return dim3(1, 64, 2);
  if (x_extent <= 2) return dim3(2, 16, 4);
  if (x_extent <= 4) return dim3(4, 16, 2);
  if (x_extent <= 8) return dim3(8, 8, 2);
  return dim3(16, 4, 2);
}

struct DftFields {
  BeamzBuffer values[6];
  float dt;
};

DftFields MakeDftFields(const BeamzLaunch& h_launch,
                        const BeamzLaunch& e_launch) {
  DftFields fields{};
  for (int component = 0; component < 3; ++component) {
    fields.values[component] = e_launch.outputs[component];
    fields.values[3 + component] = h_launch.outputs[component];
  }
  fields.dt = e_launch.dt;
  return fields;
}

__device__ __forceinline__ int64_t ElementCount(const BeamzBuffer& value) {
  int64_t elements = 1;
  for (int axis = 0; axis < value.rank; ++axis) elements *= value.dims[axis];
  return elements;
}

__device__ __forceinline__ bool SourceCellConstrained(
    const BeamzBuffer& target, int component, int phase, int metallic_edges,
    int z, int y, int x) {
  const int normal_axis = 2 - component;
  const int coordinates[3] = {z, y, x};
  if (phase == 0) {
    const int coordinate = coordinates[normal_axis];
    return (coordinate == 0 &&
            (metallic_edges & (1 << (2 * normal_axis)))) ||
           (coordinate == target.dims[normal_axis] - 1 &&
            (metallic_edges & (1 << (2 * normal_axis + 1))));
  }
  for (int axis = 0; axis < 3; ++axis) {
    if (axis == normal_axis) continue;
    const int coordinate = coordinates[axis];
    if ((coordinate == 0 && (metallic_edges & (1 << (2 * axis)))) ||
        (coordinate == target.dims[axis] - 1 &&
         (metallic_edges & (1 << (2 * axis + 1))))) {
      return true;
    }
  }
  return false;
}

template <bool Atomic>
__device__ __forceinline__ void ApplySourceGroupCell(
    BeamzBuffer target, BeamzSourceGroupLaunch group, int source_index,
    int step_offset, int metallic_edges, int z, int y, int x) {
  if (z >= group.coefficients.dims[1] ||
      y >= group.coefficients.dims[2] ||
      x >= group.coefficients.dims[3]) {
    return;
  }
  const auto* starts = static_cast<const int32_t*>(group.starts.data);
  const int64_t target_z = static_cast<int64_t>(starts[3 * source_index]) + z;
  const int64_t target_y =
      static_cast<int64_t>(starts[3 * source_index + 1]) + y;
  const int64_t target_x =
      static_cast<int64_t>(starts[3 * source_index + 2]) + x;
  if (target_z < 0 || target_z >= target.dims[0] || target_y < 0 ||
      target_y >= target.dims[1] || target_x < 0 ||
      target_x >= target.dims[2]) {
    return;
  }
  // Sources injected after a field update are followed by PEC restoration in the
  // canonical step. Skipping those constrained additions is equivalent because
  // the native field update has already written zero to every constrained cell.
  if (group.timing != 0 &&
      SourceCellConstrained(target, group.component,
                            group.timing == 1 ? 0 : 1, metallic_edges,
                            static_cast<int>(target_z),
                            static_cast<int>(target_y),
                            static_cast<int>(target_x))) {
    return;
  }
  int waveform_index =
      static_cast<const int32_t*>(group.current_step.data)[0] + step_offset;
  waveform_index = waveform_index < 0 ? 0 : waveform_index;
  waveform_index =
      waveform_index >= group.waveforms.dims[1]
          ? static_cast<int>(group.waveforms.dims[1]) - 1
          : waveform_index;
  const int waveform_offset =
      source_index * static_cast<int>(group.waveforms.dims[1]) +
      waveform_index;
  const int coefficient_offset =
      ((source_index * static_cast<int>(group.coefficients.dims[1]) + z) *
           static_cast<int>(group.coefficients.dims[2]) +
       y) *
          static_cast<int>(group.coefficients.dims[3]) +
      x;
  const int target_offset =
      (static_cast<int>(target_z) * static_cast<int>(target.dims[1]) +
       static_cast<int>(target_y)) *
          static_cast<int>(target.dims[2]) +
      static_cast<int>(target_x);
  const float contribution =
      static_cast<const float*>(group.coefficients.data)[coefficient_offset] *
      static_cast<const float*>(group.waveforms.data)[waveform_offset];
  if constexpr (Atomic) {
    atomicAdd(static_cast<float*>(target.data) + target_offset, contribution);
  } else {
    static_cast<float*>(target.data)[target_offset] += contribution;
  }
}

template <int Timing>
__global__ void ApplySingleSourceGroup(BeamzBuffer target,
                                       BeamzSourceGroupLaunch group,
                                       int step_offset,
                                       int metallic_edges) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  const int z = blockIdx.z * blockDim.z + threadIdx.z;
  if (z >= group.coefficients.dims[1] ||
      y >= group.coefficients.dims[2] ||
      x >= group.coefficients.dims[3]) {
    return;
  }
  const auto* starts = static_cast<const int32_t*>(group.starts.data);
  const int64_t target_z = static_cast<int64_t>(starts[0]) + z;
  const int64_t target_y = static_cast<int64_t>(starts[1]) + y;
  const int64_t target_x = static_cast<int64_t>(starts[2]) + x;
  if (target_z < 0 || target_z >= target.dims[0] || target_y < 0 ||
      target_y >= target.dims[1] || target_x < 0 ||
      target_x >= target.dims[2]) {
    return;
  }
  if constexpr (Timing != 0) {
    if (SourceCellConstrained(target, group.component, Timing == 1 ? 0 : 1,
                              metallic_edges, static_cast<int>(target_z),
                              static_cast<int>(target_y),
                              static_cast<int>(target_x))) {
      return;
    }
  }
  int waveform_index =
      static_cast<const int32_t*>(group.current_step.data)[0] + step_offset;
  waveform_index = waveform_index < 0 ? 0 : waveform_index;
  waveform_index =
      waveform_index >= group.waveforms.dims[1]
          ? static_cast<int>(group.waveforms.dims[1]) - 1
          : waveform_index;
  const int coefficient_offset =
      (z * static_cast<int>(group.coefficients.dims[2]) + y) *
          static_cast<int>(group.coefficients.dims[3]) +
      x;
  const int target_offset =
      (static_cast<int>(target_z) * static_cast<int>(target.dims[1]) +
       static_cast<int>(target_y)) *
          static_cast<int>(target.dims[2]) +
      static_cast<int>(target_x);
  static_cast<float*>(target.data)[target_offset] +=
      static_cast<const float*>(group.coefficients.data)[coefficient_offset] *
      static_cast<const float*>(group.waveforms.data)[waveform_index];
}

__device__ __forceinline__ void ApplyCoincidentSourceGroupCell(
    BeamzBuffer target, BeamzSourceGroupLaunch group, int step_offset,
    int metallic_edges, int z, int y, int x) {
  if (z >= group.coefficients.dims[1] ||
      y >= group.coefficients.dims[2] ||
      x >= group.coefficients.dims[3]) {
    return;
  }
  const auto* starts = static_cast<const int32_t*>(group.starts.data);
  const int64_t target_z = static_cast<int64_t>(starts[0]) + z;
  const int64_t target_y = static_cast<int64_t>(starts[1]) + y;
  const int64_t target_x = static_cast<int64_t>(starts[2]) + x;
  if (target_z < 0 || target_z >= target.dims[0] || target_y < 0 ||
      target_y >= target.dims[1] || target_x < 0 ||
      target_x >= target.dims[2]) {
    return;
  }
  if (group.timing != 0 &&
      SourceCellConstrained(target, group.component,
                            group.timing == 1 ? 0 : 1, metallic_edges,
                            static_cast<int>(target_z),
                            static_cast<int>(target_y),
                            static_cast<int>(target_x))) {
    return;
  }
  int waveform_index =
      static_cast<const int32_t*>(group.current_step.data)[0] + step_offset;
  waveform_index = waveform_index < 0 ? 0 : waveform_index;
  waveform_index =
      waveform_index >= group.waveforms.dims[1]
          ? static_cast<int>(group.waveforms.dims[1]) - 1
          : waveform_index;
  const int target_offset =
      (static_cast<int>(target_z) * static_cast<int>(target.dims[1]) +
       static_cast<int>(target_y)) *
          static_cast<int>(target.dims[2]) +
      static_cast<int>(target_x);
  float value = static_cast<float*>(target.data)[target_offset];
  const int coefficient_stride =
      static_cast<int>(group.coefficients.dims[1] *
                       group.coefficients.dims[2] *
                       group.coefficients.dims[3]);
  const int coefficient_cell =
      (z * static_cast<int>(group.coefficients.dims[2]) + y) *
          static_cast<int>(group.coefficients.dims[3]) +
      x;
  const auto* coefficients =
      static_cast<const float*>(group.coefficients.data);
  const auto* waveforms = static_cast<const float*>(group.waveforms.data);
  const int waveform_stride = static_cast<int>(group.waveforms.dims[1]);
  for (int source = 0; source < group.coefficients.dims[0]; ++source) {
    value += coefficients[source * coefficient_stride + coefficient_cell] *
             waveforms[source * waveform_stride + waveform_index];
  }
  static_cast<float*>(target.data)[target_offset] = value;
}

template <bool Atomic>
__global__ void ApplySourceGroupBatched(BeamzBuffer target,
                                        BeamzSourceGroupLaunch group,
                                        int z_blocks, int step_offset,
                                        int metallic_edges) {
  const int source_index = blockIdx.z / z_blocks;
  const int source_block_z = blockIdx.z - source_index * z_blocks;
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  const int z = source_block_z * blockDim.z + threadIdx.z;
  // The compiler marks only statically disjoint slabs as non-atomic. The
  // conservative default preserves additive semantics for every other group.
  ApplySourceGroupCell<Atomic>(target, group, source_index, step_offset,
                               metallic_edges, z, y, x);
}

__global__ void ApplyCoincidentSourceGroup(BeamzBuffer target,
                                            BeamzSourceGroupLaunch group,
                                            int step_offset,
                                            int metallic_edges) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  const int z = blockIdx.z * blockDim.z + threadIdx.z;
  ApplyCoincidentSourceGroupCell(target, group, step_offset, metallic_edges, z,
                                 y, x);
}

cudaError_t LaunchSourceGroup(cudaStream_t stream, const BeamzLaunch& launch,
                              const BeamzBuffer& target,
                              const BeamzSourceGroupLaunch& group,
                              int32_t step) {
  const dim3 threads = SourceThreads(group.coefficients.dims[3]);
  const int z_blocks =
      (group.coefficients.dims[1] + threads.z - 1) / threads.z;
  const bool single = group.coefficients.dims[0] == 1;
  const bool coincident = !single && group.coincident != 0;
  const int launch_z_blocks =
      !single && !coincident ? z_blocks * group.coefficients.dims[0] : z_blocks;
  const dim3 blocks(
      (group.coefficients.dims[3] + threads.x - 1) / threads.x,
      (group.coefficients.dims[2] + threads.y - 1) / threads.y,
      launch_z_blocks);
  if (single) {
    if (group.timing == 0) {
      ApplySingleSourceGroup<0><<<blocks, threads, 0, stream>>>(
          target, group, step, launch.metallic_edges);
    } else if (group.timing == 1) {
      ApplySingleSourceGroup<1><<<blocks, threads, 0, stream>>>(
          target, group, step, launch.metallic_edges);
    } else {
      ApplySingleSourceGroup<2><<<blocks, threads, 0, stream>>>(
          target, group, step, launch.metallic_edges);
    }
    return cudaPeekAtLastError();
  }
  if (coincident) {
    ApplyCoincidentSourceGroup<<<blocks, threads, 0, stream>>>(
        target, group, step, launch.metallic_edges);
    return cudaPeekAtLastError();
  }
  if (group.disjoint != 0) {
    ApplySourceGroupBatched<false><<<blocks, threads, 0, stream>>>(
        target, group, z_blocks, step, launch.metallic_edges);
  } else {
    ApplySourceGroupBatched<true><<<blocks, threads, 0, stream>>>(
        target, group, z_blocks, step, launch.metallic_edges);
  }
  return cudaPeekAtLastError();
}

template <bool SingleMonitor>
__global__ void AccumulateDftGroups(DftFields fields,
                                    BeamzDftGroupLaunch monitors,
                                    int step_offset) {
  const int point = blockIdx.x * blockDim.x + threadIdx.x;
  const int frequency = blockIdx.y * blockDim.y + threadIdx.y;
  const int lane = blockIdx.z * blockDim.z + threadIdx.z;
  const int monitor = SingleMonitor ? 0 : lane / 6;
  const int component = SingleMonitor ? lane : lane % 6;
  if constexpr (!SingleMonitor) {
    if (monitor >= monitors.monitor_count) return;
  }

  const auto* counts = static_cast<const int32_t*>(monitors.counts.data);
  const int frequency_count = counts[5 * monitor];
  const int point_count = counts[5 * monitor + 1];
  const int interval = counts[5 * monitor + 2] > 0
                           ? counts[5 * monitor + 2]
                           : 1;
  const int value_offset = counts[5 * monitor + 3];
  const int weight_offset = counts[5 * monitor + 4];
  const int max_frequency_count = static_cast<int>(monitors.frequencies.dims[1]);
  const int max_points = static_cast<int>(monitors.indices.dims[2]);
  if (frequency_count < 1 || frequency_count > max_frequency_count ||
      point_count < 1 || point_count > max_points || value_offset < 0 ||
      weight_offset < 0 || frequency >= frequency_count) {
    return;
  }
  const int64_t value_count =
      6LL * frequency_count * point_count;
  if (static_cast<int64_t>(value_offset) + value_count >
          monitors.dft_re.dims[0] ||
      static_cast<int64_t>(value_offset) + value_count >
          monitors.dft_im.dims[0] ||
      static_cast<int64_t>(weight_offset) + frequency_count >
          monitors.dft_weight.dims[0]) {
    return;
  }

  const int64_t absolute_step =
      static_cast<int64_t>(
          static_cast<const int32_t*>(monitors.current_step.data)[0]) +
      step_offset;
  if (absolute_step % interval != 0) return;
  const float time = static_cast<const float*>(monitors.time.data)[0] +
                     static_cast<float>(step_offset + 1) * fields.dt;
  const auto* windows = static_cast<const float*>(monitors.windows.data);
  const float start = windows[3 * monitor];
  const float end = windows[3 * monitor + 1];
  if (time < start || time > end) return;

  const auto* codes = static_cast<const int32_t*>(monitors.codes.data);
  if ((codes[2 * monitor] != 0 && codes[2 * monitor] != 1) ||
      (codes[2 * monitor + 1] != 0 && codes[2 * monitor + 1] != 1)) {
    return;
  }
  float window = 0.0f;
  float phase_sin = 0.0f;
  float phase_cos = 0.0f;
  if (threadIdx.x == 0) {
    window = 1.0f;
    if (codes[2 * monitor] == 1 && isfinite(end) && end > start) {
      const float tau =
          fminf(fmaxf((time - start) / (end - start), 0.0f), 1.0f);
      window = 0.5f * (1.0f - cosf(6.2831853071795864769f * tau));
    }
    const float frequency_hz =
        static_cast<const float*>(monitors.frequencies.data)
            [monitor * max_frequency_count + frequency];
    sincosf(6.2831853071795864769f * frequency_hz * time, &phase_sin,
            &phase_cos);
  }
  window = __shfl_sync(0xffffffff, window, 0);
  phase_sin = __shfl_sync(0xffffffff, phase_sin, 0);
  phase_cos = __shfl_sync(0xffffffff, phase_cos, 0);
  if (point >= point_count) return;
  if (component == 0 && point == 0) {
    static_cast<float*>(monitors.dft_weight.data)
        [weight_offset + frequency] += window;
  }
  const float mask = static_cast<const float*>(monitors.component_masks.data)
      [monitor * 6 + component];
  if (mask == 0.0f) return;

  const int neighbors = static_cast<int>(monitors.indices.dims[3]);
  const int plan_base = ((monitor * 6 + component) * max_points + point) *
                        neighbors;
  const BeamzBuffer& field = fields.values[component];
  float sample = 0.0f;
  for (int neighbor = 0; neighbor < neighbors; ++neighbor) {
    const int gather_offset = plan_base + neighbor;
    const int field_offset =
        static_cast<const int32_t*>(monitors.indices.data)[gather_offset];
    if (field_offset >= 0 && field_offset < ElementCount(field)) {
      sample += static_cast<const float*>(field.data)[field_offset] *
                static_cast<const float*>(monitors.weights.data)[gather_offset];
    }
  }

  float scale = window;
  if (codes[2 * monitor + 1] == 1) {
    const float length_unit = windows[3 * monitor + 2];
    if (!isfinite(length_unit) || length_unit <= 0.0f) return;
    scale *= fields.dt * static_cast<float>(interval) * 299792458.0f /
             length_unit / sqrtf(6.2831853071795864769f);
  }
  const int accumulator_offset =
      value_offset +
      (component * frequency_count + frequency) * point_count +
      point;
  static_cast<float*>(monitors.dft_re.data)[accumulator_offset] +=
      scale * sample * phase_cos;
  static_cast<float*>(monitors.dft_im.data)[accumulator_offset] +=
      scale * sample * phase_sin;
}

cudaError_t LaunchDftGroups(cudaStream_t stream, const BeamzLaunch& h_launch,
                            const BeamzLaunch& e_launch,
                            const BeamzDftGroupLaunch& monitors,
                            int32_t step) {
  const DftFields fields = MakeDftFields(h_launch, e_launch);
  const int frequency_threads =
      monitors.monitor_count == 1 && monitors.frequencies.dims[1] == 3 ? 3 : 2;
  const dim3 threads(32, frequency_threads, 2);
  const dim3 blocks(
      (monitors.indices.dims[2] + threads.x - 1) / threads.x,
      (monitors.frequencies.dims[1] + threads.y - 1) / threads.y,
      (monitors.monitor_count * 6 + threads.z - 1) / threads.z);
  if (monitors.monitor_count == 1) {
    AccumulateDftGroups<true><<<blocks, threads, 0, stream>>>(
        fields, monitors, step);
  } else {
    AccumulateDftGroups<false><<<blocks, threads, 0, stream>>>(
        fields, monitors, step);
  }
  return cudaPeekAtLastError();
}

}  // namespace

cudaError_t BeamzEnqueueSourceGroup(
    cudaStream_t stream, const BeamzLaunch& launch, const BeamzBuffer& target,
    const BeamzSourceGroupLaunch& group, int32_t step) {
  return LaunchSourceGroup(stream, launch, target, group, step);
}

cudaError_t BeamzEnqueueDftGroups(cudaStream_t stream,
                                  const BeamzLaunch& h_launch,
                                  const BeamzLaunch& e_launch,
                                  const BeamzDftGroupLaunch& monitors,
                                  int32_t step) {
  return LaunchDftGroups(stream, h_launch, e_launch, monitors, step);
}
