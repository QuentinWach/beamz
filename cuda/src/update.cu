#include <cuda_runtime_api.h>

#include <cuda_bf16.h>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include "kernels.h"
#include "launch.h"

namespace {

constexpr int kTileX = 32;
constexpr int kTileY = 4;
constexpr int kTileZ = 2;
constexpr int kPressureTileZ = 1;
constexpr int kFusedCoreX = 32;
constexpr int kFusedCoreY = 16;
constexpr int kFusedCoreZ = 2;
constexpr int kFusedSharedX = kFusedCoreX + 1;
constexpr int kFusedSharedY = kFusedCoreY + 1;
constexpr int kFusedSharedZ = kFusedCoreZ + 1;
constexpr int kFusedVolume = kFusedSharedX * kFusedSharedY * kFusedSharedZ;
constexpr size_t kFusedSharedBytes = 3 * kFusedVolume * sizeof(float);

// The temporal Yee kernel needs only the phase payload, not the full ABI
// descriptor (versioning, CPML slots, or unused result slots).
struct FusedYeePhase {
  float inv_resolution;
  BeamzBuffer inputs[12];
  BeamzBuffer metrics[3];
  BeamzBuffer outputs[3];
};

FusedYeePhase MakeFusedYeePhase(const BeamzLaunch& launch) {
  FusedYeePhase phase{};
  phase.inv_resolution = launch.inv_resolution;
  for (int index = 0; index < 12; ++index) {
    phase.inputs[index] = launch.inputs[index];
  }
  for (int axis = 0; axis < 3; ++axis) {
    phase.metrics[axis] = launch.metrics[axis];
    phase.outputs[axis] = launch.outputs[axis];
  }
  return phase;
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

bool SameShape(const BeamzBuffer& left, const BeamzBuffer& right) {
  if (left.rank != right.rank) return false;
  for (int axis = 0; axis < left.rank; ++axis) {
    if (left.dims[axis] != right.dims[axis]) return false;
  }
  return true;
}

bool IsPositive3D(const BeamzBuffer& value) {
  return value.rank == 3 && value.dims[0] > 0 && value.dims[1] > 0 &&
         value.dims[2] > 0 && value.data != nullptr;
}

bool BroadcastsTo(const BeamzBuffer& value, const BeamzBuffer& target) {
  if (value.rank == 0) return value.data != nullptr;
  if (value.rank != 3 || value.data == nullptr) return false;
  for (int axis = 0; axis < 3; ++axis) {
    if (value.dims[axis] != 1 && value.dims[axis] != target.dims[axis]) {
      return false;
    }
  }
  return true;
}

// CUDA's three phase outputs are the Yee components of one vector field.  The
// curl inputs must be the complementary staggered field: forward H curls use
// one extra cell on their derivative axis; backward E curls use one fewer.
// Requiring this exact relationship catches mismatched FFI buffers before any
// kernel computes a flat address from a foreign shape.
bool IsCompatibleCurlInput(const BeamzBuffer& output,
                           const BeamzBuffer& source, int axis,
                           int phase) {
  if (!IsPositive3D(source)) return false;
  for (int dimension = 0; dimension < 3; ++dimension) {
    const int64_t expected =
        output.dims[dimension] + (dimension == axis ? (phase == 0 ? 1 : -1) : 0);
    if (expected <= 0 || source.dims[dimension] != expected) return false;
  }
  return true;
}

int64_t Elements(const BeamzBuffer& value) {
  int64_t elements = 1;
  for (int axis = 0; axis < value.rank; ++axis) elements *= value.dims[axis];
  return elements;
}

int MaxOutputExtent(const BeamzLaunch& launch, int axis) {
  int extent = 0;
  for (int component = 0; component < 3; ++component) {
    const int value = static_cast<int>(launch.outputs[component].dims[axis]);
    extent = value > extent ? value : extent;
  }
  return extent;
}

bool IsCpmlProfile(const BeamzBuffer& value, int axis) {
  if (!HasType(value, kBeamzF32) || !IsPositive3D(value)) return false;
  for (int dimension = 0; dimension < 3; ++dimension) {
    if (dimension != axis && value.dims[dimension] != 1) return false;
  }
  return true;
}

bool IsCpmlPsi(const BeamzBuffer& value, const BeamzBuffer& target, int axis) {
  if (!IsPositive3D(value) || value.dims[axis] > target.dims[axis]) return false;
  for (int dimension = 0; dimension < 3; ++dimension) {
    if (dimension != axis && value.dims[dimension] != target.dims[dimension]) {
      return false;
    }
  }
  return true;
}

cudaError_t ValidatePhase(const BeamzLaunch& launch) {
  if (launch.phase < 0 || launch.phase > 1 || launch.metric_kind < 0 ||
      launch.metric_kind > 2 || (launch.nterms != 0 && launch.nterms != 6) ||
      launch.metallic_edges < 0 || launch.metallic_edges > 63 ||
      !std::isfinite(launch.dt) || !std::isfinite(launch.resolution) ||
      !std::isfinite(launch.inv_resolution) || launch.dt <= 0.0f ||
      launch.resolution <= 0.0f || launch.inv_resolution <= 0.0f ||
      launch.uniform_cpml_thickness < 0 ||
      (launch.nterms == 0 && launch.uniform_cpml_thickness != 0)) {
    return cudaErrorInvalidValue;
  }
  const int input_count = launch.nterms == 0 ? 12 : 13 + 4 * launch.nterms;
  const int output_count = 3 + launch.nterms;
  for (int index = 0; index < input_count; ++index) {
    if (!FitsIntOffsets(launch.inputs[index])) return cudaErrorInvalidValue;
  }
  for (int index = 0; index < output_count; ++index) {
    if (!FitsIntOffsets(launch.outputs[index])) return cudaErrorInvalidValue;
  }
  for (int component = 0; component < 6; ++component) {
    if (!HasType(launch.inputs[component], kBeamzF32) ||
        !IsPositive3D(launch.inputs[component])) {
      return cudaErrorInvalidValue;
    }
  }
  for (int component = 0; component < 3; ++component) {
    if (!HasType(launch.outputs[component], kBeamzF32) ||
        !IsPositive3D(launch.outputs[component]) ||
        !SameShape(launch.inputs[component], launch.outputs[component])) {
      return cudaErrorInvalidValue;
    }
  }
  constexpr int first_source[3] = {2, 0, 1};
  constexpr int second_source[3] = {1, 2, 0};
  constexpr int first_axis[3] = {1, 0, 2};
  constexpr int second_axis[3] = {0, 2, 1};
  for (int component = 0; component < 3; ++component) {
    const BeamzBuffer& output = launch.outputs[component];
    if (!IsCompatibleCurlInput(output,
                               launch.inputs[3 + first_source[component]],
                               first_axis[component], launch.phase) ||
        !IsCompatibleCurlInput(output,
                               launch.inputs[3 + second_source[component]],
                               second_axis[component], launch.phase)) {
      return cudaErrorInvalidValue;
    }
  }
  const bool packed_material =
      launch.phase == 1 &&
      HasType(launch.inputs[9], kBeamzS32) &&
      HasType(launch.inputs[10], kBeamzS32) &&
      HasType(launch.inputs[11], kBeamzS32);
  for (int component = 0; component < 3; ++component) {
    const BeamzBuffer& decay = launch.inputs[6 + component];
    const BeamzBuffer& source = launch.inputs[9 + component];
    const bool ordinary_material =
        HasType(decay, kBeamzF32) && HasType(source, kBeamzF32) &&
        BroadcastsTo(decay, launch.outputs[component]) &&
        BroadcastsTo(source, launch.outputs[component]);
    const bool packed_component =
        packed_material && HasType(decay, kBeamzF32) && decay.rank == 1 &&
        source.rank == 1 && decay.data != nullptr && source.data != nullptr &&
        decay.dims[0] > 0 && source.dims[0] >= 0 &&
        Elements(source) >= (Elements(launch.outputs[component]) + 3) / 4;
    if (!ordinary_material && !packed_component) return cudaErrorInvalidValue;
  }
  if (launch.nterms != 0) {
    const BeamzBuffer& metadata = launch.inputs[12];
    if (!HasType(metadata, kBeamzS32) || metadata.rank != 2 ||
        metadata.dims[0] != 6 || metadata.dims[1] != 5 ||
        metadata.data == nullptr) {
      return cudaErrorInvalidValue;
    }
    for (int index = 13; index < 13 + 3 * launch.nterms; ++index) {
      if (launch.inputs[index].data == nullptr) return cudaErrorInvalidValue;
    }
    const int psi_input_base = 13 + 3 * launch.nterms;
    constexpr int cpml_axes[6] = {1, 0, 0, 2, 2, 1};
    for (int term = 0; term < launch.nterms; ++term) {
      const BeamzBuffer& input = launch.inputs[psi_input_base + term];
      const BeamzBuffer& output = launch.outputs[3 + term];
      const BeamzBuffer& target = launch.outputs[term / 2];
      const bool supported_type = HasType(input, kBeamzF32) ||
                                  HasType(input, kBeamzBF16);
      if (!supported_type || input.element_type != output.element_type ||
          !IsCpmlPsi(input, target, cpml_axes[term]) ||
          !IsCpmlPsi(output, target, cpml_axes[term]) ||
          !SameShape(input, output)) {
        return cudaErrorInvalidValue;
      }
      for (int coefficient = 0; coefficient < 3; ++coefficient) {
        const BeamzBuffer& profile = launch.inputs[13 + 3 * term + coefficient];
        if (!IsCpmlProfile(profile, cpml_axes[term])) {
          return cudaErrorInvalidValue;
        }
      }
      if (launch.uniform_cpml_thickness > 0) {
        const int64_t packed_extent =
            2 * static_cast<int64_t>(launch.uniform_cpml_thickness);
        if (packed_extent > target.dims[cpml_axes[term]] ||
            input.dims[cpml_axes[term]] != packed_extent) {
          return cudaErrorInvalidValue;
        }
        for (int coefficient = 0; coefficient < 3; ++coefficient) {
          if (launch.inputs[13 + 3 * term + coefficient]
                  .dims[cpml_axes[term]] != packed_extent) {
            return cudaErrorInvalidValue;
          }
        }
      }
    }
  }
  for (int axis = 0; axis < 3; ++axis) {
    const BeamzBuffer& metric = launch.metrics[axis];
    if (!FitsIntOffsets(metric) || !HasType(metric, kBeamzF32) ||
        (launch.metric_kind == 1 &&
         (metric.rank != 0 || metric.data == nullptr)) ||
        (launch.metric_kind == 2 &&
         (metric.rank != 1 ||
          metric.dims[0] < MaxOutputExtent(launch, axis) ||
          metric.data == nullptr))) {
      return cudaErrorInvalidValue;
    }
  }
  return cudaSuccess;
}

__device__ __forceinline__ int Offset(const BeamzBuffer& value, int z, int y,
                                      int x) {
  if (value.rank == 0) return 0;
  const int iz = value.dims[0] == 1 ? 0 : z;
  const int iy = value.dims[1] == 1 ? 0 : y;
  const int ix = value.dims[2] == 1 ? 0 : x;
  return (iz * static_cast<int>(value.dims[1]) + iy) *
             static_cast<int>(value.dims[2]) +
         ix;
}

__device__ __forceinline__ float Read(const BeamzBuffer& value, int z, int y,
                                      int x) {
  return static_cast<const float*>(value.data)[Offset(value, z, y, x)];
}

__device__ __forceinline__ float Read3D(const BeamzBuffer& value, int z, int y,
                                        int x) {
  const int offset = (z * static_cast<int>(value.dims[1]) + y) *
                         static_cast<int>(value.dims[2]) +
                     x;
  return static_cast<const float*>(value.data)[offset];
}

template <int MetricKind, typename Launch>
__device__ __forceinline__ float MetricScale(const Launch& launch, int axis,
                                             int coordinate) {
  if constexpr (MetricKind == 0) {
    return launch.inv_resolution;
  } else if constexpr (MetricKind == 1) {
    return static_cast<const float*>(launch.metrics[axis].data)[0];
  } else {
    return static_cast<const float*>(launch.metrics[axis].data)[coordinate];
  }
}

template <int MetricKind, typename Launch>
__device__ __forceinline__ float ForwardDifference(const BeamzBuffer& value,
                                                   int axis, int z, int y,
                                                   int x,
                                                   const Launch& launch) {
  const int coordinate = axis == 0 ? z : (axis == 1 ? y : x);
  int next_z = z, next_y = y, next_x = x;
  if (axis == 0) {
    if (z + 1 >= value.dims[0]) return 0.0f;
    ++next_z;
  } else if (axis == 1) {
    if (y + 1 >= value.dims[1]) return 0.0f;
    ++next_y;
  } else {
    if (x + 1 >= value.dims[2]) return 0.0f;
    ++next_x;
  }
  return (Read3D(value, next_z, next_y, next_x) - Read3D(value, z, y, x)) *
         MetricScale<MetricKind>(launch, axis, coordinate);
}

template <int MetricKind, bool HasMetallicEdges = true>
__device__ __forceinline__ float BoundaryDifference(const BeamzBuffer& value,
                                                    int axis, int z, int y,
                                                    int x, int edge_mask,
                                                    const BeamzLaunch& launch) {
  const int coordinate = axis == 0 ? z : (axis == 1 ? y : x);
  const float inv_dx = MetricScale<MetricKind>(launch, axis, coordinate);
  const int size = static_cast<int>(value.dims[axis]);
  if (coordinate == 0) {
    if constexpr (!HasMetallicEdges) return 0.0f;
    const bool metallic = edge_mask & (1 << (2 * axis));
    return metallic ? Read3D(value, z, y, x) * inv_dx : 0.0f;
  }
  if (coordinate == size) {
    if constexpr (!HasMetallicEdges) return 0.0f;
    int last_z = z, last_y = y, last_x = x;
    if (axis == 0) last_z = size - 1;
    if (axis == 1) last_y = size - 1;
    if (axis == 2) last_x = size - 1;
    const bool metallic = edge_mask & (1 << (2 * axis + 1));
    return metallic ? -Read3D(value, last_z, last_y, last_x) * inv_dx : 0.0f;
  }
  int low_z = z, low_y = y, low_x = x;
  if (axis == 0) --low_z;
  if (axis == 1) --low_y;
  if (axis == 2) --low_x;
  return (Read3D(value, z, y, x) - Read3D(value, low_z, low_y, low_x)) * inv_dx;
}

template <int Term, bool UniformCpml = false, int PsiType = -1>
__device__ __forceinline__ float CorrectCpml(float derivative, int z, int y,
                                             int x, const BeamzLaunch& launch) {
  // The 3D compiler emits the six curl terms in this fixed derivative order.
  // CPML coefficient buffers are 1D profiles along their derivative axis.
  constexpr int axes[6] = {1, 0, 0, 2, 2, 1};
  constexpr int axis = axes[Term];
  constexpr float sign = Term % 2 == 0 ? 1.0f : -1.0f;
  int low;
  int high;
  if constexpr (UniformCpml) {
    low = high = launch.uniform_cpml_thickness;
  } else {
    const auto* descriptor =
        static_cast<const int32_t*>(launch.inputs[12].data);
    low = descriptor[Term * 5 + 2];
    high = descriptor[Term * 5 + 3];
  }
  const BeamzBuffer& target = launch.outputs[Term / 2];
  const int coordinate = axis == 0 ? z : (axis == 1 ? y : x);
  const int axis_size = static_cast<int>(target.dims[axis]);
  if (low < 0 || high < 0 ||
      static_cast<int64_t>(low) + high > axis_size) {
    // Metadata arrives in a device buffer, so the host cannot inspect its
    // values without synchronizing every launch. Invalid foreign metadata is a
    // no-op recurrence rather than an out-of-bounds packed-slab access.
    return sign * derivative;
  }
  int packed = -1;
  if (coordinate < low) {
    packed = coordinate;
  } else if (coordinate >= axis_size - high) {
    packed = low + coordinate - (axis_size - high);
  }
  if (packed < 0) return sign * derivative;

  int pz = z, py = y, px = x;
  if (axis == 0) pz = packed;
  if (axis == 1) py = packed;
  if (axis == 2) px = packed;
  const int coefficient_base = 13 + 3 * Term;
  const int psi_base = 13 + 3 * launch.nterms;
  const BeamzBuffer& psi_input = launch.inputs[psi_base + Term];
  const BeamzBuffer& psi_output = launch.outputs[3 + Term];
  const BeamzBuffer& a = launch.inputs[coefficient_base];
  const BeamzBuffer& b = launch.inputs[coefficient_base + 1];
  const BeamzBuffer& inv_kappa = launch.inputs[coefficient_base + 2];
  if (pz < 0 || pz >= psi_output.dims[0] || py < 0 ||
      py >= psi_output.dims[1] || px < 0 || px >= psi_output.dims[2] ||
      packed >= a.dims[axis] || packed >= b.dims[axis] ||
      packed >= inv_kappa.dims[axis]) {
    return sign * derivative;
  }
  const int psi_offset = (pz * static_cast<int>(psi_output.dims[1]) + py) *
                             static_cast<int>(psi_output.dims[2]) +
                         px;
  float old_psi;
  if constexpr (PsiType == kBeamzBF16) {
    old_psi = __bfloat162float(
        static_cast<const __nv_bfloat16*>(psi_input.data)[psi_offset]);
  } else if constexpr (PsiType == kBeamzF32) {
    old_psi = static_cast<const float*>(psi_input.data)[psi_offset];
  } else {
    old_psi = psi_input.element_type == kBeamzBF16
                  ? __bfloat162float(static_cast<const __nv_bfloat16*>(
                                         psi_input.data)[psi_offset])
                  : static_cast<const float*>(psi_input.data)[psi_offset];
  }
  const float next_psi =
      static_cast<const float *>(
          b.data)[packed] *
          old_psi +
      static_cast<const float*>(a.data)[packed] *
          derivative;
  if constexpr (PsiType == kBeamzBF16) {
    static_cast<__nv_bfloat16*>(psi_output.data)[psi_offset] =
        __float2bfloat16_rn(next_psi);
  } else if constexpr (PsiType == kBeamzF32) {
    static_cast<float*>(psi_output.data)[psi_offset] = next_psi;
  } else if (psi_output.element_type == kBeamzBF16) {
    static_cast<__nv_bfloat16*>(psi_output.data)[psi_offset] =
        __float2bfloat16_rn(next_psi);
  } else {
    static_cast<float*>(psi_output.data)[psi_offset] = next_psi;
  }
  return sign *
         (derivative * static_cast<const float*>(
                           inv_kappa.data)[packed] +
          next_psi);
}

template <int Phase, int Component, bool Cpml, int MetricKind,
          bool HasMetallicEdges = true, bool UniformCpml = false,
          int PsiType = -1, bool PackedLosslessMaterial = false>
__device__ __forceinline__ void UpdateComponent(const BeamzLaunch& launch,
                                                int z, int y, int x) {
  const BeamzBuffer& input = launch.inputs[Component];
  const BeamzBuffer& output = launch.outputs[Component];
  if (z >= output.dims[0] || y >= output.dims[1] || x >= output.dims[2]) return;
  const int linear = (z * static_cast<int>(output.dims[1]) + y) *
                         static_cast<int>(output.dims[2]) +
                     x;
  constexpr int normal_axis = 2 - Component;
  constexpr bool constrained = Phase == 0;
  const int coordinate = normal_axis == 0 ? z : (normal_axis == 1 ? y : x);
  const int axis_size = static_cast<int>(output.dims[normal_axis]);
  const bool on_low_wall =
      coordinate == 0 && (launch.metallic_edges & (1 << (2 * normal_axis)));
  const bool on_high_wall =
      coordinate == axis_size - 1 &&
      (launch.metallic_edges & (1 << (2 * normal_axis + 1)));
  bool zero_on_wall = false;
  if constexpr (HasMetallicEdges) {
    if constexpr (constrained) {
      zero_on_wall = on_low_wall || on_high_wall;
    } else {
      for (int axis = 0; axis < 3; ++axis) {
        if (axis == normal_axis) continue;
        const int axis_coordinate = axis == 0 ? z : (axis == 1 ? y : x);
        const int size = static_cast<int>(output.dims[axis]);
        if ((axis_coordinate == 0 &&
             (launch.metallic_edges & (1 << (2 * axis)))) ||
            (axis_coordinate == size - 1 &&
             (launch.metallic_edges & (1 << (2 * axis + 1))))) {
          zero_on_wall = true;
          break;
        }
      }
    }
  }
  if constexpr (!Cpml) {
    if (zero_on_wall) {
      static_cast<float*>(output.data)[linear] = 0.0f;
      return;
    }
  }
  constexpr int first_source[3] = {2, 0, 1};
  constexpr int second_source[3] = {1, 2, 0};
  constexpr int first_axis[3] = {1, 0, 2};
  constexpr int second_axis[3] = {0, 2, 1};
  const BeamzBuffer& first = launch.inputs[3 + first_source[Component]];
  const BeamzBuffer& second = launch.inputs[3 + second_source[Component]];
  float derivative0;
  float derivative1;
  if constexpr (Phase == 0) {
    derivative0 =
        ForwardDifference<MetricKind>(first, first_axis[Component], z, y, x,
                                      launch);
    derivative1 =
        ForwardDifference<MetricKind>(second, second_axis[Component], z, y, x,
                                      launch);
  } else {
    derivative0 = BoundaryDifference<MetricKind, HasMetallicEdges>(
        first, first_axis[Component], z, y, x, launch.metallic_edges, launch);
    derivative1 = BoundaryDifference<MetricKind, HasMetallicEdges>(
        second, second_axis[Component], z, y, x, launch.metallic_edges, launch);
  }
  float curl;
  if constexpr (Cpml) {
    curl = CorrectCpml<2 * Component, UniformCpml, PsiType>(
               derivative0, z, y, x, launch) +
           CorrectCpml<2 * Component + 1, UniformCpml, PsiType>(
               derivative1, z, y, x, launch);
  } else {
    curl = derivative0 - derivative1;
  }
  if (zero_on_wall) {
    // CPML memory still evolves where an absorbing face intersects a PEC face;
    // only the constrained field value is masked after the recurrence update.
    static_cast<float*>(output.data)[linear] = 0.0f;
    return;
  }

  const float old_field = static_cast<const float*>(input.data)[linear];
  float decay;
  float source;
  if constexpr (PackedLosslessMaterial) {
    decay = 1.0f;
    const auto* packed =
        static_cast<const uint32_t*>(launch.inputs[9 + Component].data);
    const uint32_t word = packed[linear >> 2];
    const uint32_t code = (word >> (8 * (linear & 3))) & 0xffu;
    source = code < launch.inputs[6 + Component].dims[0]
                 ? static_cast<const float*>(launch.inputs[6 + Component].data)[code]
                 : 0.0f;
  } else {
    decay = Read(launch.inputs[6 + Component], z, y, x);
    source = Read(launch.inputs[9 + Component], z, y, x);
  }
  if constexpr (Phase == 0) {
    static_cast<float*>(output.data)[linear] =
        decay * old_field - source * curl;
  } else {
    static_cast<float*>(output.data)[linear] =
        decay * old_field + source * curl;
  }
}

template <int Phase, bool Cpml, int MetricKind, bool HasMetallicEdges = true,
          bool UniformCpml = false, int PsiType = -1,
          bool PackedLosslessMaterial = false>
__global__ void UpdateFusedComponents(BeamzLaunch launch) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  const int z = blockIdx.z * blockDim.z + threadIdx.z;
  UpdateComponent<Phase, 0, Cpml, MetricKind, HasMetallicEdges, UniformCpml,
                  PsiType, PackedLosslessMaterial>(launch, z, y, x);
  UpdateComponent<Phase, 1, Cpml, MetricKind, HasMetallicEdges, UniformCpml,
                  PsiType, PackedLosslessMaterial>(launch, z, y, x);
  UpdateComponent<Phase, 2, Cpml, MetricKind, HasMetallicEdges, UniformCpml,
                  PsiType, PackedLosslessMaterial>(launch, z, y, x);
}

template <int Phase>
__device__ __forceinline__ float UncheckedDifference(
    const BeamzBuffer& value, int axis, int z, int y, int x, float inv_dx) {
  int neighbor_z = z, neighbor_y = y, neighbor_x = x;
  if (axis == 0) neighbor_z += Phase == 0 ? 1 : -1;
  if (axis == 1) neighbor_y += Phase == 0 ? 1 : -1;
  if (axis == 2) neighbor_x += Phase == 0 ? 1 : -1;
  const float center = Read3D(value, z, y, x);
  const float neighbor = Read3D(value, neighbor_z, neighbor_y, neighbor_x);
  return (Phase == 0 ? neighbor - center : center - neighbor) * inv_dx;
}

template <int Phase, int Component>
__device__ __forceinline__ void UpdateFullPecScalarComponent(
    const BeamzLaunch& launch, int z, int y, int x) {
  const BeamzBuffer& input = launch.inputs[Component];
  const BeamzBuffer& output = launch.outputs[Component];
  if (z >= output.dims[0] || y >= output.dims[1] || x >= output.dims[2]) return;
  const int linear = (z * static_cast<int>(output.dims[1]) + y) *
                         static_cast<int>(output.dims[2]) +
                     x;
  constexpr int normal_axis = 2 - Component;
  if constexpr (Phase == 0) {
    const int coordinate = normal_axis == 0 ? z : (normal_axis == 1 ? y : x);
    if (coordinate == 0 || coordinate == output.dims[normal_axis] - 1) {
      static_cast<float*>(output.data)[linear] = 0.0f;
      return;
    }
  } else {
    for (int axis = 0; axis < 3; ++axis) {
      if (axis == normal_axis) continue;
      const int coordinate = axis == 0 ? z : (axis == 1 ? y : x);
      if (coordinate == 0 || coordinate == output.dims[axis] - 1) {
        static_cast<float*>(output.data)[linear] = 0.0f;
        return;
      }
    }
  }

  constexpr int first_source[3] = {2, 0, 1};
  constexpr int second_source[3] = {1, 2, 0};
  constexpr int first_axis[3] = {1, 0, 2};
  constexpr int second_axis[3] = {0, 2, 1};
  const float derivative0 = UncheckedDifference<Phase>(
      launch.inputs[3 + first_source[Component]], first_axis[Component], z, y,
      x, launch.inv_resolution);
  const float derivative1 = UncheckedDifference<Phase>(
      launch.inputs[3 + second_source[Component]], second_axis[Component], z,
      y, x, launch.inv_resolution);
  const float old_field = static_cast<const float*>(input.data)[linear];
  const float decay =
      static_cast<const float*>(launch.inputs[6 + Component].data)[0];
  const float source =
      static_cast<const float*>(launch.inputs[9 + Component].data)[0];
  const float curl = derivative0 - derivative1;
  static_cast<float*>(output.data)[linear] =
      Phase == 0 ? decay * old_field - source * curl
                 : decay * old_field + source * curl;
}

template <int Phase>
__global__ void UpdateFusedFullPecScalarComponents(BeamzLaunch launch) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  const int z = blockIdx.z * blockDim.z + threadIdx.z;
  UpdateFullPecScalarComponent<Phase, 0>(launch, z, y, x);
  UpdateFullPecScalarComponent<Phase, 1>(launch, z, y, x);
  UpdateFullPecScalarComponent<Phase, 2>(launch, z, y, x);
}


__device__ __forceinline__ int FusedOffset(int z, int y, int x) {
  return (z * kFusedSharedY + y) * kFusedSharedX + x;
}

__device__ __forceinline__ bool BufferContains(const BeamzBuffer& value, int z,
                                                int y, int x) {
  return z >= 0 && y >= 0 && x >= 0 && z < value.dims[0] &&
         y < value.dims[1] && x < value.dims[2];
}

__host__ __device__ __forceinline__ int Min3(int a, int b, int c) {
  return a < b ? (a < c ? a : c) : (b < c ? b : c);
}

struct PhaseGeometry {
  int max_z;
  int max_y;
  int max_x;
};

PhaseGeometry MakePhaseGeometry(const BeamzLaunch& launch) {
  PhaseGeometry geometry{};
  for (int component = 0; component < 3; ++component) {
    const BeamzBuffer& output = launch.outputs[component];
    geometry.max_z =
        output.dims[0] > geometry.max_z ? output.dims[0] : geometry.max_z;
    geometry.max_y =
        output.dims[1] > geometry.max_y ? output.dims[1] : geometry.max_y;
    geometry.max_x =
        output.dims[2] > geometry.max_x ? output.dims[2] : geometry.max_x;
  }
  return geometry;
}

struct CpmlGeometry {
  PhaseGeometry field;
  int low;
  int high_z;
  int high_y;
  int high_x;
};

CpmlGeometry MakeCpmlGeometry(const BeamzLaunch& launch) {
  CpmlGeometry geometry{};
  geometry.field = MakePhaseGeometry(launch);
  geometry.low = launch.uniform_cpml_thickness;
  geometry.high_z =
      Min3(static_cast<int>(launch.outputs[0].dims[0]),
           static_cast<int>(launch.outputs[1].dims[0]),
           static_cast<int>(launch.outputs[2].dims[0])) -
      geometry.low;
  geometry.high_y =
      Min3(static_cast<int>(launch.outputs[0].dims[1]),
           static_cast<int>(launch.outputs[1].dims[1]),
           static_cast<int>(launch.outputs[2].dims[1])) -
      geometry.low;
  geometry.high_x =
      Min3(static_cast<int>(launch.outputs[0].dims[2]),
           static_cast<int>(launch.outputs[1].dims[2]),
           static_cast<int>(launch.outputs[2].dims[2])) -
      geometry.low;
  return geometry;
}

template <bool ScalarCoefficients, int MetricKind, int Component>
__device__ __forceinline__ float FusedHValue(const FusedYeePhase& launch,
                                             int z, int y, int x) {
  const BeamzBuffer& output = launch.outputs[Component];
  if (!BufferContains(output, z, y, x)) return 0.0f;
  constexpr int normal_axis = 2 - Component;
  const int coordinate = normal_axis == 0 ? z : (normal_axis == 1 ? y : x);
  if (coordinate == 0 || coordinate == output.dims[normal_axis] - 1) {
    return 0.0f;
  }
  constexpr int first_source[3] = {2, 0, 1};
  constexpr int second_source[3] = {1, 2, 0};
  constexpr int first_axis[3] = {1, 0, 2};
  constexpr int second_axis[3] = {0, 2, 1};
  const float derivative0 = ForwardDifference<MetricKind>(
      launch.inputs[3 + first_source[Component]], first_axis[Component], z, y,
      x, launch);
  const float derivative1 = ForwardDifference<MetricKind>(
      launch.inputs[3 + second_source[Component]], second_axis[Component], z,
      y, x, launch);
  const int linear = (z * static_cast<int>(output.dims[1]) + y) *
                         static_cast<int>(output.dims[2]) +
                     x;
  const float old_field =
      static_cast<const float*>(launch.inputs[Component].data)[linear];
  const float decay =
      ScalarCoefficients
          ? static_cast<const float*>(launch.inputs[6 + Component].data)[0]
          : Read3D(launch.inputs[6 + Component], z, y, x);
  const float source =
      ScalarCoefficients
          ? static_cast<const float*>(launch.inputs[9 + Component].data)[0]
          : Read3D(launch.inputs[9 + Component], z, y, x);
  return decay * old_field - source * (derivative0 - derivative1);
}

template <bool ScalarCoefficients, int MetricKind, int Component>
__device__ __forceinline__ float FusedEValue(const FusedYeePhase& launch,
                                             const float* h_fields,
                                             int local_z, int local_y,
                                             int local_x, int z, int y, int x) {
  const BeamzBuffer& output = launch.outputs[Component];
  if (!BufferContains(output, z, y, x)) return 0.0f;
  constexpr int normal_axis = 2 - Component;
  const int coordinates[3] = {z, y, x};
  for (int axis = 0; axis < 3; ++axis) {
    if (axis != normal_axis &&
        (coordinates[axis] == 0 ||
         coordinates[axis] == output.dims[axis] - 1)) {
      return 0.0f;
    }
  }
  constexpr int first_source[3] = {2, 0, 1};
  constexpr int second_source[3] = {1, 2, 0};
  constexpr int first_axis[3] = {1, 0, 2};
  constexpr int second_axis[3] = {0, 2, 1};
  auto difference = [&](int source_component, int axis) {
    int neighbor_z = local_z;
    int neighbor_y = local_y;
    int neighbor_x = local_x;
    if (axis == 0) --neighbor_z;
    if (axis == 1) --neighbor_y;
    if (axis == 2) --neighbor_x;
    const float center = h_fields[source_component * kFusedVolume +
                                  FusedOffset(local_z, local_y, local_x)];
    const float neighbor = h_fields[source_component * kFusedVolume +
                                    FusedOffset(neighbor_z, neighbor_y,
                                                neighbor_x)];
    const int coordinate = axis == 0 ? z : (axis == 1 ? y : x);
    return (center - neighbor) *
           MetricScale<MetricKind>(launch, axis, coordinate);
  };
  const float curl = difference(first_source[Component], first_axis[Component]) -
                     difference(second_source[Component],
                                second_axis[Component]);
  const int linear = (z * static_cast<int>(output.dims[1]) + y) *
                         static_cast<int>(output.dims[2]) +
                     x;
  const float old_field =
      static_cast<const float*>(launch.inputs[Component].data)[linear];
  const float decay =
      ScalarCoefficients
          ? static_cast<const float*>(launch.inputs[6 + Component].data)[0]
          : Read3D(launch.inputs[6 + Component], z, y, x);
  const float source =
      ScalarCoefficients
          ? static_cast<const float*>(launch.inputs[9 + Component].data)[0]
          : Read3D(launch.inputs[9 + Component], z, y, x);
  return decay * old_field + source * curl;
}

template <int Phase, int PsiType, bool PackedLosslessMaterial,
          bool HasMetallicEdges, bool WarpTile = false, int TileX = 64,
          int TileY = 4, int CoreTileX = TileX, int CoreTileY = TileY>
__device__ __forceinline__ void UpdateCombinedCpmlQueueBlock(
    const BeamzLaunch& launch, int max_z, int max_y, int max_x, int high_z,
    int high_y, int high_x, int z_region_blocks, int y_region_blocks,
    int shell_blocks, int core_x_blocks, int core_y_blocks, int block) {
  constexpr int tile_x = WarpTile ? 32 : TileX;
  constexpr int tile_y = WarpTile ? 1 : TileY;
  constexpr int core_tile_x = WarpTile ? 32 : CoreTileX;
  constexpr int core_tile_y = WarpTile ? 1 : CoreTileY;
  const int local_x_thread = WarpTile ? (threadIdx.x & 31) : threadIdx.x;
  const int local_y_thread = WarpTile ? 0 : threadIdx.y;
  const int low = launch.uniform_cpml_thickness;
  int z;
  int y;
  int x;
  if (block < shell_blocks) {
    const int x_blocks = (max_x + tile_x - 1) / tile_x;
    if (block < z_region_blocks) {
      const int y_blocks = (max_y + tile_y - 1) / tile_y;
      const int block_z = block / (x_blocks * y_blocks);
      block -= block_z * x_blocks * y_blocks;
      const int block_y = block / x_blocks;
      const int block_x = block - block_y * x_blocks;
      z = block_z < low ? block_z : high_z + block_z - low;
      y = block_y * tile_y + local_y_thread;
      x = block_x * tile_x + local_x_thread;
      if (y >= max_y || x >= max_x) return;
    } else if ((block -= z_region_blocks, block < y_region_blocks)) {
      const int outer_y = low + max_y - high_y;
      const int y_blocks = (outer_y + tile_y - 1) / tile_y;
      const int block_z = block / (x_blocks * y_blocks);
      block -= block_z * x_blocks * y_blocks;
      const int block_y = block / x_blocks;
      const int block_x = block - block_y * x_blocks;
      z = low + block_z;
      const int local_y = block_y * tile_y + local_y_thread;
      y = local_y < low ? local_y : high_y + local_y - low;
      x = block_x * tile_x + local_x_thread;
      if (local_y >= outer_y || x >= max_x) return;
    } else {
      block -= y_region_blocks;
      const int outer_x = low + max_x - high_x;
      const int inner_y = high_y - low;
      const int region_x_blocks = (outer_x + tile_x - 1) / tile_x;
      const int y_blocks = (inner_y + tile_y - 1) / tile_y;
      const int block_z = block / (region_x_blocks * y_blocks);
      block -= block_z * region_x_blocks * y_blocks;
      const int block_y = block / region_x_blocks;
      const int block_x = block - block_y * region_x_blocks;
      z = low + block_z;
      y = low + block_y * tile_y + local_y_thread;
      const int local_x = block_x * tile_x + local_x_thread;
      x = local_x < low ? local_x : high_x + local_x - low;
      if (y >= high_y || local_x >= outer_x) return;
    }
    UpdateComponent<Phase, 0, true, 0, HasMetallicEdges, true, PsiType,
                    PackedLosslessMaterial>(launch, z, y, x);
    UpdateComponent<Phase, 1, true, 0, HasMetallicEdges, true, PsiType,
                    PackedLosslessMaterial>(launch, z, y, x);
    UpdateComponent<Phase, 2, true, 0, HasMetallicEdges, true, PsiType,
                    PackedLosslessMaterial>(launch, z, y, x);
    return;
  }

  block -= shell_blocks;
  const int block_z = block / (core_x_blocks * core_y_blocks);
  block -= block_z * core_x_blocks * core_y_blocks;
  const int block_y = block / core_x_blocks;
  const int block_x = block - block_y * core_x_blocks;
  const int linear_thread = threadIdx.y * blockDim.x + threadIdx.x;
  const int core_x_thread =
      WarpTile ? local_x_thread : linear_thread % core_tile_x;
  const int core_y_thread =
      WarpTile ? local_y_thread : linear_thread / core_tile_x;
  z = low + block_z;
  y = low + block_y * core_tile_y + core_y_thread;
  x = low + block_x * core_tile_x + core_x_thread;
  if (z >= high_z || y >= high_y || x >= high_x) return;
  UpdateComponent<Phase, 0, false, 0, HasMetallicEdges, false, -1,
                  PackedLosslessMaterial>(launch, z, y, x);
  UpdateComponent<Phase, 1, false, 0, HasMetallicEdges, false, -1,
                  PackedLosslessMaterial>(launch, z, y, x);
  UpdateComponent<Phase, 2, false, 0, HasMetallicEdges, false, -1,
                  PackedLosslessMaterial>(launch, z, y, x);
}

template <int Phase, int PsiType, bool PackedLosslessMaterial,
          bool HasMetallicEdges, int TileX, int TileY, int CoreTileX,
          int CoreTileY>
__global__ void UpdateCombinedCpmlQueue(
    BeamzLaunch launch, int max_z, int max_y, int max_x, int high_z,
    int high_y, int high_x, int z_region_blocks, int y_region_blocks,
    int shell_blocks, int core_x_blocks, int core_y_blocks) {
  UpdateCombinedCpmlQueueBlock<Phase, PsiType, PackedLosslessMaterial,
                               HasMetallicEdges, false, TileX, TileY,
                               CoreTileX, CoreTileY>(
      launch, max_z, max_y, max_x, high_z, high_y, high_x, z_region_blocks,
      y_region_blocks, shell_blocks, core_x_blocks, core_y_blocks, blockIdx.x);
}

// Fuse a complete leapfrog timestep without a device-wide barrier.  Each block
// redundantly computes the one-cell low halo of H in shared memory, then uses it
// to update its disjoint E/H core into a frozen out-of-place destination.
template <bool ScalarCoefficients, int MetricKind>
__global__ void FusedFullStepPec(FusedYeePhase h_launch,
                                 FusedYeePhase e_launch) {
  extern __shared__ float h_fields[];
  const int thread = threadIdx.y * blockDim.x + threadIdx.x;
  const int threads = blockDim.x * blockDim.y;
  const int origin_x = blockIdx.x * kFusedCoreX - 1;
  const int origin_y = blockIdx.y * kFusedCoreY - 1;
  const int origin_z = blockIdx.z * kFusedCoreZ - 1;
  for (int index = thread; index < kFusedVolume; index += threads) {
    const int local_x = index % kFusedSharedX;
    const int local_y = (index / kFusedSharedX) % kFusedSharedY;
    const int local_z = index / (kFusedSharedX * kFusedSharedY);
    const int x = origin_x + local_x;
    const int y = origin_y + local_y;
    const int z = origin_z + local_z;
    h_fields[index] =
        FusedHValue<ScalarCoefficients, MetricKind, 0>(h_launch, z, y, x);
    h_fields[kFusedVolume + index] =
        FusedHValue<ScalarCoefficients, MetricKind, 1>(h_launch, z, y, x);
    h_fields[2 * kFusedVolume + index] =
        FusedHValue<ScalarCoefficients, MetricKind, 2>(h_launch, z, y, x);
  }
  __syncthreads();

  const int local_x = threadIdx.x + 1;
  const int local_y = threadIdx.y + 1;
  const int x = origin_x + local_x;
  const int y = origin_y + local_y;
  for (int core_z = 0; core_z < kFusedCoreZ; ++core_z) {
    const int local_z = core_z + 1;
    const int z = origin_z + local_z;
    const int center = FusedOffset(local_z, local_y, local_x);
    for (int component = 0; component < 3; ++component) {
      const BeamzBuffer& h_output = h_launch.outputs[component];
      if (BufferContains(h_output, z, y, x)) {
        const int linear = (z * static_cast<int>(h_output.dims[1]) + y) *
                               static_cast<int>(h_output.dims[2]) +
                           x;
        static_cast<float*>(h_output.data)[linear] =
            h_fields[component * kFusedVolume + center];
      }
    }
    const float e0 =
        FusedEValue<ScalarCoefficients, MetricKind, 0>(
            e_launch, h_fields, local_z, local_y, local_x, z, y, x);
    const float e1 =
        FusedEValue<ScalarCoefficients, MetricKind, 1>(
            e_launch, h_fields, local_z, local_y, local_x, z, y, x);
    const float e2 =
        FusedEValue<ScalarCoefficients, MetricKind, 2>(
            e_launch, h_fields, local_z, local_y, local_x, z, y, x);
    const float values[3] = {e0, e1, e2};
    for (int component = 0; component < 3; ++component) {
      const BeamzBuffer& e_output = e_launch.outputs[component];
      if (BufferContains(e_output, z, y, x)) {
        const int linear = (z * static_cast<int>(e_output.dims[1]) + y) *
                               static_cast<int>(e_output.dims[2]) +
                           x;
        static_cast<float*>(e_output.data)[linear] = values[component];
      }
    }
  }
}

template <int MetricKind, bool HasMetallicEdges, bool UniformCpml>
void LaunchFusedUpdate(cudaStream_t stream, const BeamzLaunch& launch,
                       dim3 blocks, dim3 threads) {
  int psi_type = -1;
  if (launch.nterms != 0) {
    constexpr int psi_input_base = 13 + 3 * 6;
    psi_type = launch.outputs[3].element_type;
    for (int term = 0; term < 6; ++term) {
      if (launch.inputs[psi_input_base + term].element_type != psi_type ||
          launch.outputs[3 + term].element_type != psi_type) {
        psi_type = -1;
        break;
      }
    }
  }
  const bool packed_lossless_material =
      launch.phase == 1 && launch.inputs[6].rank == 1 &&
      launch.inputs[7].rank == 1 && launch.inputs[8].rank == 1 &&
      launch.inputs[9].rank == 1 && launch.inputs[10].rank == 1 &&
      launch.inputs[11].rank == 1 &&
      launch.inputs[9].element_type == kBeamzS32 &&
      launch.inputs[10].element_type == kBeamzS32 &&
      launch.inputs[11].element_type == kBeamzS32;
  if (launch.phase == 0 && launch.nterms == 0) {
    UpdateFusedComponents<0, false, MetricKind, HasMetallicEdges, UniformCpml>
        <<<blocks, threads, 0, stream>>>(launch);
  } else if (launch.phase == 0) {
    if (psi_type == kBeamzBF16) {
      UpdateFusedComponents<0, true, MetricKind, HasMetallicEdges, UniformCpml,
                            kBeamzBF16>
          <<<blocks, threads, 0, stream>>>(launch);
    } else if (psi_type == kBeamzF32) {
      UpdateFusedComponents<0, true, MetricKind, HasMetallicEdges, UniformCpml,
                            kBeamzF32>
          <<<blocks, threads, 0, stream>>>(launch);
    } else {
      UpdateFusedComponents<0, true, MetricKind, HasMetallicEdges, UniformCpml>
          <<<blocks, threads, 0, stream>>>(launch);
    }
  } else if (launch.nterms == 0) {
    if (packed_lossless_material) {
      UpdateFusedComponents<1, false, MetricKind, HasMetallicEdges, UniformCpml,
                            -1, true><<<blocks, threads, 0, stream>>>(launch);
    } else {
      UpdateFusedComponents<1, false, MetricKind, HasMetallicEdges, UniformCpml>
          <<<blocks, threads, 0, stream>>>(launch);
    }
  } else {
    if (psi_type == kBeamzBF16 && packed_lossless_material) {
      UpdateFusedComponents<1, true, MetricKind, HasMetallicEdges, UniformCpml,
                            kBeamzBF16, true>
          <<<blocks, threads, 0, stream>>>(launch);
    } else if (psi_type == kBeamzF32 && packed_lossless_material) {
      UpdateFusedComponents<1, true, MetricKind, HasMetallicEdges, UniformCpml,
                            kBeamzF32, true>
          <<<blocks, threads, 0, stream>>>(launch);
    } else if (packed_lossless_material) {
      UpdateFusedComponents<1, true, MetricKind, HasMetallicEdges, UniformCpml,
                            -1, true><<<blocks, threads, 0, stream>>>(launch);
    } else if (psi_type == kBeamzBF16) {
      UpdateFusedComponents<1, true, MetricKind, HasMetallicEdges, UniformCpml,
                            kBeamzBF16>
          <<<blocks, threads, 0, stream>>>(launch);
    } else if (psi_type == kBeamzF32) {
      UpdateFusedComponents<1, true, MetricKind, HasMetallicEdges, UniformCpml,
                            kBeamzF32>
          <<<blocks, threads, 0, stream>>>(launch);
    } else {
      UpdateFusedComponents<1, true, MetricKind, HasMetallicEdges, UniformCpml>
          <<<blocks, threads, 0, stream>>>(launch);
    }
  }
}

template <int MetricKind, bool HasMetallicEdges>
void LaunchFusedUpdateForBoundary(cudaStream_t stream,
                                  const BeamzLaunch& launch, dim3 blocks,
                                  dim3 threads) {
  if (launch.nterms != 0 && launch.uniform_cpml_thickness > 0) {
    LaunchFusedUpdate<MetricKind, HasMetallicEdges, true>(stream, launch,
                                                          blocks, threads);
  } else {
    LaunchFusedUpdate<MetricKind, HasMetallicEdges, false>(stream, launch,
                                                           blocks, threads);
  }
}


}  // namespace

cudaError_t BeamzValidatePhase(const BeamzLaunch& launch) {
  return ValidatePhase(launch);
}

int BeamzLaunchStreamed(void* raw_stream, const BeamzLaunch& launch) {
  if (cudaError_t error = BeamzValidatePhase(launch); error != cudaSuccess) {
    return static_cast<int>(error);
  }
  auto stream = reinterpret_cast<cudaStream_t>(raw_stream);
  const PhaseGeometry geometry = MakePhaseGeometry(launch);
  const int y_blocks = (geometry.max_y + kTileY - 1) / kTileY;
  const int tile_z =
      launch.nterms != 0 || launch.metric_kind != 0 ? kPressureTileZ : kTileZ;
  const dim3 threads(kTileX, kTileY, tile_z);
  const dim3 fused_blocks((geometry.max_x + kTileX - 1) / kTileX, y_blocks,
                          (geometry.max_z + tile_z - 1) / tile_z);
  const bool scalar_coefficients =
      launch.nterms == 0 && launch.inputs[6].rank == 0 &&
      launch.inputs[7].rank == 0 && launch.inputs[8].rank == 0 &&
      launch.inputs[9].rank == 0 && launch.inputs[10].rank == 0 &&
      launch.inputs[11].rank == 0;
  if (launch.metric_kind == 0 && scalar_coefficients &&
      launch.metallic_edges == 63) {
    if (launch.phase == 0) {
      UpdateFusedFullPecScalarComponents<0>
          <<<fused_blocks, threads, 0, stream>>>(launch);
    } else {
      UpdateFusedFullPecScalarComponents<1>
          <<<fused_blocks, threads, 0, stream>>>(launch);
    }
  } else if (launch.metric_kind == 0) {
    if (launch.metallic_edges == 0) {
      LaunchFusedUpdateForBoundary<0, false>(stream, launch, fused_blocks,
                                              threads);
    } else {
      LaunchFusedUpdateForBoundary<0, true>(stream, launch, fused_blocks,
                                             threads);
    }
  } else if (launch.metric_kind == 1) {
    if (launch.metallic_edges == 0) {
      LaunchFusedUpdateForBoundary<1, false>(stream, launch, fused_blocks,
                                              threads);
    } else {
      LaunchFusedUpdateForBoundary<1, true>(stream, launch, fused_blocks,
                                             threads);
    }
  } else {
    if (launch.metallic_edges == 0) {
      LaunchFusedUpdateForBoundary<2, false>(stream, launch, fused_blocks,
                                              threads);
    } else {
      LaunchFusedUpdateForBoundary<2, true>(stream, launch, fused_blocks,
                                             threads);
    }
  }
  return static_cast<int>(cudaPeekAtLastError());
}

cudaError_t BeamzEnqueuePhase(void* raw_stream, const BeamzLaunch& launch) {
  return static_cast<cudaError_t>(BeamzLaunchStreamed(raw_stream, launch));
}

bool HasPackedLosslessMaterial(const BeamzLaunch& launch) {
  return launch.inputs[6].rank == 1 && launch.inputs[7].rank == 1 &&
         launch.inputs[8].rank == 1 && launch.inputs[9].rank == 1 &&
         launch.inputs[10].rank == 1 && launch.inputs[11].rank == 1 &&
         launch.inputs[9].element_type == kBeamzS32 &&
         launch.inputs[10].element_type == kBeamzS32 &&
         launch.inputs[11].element_type == kBeamzS32;
}

int UniformPsiType(const BeamzLaunch& launch) {
  constexpr int psi_input_base = 13 + 3 * 6;
  int psi_type = launch.outputs[3].element_type;
  for (int term = 0; term < 6; ++term) {
    if (launch.inputs[psi_input_base + term].element_type != psi_type ||
        launch.outputs[3 + term].element_type != psi_type) {
      return -1;
    }
  }
  return psi_type;
}

bool CpmlCoreScheduleSupported(const BeamzLaunch& h_launch,
                               const BeamzLaunch& e_launch) {
  if (h_launch.nterms != 6 || e_launch.nterms != 6 ||
      h_launch.metric_kind != 0 ||
      e_launch.metric_kind != 0 || h_launch.uniform_cpml_thickness <= 0 ||
      e_launch.uniform_cpml_thickness != h_launch.uniform_cpml_thickness ||
      !HasPackedLosslessMaterial(e_launch)) {
    return false;
  }
  for (int material = 0; material < 6; ++material) {
    if (h_launch.inputs[6 + material].rank != 0) return false;
  }
  const CpmlGeometry geometry = MakeCpmlGeometry(h_launch);
  return geometry.high_z > geometry.low &&
         geometry.high_y > geometry.low && geometry.high_x > geometry.low;
}

template <int Phase, int PsiType, bool PackedLosslessMaterial>
void LaunchCombinedCpmlQueueForType(cudaStream_t stream,
                                    const BeamzLaunch& launch,
                                    const CpmlGeometry& geometry) {
  constexpr int tile_x = PsiType == kBeamzBF16 ? 32 : 64;
  constexpr int tile_y = 4;
  constexpr int core_tile_x = PsiType == kBeamzBF16 ? 64 : tile_x;
  constexpr int core_tile_y = PsiType == kBeamzBF16 ? 2 : tile_y;
  static_assert(tile_x * tile_y == core_tile_x * core_tile_y);
  const int max_z = geometry.field.max_z;
  const int max_y = geometry.field.max_y;
  const int max_x = geometry.field.max_x;
  const int low = geometry.low;
  const int high_z = geometry.high_z;
  const int high_y = geometry.high_y;
  const int high_x = geometry.high_x;
  const int x_blocks = (max_x + tile_x - 1) / tile_x;
  const int z_region_blocks =
      x_blocks * ((max_y + tile_y - 1) / tile_y) *
      (low + max_z - high_z);
  const int y_region_blocks =
      x_blocks * ((low + max_y - high_y + tile_y - 1) / tile_y) *
      (high_z - low);
  const int x_region_blocks =
      ((low + max_x - high_x + tile_x - 1) / tile_x) *
      ((high_y - low + tile_y - 1) / tile_y) * (high_z - low);
  const int shell_blocks =
      z_region_blocks + y_region_blocks + x_region_blocks;
  const int core_x_blocks =
      (high_x - low + core_tile_x - 1) / core_tile_x;
  const int core_y_blocks =
      (high_y - low + core_tile_y - 1) / core_tile_y;
  const int core_blocks = core_x_blocks * core_y_blocks * (high_z - low);
  const dim3 threads(tile_x, tile_y, 1);
  if (launch.metallic_edges == 0) {
    UpdateCombinedCpmlQueue<Phase, PsiType, PackedLosslessMaterial, false,
                            tile_x, tile_y, core_tile_x, core_tile_y>
        <<<shell_blocks + core_blocks, threads, 0, stream>>>(
            launch, max_z, max_y, max_x, high_z, high_y, high_x,
            z_region_blocks, y_region_blocks, shell_blocks, core_x_blocks,
            core_y_blocks);
  } else {
    UpdateCombinedCpmlQueue<Phase, PsiType, PackedLosslessMaterial, true,
                            tile_x, tile_y, core_tile_x, core_tile_y>
        <<<shell_blocks + core_blocks, threads, 0, stream>>>(
            launch, max_z, max_y, max_x, high_z, high_y, high_x,
            z_region_blocks, y_region_blocks, shell_blocks, core_x_blocks,
            core_y_blocks);
  }
}

cudaError_t LaunchCombinedCpmlQueuePhase(cudaStream_t stream,
                                         const BeamzLaunch& launch) {
  const CpmlGeometry geometry = MakeCpmlGeometry(launch);
  const int psi_type = UniformPsiType(launch);
  const bool packed = launch.phase == 1 && HasPackedLosslessMaterial(launch);
  if (launch.phase == 0) {
    if (psi_type == kBeamzBF16) {
      LaunchCombinedCpmlQueueForType<0, kBeamzBF16, false>(
          stream, launch, geometry);
    } else if (psi_type == kBeamzF32) {
      LaunchCombinedCpmlQueueForType<0, kBeamzF32, false>(
          stream, launch, geometry);
    } else {
      LaunchCombinedCpmlQueueForType<0, -1, false>(
          stream, launch, geometry);
    }
  } else if (packed && psi_type == kBeamzBF16) {
    LaunchCombinedCpmlQueueForType<1, kBeamzBF16, true>(
        stream, launch, geometry);
  } else if (packed && psi_type == kBeamzF32) {
    LaunchCombinedCpmlQueueForType<1, kBeamzF32, true>(
        stream, launch, geometry);
  } else if (packed) {
    LaunchCombinedCpmlQueueForType<1, -1, true>(
        stream, launch, geometry);
  } else if (psi_type == kBeamzBF16) {
    LaunchCombinedCpmlQueueForType<1, kBeamzBF16, false>(
        stream, launch, geometry);
  } else if (psi_type == kBeamzF32) {
    LaunchCombinedCpmlQueueForType<1, kBeamzF32, false>(
        stream, launch, geometry);
  } else {
    LaunchCombinedCpmlQueueForType<1, -1, false>(
        stream, launch, geometry);
  }
  return cudaPeekAtLastError();
}

bool BeamzCpmlScheduleSupported(const BeamzLaunch& h_launch,
                                const BeamzLaunch& e_launch) {
  return CpmlCoreScheduleSupported(h_launch, e_launch);
}

cudaError_t BeamzEnqueueCpmlPhase(cudaStream_t stream,
                                  const BeamzLaunch& launch) {
  return LaunchCombinedCpmlQueuePhase(stream, launch);
}

cudaError_t BeamzEnqueueFusedFullStep(cudaStream_t stream,
                                      const BeamzLaunch& h_launch,
                                      const BeamzLaunch& e_launch) {
  const FusedYeePhase h_phase = MakeFusedYeePhase(h_launch);
  const FusedYeePhase e_phase = MakeFusedYeePhase(e_launch);
  const bool scalar_coefficients =
      h_launch.inputs[6].rank == 0 && h_launch.inputs[7].rank == 0 &&
      h_launch.inputs[8].rank == 0 && h_launch.inputs[9].rank == 0 &&
      h_launch.inputs[10].rank == 0 && h_launch.inputs[11].rank == 0 &&
      e_launch.inputs[6].rank == 0 && e_launch.inputs[7].rank == 0 &&
      e_launch.inputs[8].rank == 0 && e_launch.inputs[9].rank == 0 &&
      e_launch.inputs[10].rank == 0 && e_launch.inputs[11].rank == 0;
  PhaseGeometry geometry = MakePhaseGeometry(h_launch);
  const PhaseGeometry e_geometry = MakePhaseGeometry(e_launch);
  geometry.max_x =
      e_geometry.max_x > geometry.max_x ? e_geometry.max_x : geometry.max_x;
  geometry.max_y =
      e_geometry.max_y > geometry.max_y ? e_geometry.max_y : geometry.max_y;
  geometry.max_z =
      e_geometry.max_z > geometry.max_z ? e_geometry.max_z : geometry.max_z;
  const dim3 threads(kFusedCoreX, kFusedCoreY);
  const dim3 blocks(
      (geometry.max_x + kFusedCoreX - 1) / kFusedCoreX,
      (geometry.max_y + kFusedCoreY - 1) / kFusedCoreY,
      (geometry.max_z + kFusedCoreZ - 1) / kFusedCoreZ);
  if (h_launch.metric_kind == 0) {
    if (scalar_coefficients) {
      FusedFullStepPec<true, 0>
          <<<blocks, threads, kFusedSharedBytes, stream>>>(h_phase, e_phase);
    } else {
      FusedFullStepPec<false, 0>
          <<<blocks, threads, kFusedSharedBytes, stream>>>(h_phase, e_phase);
    }
  } else if (h_launch.metric_kind == 1) {
    if (scalar_coefficients) {
      FusedFullStepPec<true, 1>
          <<<blocks, threads, kFusedSharedBytes, stream>>>(h_phase, e_phase);
    } else {
      FusedFullStepPec<false, 1>
          <<<blocks, threads, kFusedSharedBytes, stream>>>(h_phase, e_phase);
    }
  } else if (scalar_coefficients) {
    FusedFullStepPec<true, 2>
        <<<blocks, threads, kFusedSharedBytes, stream>>>(h_phase, e_phase);
  } else {
    FusedFullStepPec<false, 2>
        <<<blocks, threads, kFusedSharedBytes, stream>>>(h_phase, e_phase);
  }
  return cudaPeekAtLastError();
}
