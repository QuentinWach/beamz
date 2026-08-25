#include "launch.h"

#include <cuda_runtime_api.h>

#include <cstdint>

#include "kernels.h"
#include "yee_primitives.cuh"

namespace {

constexpr int kTileX = 32;
constexpr int kTileY = 4;
constexpr int kTileZ = 2;
constexpr int kAxis0Elements = (kTileZ + 2) * kTileY * kTileX;
constexpr int kAxis1Elements = kTileZ * (kTileY + 2) * kTileX;
constexpr int kAxis2Elements = kTileZ * kTileY * (kTileX + 2);
constexpr int kMaxSharedElements = kAxis0Elements;

cudaError_t ValidateHopperPhase(const BeamzLaunch& launch) {
  if (cudaError_t error = BeamzValidatePhase(launch); error != cudaSuccess) {
    return error;
  }
  if (launch.metric_kind != 0) return cudaErrorInvalidValue;
  // The tiled experiment stages FP32 recurrence state.  The shared validator
  // accepts BF16 for streamed kernels, but Hopper deliberately rejects it
  // rather than silently taking a numerically different path.
  for (int term = 0; term < launch.nterms; ++term) {
    if (launch.inputs[31 + term].element_type != kBeamzF32 ||
        launch.outputs[3 + term].element_type != kBeamzF32) {
      return cudaErrorInvalidValue;
    }
  }
  return cudaSuccess;
}

__device__ __forceinline__ int64_t Offset(const BeamzBuffer& value, int z,
                                          int y, int x) {
  if (value.rank == 0) return 0;
  const int iz = value.dims[0] == 1 ? 0 : z;
  const int iy = value.dims[1] == 1 ? 0 : y;
  const int ix = value.dims[2] == 1 ? 0 : x;
  return (static_cast<int64_t>(iz) * value.dims[1] + iy) * value.dims[2] + ix;
}

__device__ __forceinline__ float Read(const BeamzBuffer& value, int z, int y,
                                      int x) {
  return static_cast<const float*>(value.data)[Offset(value, z, y, x)];
}

__device__ __forceinline__ float ReadChecked(const BeamzBuffer& value, int z,
                                             int y, int x) {
  if (z < 0 || y < 0 || x < 0 || z >= value.dims[0] || y >= value.dims[1] ||
      x >= value.dims[2]) {
    return 0.0f;
  }
  return Read(value, z, y, x);
}

__device__ __forceinline__ int DirectionalElements(int axis) {
  return axis == 0 ? kAxis0Elements
                   : (axis == 1 ? kAxis1Elements : kAxis2Elements);
}

__device__ __forceinline__ int DirectionalOffset(int axis, int z, int y,
                                                 int x) {
  if (axis == 0) return (z * kTileY + y) * kTileX + x;
  if (axis == 1) return (z * (kTileY + 2) + y) * kTileX + x;
  return (z * kTileY + y) * (kTileX + 2) + x;
}

__device__ __forceinline__ void StageDirectional(
    float* tile, const BeamzBuffer& source, int axis, int index, int base_z,
    int base_y, int base_x) {
  int value = index;
  const int width = axis == 2 ? kTileX + 2 : kTileX;
  const int height = axis == 1 ? kTileY + 2 : kTileY;
  const int local_x = value % width;
  value /= width;
  const int local_y = value % height;
  const int local_z = value / height;
  const int global_x = base_x + local_x - (axis == 2 ? 1 : 0);
  const int global_y = base_y + local_y - (axis == 1 ? 1 : 0);
  const int global_z = base_z + local_z - (axis == 0 ? 1 : 0);
  tile[index] = ReadChecked(source, global_z, global_y, global_x);
}

__device__ __forceinline__ float BoundaryDifference(
    const BeamzBuffer& value, int axis, int z, int y, int x, int edge_mask,
    float inv_dx) {
  const int coordinate = axis == 0 ? z : (axis == 1 ? y : x);
  const int size = static_cast<int>(value.dims[axis]);
  if (coordinate == 0) {
    return edge_mask & (1 << (2 * axis)) ? Read(value, z, y, x) * inv_dx
                                         : 0.0f;
  }
  if (coordinate == size) {
    int last_z = z, last_y = y, last_x = x;
    if (axis == 0) last_z = size - 1;
    if (axis == 1) last_y = size - 1;
    if (axis == 2) last_x = size - 1;
    return edge_mask & (1 << (2 * axis + 1))
               ? -Read(value, last_z, last_y, last_x) * inv_dx
               : 0.0f;
  }
  int low_z = z, low_y = y, low_x = x;
  if (axis == 0) --low_z;
  if (axis == 1) --low_y;
  if (axis == 2) --low_x;
  return (Read(value, z, y, x) - Read(value, low_z, low_y, low_x)) *
         inv_dx;
}

__device__ __forceinline__ float CorrectCpml(
    float derivative, int term, int z, int y, int x,
    const BeamzLaunch& launch) {
  const auto* descriptor = static_cast<const int32_t*>(launch.inputs[12].data);
  const int axis = beamz::cuda::yee::CpmlAxis(term);
  const int low = descriptor[term * 5 + 2];
  const int high = descriptor[term * 5 + 3];
  const float sign = beamz::cuda::yee::CpmlSign(term);
  const BeamzBuffer& target = launch.outputs[term / 2];
  const int coordinate = axis == 0 ? z : (axis == 1 ? y : x);
  const int axis_size = static_cast<int>(target.dims[axis]);
  int packed = -1;
  if (!beamz::cuda::yee::CpmlPackedCoordinate(coordinate, axis_size, low, high,
                                               &packed)) {
    return sign * derivative;
  }
  int pz = z, py = y, px = x;
  if (axis == 0) pz = packed;
  if (axis == 1) py = packed;
  if (axis == 2) px = packed;
  const int coefficient_base = 13 + 3 * term;
  const int psi_base = 13 + 3 * launch.nterms;
  const BeamzBuffer& psi_input = launch.inputs[psi_base + term];
  const BeamzBuffer& psi_output = launch.outputs[3 + term];
  const BeamzBuffer& a = launch.inputs[coefficient_base];
  const BeamzBuffer& b = launch.inputs[coefficient_base + 1];
  const BeamzBuffer& inv_kappa = launch.inputs[coefficient_base + 2];
  if (pz < 0 || pz >= psi_output.dims[0] || py < 0 ||
      py >= psi_output.dims[1] || px < 0 || px >= psi_output.dims[2] ||
      packed >= a.dims[axis] || packed >= b.dims[axis] ||
      packed >= inv_kappa.dims[axis]) {
    return sign * derivative;
  }
  const int64_t psi_offset = Offset(psi_output, pz, py, px);
  const float old_psi = static_cast<const float*>(psi_input.data)[psi_offset];
  const float next_psi = beamz::cuda::yee::AdvanceCpmlPsi(
      Read(b, pz, py, px), old_psi, Read(a, pz, py, px), derivative);
  static_cast<float*>(psi_output.data)[psi_offset] = next_psi;
  return beamz::cuda::yee::CorrectCpmlDerivative(
      sign, derivative, Read(inv_kappa, pz, py, px), next_psi);
}

__device__ __forceinline__ void DerivativePlan(int component, int* first_source,
                                               int* second_source,
                                               int* first_axis,
                                               int* second_axis) {
  const auto first = beamz::cuda::yee::FirstCurlTerm(component);
  const auto second = beamz::cuda::yee::SecondCurlTerm(component);
  *first_source = first.source_component;
  *second_source = second.source_component;
  *first_axis = first.derivative_axis;
  *second_axis = second.derivative_axis;
}

__device__ __forceinline__ float SharedForward(const float* tile, int axis,
                                               int lz, int ly, int lx,
                                               float inv_dx) {
  int cz = lz + (axis == 0 ? 1 : 0);
  int cy = ly + (axis == 1 ? 1 : 0);
  int cx = lx + (axis == 2 ? 1 : 0);
  int nz = cz, ny = cy, nx = cx;
  if (axis == 0) ++nz;
  if (axis == 1) ++ny;
  if (axis == 2) ++nx;
  return (tile[DirectionalOffset(axis, nz, ny, nx)] -
          tile[DirectionalOffset(axis, cz, cy, cx)]) *
         inv_dx;
}

__device__ __forceinline__ float SharedBackward(const float* tile, int axis,
                                                int lz, int ly, int lx,
                                                float inv_dx) {
  int cz = lz + (axis == 0 ? 1 : 0);
  int cy = ly + (axis == 1 ? 1 : 0);
  int cx = lx + (axis == 2 ? 1 : 0);
  int pz = cz, py = cy, px = cx;
  if (axis == 0) --pz;
  if (axis == 1) --py;
  if (axis == 2) --px;
  return (tile[DirectionalOffset(axis, cz, cy, cx)] -
          tile[DirectionalOffset(axis, pz, py, px)]) *
         inv_dx;
}

__global__ __launch_bounds__(256, 2) void UpdateTiled(BeamzLaunch launch,
                                                       int component) {
  __shared__ float first_tile[kMaxSharedElements];
  __shared__ float second_tile[kMaxSharedElements];
  int first_source, second_source, first_axis, second_axis;
  DerivativePlan(component, &first_source, &second_source, &first_axis,
                 &second_axis);
  const BeamzBuffer& first = launch.inputs[3 + first_source];
  const BeamzBuffer& second = launch.inputs[3 + second_source];
  const int base_x = blockIdx.x * kTileX;
  const int base_y = blockIdx.y * kTileY;
  const int base_z = blockIdx.z * kTileZ;
  const int thread_linear =
      (threadIdx.z * blockDim.y + threadIdx.y) * blockDim.x + threadIdx.x;
  const int first_elements = DirectionalElements(first_axis);
  const int second_elements = DirectionalElements(second_axis);
  const int staged_elements =
      first_elements > second_elements ? first_elements : second_elements;
  for (int index = thread_linear; index < staged_elements;
       index += blockDim.x * blockDim.y * blockDim.z) {
    if (index < first_elements) {
      StageDirectional(first_tile, first, first_axis, index, base_z, base_y,
                       base_x);
    }
    if (index < second_elements) {
      StageDirectional(second_tile, second, second_axis, index, base_z, base_y,
                       base_x);
    }
  }
  __syncthreads();

  const BeamzBuffer& input = launch.inputs[component];
  const BeamzBuffer& output = launch.outputs[component];
  const int x = base_x + threadIdx.x;
  const int y = base_y + threadIdx.y;
  const int z = base_z + threadIdx.z;
  if (x >= output.dims[2] || y >= output.dims[1] || z >= output.dims[0]) return;
  const bool zero_on_wall = beamz::cuda::yee::PecConstrained(
      output, launch.phase, component, launch.metallic_edges, z, y, x);
  if (launch.nterms == 0 && zero_on_wall) {
    static_cast<float*>(output.data)[Offset(output, z, y, x)] = 0.0f;
    return;
  }
  const int lx = threadIdx.x;
  const int ly = threadIdx.y;
  const int lz = threadIdx.z;
  const float inv_dx = 1.0f / launch.resolution;
  float derivative0;
  float derivative1;
  if (launch.phase == 0) {
    const int c0 = first_axis == 0 ? z : (first_axis == 1 ? y : x);
    const int c1 = second_axis == 0 ? z : (second_axis == 1 ? y : x);
    derivative0 = c0 + 1 < first.dims[first_axis]
                      ? SharedForward(first_tile, first_axis, lz, ly, lx, inv_dx)
                      : 0.0f;
    derivative1 = c1 + 1 < second.dims[second_axis]
                      ? SharedForward(second_tile, second_axis, lz, ly, lx, inv_dx)
                      : 0.0f;
  } else {
    const int c0 = first_axis == 0 ? z : (first_axis == 1 ? y : x);
    const int c1 = second_axis == 0 ? z : (second_axis == 1 ? y : x);
    derivative0 = c0 > 0 && c0 < first.dims[first_axis]
                      ? SharedBackward(first_tile, first_axis, lz, ly, lx, inv_dx)
                      : BoundaryDifference(first, first_axis, z, y, x,
                                           launch.metallic_edges, inv_dx);
    derivative1 = c1 > 0 && c1 < second.dims[second_axis]
                      ? SharedBackward(second_tile, second_axis, lz, ly, lx, inv_dx)
                      : BoundaryDifference(second, second_axis, z, y, x,
                                           launch.metallic_edges, inv_dx);
  }
  const float curl =
      launch.nterms
          ? CorrectCpml(derivative0, 2 * component, z, y, x, launch) +
                CorrectCpml(derivative1, 2 * component + 1, z, y, x, launch)
          : derivative0 - derivative1;
  const int64_t linear = Offset(output, z, y, x);
  if (zero_on_wall) {
    // Match streamed CPML semantics: advance psi before masking a PEC field.
    static_cast<float*>(output.data)[linear] = 0.0f;
    return;
  }
  const float old_field = static_cast<const float*>(input.data)[linear];
  const bool packed_lossless_material =
      launch.phase == 1 && launch.inputs[6 + component].rank == 1 &&
      launch.inputs[9 + component].rank == 1 &&
      launch.inputs[9 + component].element_type == kBeamzS32;
  float decay;
  float source;
  if (packed_lossless_material) {
    decay = 1.0f;
    source = beamz::cuda::yee::PackedMaterialSource(
        launch.inputs[6 + component], launch.inputs[9 + component], linear);
  } else {
    decay = Read(launch.inputs[6 + component], z, y, x);
    source = Read(launch.inputs[9 + component], z, y, x);
  }
  static_cast<float*>(output.data)[linear] = beamz::cuda::yee::AdvanceYeeField(
      launch.phase, old_field, decay, source, curl);
}

}  // namespace

int BeamzLaunchHopper(void* raw_stream, const BeamzLaunch& launch) {
  if (cudaError_t error = ValidateHopperPhase(launch); error != cudaSuccess) {
    return static_cast<int>(error);
  }
  auto stream = reinterpret_cast<cudaStream_t>(raw_stream);
  const dim3 threads(kTileX, kTileY, kTileZ);
  for (int component = 0; component < 3; ++component) {
    const BeamzBuffer& output = launch.outputs[component];
    const dim3 blocks((output.dims[2] + kTileX - 1) / kTileX,
                      (output.dims[1] + kTileY - 1) / kTileY,
                      (output.dims[0] + kTileZ - 1) / kTileZ);
    UpdateTiled<<<blocks, threads, 0, stream>>>(launch, component);
  }
  return static_cast<int>(cudaPeekAtLastError());
}
