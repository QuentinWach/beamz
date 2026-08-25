#ifndef BEAMZ_CUDA_LAUNCH_H_
#define BEAMZ_CUDA_LAUNCH_H_

#include <cstdint>

enum BeamzElementType : int32_t {
  kBeamzF32 = 0,
  kBeamzS32 = 1,
  kBeamzBF16 = 2,
};

struct BeamzBuffer {
  void* data;
  int32_t rank;
  int32_t element_type;
  int64_t dims[4];
};

struct BeamzLaunch {
  int32_t abi_version;
  int32_t cuda_flags;
  int32_t phase;
  int32_t nterms;
  int32_t metric_kind;
  float dt;
  float resolution;
  float inv_resolution;
  float dt_over_eps;
  float dt_over_mu;
  int32_t metallic_edges;
  int32_t uniform_cpml_thickness;
  BeamzBuffer inputs[37];
  BeamzBuffer metrics[3];
  BeamzBuffer outputs[9];
};

struct BeamzSourceGroupLaunch {
  BeamzBuffer coefficients;
  BeamzBuffer waveforms;
  BeamzBuffer starts;
  BeamzBuffer current_step;
  int32_t component;
  int32_t timing;
  int32_t coincident;
  // Proven by the Python compiler from static source origins and extents. The
  // source kernel can replace atomics with ordinary adds only for this case.
  int32_t disjoint;
};

struct BeamzDftGroupLaunch {
  BeamzBuffer indices;
  BeamzBuffer weights;
  BeamzBuffer frequencies;
  BeamzBuffer component_masks;
  BeamzBuffer counts;
  BeamzBuffer codes;
  BeamzBuffer windows;
  BeamzBuffer dft_re;
  BeamzBuffer dft_im;
  BeamzBuffer dft_weight;
  BeamzBuffer time;
  BeamzBuffer current_step;
  int32_t monitor_count;
};

// A non-owning, complete native timestep program assembled by the FFI decoder.
// One field bank describes in-place graph execution; two banks describe the
// temporal ping-pong schedule. Source and monitor pointers remain valid for the
// duration of graph lookup/capture because their storage belongs to the handler.
struct BeamzProgramLaunch {
  BeamzLaunch h_ab;
  BeamzLaunch e_ab;
  BeamzLaunch h_ba;
  BeamzLaunch e_ba;
  const BeamzSourceGroupLaunch* source_groups;
  int32_t source_group_count;
  const BeamzDftGroupLaunch* monitors;
  int32_t field_bank_count;
  int32_t nsteps;
  int32_t graph_cache_capacity = 32;
  // One immutable compiler decision, cross-validated once before capture. It
  // prevents individual launchers from re-inferring incompatible fast paths.
  int32_t schedule_flags = 0;
};

// Returns zero after enqueueing all work, otherwise a CUDA runtime error code.
int BeamzLaunchStreamed(void* stream, const BeamzLaunch& launch);
int BeamzLaunchProgram(void* stream, const BeamzProgramLaunch& program);
int BeamzLaunchHopper(void* stream, const BeamzLaunch& launch);

#endif  // BEAMZ_CUDA_LAUNCH_H_
