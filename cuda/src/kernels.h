#ifndef BEAMZ_CUDA_KERNELS_H_
#define BEAMZ_CUDA_KERNELS_H_

#include <cuda_runtime_api.h>

#include "launch.h"

// Leaf launchers enqueue one operation and own no graph or timestep policy.
cudaError_t BeamzValidatePhase(const BeamzLaunch& launch);
cudaError_t BeamzEnqueuePhase(void* stream, const BeamzLaunch& launch);
cudaError_t BeamzEnqueueCpmlPhase(cudaStream_t stream,
                                  const BeamzLaunch& launch);
cudaError_t BeamzEnqueueFusedFullStep(cudaStream_t stream,
                                      const BeamzLaunch& h_launch,
                                      const BeamzLaunch& e_launch);
cudaError_t BeamzEnqueueSourceGroup(cudaStream_t stream,
                                    const BeamzLaunch& launch,
                                    const BeamzBuffer& target,
                                    const BeamzSourceGroupLaunch& group,
                                    int32_t step);
cudaError_t BeamzEnqueueDftGroups(cudaStream_t stream,
                                  const BeamzLaunch& h_launch,
                                  const BeamzLaunch& e_launch,
                                  const BeamzDftGroupLaunch& monitors,
                                  int32_t step);

#endif  // BEAMZ_CUDA_KERNELS_H_
