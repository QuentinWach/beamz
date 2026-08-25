#include "graph.h"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <mutex>
#include <unordered_map>

namespace {

// A graph executable stays alive until the event recorded after its most recent
// replay completes. The event gives cache eviction a non-blocking lifetime
// check instead of assuming that submission means completion.
struct GraphCacheEntry {
  cudaGraphExec_t executable = nullptr;
  cudaEvent_t completion = nullptr;
  uint64_t last_used = 0;
  bool completion_recorded = false;
};

struct DeviceGraphCache {
  std::unordered_map<std::string, GraphCacheEntry> entries;
  uint64_t use_clock = 0;
};

struct GraphCache {
  std::mutex mutex;
  // Graph executables and CUDA events belong to their creating device. Keep
  // independent LRU clocks per device so eviction never destroys a resource
  // while another device is current on the calling thread.
  std::unordered_map<int, DeviceGraphCache> devices;
};

GraphCache& CachedGraphs() {
  // CUDA contexts may already be gone when static destructors run. Retain this
  // deliberately bounded cache until process teardown instead.
  static auto* cache = new GraphCache();
  return *cache;
}

void DestroyGraphEntry(GraphCacheEntry entry) {
  if (entry.completion != nullptr) cudaEventDestroy(entry.completion);
  if (entry.executable != nullptr) cudaGraphExecDestroy(entry.executable);
}

// Evict one completed least-recently-used entry at a time. An oversubscribed
// cache is intentional when every executable is in flight: retaining a small
// temporary overflow is both safer and faster than synchronizing the stream or
// destroying an executable that another caller's work still references.
void PruneCompletedEntriesLocked(DeviceGraphCache* cache, size_t capacity) {
  while (cache->entries.size() > capacity) {
    auto victim = cache->entries.end();
    uint64_t victim_use = std::numeric_limits<uint64_t>::max();
    for (auto it = cache->entries.begin(); it != cache->entries.end(); ++it) {
      GraphCacheEntry& entry = it->second;
      if (!entry.completion_recorded ||
          cudaEventQuery(entry.completion) != cudaSuccess) {
        continue;
      }
      if (entry.last_used < victim_use) {
        victim = it;
        victim_use = entry.last_used;
      }
    }
    if (victim == cache->entries.end()) return;
    GraphCacheEntry entry = victim->second;
    cache->entries.erase(victim);
    // cudaEventQuery above established that this executable is no longer
    // referenced by its stream, so releasing it cannot race another caller.
    DestroyGraphEntry(entry);
  }
}

cudaError_t LaunchCachedEntryLocked(DeviceGraphCache* cache,
                                    GraphCacheEntry* entry,
                                    cudaStream_t stream, bool* launched) {
  *launched = false;
  const cudaError_t launch_error = cudaGraphLaunch(entry->executable, stream);
  if (launch_error != cudaSuccess) return launch_error;
  *launched = true;

  entry->last_used = ++cache->use_clock;
  const cudaError_t event_error = cudaEventRecord(entry->completion, stream);
  // Do not use an older event to free a newer in-flight launch. A context error
  // is returned to the caller and the process-lifetime cache remains safe.
  entry->completion_recorded = event_error == cudaSuccess;
  return event_error;
}

template <typename T>
void Append(std::string* key, const T& value) {
  key->append(reinterpret_cast<const char*>(&value), sizeof(value));
}

void AppendBuffer(std::string* key, const BeamzBuffer& value) {
  Append(key, value.data);
  Append(key, value.rank);
  Append(key, value.element_type);
  for (int axis = 0; axis < 4; ++axis) Append(key, value.dims[axis]);
}

void AppendLaunch(std::string* key, const BeamzLaunch& value) {
  Append(key, value.abi_version);
  Append(key, value.cuda_flags);
  Append(key, value.phase);
  Append(key, value.nterms);
  Append(key, value.metric_kind);
  Append(key, value.dt);
  Append(key, value.resolution);
  Append(key, value.inv_resolution);
  Append(key, value.dt_over_eps);
  Append(key, value.dt_over_mu);
  Append(key, value.metallic_edges);
  Append(key, value.uniform_cpml_thickness);
  for (const BeamzBuffer& buffer : value.inputs) AppendBuffer(key, buffer);
  for (const BeamzBuffer& buffer : value.metrics) AppendBuffer(key, buffer);
  for (const BeamzBuffer& buffer : value.outputs) AppendBuffer(key, buffer);
}

void AppendSourceGroup(std::string* key,
                       const BeamzSourceGroupLaunch& value) {
  AppendBuffer(key, value.coefficients);
  AppendBuffer(key, value.waveforms);
  AppendBuffer(key, value.starts);
  AppendBuffer(key, value.current_step);
  Append(key, value.component);
  Append(key, value.timing);
  Append(key, value.coincident);
}

void AppendMonitors(std::string* key, const BeamzDftGroupLaunch& value) {
  AppendBuffer(key, value.indices);
  AppendBuffer(key, value.weights);
  AppendBuffer(key, value.frequencies);
  AppendBuffer(key, value.component_masks);
  AppendBuffer(key, value.counts);
  AppendBuffer(key, value.codes);
  AppendBuffer(key, value.windows);
  AppendBuffer(key, value.dft_re);
  AppendBuffer(key, value.dft_im);
  AppendBuffer(key, value.dft_weight);
  AppendBuffer(key, value.time);
  AppendBuffer(key, value.current_step);
  Append(key, value.monitor_count);
}

}  // namespace

std::string BeamzGraphKey(const char* schedule, void* stream,
                          const BeamzProgramLaunch& program) {
  std::string key(schedule);
  key.push_back('\0');
  Append(&key, stream);
  Append(&key, program.field_bank_count);
  Append(&key, program.nsteps);
  AppendLaunch(&key, program.h_ab);
  AppendLaunch(&key, program.e_ab);
  if (program.field_bank_count == 2) {
    AppendLaunch(&key, program.h_ba);
    AppendLaunch(&key, program.e_ba);
  }
  Append(&key, program.source_group_count);
  for (int32_t index = 0; index < program.source_group_count; ++index) {
    AppendSourceGroup(&key, program.source_groups[index]);
  }
  const bool has_monitors = program.monitors != nullptr;
  Append(&key, has_monitors);
  if (has_monitors) AppendMonitors(&key, *program.monitors);
  return key;
}

cudaError_t BeamzLaunchGraph(cudaStream_t stream, const std::string& key,
                             bool cache_enabled, int32_t cache_capacity,
                             const std::function<cudaError_t()>& enqueue) {
  cache_enabled = cache_enabled && cache_capacity > 0;
  const size_t capacity = static_cast<size_t>(cache_capacity);
  GraphCache& cache = CachedGraphs();
  int device = -1;
  if (cache_enabled && cudaGetDevice(&device) != cudaSuccess) {
    // Capturing is still useful after a cache bookkeeping failure, but never
    // insert a resource whose owning device is unknown.
    cache_enabled = false;
  }
  if (cache_enabled) {
    std::lock_guard<std::mutex> lock(cache.mutex);
    DeviceGraphCache& device_cache = cache.devices[device];
    PruneCompletedEntriesLocked(&device_cache, capacity);
    const auto cached = device_cache.entries.find(key);
    if (cached != device_cache.entries.end()) {
      bool launched = false;
      return LaunchCachedEntryLocked(&device_cache, &cached->second, stream,
                                     &launched);
    }
  }

  cudaGraph_t graph = nullptr;
  cudaGraphExec_t executable = nullptr;
  cudaError_t error =
      cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal);
  if (error != cudaSuccess) {
    (void)cudaGetLastError();
    return enqueue();
  }
  error = enqueue();
  const cudaError_t end_error = cudaStreamEndCapture(stream, &graph);
  if (error == cudaSuccess) error = end_error;
  if (error == cudaSuccess) error = cudaGraphInstantiate(&executable, graph, 0);
  if (error == cudaSuccess && cache_enabled) {
    GraphCacheEntry new_entry{};
    new_entry.executable = executable;
    error =
        cudaEventCreateWithFlags(&new_entry.completion, cudaEventDisableTiming);
    if (error != cudaSuccess) {
      // An event-allocation failure only disables the optional cache. The
      // already-instantiated graph remains a valid one-shot launch.
      cache_enabled = false;
      error = cudaGraphLaunch(executable, stream);
    } else {
      GraphCacheEntry duplicate{};
      bool destroy_duplicate = false;
      GraphCacheEntry failed_entry{};
      bool destroy_failed_entry = false;
      {
        std::lock_guard<std::mutex> lock(cache.mutex);
        DeviceGraphCache& device_cache = cache.devices[device];
        PruneCompletedEntriesLocked(&device_cache, capacity);
        const auto [entry, inserted] =
            device_cache.entries.emplace(key, new_entry);
        if (!inserted) {
          // Captures can race between host threads. Reuse the first complete
          // executable and dispose of the unlaunched duplicate after unlocking.
          duplicate = new_entry;
          destroy_duplicate = true;
          executable = nullptr;
        } else {
          executable = nullptr;  // Cache now owns the executable and its event.
        }
        bool launched = false;
        error = LaunchCachedEntryLocked(&device_cache, &entry->second, stream,
                                        &launched);
        if (inserted && !launched) {
          // A failed launch never reached the stream, so its event and
          // executable can be removed immediately.
          failed_entry = entry->second;
          device_cache.entries.erase(entry);
          destroy_failed_entry = true;
        }
      }
      if (destroy_duplicate) DestroyGraphEntry(duplicate);
      if (destroy_failed_entry) DestroyGraphEntry(failed_entry);
    }
  } else if (error == cudaSuccess) {
    error = cudaGraphLaunch(executable, stream);
  }
  if (!cache_enabled && executable != nullptr) {
    cudaGraphExecDestroy(executable);
  }
  if (graph != nullptr) cudaGraphDestroy(graph);
  return error;
}
