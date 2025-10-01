import os
import jax
import jax.numpy as jnp
from jax import jit, vmap, pmap
from jax import config as jax_config
import time
from typing import Optional, Tuple, Dict, Any
import numpy as np
from beamz.simulation.backends.base import Backend

# TODO: Make this go brrrr on RTX
class JAXBackend(Backend):
    """High-performance JAX backend for FDTD with JIT compilation and multi-device support."""

    backend_name = "jax"

    def __init__(self, device="auto", **kwargs):
        super().__init__()
        
        # Device configuration
        self.device = self._select_device(device)
        self.dtype = kwargs.get("dtype", jnp.float32)
        
        # Performance settings
        self.use_jit = kwargs.get("use_jit", True)
        self.use_64bit = kwargs.get("use_64bit", False)
        
        # Enable 64-bit mode if requested
        if self.use_64bit:
            jax_config.update("jax_enable_x64", True)
            self.dtype = jnp.float64
        
        # XLA flags for optimization
        xla_flags = kwargs.get("xla_flags", {})
        self._set_xla_flags(xla_flags)
        
        # Multi-device setup
        self.devices = jax.devices()
        self.num_devices = len(self.devices)
        self.use_pmap = kwargs.get("use_pmap", self.num_devices > 1)
        
        # Initialize optimized field update functions
        self._init_field_update_functions()
        
        # Performance tracking
        self._timers = {}
        
        print(f"JAX Backend initialized:")
        print(f"  Device: {self.device}")
        print(f"  Available devices: {[str(d) for d in self.devices]}")
        print(f"  JIT compilation: {'enabled' if self.use_jit else 'disabled'}")
        print(f"  64-bit precision: {'enabled' if self.use_64bit else 'disabled'}")

    def _select_device(self, device_str: str) -> str:
        """Select the appropriate device."""
        d = device_str.lower()
        if d == "auto":
            # Get all available devices and check their platforms
            available_devices = jax.devices()
            available_platforms = [device.platform for device in available_devices]
            
            if "gpu" in available_platforms:
                return "gpu"
            elif "tpu" in available_platforms:
                return "tpu"
            else:
                return "cpu"
        return d

    def _set_xla_flags(self, flags: Dict[str, Any]):
        """Set XLA optimization flags for better performance."""
        default_flags = {
            "xla_gpu_triton_gemm_any": True,
            "xla_gpu_enable_latency_hiding_scheduler": True,
        }
        default_flags.update(flags)
        
        xla_flags_str = " ".join(f"--{key}={value}" for key, value in default_flags.items())
        os.environ.setdefault("XLA_FLAGS", xla_flags_str)

    def _init_field_update_functions(self):
        """Initialize optimized field update functions with JIT compilation."""
        if self.use_jit:
            # JIT compile the core FDTD update functions for maximum performance
            self.update_h_fields_compiled = jit(self._update_h_fields_core)
            self.update_e_field_compiled = jit(self._update_e_field_core)
            
            # Compile fused update for even better performance
            self.update_fields_fused_compiled = jit(self._update_fields_fused)
            
            if self.use_pmap and self.num_devices > 1:
                # Parallel versions for multi-device setups
                self.update_h_fields_pmap = pmap(self._update_h_fields_core)
                self.update_e_field_pmap = pmap(self._update_e_field_core)
                print(f"  Multi-device parallelization: enabled ({self.num_devices} devices)")
        else:
            # Non-compiled versions (slower but useful for debugging)
            self.update_h_fields_compiled = self._update_h_fields_core
            self.update_e_field_compiled = self._update_e_field_core
            self.update_fields_fused_compiled = self._update_fields_fused

    def _resolve_dtype(self, dtype):
        resolved = dtype if dtype is not None else self.dtype
        if not self.use_64bit:
            if resolved in (np.float64, jnp.float64):
                resolved = jnp.float32
            elif resolved in (np.complex128, jnp.complex128):
                resolved = jnp.complex64
        return resolved

    def zeros(self, shape, dtype=None):
        """Create an array of zeros with the given shape and optional dtype."""
        resolved_dtype = self._resolve_dtype(dtype)
        return jnp.zeros(shape, dtype=resolved_dtype)

    def ones(self, shape, dtype=None):
        """Create an array of ones with the given shape and optional dtype."""
        resolved_dtype = self._resolve_dtype(dtype)
        return jnp.ones(shape, dtype=resolved_dtype)
        
    def copy(self, array):
        """Create a copy of the array."""
        return jnp.array(array)

    def from_numpy(self, arr):
        """Convert NumPy array to a JAX array while preserving meaningful dtypes."""
        jax_array = jnp.asarray(arr)

        target_np_dtype = np.dtype(self.dtype)
        if (jax_array.dtype != self.dtype and
                np.issubdtype(jax_array.dtype, np.floating) and
                np.issubdtype(target_np_dtype, np.floating)):
            jax_array = jax_array.astype(self.dtype)
        elif (jax_array.dtype != self.dtype and
              np.issubdtype(jax_array.dtype, np.complexfloating) and
              np.issubdtype(target_np_dtype, np.complexfloating)):
            jax_array = jax_array.astype(self.dtype)

        return jax_array

    def to_numpy(self, arr):
        """Convert JAX array to NumPy array."""
        return np.array(arr)
        
    def roll(self, array, shift, axis=None):
        """Roll array elements along a given axis."""
        return jnp.roll(array, shift, axis=axis)

    @staticmethod
    def _update_h_fields_core(Hx, Hy, Ez, sigma, dx, dy, dt, mu0, eps0):
        """Core H-field update function optimized for JAX with maximum performance."""
        # Constants (computed once)
        dt_mu0 = dt / mu0
        half = dt_mu0 * 0.5
        mu0_eps0 = mu0 / eps0

        # Compute curl of E efficiently
        curl_e_x = (Ez[:, 1:] - Ez[:, :-1]) / dy
        curl_e_y = (Ez[1:, :] - Ez[:-1, :]) / dx
        
        # Magnetic conductivity
        sigma_m_x = sigma[:, :-1] * mu0_eps0
        sigma_m_y = sigma[:-1, :] * mu0_eps0
        
        # Update coefficients with numerical stability
        denom_x = 1.0 + sigma_m_x * half
        denom_y = 1.0 + sigma_m_y * half
        
        factor_x = (1.0 - sigma_m_x * half) / denom_x
        factor_y = (1.0 - sigma_m_y * half) / denom_y
        
        source_x = dt_mu0 / denom_x
        source_y = dt_mu0 / denom_y
        
        # Update H fields in-place style operations (JAX will optimize)
        Hx_new = factor_x * Hx - source_x * curl_e_x
        Hy_new = factor_y * Hy + source_y * curl_e_y
        
        return Hx_new, Hy_new

    @staticmethod
    def _update_e_field_core(Ez, Hx, Hy, sigma, eps_r, dx, dy, dt, eps0):
        """Core E-field update function optimized for JAX with maximum performance."""
        # Constants
        dt_eps0 = dt / eps0
        half = dt_eps0 * 0.5
        
        # Compute curl of H for interior points efficiently
        curl_h = jnp.zeros_like(Ez)
        curl_h_x = (Hx[1:-1, 1:] - Hx[1:-1, :-1]) / dy
        curl_h_y = (Hy[1:, 1:-1] - Hy[:-1, 1:-1]) / dx
        
        # Use JAX's efficient array update syntax
        curl_h = curl_h.at[1:-1, 1:-1].set(curl_h_x - curl_h_y)
        
        # Interior regions (vectorized operations)
        sigma_interior = sigma[1:-1, 1:-1]
        eps_r_interior = eps_r[1:-1, 1:-1]
        ez_interior = Ez[1:-1, 1:-1]
        curl_interior = curl_h[1:-1, 1:-1]
        
        # Update coefficients (fully vectorized)
        s_half = sigma_interior * half / eps_r_interior
        denom = 1.0 + s_half
        factor1 = (1.0 - s_half) / denom
        factor2 = dt_eps0 / (eps_r_interior * denom)
        
        # Update E field efficiently
        ez_new_interior = factor1 * ez_interior + factor2 * curl_interior
        Ez_new = Ez.at[1:-1, 1:-1].set(ez_new_interior)
        
        return Ez_new

    @staticmethod
    def _update_fields_fused(Hx, Hy, Ez, sigma, eps_r, dx, dy, dt, mu0, eps0):
        """Fused field update combining H and E updates for maximum efficiency."""
        # Update H fields first
        Hx_new, Hy_new = JAXBackend._update_h_fields_core(
            Hx, Hy, Ez, sigma, dx, dy, dt, mu0, eps0
        )
        
        # Update E field using the new H fields
        Ez_new = JAXBackend._update_e_field_core(
            Ez, Hx_new, Hy_new, sigma, eps_r, dx, dy, dt, eps0
        )
        
        return Hx_new, Hy_new, Ez_new

    def update_h_fields(self, Hx, Hy, Ez, sigma, dx, dy, dt, mu0, eps0):
        """Update magnetic field components with maximum performance."""
        if self.use_pmap and self.num_devices > 1:
            # Use parallel map for multi-device acceleration
            return self.update_h_fields_pmap(Hx, Hy, Ez, sigma, dx, dy, dt, mu0, eps0)
        else:
            # Use JIT-compiled single device version
            return self.update_h_fields_compiled(Hx, Hy, Ez, sigma, dx, dy, dt, mu0, eps0)

    def update_e_field(self, Ez, Hx, Hy, sigma, eps_r, dx, dy, dt, eps0):
        """Update electric field component with maximum performance."""
        if self.use_pmap and self.num_devices > 1:
            # Use parallel map for multi-device acceleration
            return self.update_e_field_pmap(Ez, Hx, Hy, sigma, eps_r, dx, dy, dt, eps0)
        else:
            # Use JIT-compiled single device version
            return self.update_e_field_compiled(Ez, Hx, Hy, sigma, eps_r, dx, dy, dt, eps0)

    def update_fields_fused(self, Hx, Hy, Ez, sigma, eps_r, dx, dy, dt, mu0, eps0):
        """Fused field update for absolute maximum performance."""
        return self.update_fields_fused_compiled(Hx, Hy, Ez, sigma, eps_r, dx, dy, dt, mu0, eps0)

    # Performance utilities
    def benchmark_function(self, fn, *args, num_runs=10):
        """Benchmark a JAX function with proper timing."""
        # Warm up JIT compilation
        for _ in range(3):
            result = fn(*args)
        
        # Ensure compilation is complete
        if hasattr(result, 'block_until_ready'):
            result.block_until_ready()
        
        # Time actual execution
        start_time = time.time()
        for _ in range(num_runs):
            result = fn(*args)
            if hasattr(result, 'block_until_ready'):
                result.block_until_ready()  # Wait for GPU computation to complete
        end_time = time.time()
        
        avg_time = (end_time - start_time) / num_runs
        return avg_time, result

    def memory_usage(self):
        """Get current memory usage."""
        try:
            if self.device == "gpu" and jax.devices("gpu"):
                # Get memory usage for GPU devices
                memory_info = {}
                for i, device in enumerate(jax.devices("gpu")):
                    try:
                        memory_info[f"gpu_{i}"] = device.memory_stats()
                    except:
                        memory_info[f"gpu_{i}"] = "unavailable"
                return memory_info
            return None
        except:
            return None

    def get_device_info(self):
        """Get comprehensive information about available devices."""
        info = {
            "devices": [str(device) for device in self.devices],
            "num_devices": self.num_devices,
            "device_type": self.device,
            "backend_name": "jax",
            "jit_enabled": self.use_jit,
            "dtype": str(self.dtype),
            "memory_usage": self.memory_usage()
        }
        return info

    def warmup_compilation(self, field_shapes):
        """Pre-compile functions with example field shapes to avoid first-call overhead."""
        print("Warming up JAX compilation...")
        
        # Create dummy fields for compilation
        Hx = jnp.zeros(field_shapes["Hx"], dtype=self.dtype)
        Hy = jnp.zeros(field_shapes["Hy"], dtype=self.dtype)
        Ez = jnp.zeros(field_shapes["Ez"], dtype=self.dtype)
        sigma = jnp.ones_like(Ez) * 0.01
        eps_r = jnp.ones_like(Ez) * 2.25
        
        # Dummy parameters
        dx, dy, dt = 1e-8, 1e-8, 1e-15
        mu0, eps0 = 4*jnp.pi*1e-7, 8.854e-12
        
        # Trigger compilation
        _ = self.update_h_fields(Hx, Hy, Ez, sigma, dx, dy, dt, mu0, eps0)
        _ = self.update_e_field(Ez, Hx, Hy, sigma, eps_r, dx, dy, dt, eps0)
        
        print("JAX compilation complete!")

    def compare_with_numpy(self, numpy_backend, test_shapes, num_runs=10):
        """Compare performance with NumPy backend."""
        print(f"Comparing JAX vs NumPy performance...")
        
        results = {"jax": {}, "numpy": {}}
        
        # Test array operations
        for name, shape in test_shapes.items():
            print(f"  Testing {name} with shape {shape}...")
            
            # JAX test
            a_jax = self.ones(shape)
            b_jax = self.ones(shape)
            
            def jax_add_op(a, b):
                return a + b
            
            jax_time, _ = self.benchmark_function(jax_add_op, a_jax, b_jax, num_runs)
            results["jax"][name] = jax_time
            
            # NumPy test
            a_np = numpy_backend.ones(shape)
            b_np = numpy_backend.ones(shape)
            
            start = time.time()
            for _ in range(num_runs):
                _ = a_np + b_np
            numpy_time = (time.time() - start) / num_runs
            results["numpy"][name] = numpy_time
            
            speedup = numpy_time / jax_time
            print(f"    JAX: {jax_time:.6f}s, NumPy: {numpy_time:.6f}s, Speedup: {speedup:.2f}x")
        
        return results 