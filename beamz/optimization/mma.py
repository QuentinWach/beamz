"""Method of Moving Asymptotes (MMA) Optimizer in JAX.

Based on the original paper:
Svanberg, K. (1987). The method of moving asymptotes—a new method for structural optimization.
International journal for numerical methods in engineering, 24(2), 359-373.
"""

import jax
import jax.numpy as jnp
from typing import NamedTuple, Optional

class MMAState(NamedTuple):
    """State for the MMA optimizer."""
    step: int
    x_old: jnp.ndarray      # x^{(k-1)}
    x_old_old: jnp.ndarray  # x^{(k-2)}
    low: jnp.ndarray        # Lower asymptote L_j
    upp: jnp.ndarray        # Upper asymptote U_j
    alpha: jnp.ndarray      # Moving asymptote parameters
    beta: jnp.ndarray       # Moving asymptote parameters

def init_mma(x0: jnp.ndarray) -> MMAState:
    """Initialize MMA state."""
    return MMAState(
        step=0,
        x_old=x0,
        x_old_old=x0,
        low=jnp.zeros_like(x0),
        upp=jnp.ones_like(x0),
        alpha=jnp.zeros_like(x0),
        beta=jnp.zeros_like(x0)
    )

@jax.jit
def update_asymptotes(x_val, x_old, x_old_old, low, upp, x_min, x_max, iter_count):
    """
    Update moving asymptotes based on oscillation of design variables.
    L_j^{(k)} and U_j^{(k)}
    """
    # Heuristic parameters (from Svanberg 1987 / standard implementations)
    # 0.7 for shrinking (oscillation), 1.2 for expanding (monotonic)
    # The first two iterations are special.
    
    # Check oscillation: (x_k - x_{k-1}) * (x_{k-1} - x_{k-2})
    # > 0: Monotonic -> Expand asymptotes (more aggressive)
    # < 0: Oscillating -> Shrink asymptotes (more conservative)
    
    delta_k = x_val - x_old
    delta_old = x_old - x_old_old
    
    signs = delta_k * delta_old
    
    gamma = jnp.where(signs > 0, 1.2, jnp.where(signs < 0, 0.7, 1.0))
    
    # Special handling for first two iterations (iter_count starts at 0)
    # Iter 0 (k=1): No history
    # Iter 1 (k=2): One history
    
    # Calculate new asymptotes based on current range
    # Basic rule: L_j = x_j - s_0 * (x_max - x_min)
    # U_j = x_j + s_0 * (x_max - x_min)
    # But updated via gamma
    
    # We use the standard update rule:
    # L_new = x - gamma * (x - L_old)
    # U_new = x + gamma * (U_old - x)
    
    # Initial guess for asymptotes if iter < 2
    # Typically L = x - 0.5*(max-min), U = x + 0.5*(max-min)
    range_width = x_max - x_min
    l_init = x_val - 0.5 * range_width
    u_init = x_val + 0.5 * range_width
    
    l_updated = x_val - gamma * (x_val - low)
    u_updated = x_val + gamma * (upp - x_val)
    
    # Apply initial logic
    l_final = jnp.where(iter_count < 2, l_init, l_updated)
    u_final = jnp.where(iter_count < 2, u_init, u_updated)
    
    # Safety checks (asymptotes shouldn't be too close to x)
    # Typically ensure |x - L| >= 0.01 * range
    min_dist = 0.01 * range_width
    l_final = jnp.minimum(l_final, x_val - min_dist)
    u_final = jnp.maximum(u_final, x_val + min_dist)
    
    return l_final, u_final

@jax.jit
def mma_subproblem_dual_grad(y, p, q, b, constraints_val, grad_g_p, grad_g_q, num_constraints):
    """
    Gradient of the dual objective function for the convex subproblem.
    Dual variables y (lagrange multipliers for constraints).
    We solve for x(y) analytically, then compute gradient w.r.t y.
    """
    # x_j(y) depends on signs of p_ij + sum(y_i * p_ij)
    # In MMA approximation:
    # f_0(x) ~ sum (p_0j / (U_j - x_j) + q_0j / (x_j - L_j))
    # g_i(x) ~ sum (p_ij / (U_j - x_j) + q_ij / (x_j - L_j)) - a_i * z - b_i
    
    # Dual function is concave. We maximize it (minimize negative).
    # This function is usually solved via Newton's method on y.
    # Implementation details are complex; simplified dual step:
    # return grad_dual
    pass 

# For now, we implement a simplified MMA update that assumes a single objective 
# and separable box constraints, or we use a simple rational approximation step.
# Full dual solver in JAX is non-trivial.
# We will implement the explicit primal update approximation if constraints are simple,
# or a simple dual ascent.

@jax.jit
def mma_solve_subproblem(
    x_val, low, upp, 
    p0, q0, # Objective coefficients
    p, q,   # Constraint coefficients (M constraints x N vars)
    b,      # Constraint limits (g_i <= b_i)
    x_min, x_max
):
    """
    Solve the separable convex subproblem.
    Minimize: sum(p0_j / (U_j - x_j) + q0_j / (x_j - L_j)) + ...
    Subject to: sum(p_ij / (...) + ...) <= b_i
    
    Since implementing a robust dual solver in JAX from scratch is risky,
    we'll use a simplified heuristic if no constraints (which matches the current state),
    and a basic dual update for constraints.
    """
    # If no constraints (p, q empty), solution is analytical per variable
    # d/dx = p0/(U-x)^2 - q0/(x-L)^2 = 0
    # sqrt(p0)/(U-x) = sqrt(q0)/(x-L)
    # sqrt(p0)*(x-L) = sqrt(q0)*(U-x)
    # x * (sqrt(p0) + sqrt(q0)) = sqrt(p0)*L + sqrt(q0)*U
    # x = (sqrt(p0)*L + sqrt(q0)*U) / (sqrt(p0) + sqrt(q0))
    
    # With constraints, we need Lagrange multipliers (lambda)
    # L(x, lambda) = f_approx(x) + sum(lambda_i * (g_approx_i(x) - b_i))
    #              = sum_j [ (p0_j + sum(lam*p_ij))/(U-x) + (q0_j + sum(lam*q_ij))/(x-L) ] - const
    # Let P_j(lam) = p0_j + sum(lam_i * p_ij)
    # Let Q_j(lam) = q0_j + sum(lam_i * q_ij)
    # Optimum x_j(lam) = (sqrt(P_j)*L_j + sqrt(Q_j)*U_j) / (sqrt(P_j) + sqrt(Q_j))
    
    # We need to find lam >= 0 to satisfy constraints.
    # Since we likely have few constraints (Volume, Linewidth), we can use a simple loop.
    
    num_constraints = p.shape[0]
    
    if num_constraints == 0:
        # Unconstrained (analytical)
        P = p0
        Q = q0
        # Avoid zero division
        sqrt_P = jnp.sqrt(jnp.maximum(P, 1e-10))
        sqrt_Q = jnp.sqrt(jnp.maximum(Q, 1e-10))
        
        x_new = (sqrt_P * low + sqrt_Q * upp) / (sqrt_P + sqrt_Q)
        return jnp.clip(x_new, x_min, x_max)
    
    # Simple dual ascent for constraints
    lam = jnp.zeros(num_constraints)
    
    # Fixed number of dual iterations for JIT
    for _ in range(15):
        # 1. Compute x(lam)
        P = p0 + lam @ p
        Q = q0 + lam @ q
        
        sqrt_P = jnp.sqrt(jnp.maximum(P, 1e-10))
        sqrt_Q = jnp.sqrt(jnp.maximum(Q, 1e-10))
        
        x_new = (sqrt_P * low + sqrt_Q * upp) / (sqrt_P + sqrt_Q)
        x_new = jnp.clip(x_new, x_min, x_max)
        
        # 2. Evaluate constraints
        # g_i(x) ~ sum (p_ij / (U_j - x_j) + q_ij / (x_j - L_j)) - b_i
        # Using simplified term sum
        term_p = p / (upp - x_new)
        term_q = q / (x_new - low)
        g_val = jnp.sum(term_p + term_q, axis=1) - b
        
        # 3. Update lambda (ascent)
        # lam = max(0, lam + step * g_val)
        # Step size heuristic? 
        # Using a small fixed step or adaptive is tricky.
        # Ideally we use Newton step on the dual.
        # For robustness in this plan, we stick to a simple update or just return unconstrained if 0 constraints active.
        # Let's assume user constraints are penalties for now unless explicit.
        lam = jnp.maximum(0.0, lam + 0.1 * g_val) # Naive
        
    return x_new

@jax.jit
def mma_update(
    state: MMAState,
    x_val: jnp.ndarray,
    grad_f: jnp.ndarray,
    constraints: jnp.ndarray,      # Values g_i(x)
    grad_g: jnp.ndarray,           # Gradients dg_i/dx (M x N)
    x_min: float = 0.0,
    x_max: float = 1.0,
    move_limit: float = 0.2
):
    """
    Perform one MMA iteration.
    
    Args:
        state: Current MMA state.
        x_val: Current design variables.
        grad_f: Gradient of objective function.
        constraints: Array of constraint values (g_i <= 0). 
                     (Note: MMA formulates as g_i(x) <= b_i, but we pass g_i(x) - b_i)
        grad_g: Matrix of constraint gradients (M constraints, N vars).
    """
    
    # 1. Update Asymptotes
    low, upp = update_asymptotes(
        x_val, state.x_old, state.x_old_old, 
        state.low, state.upp, x_min, x_max, state.step
    )
    
    # 2. Approximate Objective f(x)
    # f(x) ~ sum (p0_j / (U_j - x_j) + q0_j / (x_j - L_j)) + r0
    # p0_j = (U_j - x_j)^2 * max(0, df/dx_j)
    # q0_j = (x_j - L_j)^2 * max(0, -df/dx_j)
    # Plus a small regularization term (rho) sometimes added
    
    p0 = (upp - x_val)**2 * jnp.maximum(0.0, grad_f)
    q0 = (x_val - low)**2 * jnp.maximum(0.0, -grad_f)
    
    # 3. Approximate Constraints g_i(x)
    # Similar form.
    # We assume 'constraints' input is (g_i - limit), so we want <= 0.
    # The approximation matches value and gradient at x_val.
    # b_i = - (g_approx(x_val) - g_val) ... 
    # Actually, standard MMA linearizes g_i around x_val using the rational form.
    # p_ij = (U_j - x_j)^2 * max(0, dg_i/dx_j)
    # q_ij = (x_j - L_j)^2 * max(0, -dg_i/dx_j)
    # constant term such that g_approx(x_val) = g_i(x_val)
    # So the constraint is: g_approx(x) <= 0
    # sum(...) + r_i <= 0
    # effectively: sum(...) <= b_i where b_i = -r_i
    
    if grad_g.shape[0] > 0:
        p_const = (upp - x_val)**2 * jnp.maximum(0.0, grad_g)
        q_const = (x_val - low)**2 * jnp.maximum(0.0, -grad_g)
        
        # Calculate limit 'b'
        # g_approx(x) = sum(p/(U-x) + q/(x-L)) - sum(p/(U-x_val) + q/(x_val-L)) + g_val
        # we want g_approx(x) <= 0
        # sum(new) <= sum(old) - g_val
        
        term_p_old = p_const / (upp - x_val)
        term_q_old = q_const / (x_val - low)
        
        b = jnp.sum(term_p_old + term_q_old, axis=1) - constraints
    else:
        p_const = jnp.zeros((0, x_val.shape[0]))
        q_const = jnp.zeros((0, x_val.shape[0]))
        b = jnp.zeros((0,))
        
    # 4. Solve Subproblem
    # Apply move limits to box constraints for subproblem
    x_min_sub = jnp.maximum(x_min, x_val - move_limit * (x_max - x_min))
    x_max_sub = jnp.minimum(x_max, x_val + move_limit * (x_max - x_min))
    
    # Also enforce 0.9 * L <= x <= 0.9 * U rules implicitly via bounds or trust region
    # Standard MMA: 0.9 L + 0.1 x <= x_new <= 0.9 U + 0.1 x
    # We stick to box limits for simplicity
    
    x_new = mma_solve_subproblem(
        x_val, low, upp, 
        p0, q0, 
        p_const, q_const, b,
        x_min_sub, x_max_sub
    )
    
    # 5. Return updated state and new x
    new_state = MMAState(
        step=state.step + 1,
        x_old=x_val,
        x_old_old=state.x_old,
        low=low,
        upp=upp,
        alpha=state.alpha,
        beta=state.beta
    )
    
    return x_new, new_state

