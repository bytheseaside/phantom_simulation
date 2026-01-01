import numpy as np
from vtkmodules.util.numpy_support import vtk_to_numpy, numpy_to_vtk

# ============================================================================
# PARAMETERS 
# ============================================================================

V0 = 0.0        # potential at r0 [V]
V2 = 1.0        # potential at r2 [V]

r0 = 0.002      # inner radius [m]
r1 = 0.080      # interface radius [m]
r2 = 0.08320     # outer radius [m]

sigma1 = 4   # conductivity inner shell [S/m]
sigma2 = 0.33      # conductivity outer shell [S/m]


epsilon = 1e-12 # small value to avoid division by zero

# ============================================================================
# ANALYTICAL SOLUTION COEFFICIENTS
# ============================================================================
# For nested spherical shells:
# Shell 1 (r0 ≤ r ≤ r1): u₁(r) = A₁ + B₁/r
# Shell 2 (r1 ≤ r ≤ r2): u₂(r) = A₂ + B₂/r
#
# Boundary conditions:
# 1. u₁(r0) = V0
# 2. u₂(r2) = V2
# 3. u₁(r1) = u₂(r1)                    [continuity]
# 4. σ1·∂u₁/∂r|r1 = σ2·∂u₂/∂r|r1      [flux continuity]

# Solving the system:
denominator = r0 * r1 * sigma1 - r1 * r2 * sigma2 + r0 * r2 * (sigma2 - sigma1)
B1 = (- r1 * r2 * V2 * sigma2  + r0 * V0 * ((r1 * sigma1 )+ r2 * (sigma2 - sigma1))) / denominator
A1 =   r0 * r1 * r2 * sigma2 * (V2 - V0) / denominator

B2 = (r0 * r1 * V0 * sigma1 - r1 * r2 * V2 * sigma2 + r0 * r2 * V2 * (sigma2 - sigma1)) / denominator
A2 = r0 * r1 * r2 * sigma1 * (V2 - V0) / denominator

print("="*60)
print("ANALYTICAL SOLUTION COEFFICIENTS")
print("="*60)
print(f"Shell 1 (r0={r0:.4f} to r1={r1:.4f}): u₁(r) = {A1:.6f} + {B1:.6f}/r")
print(f"Shell 2 (r1={r1:.4f} to r2={r2:.4f}): u₂(r) = {A2:.6f} + {B2:.6f}/r")
print(f"u at interface r1: {A1 + B1/r1:.6f} (shell 1) vs {A2 + B2/r1:.6f} (shell 2)")
print()

# ============================================================================
# GET INPUT DATA
# ============================================================================
input_ds = self.GetInputDataObject(0, 0)

# Pass geometry + existing arrays through
output.CopyStructure(input_ds)
output.GetPointData().PassData(input_ds.GetPointData())
output.GetCellData().PassData(input_ds.GetCellData())

# Get points as Nx3 numpy array
pts_vtk = input_ds.GetPoints().GetData()
pts = vtk_to_numpy(pts_vtk)  # shape (N,3)

# ============================================================================
# COMPUTE RADIAL DISTANCE
# ============================================================================
r = np.linalg.norm(pts, axis=1)

# ============================================================================
# COMPUTE ANALYTICAL SOLUTION (VECTORIZED)
# ============================================================================
# Create mask for each region
mask_shell1 = (r >= r0) & (r <= r1)
mask_shell2 = (r > r1)

# Initialize solution array
u_ana = np.zeros_like(r)

# Shell 1: u₁(r) = A₁/r + B₁
u_ana[mask_shell1] = A1 / r[mask_shell1] + B1

# Shell 2: u₂(r) = A₂/r + B₂  
u_ana[mask_shell2] = A2 / r[mask_shell2] + B2

# ============================================================================
# GET NUMERICAL SOLUTION
# ============================================================================
u_num = vtk_to_numpy(input_ds.GetPointData().GetArray("u"))

# ============================================================================
# COMPUTE ERRORS
# ============================================================================
# Absolute error
err_abs = np.abs(u_num - u_ana)

# Relative error: err_abs / max(|u_ana|, epsilon)
err_rel = err_abs / np.maximum(np.abs(u_ana), epsilon)

# ============================================================================
# COMPUTE ANALYTICAL GRADIENT (for flux checks)
# ============================================================================
# ∂u/∂r for each shell
du_dr = np.zeros_like(r)

# Shell 1: ∂u₁/∂r = -A1/r²
du_dr[mask_shell1] = -A1 / r[mask_shell1]**2

# Shell 2: ∂u₂/∂r = -A2/r²
du_dr[mask_shell2] = -A2 / r[mask_shell2]**2

# Flux (σ·∂u/∂r)
flux_ana = np.where(mask_shell1, sigma1 * du_dr, sigma2 * du_dr)

# ============================================================================
# COMPUTE ANALYTICAL SECOND RADIAL DERIVATIVE
# ============================================================================
# d2u/dr2 for u = A/r + B  => d2u/dr2 = 2A/r^3
d2u_dr2_ana = np.zeros_like(r)
d2u_dr2_ana[mask_shell1] = 2.0 * A1 / (r[mask_shell1]**3)
d2u_dr2_ana[mask_shell2] = 2.0 * A2 / (r[mask_shell2]**3)

# ============================================================================
# COMPUTE CONDUCTIVITY (σ)
# ============================================================================
# Based on which shell each point is in
sigma = np.where(mask_shell1, sigma1, sigma2)

# ============================================================================
# ADD ARRAYS TO OUTPUT
# ============================================================================
# Analytical solution
vtk_u_ana = numpy_to_vtk(u_ana, deep=True)
vtk_u_ana.SetName("u_ana")
output.GetPointData().AddArray(vtk_u_ana)

# Absolute error
vtk_err_abs = numpy_to_vtk(err_abs, deep=True)
vtk_err_abs.SetName("err_abs")
output.GetPointData().AddArray(vtk_err_abs)

# Relative error
vtk_err_rel = numpy_to_vtk(err_rel, deep=True)
vtk_err_rel.SetName("err_rel")
output.GetPointData().AddArray(vtk_err_rel)

# Analytical gradient
vtk_du_dr = numpy_to_vtk(du_dr, deep=True)
vtk_du_dr.SetName("du_dr_ana")
output.GetPointData().AddArray(vtk_du_dr)

# Analytical flux
vtk_flux_ana = numpy_to_vtk(flux_ana, deep=True)
vtk_flux_ana.SetName("flux_ana")
output.GetPointData().AddArray(vtk_flux_ana)

# Conductivity
vtk_d2u = numpy_to_vtk(d2u_dr2_ana, deep=True)
vtk_d2u.SetName("d2u_dr2_ana")
output.GetPointData().AddArray(vtk_d2u)

# Shell indicator (0=shell1, 1=shell2)
shell_id = np.where(mask_shell1, 0, 1)
vtk_shell = numpy_to_vtk(shell_id, deep=True)
vtk_shell.SetName("shell_id")
output.GetPointData().AddArray(vtk_shell)

# Sigma 
vtk_sigma = numpy_to_vtk(sigma, deep=True)
vtk_sigma.SetName("sigma")
output.GetPointData().AddArray(vtk_sigma)
