"""script to simulate detecton of small hot spot in cylindrical object + scanner"""
#TODO: lesion size, profile plots

from __future__ import annotations
from array_api_strict._array_object import Array
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import array_api_compat.numpy as np

import parallelproj
import pymirc.viewer as pv
from scipy.ndimage import gaussian_filter
import array_api_compat.cupy as xp

import time

# choose a device (CPU or CUDA GPU)
if "numpy" in xp.__name__:
    # using numpy, device must be cpu
    dev = "cpu"
elif "cupy" in xp.__name__:
    # using cupy, only cuda devices are possible
    dev = xp.cuda.Device(0)
elif "torch" in xp.__name__:
    # using torch valid choices are 'cpu' or 'cuda'
    if parallelproj.cuda_present:
        dev = "cuda"
    else:
        dev = "cpu"

# %% parse the command line

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--num_rings", type=int, default=25)
parser.add_argument("--axial_fov", type=float, default=25.0)
parser.add_argument("--ring_diameter", type=float, default=76.0)
parser.add_argument("--crystal_size", type=float, default=1.0)
parser.add_argument("--scanner_fwhm", type=float, default=2.0)
parser.add_argument("--counts", type=int, default=int(1e8))
parser.add_argument("--num_iter", type=int, default=100)
parser.add_argument("--contrast", type=float, default=40.0)
parser.add_argument("--spot_size", type=int, default=2)

args = parser.parse_args()

num_rings = args.num_rings
axial_fov = args.axial_fov
ring_diameter = args.ring_diameter
crystal_size = args.crystal_size
scanner_fwhm = args.scanner_fwhm
counts = args.counts
num_iter = args.num_iter
contrast = args.contrast
spot_size = args.spot_size

# %%
# Setup of the forward model :math:`\bar{y}(x) = A x + s`
# --------------------------------------------------------
#
# We setup a linear forward operator :math:`A` consisting of an
# image-based resolution model, a non-TOF PET projector and an attenuation model
#
# .. note::
#     The MLEM implementation below works with all linear operators that
#     subclass :class:`.LinearOperator` (e.g. the high-level projectors).

scanner = parallelproj.RegularPolygonPETScannerGeometry(
    xp,
    dev,
    radius=0.5*ring_diameter,
    num_sides=2*int(0.5*ring_diameter*xp.pi/crystal_size),
    num_lor_endpoints_per_side=1,
    lor_spacing=2.0,
    ring_positions= axial_fov * xp.linspace(-1, 1, num_rings),
    symmetry_axis=2,
)

voxel_size = (0.25, 0.25, 0.25)
img_shape = (2*int(24/voxel_size[0]) + 1, 2*int(24/voxel_size[1]) + 1, 2*int(24/voxel_size[2]) + 1)

# %%
# setup the LOR descriptor that defines the sinogram

lor_desc = parallelproj.RegularPolygonPETLORDescriptor(
    scanner,
    radial_trim=20,
    sinogram_order=parallelproj.SinogramSpatialAxisOrder.RVP,
)

proj = parallelproj.RegularPolygonPETProjector(
    lor_desc, img_shape=img_shape, voxel_size=voxel_size
)


# setup a simple test image containing a few "hot rods"

X0, X1 = xp.meshgrid(xp.linspace(-1,1,img_shape[0]), xp.linspace(-1,1,img_shape[1]), indexing='ij')
R = xp.sqrt(X0**2 + X1**2)

x_true = xp.zeros(proj.in_shape, device=dev, dtype=xp.float32)
for i in range(15, img_shape[2]-15):
    x_true[:,:,i] = xp.astype(R < 0.8, xp.float32)

s = [i//2 - spot_size // 2 for i in img_shape]
x_true[s[0]:(s[0]+2),s[1]:(s[1]+2),s[2]:(s[2]+2)] = contrast

# %%
# Attenuation image and sinogram setup
# ------------------------------------

# setup an attenuation image
x_att = 0.01 * xp.astype(x_true > 0, xp.float32)
# calculate the attenuation sinogram
att_sino = xp.exp(-proj(x_att))


# %%
# Complete PET forward model setup
# --------------------------------
#

att_op = parallelproj.ElementwiseMultiplicationOperator(att_sino)

res_model = parallelproj.GaussianFilterOperator(
    proj.in_shape, sigma=scanner_fwhm / (2.35 * proj.voxel_size)
)

# compose all 3 operators into a single linear operator
pet_lin_op = parallelproj.CompositeLinearOperator((att_op, proj, res_model))


# %%
# Simulation of projection data
# -----------------------------
#
# We setup an arbitrary ground truth :math:`x_{true}` and simulate
# noise-free and noisy data :math:`y` by adding Poisson noise.

# simulated noise-free data
noise_free_data = pet_lin_op(x_true)

# generate a contant contamination sinogram
contamination = xp.full(
    noise_free_data.shape,
    0.5 * float(xp.mean(noise_free_data)),
    device=dev,
    dtype=xp.float32,
)

noise_free_data += contamination

scale_fac = counts / float(xp.sum(noise_free_data))
noise_free_data *= scale_fac
contamination *= scale_fac

# add Poisson noise
np.random.seed(1)
y = xp.asarray(
    np.random.poisson(parallelproj.to_numpy_array(noise_free_data)),
    device=dev,
    dtype=xp.float64,
)

# %%
# EM update to minimize :math:`f(x)`
# ----------------------------------
#
# The EM update that can be used in MLEM or OSEM is given by cite:p:`Dempster1977` :cite:p:`Shepp1982` :cite:p:`Lange1984` :cite:p:`Hudson1994`
#
# .. math::
#     x^+ = \frac{x}{(A^k)^H 1} (A^k)^H \frac{y^k}{A^k x + s^k}
#
# to calculate the minimizer of :math:`f(x)` iteratively.
#
# To monitor the convergence we calculate the relative cost
#
# .. math::
#    \frac{f(x) - f(x^*)}{|f(x^*)|}
#
# and the distance to the optimal point
#
# .. math::
#    \frac{\|x - x^*\|}{\|x^*\|}.
#
#
# We setup a function that calculates a single MLEM/OSEM
# update given the current solution, a linear forward operator,
# data, contamination and the adjoint of ones.


def em_update(
    x_cur: Array,
    data: Array,
    op: parallelproj.LinearOperator,
    s: Array,
    adjoint_ones: Array,
) -> Array:
    """EM update

    Parameters
    ----------
    x_cur : Array
        current solution
    data : Array
        data
    op : parallelproj.LinearOperator
        linear forward operator
    s : Array
        contamination
    adjoint_ones : Array
        adjoint of ones

    Returns
    -------
    Array
        _description_
    """
    ybar = op(x_cur) + s
    return x_cur * op.adjoint(data / ybar) / adjoint_ones


# %%
# Run the MLEM iterations
# -----------------------

# initialize x
x = xp.zeros(proj.in_shape, device=dev, dtype=xp.float32)
for i in range(15, img_shape[2]-15):
    x[:,:,i] = xp.astype(R < 1.0, xp.float32)

# calculate A^H 1
adjoint_ones = pet_lin_op.adjoint(
    xp.ones(pet_lin_op.out_shape, dtype=xp.float32, device=dev)
)

for i in range(num_iter):
    print(f"MLEM iteration {(i + 1):03} / {num_iter:03}", end="\r")
    x = em_update(x, y, pet_lin_op, contamination, adjoint_ones)

x /= scale_fac

# %%

x_np = parallelproj.to_numpy_array(x)
x_np_2mm = gaussian_filter(x_np, parallelproj.to_numpy_array(2.0 / (2.35 * proj.voxel_size)))
x_np_3mm = gaussian_filter(x_np, parallelproj.to_numpy_array(3.0 / (2.35 * proj.voxel_size)))
x_true_np = parallelproj.to_numpy_array(x_true)

ims = dict(vmin = 0.0, vmax = 2.0)
r = (np.arange(img_shape[0])- 0.5 * img_shape[0] + 0.5) * voxel_size[0]
c = [i//2 for i in img_shape]

pdf_str = f"sim_{time.strftime("%Y%m%d-%H%M%S")}.pdf"

with PdfPages(pdf_str) as pdf:
    fig, ax = plt.subplots(2, 1, figsize=(8, 8), sharex=True)
    for axx in ax.ravel():
      axx.plot(r, x_true_np[:, c[1], c[2]], label="ground truth")
      axx.plot(r, x_np[:, c[1], c[2]], label="MLEM")
      axx.plot(r, x_np_2mm[:, c[1], c[2]], label="MLEM 2mm sm.")
      axx.plot(r, x_np_3mm[:, c[1], c[2]], label="MLEM 3mm sm.")
      axx.grid(ls = ':')
      axx.legend()
      axx.set_xlabel('x [mm]')
    ax[0].set_title(f"spot diameter {(spot_size*voxel_size[0]):.2f} mm, contrast {contrast:.1f}, scanner res {scanner_fwhm:.1f} mm, counts {counts//int(1e6)} mio., iterations {num_iter}", fontsize = "medium")
    ax[0].set_ylim(-0.05,2.05)
    ax[1].set_ylim(-0.05,5.05)
    fig.show()
    pdf.savefig()

    vi = pv.ThreeAxisViewer([x_np, x_np_2mm, x_np_3mm, x_true_np],
                             ls = '', imshow_kwargs=ims,
                             rowlabels = ['MLEM', 'MLEM 2mm smoothed', 'MLEM 3mm smoothed', 'ground truth'])
    
    pdf.savefig()

    
# %%
# show simulated scanner geometry
fig2 = plt.figure(figsize=(10, 10))
ax = fig2.add_subplot(111, projection="3d")
lor_desc.show_views(ax, views = xp.asarray([0]), planes=xp.asarray([0]))
proj.show_geometry(ax)
fig2.show()


#vi.fig.savefig(f"recons_{time_str}.png")
#fig.savefig(f"profiles_{time_str}.png")