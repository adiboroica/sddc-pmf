import sys
import os

# Get the parent directory of the current directory (subdir2)
parent_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir, os.pardir))
sys.path.append(parent_dir)

import numpy as np
import sparse

from pmf import PMFModel, ModelHyperParams
from pmf.params import (
    InitParams,
    InitXYsvd,
    InitXYconstant,
    InitXYrandomBeta,
    InitXYrandomUniform,
    InitXYrandomGamma,
)
from pmf.vi import CaviParams
from report.simulation.data import Data
from utils.distributions import truncatedgamma
from utils.logger import Logger


"""
Define parameters for the simulation.
"""

# Set the random seed for reproducibility
rng = np.random.default_rng(12345)

T = 20  # Number of time steps
N1 = 50  # Number of source nodes
N2 = 40  # Number of destination nodes
k1 = 2  # Number of clusters for source nodes
k2 = 2  # Number of clusters for destination nodes

alpha, beta = 2, 6  # Shape and rate parameters for the gamma distribution

# Sample the correction factors shape and rate parameters
qx = np.full((T, N1), 1.0)
rx = np.full((T, N1), 1.0)
qy = np.full((T, N2), 1.0)
ry = np.full((T, N2), 1.0)


"""
Simulate the data.
"""


# Step 1: Sample the interaction matrix B
B = rng.gamma(alpha, 1 / beta, size=(k1, k2))

# Step 2: Sample the source nodes cluster labels
Fx = rng.choice(k1, size=N1)

# Step 3: Sample the destination nodes cluster labels
Fy = rng.choice(k2, size=N2)

# Step 4: Sample the correction factors
rho_x = truncatedgamma.sample(qx, rx, rng=rng)
rho_y = truncatedgamma.sample(qy, ry, rng=rng)

# Step 5: Simulate the adjacency matrices
prob_matrix = B[np.ix_(Fx, Fy)]  # shape (N1, N2)
poisson_rate = np.einsum(
    "ti, tj, ij -> tij", rho_x, rho_y, prob_matrix
)  # shape (T, N1, N2)
bernoulli_rate = -np.expm1(-poisson_rate)
A = rng.binomial(1, bernoulli_rate)

# Transform the adjacency matrix in a sparse matrix
A = sparse.COO.from_numpy(A)

# Print results
print(f"T: {T}, N1: {N1}, N2: {N2}")
print("k1: {k1}, k2: {k2}")
print("Probability matrix B:", B.shape)
print("Source nodes cluster labels Fx:", Fx.shape)
print("Destination nodes cluster labels Fy:", Fy.shape)
print("Correction factors rho_x:", rho_x.shape)
print("Correction factors rho_y:", rho_y.shape)
print("Adjacency matrix A:", A)
print("")


"""
Save the data to a file.
"""


data = Data(
    k1=k1,
    k2=k2,
    Fx=Fx,
    Fy=Fy,
    probability_matrix=B,
    rho_x=rho_x,
    rho_y=rho_y,
)

# Save the data to a file
data.save("data.pkl", verbose=True)
print("")


"""
Set the folder to save the results.
"""


models_folder_path = "models"


"""
Set the hyperparameters for the model.
"""


a = 1.0
b = 1.0
c = 0.1
alpha = 1.0
beta = 1.0

hyperparams = ModelHyperParams(
    num_features=2,
    a_x=a,
    a_y=a,
    b_x=b,
    b_y=b,
    c_x=c,
    c_y=c,
    alpha_x=alpha,
    alpha_y=alpha,
    beta_x=beta,
    beta_y=beta,
)

logger = Logger("pmf_model", log_to_console=True)


"""
Set the cavi parameters.
"""

max_iterations = 10**4
eval_interval = 2
elbo_tolerance = 1e-6

compute_prediction_params = False


"""
Train the SVD model.
"""


# Initialize the model
svd_model = PMFModel(hyperparams, logger=logger)

# Fit the model
svd_model.fit(
    A,
    init_params=InitParams(
        xy_strategy=InitXYsvd(),
    ),
    cavi_params=CaviParams(
        max_iterations=max_iterations,
        eval_interval=eval_interval,
        elbo_tol=elbo_tolerance,
    ),
    compute_prediction_params=compute_prediction_params,
)

# Save the model
svd_model.save(os.path.join(models_folder_path, "model.pkl"))


"""
Train the constant model.
"""


# Initialize the model
constant_model = PMFModel(hyperparams, logger=logger)

# Fit the model
constant_model.fit(
    A,
    init_params=InitParams(
        xy_strategy=InitXYconstant(),
    ),
    cavi_params=CaviParams(
        max_iterations=max_iterations,
        eval_interval=eval_interval,
        elbo_tol=elbo_tolerance,
    ),
    compute_prediction_params=compute_prediction_params,
)

# Save the model
constant_model.save(os.path.join(models_folder_path, "model_constant.pkl"))


"""
Train the random uniform model.
"""


# Initialize the model
random_uniform_model = PMFModel(hyperparams, logger=logger)

# Fit the model
random_uniform_model.fit(
    A,
    init_params=InitParams(
        xy_strategy=InitXYrandomUniform(rng=np.random.default_rng(43))
    ),
    cavi_params=CaviParams(
        max_iterations=max_iterations,
        eval_interval=eval_interval,
        elbo_tol=elbo_tolerance,
    ),
    compute_prediction_params=compute_prediction_params,
)

# Save the model
random_uniform_model.save(os.path.join(models_folder_path, "model_random_uniform.pkl"))


"""
Train the random beta model.
"""


# Initialize the model
random_beta_model = PMFModel(hyperparams, logger=logger)

# Fit the model
random_beta_model.fit(
    A,
    init_params=InitParams(xy_strategy=InitXYrandomBeta(rng=np.random.default_rng(43))),
    cavi_params=CaviParams(
        max_iterations=max_iterations,
        eval_interval=eval_interval,
        elbo_tol=elbo_tolerance,
    ),
    compute_prediction_params=compute_prediction_params,
)

# Save the model
random_beta_model.save(os.path.join(models_folder_path, "model_random_beta.pkl"))


"""
Train the random gamma model.
"""

# Initialize the model
random_gamma_model = PMFModel(hyperparams, logger=logger)

# Fit the model
random_gamma_model.fit(
    A,
    init_params=InitParams(
        xy_strategy=InitXYrandomGamma(rng=np.random.default_rng(43))
    ),
    cavi_params=CaviParams(
        max_iterations=max_iterations,
        eval_interval=eval_interval,
        elbo_tol=elbo_tolerance,
    ),
    compute_prediction_params=compute_prediction_params,
)

# Save the model
random_gamma_model.save(os.path.join(models_folder_path, "model_random_gamma.pkl"))
