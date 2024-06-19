import sys
import os

# Get the parent directory of the current directory (subdir2)
parent_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir, os.pardir))
sys.path.append(parent_dir)

import pickle

from pmf import PMFModel, ModelHyperParams
from pmf.vi import CaviParams
from utils.data import convert_data
from utils.logger import Logger


"""
Set the hyperparameters for the model.
"""

a = 1.0
b = 1.0
c = 0.1
alpha = 1.0
beta = 1.0
num_features = 7

hyperparams = ModelHyperParams(
    num_features=num_features,
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


"""
Set the cavi parameters.
"""

max_iterations = 1000
eval_interval = 10
elbo_tolerance = 1e-5

compute_prediction_params = True


"""
Load the data.
"""

# Load the data from the pickle file
with open("huxley_2020.pkl", "rb") as stream:
    data = pickle.load(stream)

# Convert the data to the format expected by the model
data = convert_data(data)

# Set the number of training time steps
T = 10
data = data[:T]


"""
Train the model.
"""

# Initialize the model
model = PMFModel(hyperparams, logger=Logger("pmf_model", log_filename="pmf_model.log"))

# Fit the model
model.fit(
    data,
    cavi_params=CaviParams(
        max_iterations=max_iterations,
        eval_interval=eval_interval,
        elbo_tol=elbo_tolerance,
    ),
    compute_prediction_params=compute_prediction_params,
)

# Save the model
model.save("model.pkl")
