import cProfile
import pstats
import numpy as np

from pmf.initial import InitialModel, simulate_initial_model_params
from pmf.hyperparams import ModelHyperParams
from pmf.model import PMFModel
from pmf.vi.cavi import CaviParams


def main():
    # Model hyperparameters
    hyperparams = ModelHyperParams(
        num_features=3,
        a_x=1.0,
        a_y=1.0,
        b_x=1.0,
        b_y=1.0,
        c_x=1.0,
        c_y=1.0,
        alpha_x=1.0,
        alpha_y=1.0,
        beta_x=1.0,
        beta_y=1.0,
    )
    # Data dimensions
    time, N1, N2 = 11, 5, 4

    # Set the random seed
    rng = np.random.default_rng(43)

    # Simulate the initial model parameters
    initial_model_params = simulate_initial_model_params(
        hyperparams, time, N1, N2, rng=rng
    )

    # Create the initial model
    initial_model = InitialModel(hyperparams=hyperparams, params=initial_model_params)

    # Simulate some data
    data = initial_model.simulate_data()

    # Initialize the model
    model = PMFModel(hyperparams)

    # Fit the model
    model.fit(data, cavi_params=CaviParams(max_iterations=30))


def profile_main():
    print_file = "output.txt"
    profile_file = "output.dat"

    # Profile the main function
    cProfile.run("main()", profile_file)

    # Extract the stats from the output file
    stats = pstats.Stats(profile_file)

    # Create a filtered list of stats
    filtered_stats = {}
    for func, stat in stats.stats.items():
        if "venv" not in func[0]:
            filtered_stats[func] = stat

    # Replace the stats with the filtered stats
    stats.stats = filtered_stats

    # Sort the stats
    stats = stats.sort_stats("cumulative")

    # Print the stats to a file
    with open(print_file, "w") as stream:
        # Redirect output to file stream
        stats.stream = stream
        # Print the stats
        stats = stats.print_stats()


if __name__ == "__main__":
    # main()
    profile_main()
