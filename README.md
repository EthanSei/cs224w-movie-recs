# cs224w-movie-recs
Repo for the CS224w Movie Recommender Project

# Setup & Installation
1. Download the git repo via the command: `git clone https://github.com/EthanSei/cs224w-movie-recs.git`
2. In the project's root directory, run `make setup`. This should install dependencies and modularize the source code.
3. Run `make load` or `make force_load` to download the dataset and build the initial graph. Add the `env=prod` suffix to run against the production dataset. Note that this dataset is much larger. 