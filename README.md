# MinimumDominatingSets

## Data Generation

The data generation script can be found in the `data_generation/` directory. To execute the script, navigate to the directory and run

    bash -i run_datagen

This will also take care of creating a Conda environment necessary for the data generation using the `data_generation/environment.yml` file. 

## Data

All data that has already been generated can be found in the `data/` directory.

## Graph Convolutional Network

To generate the Conda environment necessary to run the training script, first execute from the root directory

    conda env create --name [env_name] --file=environments.yml

and then activate the environment using 

    conda activate [env_name]

The `train.py` script can then be run to train and test the GCN. Note that the script will look for a `test/` directory containing testing graphs during the testing phase, which is not pre-provided. `FINAL_RUN/` contains the results of the original execution of the script, and `final-results.json` contain the corresponding results.

Similarly, `extend.py` can be used to test the GCN on larger graph sizes. Note that this script requires the GCN to have already been trained and will load the model from its checkpoints. The results of the original execution of the script are found in `final-extend-results.json`.

## Visualizations

The `analysis.ipynb` notebook provides code to visualize the results once the GCN has been trained. Note that the `plot_num` can be changed in each cell to visualize different sets of results.

## Questions

Any questions can be directed to abi.kothapalli@vanderbilt.edu.

## License

MIT License

## Citations

The GCN architecture and approach used here is adapted from the following citation. ([Paper](https://proceedings.neurips.cc/paper/2018/file/8d3bba7425e7c98c50f52ca1b52d3735-Paper.pdf)) ([Github](https://github.com/isl-org/NPHard/))

> Zhuwen Li, Qifeng Chen and Vladlen Koltun. Combinatorial Optimization with Graph Convolutional Networks and Guided Tree Search. In NIPS 2018.

