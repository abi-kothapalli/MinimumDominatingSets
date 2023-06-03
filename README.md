# MinimumDominatingSets

## Data Generation

The data generation script can be found in the `data_generation/` directory. To execute the script, navigate to the directory and run

    bash -i run_datagen

This will also take care of creating a Conda environment necessary for the data generation using the `data_generation/environment.yml` file. 

## Data

All synthetic data that has already been generated can be found in the `data/` directory.

The labeled real-world datasets can be found in the `real-world-datasets/` directory. Each dataset's `.zip` file can be decompressed to obtain a `.json` file containing the graph instances with adjacency matrices and exact MDS solutions.

## Graph Convolutional Network

To generate the Conda environment necessary to run the training script, first execute from the root directory

    conda env create -n [env_name] -f environment.yml

and then activate the environment using 

    conda activate [env_name]

The `train.py` script can then be run to train and test the GCN. Note that the script will look for a `test/` directory containing testing graphs during the testing phase, which is not pre-provided.

Similarly, `extend.py` can be used to test the GCN on larger graph sizes. Note that this script requires the GCN to have already been trained and will load the model from its checkpoints. There is a `graph_model` field in this file that can be set to choose the random graph model that is used in the evaluation.

Finally, the `real_world_graphs.py` can be used to evaluate the GCN on the real-world datasets provided. Similar to the `extend.py` script, this script also requires the GCN to have already been trained. There is a `data_dir` field in this file that can be set to choose the appropriate dataset to run the experiment on.

## Questions

Any questions can be directed to abi.kothapalli@vanderbilt.edu.

## License

MIT License

## Citations

The GCN implementation is based on the following citation (MIT License). ([Paper](https://arxiv.org/abs/1609.02907)) ([GitHub](https://github.com/tkipf/gcn))

> Thomas N. Kipf and Max Welling. Semi-Supervised Classification with Graph Convolutional Networks. In ICLR 2017.

The specific GCN architecture and approach used here is adapted from the following citation (MIT License). ([Paper](https://arxiv.org/abs/1810.10659)) ([GitHub](https://github.com/isl-org/NPHard/))

> Zhuwen Li, Qifeng Chen and Vladlen Koltun. Combinatorial Optimization with Graph Convolutional Networks and Guided Tree Search. In NeurIPS 2018.

