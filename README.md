# [GlasXC](https://github.com/pyschedelicsid/GlasXC) 

`GlasXC` is an implementation of paper `Breaking the Glass Ceiling for Embedding-BasedClassifiers for Large Output Spaces - NIPS'19` in Pytorch:

- **GlasXC**: is for solving the extreme classification using simple Deep Learning architecture + GLAS Regularizer


## Setup

### Prerequisites

- Python 2.7 or 3.5
- Requirements for the project are listed in [requirements.txt](requirements.txt). In addition to these, PyTorch 0.4.1 or higher is necessary. The requirements can be installed using pip:
   ```bash
   $ pip install -r requirements.txt 
   ```
   or using conda:
   ```bash
   $ conda install --file requirements.txt
     ```

### Installing `extreme_classification`

- Clone the repository.
  ```bash
  $ git clone https://github.com/pyschedelicsid/GlasXC
  $ cd GlasXC
  ```

- Install
  ```bash
  $ python setup.py install
  ```


## Using the scripts

### GlasXC

Use `train_GlasXC.py`. A description of the options available can be found using:

```bash
$ python train_GlasXC.py --help
```

This script trains (and optionally evaluates) evaluates a model on a given dataset using the GlasXC algorithm.

## Reported Results
To run GlasXC in the configuration used in the report, use:
```bash
$ ./train_GlasXC_with_args.sh
```

To run the baseline model, use:
```bash
$ python baseline.py
```

Links to downloading each dataset used can be found [here](http://manikvarma.org/downloads/XC/XMLRepository.html), and the project report can be found [here](report/report.pdf). The configuration files used (described below) for each dataset can be found [here](setups).

## Data Format
The input data must be in the [LIBSVM](https://www.csie.ntu.edu.tw/~cjlin/libsvm/) format. An example of such a dataset is the Bibtex dataset found [here](http://manikvarma.org/downloads/XC/XMLRepository.html).

The first row in the LIBSVM format specifies dataset size and input and output dimensions. This row must be removed, and this information must be provided through configuration files, as explained below.

## Configuration files

### Deep Neural Network Architecture Configurations
For using GlasXC through `GlasXC.py`, you need to have valid neural network configurations for the encoding of the inputs, labels in the latent space and the regressor in the YAML format. An example configuration file is:
```yaml
- name: Linear
  kwargs:
    in_features: 500
    out_features: 1152

- name: LeakyReLU
  kwargs:
    negative_slope: 0.2
    inplace: True

- name: Linear
  kwargs:
    in_features: 1152
    out_features: 1836

- name: Sigmoid
```
Please note that the `name` and `kwargs` attributes have to resemble the same names as those in PyTorch.

### Optimizer Configurations
Optimizer configurations are very similar to the neural network configurations. Here you have to retain the same naming as PyTorch for optimizer names and their parameters - for example: `lr` for learning rate. Below is a sample:
```yaml
name: SGD
args:
  lr: 0.01
  momentum: 0.9

```

### Dataset Configurations
In both the scripts, you are required to specify a data root (`data_root`), dataset information file (`dataset_info`). `data_root` corresponds to the folder containing the datasets. `dataset_info` requires a YAML file in the following format:
```yaml
train_filename:
train_opts:
  num_data_points:
  input_dims:
  output_dims:

test_filename:
test_opts:
  num_data_points:
  input_dims:
  output_dims:
```

If the test dataset doesn't exist, then please remove the fields `test_filename` and `test_opts`. An example for the Bibtex dataset would be:
```yaml
train_filename: bibtex_train.txt
train_opts:
  num_data_points: 4880
  input_dims: 1836
  output_dims: 159

test_filename: bibtex_test.txt
test_opts:
  num_data_points: 2515
  input_dims: 1836
  output_dims: 159
```

## License
This code is provided under the [MIT License](LICENSE).

---
Written By: Siddhant Katyan
