# Demo PET simulations

## Installation

Using conda
```
conda env create -f environment.yaml
```

Using mamba (recommended)
```
mamba env create -f environment.yaml
```

## Installation of cupy to run on GPUs

Using conda
```
conda activate dia_pet
conda install -c conda-forge cupy
```

Using mamba (recommended)
```
mamba activate dia_pet
mamba install cupy
```

## Run

```
python mlem.py -h
```

## See also

[Documentation and examples of parallelproj](https://parallelproj.readthedocs.io/en/stable/)
