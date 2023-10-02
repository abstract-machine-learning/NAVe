# Robustness Verification of *k*-Nearest Neighbors

*k*NAVe (***k*****N**N **A**bstract **Ve**rifier) is an abstract interpretation-based tool for proving robustness and stability properties of *k*NN classifiers.

<p align="center">
	<img src="https://github.com/nicolofassina/kNAVe/blob/main/assets/abstract-classification.png" width="300" height="300">
</p>

Given a training set *D* and a test set *T*, together with a perturbation *P*, *k*NAVe symbolically computes an over-approximation of *P(x)*, the region of (possibly infinite) vectors that corresponds to feature variations of *x* in *T*, and runs a sound abstract version of the *k*NN on it, returning a superset of the labels associated with vectors in *P(x)*. Clearly, when *k*NAVe only returns one label, *k*NN is provably robust and stable for such a perturbation.

## Requirements
- Python3

## Installation
To install *k*NAVe you need to clone or download this repository and run the commands:
```[bash]
cd src
pip install ./
```
This will install the following dependencies:
- joblib
- nptyping
- numpy
- pandas
- pick
- python-dateutil
- pytz
- scikit-learn
- scipy
- six
- threadpoolctl
- tqdm

## Usage
To run *k*NAVe:
```[bash]
cd src
python3 nave.py <config_file.ini>
```
where `config_file.ini` is located in the `config` folder, or alternatively:
```[bash]
cd src
python3 nave.py <config_file.ini> log
```
to obtain also a log file in the `logs` folder. For more information on usage and configuration file, run `help.py` without arguments.

Note: You can use `...` to match multiple files. For example:
```[bash]
python3 nave.py ...
python3 nave.py str...
```
match, respectively, all files and files starting with `str` in the `config` folder.

## Results
Results are saved in 3 files:
- **details.csv**: contains classifications for all processed feature vectors
- **robustness.csv**: contains robustness results
- **stability.csv**: contains stability results

Their location depends on the specifications given in the configuration file.
