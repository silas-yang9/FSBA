##  Experiment Setting

The experiments are carried out on a server equipped with an Intel Xeon Platinum 8352V CPU @ 2.10GHz, featuring 16 vCPUs and 120GB of memory. The GPU used is an NVIDIA RTX 4090 with 24GB VRAM. The algorithms are implemented using PyTorch 2.5.1 and Python 3.12, with GPU acceleration supported by CUDA 12.4. The operating system is Ubuntu 22.04.

##  Data Hyper-Cleaning (FSBA, LFSBA, IFSBA)

To reproduce our experiment results, first generate the data by running

```
python -u data_cleaning.py --pretrain 0
```

Then run
```
cd ./australian_25
python -u Run_Australian.py
```

Then run
```
cd ./australian_50
python -u Run_Australian.py
```

Then run
```
cd ./breast_25
python -u Run_Breast.py
```

Then run
```
cd ./breast_50
python -u Run_Breast.py
```

## Hyperparemeter Optimization
All results have been saved; only plotting is required.

```
python -u plot.py
```

##  Meta-learning Experiments

### Requirements

```bash
pip install learn2learn
```

### Running Experiments

FC100 dataset

```bash
python -u F2BAfc.py
python -u IFSBA2fc.py
python -u qNBOfc.py
```

miniImageNet dataset

```bash
python -u qNBOmini.py
python -u pzobo.py
```

Medical few-shot experiments

```bash
python -u meta_f2ba_medical.py
```

---

### Plotting Results

```bash
python -u plotmini.py
python -u plotmini_fc100.py
```
### Acknowledgement

The meta-learning experiments in this repository are partially built upon the implementation from:

https://github.com/sowmaster/esjacobians

