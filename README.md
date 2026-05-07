
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


# Meta-learning Experiment Figures

This directory mainly contains few-shot meta-learning experiment code, training logs,  experimental result figures and Experiment Setting

.

## 1\. Code Files

The main code related to experiments and plotting includes:

|File Name|Function|
|-|-|
|`F2BAfc.py`|Training script for the F2BA method on FC100.|
|`IFSBA2fc.py`|Training script for the IFSBA method on FC100.|
|`qNBOfc.py`|Training script for the qNBO method on FC100.|
|`qNBOmini.py`|Training script for the qNBO method on miniImageNet.|
|`pzobo.py`|Training script for the PZOBO method on FC100 and miniImageNet.|
|`plotmini.py`|Plotting script for miniImageNet results.|
|`plotmini_fc100.py`|Plotting script for FC100 results.|
|`meta_f2ba_medical.py`|F2BA experiment script for few-shot medical image tasks.|

## 2\. Log Files

The project also contains two log directories:

```text
metalogs\\
metalogs_fc100\\
```

where:

* `.txt` files save the log outputs during training and testing;
* `.pt` files save experimental result data and the corresponding model weights;
* `metalogs\\` mainly corresponds to miniImageNet experiments;
* `metalogs_fc100\\` mainly corresponds to FC100 experiments.

## 3\. Figure Files

These figures show the results of different meta-learning/bilevel optimization methods on the **miniImageNet** and **FC100** datasets, where the test accuracy changes with running time. In the figures, the horizontal axis represents running time, and the vertical axis represents test accuracy.

* Horizontal axis: `Running time (s)`, indicating the cumulative running time during training, measured in seconds.
* Vertical axis: `Test accuracy (%)`, indicating the classification accuracy on test tasks, measured as a percentage.
* Curves: indicate the trend of the average test accuracy of different algorithms at different running times.
* Shaded areas: indicate the confidence interval range of the experimental results.
* Compared methods include:

  * `PZOBO`
  * `qNBO`
  * `F2BA`
  * `IFSBA`

|File Name|Dataset|Included Methods|Description|
|-|-|-|-|
|`compare_shadow_raw_fc100.png`|FC100|PZOBO, qNBO, F2BA, IFSBA|Overall comparison figure on the FC100 dataset, showing the trend of test accuracy of the four methods as running time changes. The training time is 1400s.|
|`compare_shadow_raw_miniimagenet.png`|miniImageNet|PZOBO, qNBO, F2BA, IFSBA|Overall comparison figure on the miniImageNet dataset, showing the trend of test accuracy of the four methods as running time changes. The training time is 1200s.|
|`fc100_3_1450.png`|FC100|PZOBO, qNBO, F2BA|Three-method comparison figure on the FC100 dataset. The training time is 1450s.|
|`fc100_4_1450.png`|FC100|PZOBO, qNBO, F2BA, IFSBA|Four-method comparison figure on the FC100 dataset. The training time is 1450s.|
|`miniimagenet_3_1300.png`|miniImageNet|PZOBO, qNBO, F2BA|Three-method comparison figure on the miniImageNet dataset. The training time is 1300s.|
|`miniimagenet_4_1300.png`|miniImageNet|PZOBO, qNBO, F2BA, IFSBA|Four-method comparison figure on the miniImageNet dataset. The training time is 1300s.|

## 4\. Experiment Setting

The experiments are carried out on a server equipped with an Intel Xeon Platinum 8352V CPU @ 2.10GHz, featuring 16 vCPUs and 120GB of memory. The GPU used is an NVIDIA RTX 4090 with 24GB VRAM. The algorithms are implemented using PyTorch 2.5.1 and Python 3.12, with GPU acceleration supported by CUDA 12.4. The operating system is Ubuntu 22.04.