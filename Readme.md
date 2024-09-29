# RetinexMac

The implementation of "Efficient Retinex-Based Framework for Low-Light Image Enhancement without Additional Networks"

<img src="./figures/fig1.png" alt="fig1"/> 

## Prerequisites
```
pip install -r requirements.txt
````

## Quick Start

### Download the training and evaluation datasets

You can refer to the following links to download the dataset, 
[LOLv1](https://daooshee.github.io/BMVC2018website/ "LOLv1").
[LOLv2](https://github.com/flyywh/CVPR-2020-Semi-Low-Light "LOLv2"). 
[LSRW](https://github.com/JianghaiSCU/R2RNet "LSRW").
[DICM][LIME][MEF][NPE][VV](https://github.com/mingcv/Bread "Tesing dataset")

### Description

Our PSNR/SSIM/LPIPS/NIQE on different test sets:

<img src="./figures/Table1.png" alt="tab1"/> 
<img src="./figures/Table2.png" alt="tab2"/> 
<img src="./figures/Table3.png" alt="tab3"/> 

### Training

***You can train data using:***

```python train.py```
	
### Testing

**We provide a pre-trained model.**

***You can test paired data using:***

```python test.py```

<img src="./figures/fig2.png" alt="fig2"/> 
<img src="./figures/fig3.png" alt="fig3"/> 


