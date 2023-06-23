# IFL: Mitigating Spurious Correlations for Self-supervised Recommendation
This is the pytorch implementation of our paper
> Mitigating Spurious Correlations for Self-supervised Recommendation
>
> Xinyu Lin, Yiyan Xu, Wenjie Wang, Yang Zhang, Fuli Feng

## Environment
- Anaconda 3
- python 3.7.15
- pytorch 1.7.0
- numpy 1.21.5

## Usage
### Data
The experimental data of XING are in './data' folder.

### Training
```
python main.py --model=$1 --dataset=$2 --lr=$3 --hidden_factor=$4 --batch_size=$5 --layers_u=$6 --layers_i=$7 --temp=$8 --alpha=$9 --beta=$10 --env=$11 --regs=$12 --regs_mask=$13 --dropout=$14 --batch_norm=$15 --log_name=$16 --gpu=$17
```
or use run.sh
```
sh run.sh model_name dataset lr hidden_factor batch_size DNN_layers_user DNN_layers_item temperature alpha beta n_env reg reg_mask dropout batchNorm log_name gpu_id
```
- The log file will be in the './code/log/' folder. 
- The explanation of hyper-parameters can be found in './code/main.py'. 
- The default hyper-parameter settings are detailed in './code/hyper-parameters.txt'.

### Example

1. Train IFL on XING:
```
cd ./code
sh run.sh IFL XING 0.01 64 1024 [] [] 0.7 0.7 0.001 2 0.0001 0.0001 [0,0] 0 log 0
```