nohup python -u main.py --model=$1 --dataset=$2 --lr=$3 --hidden_factor=$4 --batch_size=$5 --layers_u=$6 --layers_i=$7 --temp=$8 --alpha=$9 --beta=$10 --env=$11 --regs=$12 --regs_mask=$13 --dropout=$14 --batch_norm=$15 --log_name=$16 --gpu=$17 >./log/$1_$2_$3lr_$4hidden_$5bs_$6u_$7i_$8temp_$9alpha_$10beta_$11env_$12regs_$13regsMask_$14drop_$15bn_$16.txt 2>&1 &

# example
# sh run.sh IFL XING 0.01 64 1024 [] [] 0.7 0.7 0.001 2 0.0001 0.0001 [0,0] 0 log 0