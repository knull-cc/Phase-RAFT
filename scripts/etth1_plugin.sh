extra_args="$@"

root_path_name=./dataset/ETT-small
data_path_name=ETTh1.csv
model_id_name=ETTh1
data_name=ETTh1

seq_len=336
pred_len=96
random_seed=2024

python -u run.py \
  --is_training 1 \
  --root_path $root_path_name \
  --data_path $data_path_name \
  --model_id $model_id_name'_'$seq_len'_'$pred_len \
  --model iTransformer \
  --data $data_name \
  --features M \
  --seq_len $seq_len \
  --pred_len $pred_len \
  --enc_in 7 \
  --train_epochs 30 \
  --patience 5 \
  --lradj none \
  --dropout 0.1 \
  --itr 1 --batch_size 32 --learning_rate 0.0001 --random_seed $random_seed \
  $extra_args

python -u run.py \
  --is_training 1 \
  --root_path $root_path_name \
  --data_path $data_path_name \
  --model_id $model_id_name'_'$seq_len'_'$pred_len \
  --model PIBR \
  --pibr_host iTransformer \
  --pibr_fusion phase_only \
  --pibr_projector identity \
  --data $data_name \
  --features M \
  --seq_len $seq_len \
  --pred_len $pred_len \
  --enc_in 7 \
  --train_epochs 30 \
  --patience 5 \
  --lradj none \
  --dropout 0.1 \
  --itr 1 --batch_size 32 --learning_rate 0.0001 --random_seed $random_seed \
  $extra_args

python -u run.py \
  --is_training 1 \
  --root_path $root_path_name \
  --data_path $data_path_name \
  --model_id $model_id_name'_'$seq_len'_'$pred_len \
  --model PIBR \
  --pibr_host iTransformer \
  --pibr_fusion fixed_avg \
  --pibr_projector identity \
  --data $data_name \
  --features M \
  --seq_len $seq_len \
  --pred_len $pred_len \
  --enc_in 7 \
  --train_epochs 30 \
  --patience 5 \
  --lradj none \
  --dropout 0.1 \
  --itr 1 --batch_size 32 --learning_rate 0.0001 --random_seed $random_seed \
  $extra_args
