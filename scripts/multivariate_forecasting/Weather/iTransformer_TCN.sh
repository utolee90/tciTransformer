export CUDA_VISIBLE_DEVICES=0

model_name=iTransformer_TCN
data_file=weather_re.csv

# case for seq_len=96 & pred_len = 96
# inner_dim = 32, 64, 128, 256, 512,1024
# dropout = .05, .1, .2 (default 0.1)
# layer=1,2,3,4 (default 2)
# tcn_layer=2,3,4 (default 3)
# tcn_dropout=.05,.1,.2 (default 0.1)
# batch_size 8,16,32,64 (default 64)

python -u run.py \
  --is_training 1 \
  --root_path ./dataset/weather/ \
  --data_path $data_file \
  --model_id weather_96_d64_drop.05_l2 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --pred_len 96 \
  --e_layers 2 \
  --enc_in 21 \
  --dec_in 21 \
  --c_out 21 \
  --des 'Exp' \
  --d_model 64\
  --d_ff 64\
  --itr 1\
  --dropout 0.05


python -u run.py \
  --is_training 1 \
  --root_path ./dataset/weather/ \
  --data_path $data_file \
  --model_id weather_96_d128_drop.05_l2 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --pred_len 96 \
  --e_layers 2 \
  --enc_in 21 \
  --dec_in 21 \
  --c_out 21 \
  --des 'Exp' \
  --d_model 128\
  --d_ff 128\
  --itr 1 \
  --dropout 0.05


python -u run.py \
  --is_training 1 \
  --root_path ./dataset/weather/ \
  --data_path $data_file \
  --model_id weather_96_d128_drop.1_l2 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --pred_len 96 \
  --e_layers 2 \
  --enc_in 21 \
  --dec_in 21 \
  --c_out 21 \
  --des 'Exp' \
  --d_model 128\
  --d_ff 128\
  --itr 1 \
  --dropout 0.1

python -u run.py \
  --is_training 1 \
  --root_path ./dataset/weather/ \
  --data_path $data_file \
  --model_id weather_96_d256_drop.1_l2 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --pred_len 96 \
  --e_layers 2 \
  --enc_in 21 \
  --dec_in 21 \
  --c_out 21 \
  --des 'Exp' \
  --d_model 256\
  --d_ff 256\
  --itr 1 \
  --dropout 0.1

python -u run.py \
  --is_training 1 \
  --root_path ./dataset/weather/ \
  --data_path $data_file \
  --model_id weather_96_d256_drop.1_l3 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --pred_len 96 \
  --e_layers 3 \
  --enc_in 21 \
  --dec_in 21 \
  --c_out 21 \
  --des 'Exp' \
  --d_model 256\
  --d_ff 256\
  --itr 1 \
  --dropout 0.1

python -u run.py \
  --is_training 1 \
  --root_path ./dataset/weather/ \
  --data_path $data_file \
  --model_id weather_96_d128_drop.1_l3 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --pred_len 96 \
  --e_layers 3 \
  --enc_in 21 \
  --dec_in 21 \
  --c_out 21 \
  --des 'Exp' \
  --d_model 128\
  --d_ff 128\
  --itr 1 \
  --dropout 0.1

python -u run.py \
  --is_training 1 \
  --root_path ./dataset/weather/ \
  --data_path $data_file \
  --model_id weather_96_d128_drop.2_l3 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --pred_len 96 \
  --e_layers 3 \
  --enc_in 21 \
  --dec_in 21 \
  --c_out 21 \
  --des 'Exp' \
  --d_model 128\
  --d_ff 128\
  --itr 1 \
  --dropout 0.2\
  --tcn_dropout 0.2

python -u run.py \
  --is_training 1 \
  --root_path ./dataset/weather/ \
  --data_path $data_file \
  --model_id weather_96_d256_drop.2_l3 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --pred_len 96 \
  --e_layers 3 \
  --enc_in 21 \
  --dec_in 21 \
  --c_out 21 \
  --des 'Exp' \
  --d_model 256\
  --d_ff 256\
  --itr 1 \
  --dropout 0.2\
  --tcn_dropout 0.2


python -u run.py \
  --is_training 1 \
  --root_path ./dataset/weather/ \
  --data_path $data_file \
  --model_id weather_96_d256_drop.2_l4 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --pred_len 96 \
  --e_layers 3 \
  --enc_in 21 \
  --dec_in 21 \
  --c_out 21 \
  --des 'Exp' \
  --d_model 256\
  --d_ff 256\
  --itr 1 \
  --dropout 0.2\
  --tcn_dropout 0.2

python -u run.py \
  --is_training 1 \
  --root_path ./dataset/weather/ \
  --data_path $data_file \
  --model_id weather_96_d256_drop.1_l4 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --pred_len 96 \
  --e_layers 3 \
  --enc_in 21 \
  --dec_in 21 \
  --c_out 21 \
  --des 'Exp' \
  --d_model 256\
  --d_ff 256\
  --itr 1 \
  --dropout 0.1\
  --tcn_dropout 0.1