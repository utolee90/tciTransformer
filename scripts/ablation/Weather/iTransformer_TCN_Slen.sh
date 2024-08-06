export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6

model_name=iTransformer_TCN
comp_model_name=iTransformer
data_file=weather_re.csv
data_file_1=weather_re_1.csv # former half
data_file_2=weather_re_2.csv # later half
gpu=5
batch_size=32

# T=96
python -u run.py \
  --is_training 1 \
  --root_path ./dataset/weather/ \
  --data_path $data_file \
  --model_id weather_96-192_0805 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --pred_len 192 \
  --e_layers 3 \
  --enc_in 21 \
  --dec_in 21 \
  --c_out 21 \
  --des 'Exp' \
  --d_model 256\
  --d_ff 256\
  --itr 1 \
  --dropout 0.1\
  --tcn_dropout 0.1\
  --gpu $gpu\
  --batch_size $batch_size \

python -u run.py \
  --is_training 1 \
  --root_path ./dataset/weather/ \
  --data_path $data_file \
  --model_id weather_96-192_0805_idff \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --pred_len 192 \
  --e_layers 3 \
  --enc_in 21 \
  --dec_in 21 \
  --c_out 21 \
  --des 'Exp' \
  --d_model 256\
  --d_ff 1024\
  --itr 1 \
  --dropout 0.1\
  --tcn_dropout 0.1\
  --gpu $gpu\
  --batch_size $batch_size \

  python -u run.py \
  --is_training 1 \
  --root_path ./dataset/weather/ \
  --data_path $data_file \
  --model_id weather_96-336_0805 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --pred_len 336 \
  --e_layers 3 \
  --enc_in 21 \
  --dec_in 21 \
  --c_out 21 \
  --des 'Exp' \
  --d_model 256\
  --d_ff 256\
  --itr 1 \
  --dropout 0.1\
  --tcn_dropout 0.1\
  --gpu $gpu\
  --batch_size $batch_size 

python -u run.py \
  --is_training 1 \
  --root_path ./dataset/weather/ \
  --data_path $data_file \
  --model_id weather_96-720_0805 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --pred_len 720 \
  --e_layers 3 \
  --enc_in 21 \
  --dec_in 21 \
  --c_out 21 \
  --des 'Exp' \
  --d_model 256\
  --d_ff 256\
  --itr 1 \
  --dropout 0.1\
  --tcn_dropout 0.1\
  --gpu $gpu\
  --batch_size $batch_size 

# T=336

python -u run.py \
  --is_training 1 \
  --root_path ./dataset/weather/ \
  --data_path $data_file \
  --model_id weather_336_192-0805 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 336 \
  --pred_len 192 \
  --e_layers 3 \
  --enc_in 21 \
  --dec_in 21 \
  --c_out 21 \
  --des 'Exp' \
  --d_model 1024\
  --d_ff 1024\
  --itr 1 \
  --dropout 0.1\
  --tcn_dropout 0.1\
  --gpu $gpu\
  --batch_size $batch_size 

python -u run.py \
  --is_training 1 \
  --root_path ./dataset/weather/ \
  --data_path $data_file \
  --model_id weather_336_336-0805 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 336 \
  --pred_len 336 \
  --e_layers 3 \
  --enc_in 21 \
  --dec_in 21 \
  --c_out 21 \
  --des 'Exp' \
  --d_model 1024\
  --d_ff 1024\
  --itr 1 \
  --dropout 0.1\
  --tcn_dropout 0.1\
  --gpu $gpu\
  --batch_size $batch_size 

python -u run.py \
  --is_training 1 \
  --root_path ./dataset/weather/ \
  --data_path $data_file \
  --model_id weather_336_720-0805 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 336 \
  --pred_len 720 \
  --e_layers 3 \
  --enc_in 21 \
  --dec_in 21 \
  --c_out 21 \
  --des 'Exp' \
  --d_model 1024\
  --d_ff 1024\
  --itr 1 \
  --dropout 0.1\
  --tcn_dropout 0.1\
  --gpu $gpu\
  --batch_size $batch_size 

python -u run.py \
  --is_training 1 \
  --root_path ./dataset/weather/ \
  --data_path $data_file \
  --model_id weather_336_1500-0805 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 336 \
  --pred_len 1500 \
  --e_layers 3 \
  --enc_in 21 \
  --dec_in 21 \
  --c_out 21 \
  --des 'Exp' \
  --d_model 1024\
  --d_ff 1024\
  --itr 1 \
  --dropout 0.1\
  --tcn_dropout 0.1\
  --gpu $gpu\
  --batch_size $batch_size 


python -u run.py \
  --is_training 1 \
  --root_path ./dataset/weather/ \
  --data_path $data_file \
  --model_id weather_336-336_idff \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 336 \
  --pred_len 336 \
  --e_layers 3 \
  --enc_in 21 \
  --dec_in 21 \
  --c_out 21 \
  --des 'Exp' \
  --d_model 1024\
  --d_ff 4096\
  --itr 1 \
  --dropout 0.1\
  --tcn_dropout 0.1\
  --gpu $gpu\
  --batch_size $batch_size 


# iTransformer
# T=96
python -u run.py \
  --is_training 1 \
  --root_path ./dataset/weather/ \
  --data_path $data_file \
  --model_id weather_96-192_0805 \
  --model $comp_model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --pred_len 192 \
  --e_layers 3 \
  --enc_in 21 \
  --dec_in 21 \
  --c_out 21 \
  --des 'Exp' \
  --d_model 256\
  --d_ff 256\
  --itr 1 \
  --dropout 0.1\
  --tcn_dropout 0.1\
  --gpu $gpu\
  --batch_size $batch_size \

python -u run.py \
  --is_training 1 \
  --root_path ./dataset/weather/ \
  --data_path $data_file \
  --model_id weather_96-192_0805_idff \
  --model $comp_model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --pred_len 192 \
  --e_layers 3 \
  --enc_in 21 \
  --dec_in 21 \
  --c_out 21 \
  --des 'Exp' \
  --d_model 256\
  --d_ff 1024\
  --itr 1 \
  --dropout 0.1\
  --tcn_dropout 0.1\
  --gpu $gpu\
  --batch_size $batch_size \

  python -u run.py \
  --is_training 1 \
  --root_path ./dataset/weather/ \
  --data_path $data_file \
  --model_id weather_96-336_0805 \
  --model $comp_model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --pred_len 336 \
  --e_layers 3 \
  --enc_in 21 \
  --dec_in 21 \
  --c_out 21 \
  --des 'Exp' \
  --d_model 256\
  --d_ff 256\
  --itr 1 \
  --dropout 0.1\
  --tcn_dropout 0.1\
  --gpu $gpu\
  --batch_size $batch_size 

python -u run.py \
  --is_training 1 \
  --root_path ./dataset/weather/ \
  --data_path $data_file \
  --model_id weather_96-720_0805 \
  --model $comp_model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --pred_len 720 \
  --e_layers 3 \
  --enc_in 21 \
  --dec_in 21 \
  --c_out 21 \
  --des 'Exp' \
  --d_model 256\
  --d_ff 256\
  --itr 1 \
  --dropout 0.1\
  --tcn_dropout 0.1\
  --gpu $gpu\
  --batch_size $batch_size 

# T=336

python -u run.py \
  --is_training 1 \
  --root_path ./dataset/weather/ \
  --data_path $data_file \
  --model_id weather_336_192-0805 \
  --model $comp_model_name \
  --data custom \
  --features M \
  --seq_len 336 \
  --pred_len 192 \
  --e_layers 3 \
  --enc_in 21 \
  --dec_in 21 \
  --c_out 21 \
  --des 'Exp' \
  --d_model 1024\
  --d_ff 1024\
  --itr 1 \
  --dropout 0.1\
  --tcn_dropout 0.1\
  --gpu $gpu\
  --batch_size $batch_size 

python -u run.py \
  --is_training 1 \
  --root_path ./dataset/weather/ \
  --data_path $data_file \
  --model_id weather_336_336-0805 \
  --model $comp_model_name \
  --data custom \
  --features M \
  --seq_len 336 \
  --pred_len 336 \
  --e_layers 3 \
  --enc_in 21 \
  --dec_in 21 \
  --c_out 21 \
  --des 'Exp' \
  --d_model 1024\
  --d_ff 1024\
  --itr 1 \
  --dropout 0.1\
  --tcn_dropout 0.1\
  --gpu $gpu\
  --batch_size $batch_size 

python -u run.py \
  --is_training 1 \
  --root_path ./dataset/weather/ \
  --data_path $data_file \
  --model_id weather_336_720-0805 \
  --model $comp_model_name \
  --data custom \
  --features M \
  --seq_len 336 \
  --pred_len 720 \
  --e_layers 3 \
  --enc_in 21 \
  --dec_in 21 \
  --c_out 21 \
  --des 'Exp' \
  --d_model 1024\
  --d_ff 1024\
  --itr 1 \
  --dropout 0.1\
  --tcn_dropout 0.1\
  --gpu $gpu\
  --batch_size $batch_size 

python -u run.py \
  --is_training 1 \
  --root_path ./dataset/weather/ \
  --data_path $data_file \
  --model_id weather_336_1500-0805 \
  --model $comp_model_name \
  --data custom \
  --features M \
  --seq_len 336 \
  --pred_len 1500 \
  --e_layers 3 \
  --enc_in 21 \
  --dec_in 21 \
  --c_out 21 \
  --des 'Exp' \
  --d_model 1024\
  --d_ff 1024\
  --itr 1 \
  --dropout 0.1\
  --tcn_dropout 0.1\
  --gpu $gpu\
  --batch_size $batch_size 


python -u run.py \
  --is_training 1 \
  --root_path ./dataset/weather/ \
  --data_path $data_file \
  --model_id weather_336-336_idff \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 336 \
  --pred_len 336 \
  --e_layers 3 \
  --enc_in 21 \
  --dec_in 21 \
  --c_out 21 \
  --des 'Exp' \
  --d_model 1024\
  --d_ff 4096\
  --itr 1 \
  --dropout 0.1\
  --tcn_dropout 0.1\
  --gpu $gpu\
  --batch_size $batch_size 
