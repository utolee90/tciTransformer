export CUDA_VISIBLE_DEVICES=0

model_name=iTransformer_TCN
data_file=weather_re.csv
data_file_1=weather_re_1.csv # former half
data_file_2=weather_re_2.csv # later half
gpu=0
batch_size=32


python -u run.py \
  --is_training 1 \
  --root_path ./dataset/weather/ \
  --data_path $data_file \
  --model_id weather_96_0803_tr1_te8 \
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
  --tcn_dropout 0.1\
  --gpu $gpu\
  --batch_size $batch_size \
  --train_ratio 0.1 \
  --test_ratio 0.8
  
python -u run.py \
  --is_training 1 \
  --root_path ./dataset/weather/ \
  --data_path $data_file \
  --model_id weather_96_0803_tr2_te7 \
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
  --tcn_dropout 0.1\
  --gpu $gpu\
  --batch_size $batch_size \
  --train_ratio 0.2 \
  --test_ratio 0.7

python -u run.py \
  --is_training 1 \
  --root_path ./dataset/weather/ \
  --data_path $data_file \
  --model_id weather_96_0803_tr3_te6 \
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
  --tcn_dropout 0.1\
  --gpu $gpu\
  --batch_size $batch_size \
  --train_ratio 0.3 \
  --test_ratio 0.6

python -u run.py \
  --is_training 1 \
  --root_path ./dataset/weather/ \
  --data_path $data_file \
  --model_id weather_96_0803_tr4_te5 \
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
  --tcn_dropout 0.1\
  --gpu $gpu\
  --batch_size $batch_size \
  --train_ratio 0.4 \
  --test_ratio 0.5

python -u run.py \
  --is_training 1 \
  --root_path ./dataset/weather/ \
  --data_path $data_file \
  --model_id weather_96_0803_tr5_te4 \
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
  --tcn_dropout 0.1\
  --gpu $gpu\
  --batch_size $batch_size \
  --train_ratio 0.5 \
  --test_ratio 0.4

python -u run.py \
  --is_training 1 \
  --root_path ./dataset/weather/ \
  --data_path $data_file \
  --model_id weather_96_0803_tr6_te3 \
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
  --tcn_dropout 0.1\
  --gpu $gpu\
  --batch_size $batch_size \
  --train_ratio 0.6 \
  --test_ratio 0.3

python -u run.py \
  --is_training 1 \
  --root_path ./dataset/weather/ \
  --data_path $data_file \
  --model_id weather_96_0803_tr7_te2 \
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
  --tcn_dropout 0.1\
  --gpu $gpu\
  --batch_size $batch_size \
  --train_ratio 0.7 \
  --test_ratio 0.2

python -u run.py \
  --is_training 1 \
  --root_path ./dataset/weather/ \
  --data_path $data_file \
  --model_id weather_96_0803_tr8_te1 \
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
  --tcn_dropout 0.1\
  --gpu $gpu\
  --batch_size $batch_size \
  --train_ratio 0.8 \
  --test_ratio 0.1

python -u run.py \
  --is_training 1 \
  --root_path ./dataset/weather/ \
  --data_path $data_file \
  --model_id weather_48_0803 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 48 \
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
  --tcn_dropout 0.1\
  --gpu $gpu\
  --batch_size $batch_size 

python -u run.py \
  --is_training 1 \
  --root_path ./dataset/weather/ \
  --data_path $data_file \
  --model_id weather_192_0803 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 192 \
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
  --tcn_dropout 0.1\
  --gpu $gpu\
  --batch_size $batch_size 

python -u run.py \
  --is_training 1 \
  --root_path ./dataset/weather/ \
  --data_path $data_file \
  --model_id weather_336_0803 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 336 \
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
  --tcn_dropout 0.1\
  --gpu $gpu\
  --batch_size $batch_size 

python -u run.py \
  --is_training 1 \
  --root_path ./dataset/weather/ \
  --data_path $data_file \
  --model_id weather_720_0803 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 720 \
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
  --tcn_dropout 0.1\
  --gpu $gpu\
  --batch_size $batch_size 

python -u run.py \
  --is_training 1 \
  --root_path ./dataset/weather/ \
  --data_path $data_file \
  --model_id weather_1500_0803 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 1500 \
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
  --tcn_dropout 0.1\
  --gpu $gpu\
  --batch_size $batch_size 