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
  --data_path $data_file_1 \
  --model_id weather_96_0804_former \
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
  --d_model 512\
  --d_ff 512\
  --itr 1 \
  --dropout 0.1\
  --learning_rate 0.00001\
  --n_heads 8\
  --gpu $gpu\
  --batch_size $batch_size 

python -u run.py \
  --is_training 1 \
  --root_path ./dataset/weather/ \
  --data_path $data_file_2 \
  --model_id weather_96_0804_latter \
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
  --d_model 512\
  --d_ff 512\
  --itr 1 \
  --dropout 0.1\
  --learning_rate 0.00001\
  --n_heads 8\
  --gpu $gpu\
  --batch_size $batch_size

  python -u run.py \
  --is_training 1 \
  --root_path ./dataset/weather/ \
  --data_path $data_file_2 \
  --model_id weather_96_0804_latter \
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
  --d_model 512\
  --d_ff 512\
  --itr 1 \
  --dropout 0.1\
  --learning_rate 0.00001\
  --n_heads 8\
  --gpu $gpu\
  --batch_size $batch_size


#python -u run.py \
#  --is_training 1 \
#  --root_path ./dataset/weather/ \
#  --data_path $data_file \
#  --model_id weather_96_0803_noaug \
#  --model $model_name \
#  --data custom \
#  --features M \
#  --seq_len 96 \
#  --pred_len 96 \
#  --e_layers 3 \
#  --enc_in 21 \
#  --dec_in 21 \
#  --c_out 21 \
#  --des 'Exp' \
#  --d_model 256\
#  --d_ff 256\
#  --itr 1 \
#  --dropout 0.1\
#  --tcn_dropout 0.1\
#  --gpu $gpu\
#  --batch_size $batch_size \
#  --augmented_token