import matplotlib.pyplot as plt
import os
import pandas as pd

# 폴더 생성
folder_path = './expriments/data_visualization/'
if not os.path.exists(folder_path):
    os.makedirs(folder_path)


# 확장자명 확인
def get_file_extension(file_path):
    _, file_extension = os.path.splitext(file_path)
    return file_extension

# get data name
def get_name(file_path):
    data_name = os.path.splitext(file_path)[0].split('/')[-1]
    return data_name

# -------------------------------------------------------

# 파일 경로
path_list = ['./dataset/electricity/electricity.csv',
             './dataset/ETT-small/ETTh1.csv',
             './dataset/ETT-small/ETTh2.csv',
             './dataset/ETT-small/ETTm1.csv',
             './dataset/ETT-small/ETTm2.csv',
             './dataset/exchange_rate/exchange_rate.csv',
             './dataset/Solar/solar_AL.txt',
             './dataset/traffic/traffic.csv',
             './dataset/weather/weather.csv']

for file_path in path_list:

    # 파일 확장자, 데이터명 확인
    extension = get_file_extension(file_path)
    dataname = get_name(file_path)

    # 데이터 로드 (확장자에 따라 다르게 처리할 수 있음)
    if extension == '.txt':
        df = pd.read_table(file_path, sep=',')
        data =  df.iloc[:, -1]

        def dataset_visualization(name=f'{folder_path}{dataname}.png'):
            plt.figure()
            plt.plot(data.index, data, label='GroundTruth', linewidth=0.5)
            plt.legend()
            plt.savefig(name, bbox_inches='tight')

    elif extension == '.csv':
        df = pd.read_csv(file_path)
        data = df[['date','OT']]

        def dataset_visualization(name=f'{folder_path}{dataname}.png'):
            plt.figure()
            plt.plot(data.index, data['OT'], label='GroundTruth', linewidth=0.5)
            plt.legend()
            plt.savefig(name, bbox_inches='tight')

    else:
        raise ValueError("Unsupported file format")


    dataset_visualization()