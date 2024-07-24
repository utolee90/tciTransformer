import numpy as np


def RSE(pred, true):
    return np.sqrt(np.sum((true - pred) ** 2)) / np.sqrt(np.sum((true - true.mean()) ** 2))


def CORR(pred, true):
    u = ((true - true.mean(0)) * (pred - pred.mean(0))).sum(0)
    d = np.sqrt(((true - true.mean(0)) ** 2 * (pred - pred.mean(0)) ** 2).sum(0))
    return (u / d).mean(-1)


def MAE(pred, true):
    return np.mean(np.abs(pred - true))


def MSE(pred, true):
    return np.mean((pred - true) ** 2)


def RMSE(pred, true):
    return np.sqrt(MSE(pred, true))


def MAPE(pred, true):
    return np.mean(np.abs((pred - true) / true))


def MSPE(pred, true):
    return np.mean(np.square((pred - true) / true))

def SMAE(pred, true): # signed mae, 실제값 - 예측값
    return np.mean(true-pred)

def REC_CORR(pred, true): # 상관계수 3차원
    cf = np.zeros(pred.shape[0])
    for j in range(pred.shape[0]):
        pred_part = pred[j,:,-1] # last variable
        true_part = true[j,:,-1]
        cf[j] = np.corrcoef(pred_part, true_part)[0,1] # 상관계수
    
    return np.mean(cf) # 상관계수 평균
 
def RATIO_IRR(pred, true, coef=2): # 오차값 분석. 기본값 표준편차 2
    
    tot_size = np.size(pred)
    mae = MAE(pred, true)

    # Calculate absolute errors
    err = np.abs(pred - true)
    
    # Determine large errors (errors that are k times larger than MAE)
    large_errors = err > coef * mae
    
    # Calculate the ratio of large errors
    large_error_ratio = np.sum(large_errors) / np.size(true)
    
    return large_error_ratio


def metric(pred, true):
    mae = MAE(pred, true)
    mse = MSE(pred, true)
    rmse = RMSE(pred, true)
    mape = MAPE(pred, true)
    mspe = MSPE(pred, true)

    return mae, mse, rmse, mape, mspe
