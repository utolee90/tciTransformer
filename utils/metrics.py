import numpy as np
from sklearn.metrics import r2_score

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

def R2score(pred, true): # R2 score 값
    return r2_score(true, pred)

def SMAE(pred, true): # signed mae, 실제값 - 예측값
    return np.mean(true-pred)

# 상관계수 추측
def REC_CORR(pred, true, flag='mean'): 
    
    # 2차원 - 개별변수
    if pred.ndim == 2:
        if flag in  ['mean', 'average', 'me', 'avg']:
            return np.mean(np.array([np.corrcoef(pred[j,:], true[j,:])[0,1] for j in range(len(pred)) if not np.isnan(np.corrcoef(pred[j,:], true[j,:])[0,1])]))
        elif flag in ['median', 'med']:
            return np.median(np.array([np.corrcoef(pred[j,:], true[j,:])[0,1] for j in range(len(pred)) if not np.isnan(np.corrcoef(pred[j,:], true[j,:])[0,1])]))
        else:
            return 
    elif pred.ndim == 3:
        if flag == 'mean_total':
            return np.mean(np.array(
                    [
                        [np.corrcoef(pred[j,:,k], true[j,:,k])[0,1] for k in range(pred.shape[2]) if not np.isnan(np.corrcoef(pred[j,:,k], true[j,:,k])[0,1])] 
                        for j in range(len(pred))
                    ])
                )
        elif flag == 'median_total':
            return np.median(np.array(
                    [
                        [np.corrcoef(pred[j,:,k], true[j,:,k])[0,1] for k in range(pred.shape[2]) if not np.isnan(np.corrcoef(pred[j,:,k], true[j,:,k])[0,1])] 
                        for j in range(len(pred))
                    ])
                )
        elif flag in  ['mean', 'average', 'me', 'avg']:
            return np.mean([
                np.mean([np.corrcoef(pred[j,:,k], true[j,:,k])[0,1] for j in range(len(pred)) if not np.isnan(np.corrcoef(pred[j,:,k], true[j,:,k]))[0,1]])
                for k in range(pred.shape[2])
            ])
        elif flag in ['median', 'med']:
            return np.median([
                np.median([np.corrcoef(pred[j,:,k], true[j,:,k])[0,1] for j in range(len(pred)) if not np.isnan(np.corrcoef(pred[j,:,k], true[j,:,k]))[0,1]])
                for k in range(pred.shape[2])
            ])
        else:
            return 
 
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
