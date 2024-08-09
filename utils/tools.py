import os

import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.seasonal import seasonal_decompose
from .metrics import *
import time

plt.switch_backend('agg')


def adjust_learning_rate(optimizer, epoch, args):
    # lr = args.learning_rate * (0.2 ** (epoch // 2))
    if args.lradj == 'type1':
        lr_adjust = {epoch: args.learning_rate * (0.5 ** ((epoch - 1) // 1))}
    elif args.lradj == 'type2':
        lr_adjust = {
            2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6,
            10: 5e-7, 15: 1e-7, 20: 5e-8
        }
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        print('Updating learning rate to {}'.format(lr))


class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model, path):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), path + '/' + 'checkpoint.pth')
        self.val_loss_min = val_loss


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class StandardScaler():
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean


def visual(true, preds=None, name='./pic/test.pdf'):
    """
    Results visualization
    """
    plt.figure()
    plt.plot(true, label='GroundTruth', linewidth=2)
    if preds is not None:
        plt.plot(preds, label='Prediction', linewidth=2)
    plt.legend()
    plt.savefig(name, bbox_inches='tight')


def adjustment(gt, pred):
    anomaly_state = False
    for i in range(len(gt)):
        if gt[i] == 1 and pred[i] == 1 and not anomaly_state:
            anomaly_state = True
            for j in range(i, 0, -1):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
            for j in range(i, len(gt)):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
        elif gt[i] == 0:
            anomaly_state = False
        if anomaly_state:
            pred[i] = 1
    return gt, pred


def cal_accuracy(y_pred, y_true):
    return np.mean(y_pred == y_true)


# 시계열 분해 클래스
class STDecomp:
    def __init__(self, arr_input):
        # arr : ndArray - 1차원 또는 2차원. 2차원의 경우 행이 시간, 열이 변수
        self.arr = arr_input
        self.ndim = arr_input.ndim # 차원
        self.shape = arr_input.shape # 모양
        self.maximum = np.max(arr_input)
        self.mininum = np.min(arr_input)
        self.__cut = 1000000
        
        if self.ndim == 2:
            self.max_var = np.array([np.max(arr_input[:,k]) for k in range(self.shape[1])])
            self.min_var = np.array([np.min(arr_input[:,k]) for k in range(self.shape[1])])
        

    # 주변 변수
    @staticmethod    
    def find_local_maxima(arr, n=4):
        def is_local_maxima(index, data, range_n):
            # 범위 설정 (경계 조건 처리)
            start = max(index - range_n, 0)
            end = min(index + range_n + 1, len(data))
            max_value = data[index]
            # 주어진 범위 내에서 최대값인지 확인
            for i in range(start, end):
                if data[i] > max_value:
                    return False
            return True

        if arr.ndim == 1:  # 1차원 배열 처리
            result = {}
            for i in range(len(arr)):
                if is_local_maxima(i, arr, n):
                    result[i] = arr[i]
            return result
        elif arr.ndim == 2:  # 2차원 배열 처리
            results = []
            for row in arr:
                result = {}
                for i in range(len(row)):
                    if is_local_maxima(i, row, n):
                        result[i] = row[i]
                results.append(result)
            return results
        else:
            raise ValueError("Input must be a 1D or 2D array")
    
    # ACF 구하기
    def get_autocorrelation(self, cut=1000000):
        def autocorr_1d(data, cut=cut):
            n = min(len(data), cut)
            result = []
            for lag in range(1, n):  # 각 k에 대하여
                c = np.corrcoef(data[:-lag], data[lag:])[0, 1]  # 상관계수 계산
                result.append(c if not np.isnan(c) else 0)  # NaN 처리
            return result

        if self.arr.ndim == 1:  # 1차원 배열 처리
            return autocorr_1d(self.arr, cut)
        elif self.arr.ndim == 2:  # 2차원 배열 처리
            results = []
            for row in self.arr.T:
                results.append(autocorr_1d(row, cut))
            return np.array(results).T
    
    # cut 설정
    def set_cut(self, cut):
        self.__cut = cut
    
    # 주기 패턴 구하기
    def get_period_seq(self):
        autocorr = self.get_autocorrelation(self.__cut)
        if self.ndim == 1:
            autocorr_max = STDecomp.find_local_maxima(autocorr)
        elif self.ndim == 2:
            autocorr_max = STDecomp.find_local_maxima(autocorr.T)

        res = []
        if self.ndim == 1:
            for key, val in autocorr_max.items():
                if val >= 0.5:
                    res.append(key)
        elif self.ndim == 2:
            # 0.3 이상만 카운트
            res_count = {}
            for dic in autocorr_max:
                for key, val in dic.items():
                    if val >= 0.3:
                        res_count[key] = res_count[key] +1 if res_count.get(key) != None else 0

            # 좌우 합해서 1/3 이상인 데이터만 카운트
            for r in range(min(self.__cut, len(self.arr))):
                rmin = max(1, r-1)
                rmax = min(self.__cut-1, len(self.arr)-1, r+1)
                rsum = sum([res_count.get(s, 0) for s in range(rmin, rmax+1)])
                if rsum >= int(1/3 * self.shape[1]):
                    res.append(r)

            # 숫자 차이가 2 이하면 지우기
            res2 = []
            for i, u in enumerate(res):
                if i == 0 or u - res[i-1] >2:
                    res2.append(u)
            res = res2
            
        return res
    
    # 주기 구하기
    def get_period(self):
        period_list = np.array(self.get_period_seq())
        xfit = np.array(range(len(period_list))).reshape(-1, 1)
        linmodel = LinearRegression()
        linmodel.fit(xfit, period_list)

        return int(np.round(linmodel.coef_[0]))

    # 시계열 분해하기
    def get_seasonal_trend_decomposition(self):
        period = self.get_period()
        if self.ndim == 1:
            result = seasonal_decompose(self.arr, period=period, model='addictive')
            return result.seasonal, self.arr - result.seasonal
        elif self.ndim == 2:
            result = []
            for r in range(self.shape[1]):
                res_part = seasonal_decompose(self.arr[:,r], period=period, model='addictive')
                result.append((res_part.seasonal, self.arr[:,r] - res_part.seasonal))
            # 배열로 변경 [x1, x2, ... xn] -> [s1, t1, s2, t2, ... sn, tn] -> []
            res_2 = []
            for par in result:
                res_2.append(par[0]) # seasonal
                res_2.append(par[1]) # trend
            res_3 = np.array(res_2).T # seasonal/trend/ -> transpose
            return res_3
    
    

# 필터링하기
class FilterSegment:
    # 크기 조절 , arr은 2차원
    def __init__(self, arr, seq_len, pred_len, period=0):
        self.arr = arr
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.period = period
        if arr.shape[0] < seq_len + pred_len:
            raise Exception("Invalid Input")
        self.pieces = self.get_pieces()
        self.pieces_shape = self.pieces.shape
        self.filtered_pieces = np.zeros((1, seq_len, arr.shape[1]))
        self.filtered_indices = []

    # 조각내기
    def get_pieces(self):
        last_elem = len(self.arr) - self.pred_len - self.seq_len + 1
        res = []
        for r in range(last_elem):
            res.append(self.arr[r:r+self.pred_len])
        return np.array(res)

    # 조각 추가
    def add_pieces(self, piece):
        pieces = self.pieces
        if pieces.shape[1] == piece.shape[0] and pieces.shape[2] == piece.shape[1]:
            np.append(pieces, piece, axis=0)
            self.pieces = pieces
        else:
            print("Dimension does not fit")
    
    # 주기 기준으로 조각 걸러내기
    def filter_elements(self):
        pieces = self.pieces
        period = self.period
        filtered_pieces = np.zeros((1, self.seq_len, self.arr.shape[1]))
        filtered_indices = [] # 재정의
        for r in range(self.pieces_shape[0]):
            if r == 0:
                filtered_pieces[0, :, :] = pieces[0]
                filtered_indices.append(0)
            elif r <= period // 2 or self.pieces_shape[0] -r <= period // 2:
                np.append(filtered_pieces, pieces[r], axis=0)
                filtered_indices.append(r)
            # 이외 인접한 경우 = 오차가 충분하면 넣고 아니면 끼우기
            elif r-1 in filtered_indices:
                total_list = [np.std(self.arr[:r-1, u]) for u in self.arr.shape[1]]
                rmse_list = RMSE(pieces[:r-1], np.array([pieces[r] for _ in range(r)]))
                if rmse_list >= np.mean(np.array(total_list)) * 0.5:
                    np.append(filtered_pieces, pieces[r], axis=0)
                    filtered_indices.append(r)
                else:
                    continue

            # 이외 - 6개 이상 스킵 - 무조건 끼우기
            elif len(set(filtered_indices).intersection({r-t for t in range(1, 6)})) == 0:
                np.append(filtered_pieces, pieces[r], axis=0)
                filtered_indices.append(r)
            else:
                total_list = [np.std(self.arr[:r-1, u]) for u in self.arr.shape[1]]
                rmse_list = RMSE(pieces[:r-1], np.array([pieces[r] for _ in range(r)]))
                if rmse_list >= np.mean(np.array(total_list)) * 0.25:
                    np.append(filtered_pieces, pieces[r], axis=0)
                    filtered_indices.append(r)
                else:
                    continue
        
        self.filtered_pieces = filtered_pieces
        self.filtered_indices= filtered_indices

        return filtered_pieces, filtered_indices

# pytorch기반 - 그나마 제일 빠름.        
class FilterSegmentTorch:
    def __init__(self, arr, seq_len, pred_len, period=0, device='cuda'):
        self.device = device
        self.arr = torch.tensor(arr, dtype=torch.float32).to(device)
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.period = period
        if self.arr.shape[0] < seq_len + pred_len:
            raise Exception("Invalid Input")
        self.pieces = self.get_pieces()

    def get_pieces(self):
        # 계산을 위해 시작 인덱스를 생성합니다.
        start_indices = torch.arange(len(self.arr) - self.pred_len - self.seq_len + 1, device=self.device)
        index_matrix = start_indices[:, None] + torch.arange(self.seq_len, device=self.device)
        return self.arr[index_matrix]

    def filter_elements(self):
        pieces = self.pieces
        period = self.period
        filtered_pieces = []
        filtered_indices = []
        std_devs = torch.std(self.arr, dim=0)
        
        for r in range(len(pieces)):
            if r%100 == 0:
                print(f"{r+1} step::", end=' ')
            if r == 0 or r <= period // 2 or len(pieces) - r <= period // 2:
                filtered_pieces.append(pieces[r])
                filtered_indices.append(r)
            elif r-1 in filtered_indices:
                piece = pieces[r]
                rmse = torch.sqrt(torch.mean((piece - pieces[:r])**2, dim=0))
                if torch.all(rmse >= std_devs * 0.5):
                    filtered_pieces.append(piece)
                    filtered_indices.append(r)
            elif len(set(filtered_indices).intersection(range(r-5, r))) == 0:
                filtered_pieces.append(pieces[r])
                filtered_indices.append(r)

        print()
        self.filtered_pieces = torch.stack(filtered_pieces)
        self.filtered_indices = filtered_indices
        return self.filtered_pieces, self.filtered_indices
    

# 필터링하기 - 다른 방식이지만 느려서 사용안함
class FilterSegment2:
    # 크기 조절 , arr은 2차원
    def __init__(self, arr, seq_len, pred_len, period=0):
        self.arr = arr
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.period = period
        if arr.shape[0] < seq_len + pred_len:
            raise Exception("Invalid Input")
        self.pieces = self.get_pieces()
        self.pieces_shape = self.pieces.shape
        self.filtered_pieces = np.zeros((1, seq_len, arr.shape[1]))
        self.filtered_indices = []

    # 조각내기
    def get_pieces(self):
        last_elem = len(self.arr) - self.pred_len - self.seq_len + 1
        res = []
        for r in range(last_elem):
            res.append(self.arr[r:r+self.pred_len])
        return np.array(res)

    # 조각 추가
    def add_pieces(self, piece):
        pieces = self.pieces
        if pieces.shape[1] == piece.shape[0] and pieces.shape[2] == piece.shape[1]:
            np.append(pieces, piece, axis=0)
            self.pieces = pieces
        else:
            print("Dimension does not fit")
    
    # 주기 기준으로 조각 걸러내기
    def filter_elements(self):
        pieces = self.pieces
        period = self.period
        filtered_pieces = np.zeros((1, self.seq_len, self.arr.shape[1]))
        filtered_indices = [] # 재정의
        start_time = time.time()
        print("START_TIME", start_time)
        for r in range(self.pieces_shape[0]):

            if r % 100 == 0:
                print(f'step {r+1} start') 
            if r == 0:
                filtered_pieces[0, :, :] = pieces[0]
                filtered_indices.append(0)
            # 앞/뒤 반주기 기준은 무조건 넣기
            elif r <= period // 2 or self.pieces_shape[0] -r <= period // 2:
                np.append(filtered_pieces, np.array([pieces[r]]), axis=0)
                filtered_indices.append(r)
            # 이외 - 6개 이상 스킵 - 무조건 끼우기
            elif len(set(filtered_indices).intersection({r-t for t in range(1, 6)})) == 0:
                np.append(filtered_pieces, np.array([pieces[r]]), axis=0)
                filtered_indices.append(r)
            # 바로 인접한 토큰은 스킵
            elif r-1 in filtered_indices:
                continue
                
            # 나머지 - 1주기 토큰 기준으로 비교
            else:
                passing = False
                if period > 4:
                    tokens_period = pieces[max(0,r-period):r]
                    last_token = pieces[r]
                    std_last = np.mean([np.std(last_token[:, u]) for u in range(pieces[r].shape[1])]) 
                    for i, token in enumerate(tokens_period):
                        # 매우 유사한 토큰이 있거나
                        if RMSE(token, last_token) < 0.1*std_last:
                            passing = True
                            break
                        # 입력/출력 상관계수 차이가 엄청 크면
                        elif REC_CORR(token, last_token) - REC_CORR(pieces[i+self.seq_len], pieces[r+self.seq_len])>0.5:
                            passing = True
                            break

                # 비주기 - 최근 seq_len 기준 correlation이 0 미만 토큰 기준
                else:
                    
                    tokens_recent = pieces[max(0, r-self.seq_len):r]
                    last_token = pieces[r]
                    std_last = np.mean([np.std(last_token[:, u]) for u in range(pieces[r].shape[1])])
                    for i, token in enumerate(self.seq_len):
                        # 매우 유사한 토큰이 있거나
                        if RMSE(token, last_token) < 0.1*std_last:
                            passing = True
                            break
                        # 입력/출력 상관계수 차이가 엄청 크면
                        elif REC_CORR(token, last_token) - REC_CORR(pieces[i+self.seq_len], pieces[r+self.seq_len])>0.5:
                            passing = True
                            break
                    if passing: continue
                    else:
                        np.append(filtered_pieces, np.array([pieces[r]]), axis=0)
                        filtered_indices.append(r)
            


                
        self.filtered_pieces = filtered_pieces
        self.filtered_indices= filtered_indices
        print("end time", time.time())

        return filtered_pieces, filtered_indices
            