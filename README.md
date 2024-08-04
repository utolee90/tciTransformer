# TCiTransformer

The repo is about TCiTransformer, an improved model from [iTransformer: Inverted Transformers Are Effective for Time Series Forecasting](https://arxiv.org/abs/2310.06625).

# Basic Info
Please see [iTrnasformer Main Link](https://github.com/thuml/iTransformer)

## Overall Architecture

iTransformer regards **independent time series as variate tokens** to **capture multivariate correlations by attention** and **utilize layernorm and feed-forward networks to learn series representations**.

<p align="center">
<img src="./figures/architecture.png" alt="" align=center />
</p>

The pseudo-code of iTransformer is as simple as the following:

<p align="center">
<img src="./figures/algorithm.png" alt="" align=center />
</p>

## Usage 

1. Install Pytorch and necessary dependencies.

```
pip install -r requirements.txt
```

1. The datasets can be obtained from [Google Drive](https://drive.google.com/drive/folders/15Wj4pGPCU0IkBExQGXNXI-13HhpC5_nC?usp=sharing).

2. Train and evaluate the model. We provide all the above tasks under the folder ./scripts/. You can reproduce the results as the following examples:

```
# Multivariate forecasting with iTransformer
bash ./scripts/multivariate_forecasting/Traffic/iTransformer.sh

# Compare the performance of Transformer and iTransformer
bash ./scripts/boost_performance/Weather/iTransformer.sh

# Train the model with partial variates, and generalize on the unseen variates
bash ./scripts/variate_generalization/ECL/iTransformer.sh

# Test the performance on the enlarged lookback window
bash ./scripts/increasing_lookback/Traffic/iTransformer.sh

# Utilize FlashAttention for acceleration
bash ./scripts/efficient_attentions/iFlashTransformer.sh
```

### About run.py
The hyperparameter settings are stated in run.py.
* Every variable is defined like that:
```
  parser.add_argument('--model_id', type=str, required=True, default='test', help='model id')
```

To use the argument you shoud type like this:
```
  python run.py --model_id Sample_model_name ...
```
* Options of parameter
  * type : type of given variable (str, int, float, etc.)
  * required : If the option is set 'true', you should define the value in order to run the script.
  * default: the default value of
  * help : the introduction of the given variable
  * action : If the option is set, then certain action will be done. For example, 'store_true' means the parameter is set to be 'true' if user sets the parameter.

## About Source

### Overview
* `/checkpoints` - Saving last model parameter
* `/data_provider` - Defining data structure. 
* `/dataset` - Initially not given and you should download the dataset from [Google Drive](https://drive.google.com/file/d/1l51QsKvQPcqILT3DwfjCgx8Dsg2rpjot/view?usp=drive_link) (추가 데이터가 있으므로 추가 변경 예정)
* `/experiments` - Sources related to experiments
* `/figures` - Store figures used for the paper
* `/layers` - About embedded layers of model
* `/model` - Define models for time series forecasting
* `/results` - Store predicted data and actual data when tested
* `/scripts` - Scripts performing time series forecasting using model
* `/test_results` - Store graphs of targetted variable comparing prediction and real value.

## Testing 

### Adding more metrics

To verify the accuracy of our model, we use other metrics such as signed MAE and mean value of correlation of targetted variables. 
* Signed MAE - Mean difference of real value and predicted value. It checks whether predicted value tend to be underestimated(SMAE>0) or overestimated(SMAE<0). The result is better when it is close to zero.
* Mean correlation of last variable between (predicted and actual value) - Checks the tendency of coincidence between estimated value and real value. The result is ideal if it is close to 1.

## General Performance Boosting on Transformers

### Results about long-term forecasting

### Ablation Study
We will check the ablation study for TCiTransformer:
[] Find hyperparameters ideal for time series forecasting :
  * About TCN embedding layer - num of layers, 
  * About Attention
  * About Transformer layer - number of layers, inner dimension, 
  * Use augmented token instead of usual time series token
[x] Check the results depending on the length of input
[x] Check the results depending on the ratio of trained/tested data

## Citation

If you find this repo helpful, please cite our paper. 

```
@article{utolee90/tcitransformer,
  title={TCiTransformer:An elaborate way to via TCN },
  author={Lee, Yohan and Park, Sanghak and Park, Seyeon and Ji, Daeun and Jang, Beakcheol},
  journal={arXiv preprint arXiv:-},
  year={2024}
}
```

## Acknowledgement

We appreciate the following GitHub repos a lot for their valuable code and efforts.
- Reformer (https://github.com/lucidrains/reformer-pytorch)
- Informer (https://github.com/zhouhaoyi/Informer2020)
- FlashAttention (https://github.com/shreyansh26/FlashAttention-PyTorch)
- Autoformer (https://github.com/thuml/Autoformer)
- Stationary (https://github.com/thuml/Nonstationary_Transformers)
- Time-Series-Library (https://github.com/thuml/Time-Series-Library)
- lucidrains (https://github.com/lucidrains/iTransformer)

This work was also supported by NRF fund through the CCF-Ant Research Fund. 

## Contact

If you have any questions or want to use the code, feel free to contact:
* Lee, Yohan (utopiamath@yonsei.ac.kr)
