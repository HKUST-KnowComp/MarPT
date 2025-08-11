'''
This file grab result from data.py and perform MLE.
'''

import numpy as np
import math
import data as data_module
from scipy.optimize import minimize
from scipy.stats import norm
from scipy.stats import binom
import copy
import json
import warnings

original_data = data_module.data
current_data = original_data

# ignore warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)

values = [
    [ [40, 10, 68, 5], 
      [40, 10, 75, 5],
      [40, 10, 83, 5],
      [40, 10, 93, 5], 
      [40, 10, 106, 5], 
      [40, 10, 125, 5], 
      [40, 10, 150, 5], 
      [40, 10, 185, 5], 
      [40, 10, 220, 5], 
      [40, 10, 300, 5], 
      [40, 10, 400, 5], 
      [40, 10, 600, 5], 
      [40, 10, 1000, 5], 
      [40, 10, 1700, 5] ],

    [[40, 30, 54, 5], 
    [40, 30, 56, 5], 
    [40, 30, 58, 5], 
    [40, 30, 60, 5], 
    [40, 30, 62, 5], 
    [40, 30, 65, 5], 
    [40, 30, 68, 5], 
    [40, 30, 72, 5], 
    [40, 30, 77, 5], 
    [40, 30, 83, 5], 
    [40, 30, 90, 5], 
    [40, 30, 100, 5], 
    [40, 30, 110, 5], 
    [40, 30, 130, 5]],
    
    [ [25, -4, 30, -21], 
      [4, -4, 30, -21], 
      [1, -4, 30, -21], 
      [1, -4, 30, -16], 
      [1, -8, 30, -16], 
      [1, -8, 30, -14], 
      [1, -8, 30, -11] ]
]

probs = [
    [30, 70, 10, 90],
    [90, 10, 70, 30],
    [50, 50, 50, 50]
]

# value function
def v(x, alpha, lam):
    if x >= 0:
        return x**alpha
    else:
        return -lam * ((-x)**alpha)

# probability weighting function
def ori_w(p, gamma):
    if p <= 0.0:
        return 0.0
    elif p >= 1.0:
        return 1.0
    return math.exp(-(-math.log(p))**gamma)

def w(p, gamma):
    if abs(gamma-1.0) < 1e-5:
        return p
        
    if p > 0.99 or p < 0.01:
        # 极端概率对数空间计算
        log_w = gamma * math.log(p) - (1/gamma) * math.log(
            math.exp(gamma * math.log(p)) + 
            math.exp(gamma * math.log(1-p))
        )
        return math.exp(log_w)
    
    # 标准计算
    return p**gamma / (p**gamma + (1-p)**gamma)**(1/gamma)

# calculate prospect of an option
# e.g. value=[40, 30], prob=[30, 70]
def prospect_utility(params, value, prob):
    alpha, lam, gamma = params
    if all(v > 0 for v in value):
        # pure specification
        return v(value[1], alpha, lam) + w(prob[0]/100.0, gamma) * (v(value[0], alpha, lam) - v(value[1], alpha, lam))
    else:
        # mixed specification
        return w(prob[0]/100.0, gamma) * v(value[0], alpha, lam) + w(prob[1]/100.0, gamma) * v(value[1], alpha, lam)

def sigmoid(z):
    if z >= 0:
        return 1 / (1 + math.exp(-z))
    else:
        return math.exp(z) / (1 + math.exp(z))

def predict_safe_prob(params, data, series, lot_index):
    alpha, lam, gamma = params
    value = values[series][lot_index]
    prob = probs[series]

    safe_value = value[:2]
    safe_prob = prob[:2]
    risk_value = value[2:]
    risk_prob = prob[2:]

    u_safe = prospect_utility([alpha, lam, gamma], safe_value, safe_prob)
    u_risk = prospect_utility([alpha, lam, gamma], risk_value, risk_prob)
    eudiff = u_safe - u_risk
    return sigmoid(eudiff)

def nll(params, data):
    current_data = copy.deepcopy(data)
    alpha, lam, gamma = params
    total_nll = 0.0
    
    for series_idx, series_data in enumerate(current_data):
        for lot_index, obs in enumerate(series_data):
            n_safe, n_risk = obs
            total_trials = n_safe + n_risk
            p_safe = predict_safe_prob(params, current_data, series_idx, lot_index)
            
            if p_safe < 1e-10:
                p_safe = 1e-10
            if p_safe > 1 - 1e-10:
                p_safe = 1 - 1e-10
                
            log_lik = n_safe * math.log(p_safe) + n_risk * math.log(1.0 - p_safe)
            total_nll -= log_lik
            
    return total_nll

def estimate_parameters(data, initial_guesses=None):
    bounds = [(0.01, 4.0), (0.01, 4.0), (0.01, 4.0)]
    if initial_guesses is None:
        initial_guesses = [
            [0.8, 2.0, 0.6], [0.7, 1.5, 0.7], [1.0, 2.5, 0.5],
            [0.6, 1.7, 0.7], [0.65, 1.8, 0.65], [0.01, 0.1, 0.01],
            [2.8, 2.8, 2.9], [0.01, 2.09, 2.9], [1.5, 2.0, 0.1],
            [0.3, 1.2, 2.0], [0.62, 1.67, 0.70], [0.4, 1.5, 2.9],
            [0.88, 2.25, 0.65], [0.5, 1.5, 1.0], [0.6, 1.8, 0.8],
            [1.2, 1.8, 0.7], [0.4, 1.3, 1.2], [0.7, 1.2, 0.5],
            [1.0, 1.5, 0.4], [0.9, 2.2, 0.75], [0.4, 1.5, 0.8],
            [0.50, 1.35, 0.72], [0.65, 1.70, 0.55], [0.45, 1.80, 0.62],
            [0.38, 2.50, 0.85], [0.42, 1.20, 0.68],
            [0.55, 1.55, 0.48], [0.60, 2.00, 0.75],
            [0.75, 1.25, 0.65], [0.35, 3.00, 0.90]
        ]

    best_result = None
    best_nll = float('inf')
    
    for initial_guess in initial_guesses:
        result = minimize(
            nll,
            initial_guess,
            args=(data,),
            bounds=bounds,
            method='L-BFGS-B',
            options={
                'maxiter': 500,
                'ftol': 1e-12,
                'gtol': 1e-8,
                'maxls': 50
            }
        )
        if result.success and result.fun < best_nll:
            best_nll = result.fun
            best_result = result
    
    return best_result.x if best_result else None, best_result

def report_predictions(params, data, show_predictions=False):
    current_data = copy.deepcopy(data)
    alpha, lam, gamma = params
    if show_predictions:
        print("\nModel Predictions vs Actual:")
        print("Series | Index | Pred | Actual | Error")
        print("-------------------------------------")
    
    null_total_err = 0.0
    total_abs_error = 0.0
    count = 0
    
    for series_idx, series_data in enumerate(current_data):
        for lot_index, obs in enumerate(series_data):
            n_safe, n_risk = obs
            total_trials = n_safe + n_risk
            actual_prob = n_safe / total_trials
            pred_prob = predict_safe_prob(params, current_data, series_idx, lot_index)
            abs_error = abs(pred_prob - actual_prob)
            total_abs_error += abs_error

            null_total_err += abs(0.5 - actual_prob)

            count += 1
            if show_predictions:
                print(f"{series_idx+1:2d}   | {lot_index+1:2d}   | {pred_prob:.3f} | {actual_prob:.3f} | {abs_error:.3f}")
    print()
    mae = total_abs_error / count
    null_mae = null_total_err / count
    # if show_predictions:
    #     print("\nMAE:", f"{mae:.3f}")
    return mae, null_mae

def calculate_goodness_of_fit(params, data):
    current_data = copy.deepcopy(data)
    null_log_likelihood = 0.0
    for series_idx, series_data in enumerate(current_data):
        for lot_index, obs in enumerate(series_data):
            n_safe, n_risk = obs
            null_log_likelihood += n_safe * math.log(0.5) + n_risk * math.log(0.5)
    
    model_neg_log_likelihood = nll(params, data)
    model_log_likelihood = -model_neg_log_likelihood
    mcfadden_r2 = 1 - (model_log_likelihood / null_log_likelihood)
    
    return {
        'log_likelihood': model_log_likelihood,
        'null_log_likelihood': null_log_likelihood,
        'mcfadden_r2': mcfadden_r2,
    }

def compute_confidence_intervals(estimated_params, result, data, alpha=0.05, n_bootstrap=500):
    original_data = data
    predicted_probs = {}
    for series_idx, series in enumerate(original_data):
        for lot_index, obs in enumerate(series):
            p = predict_safe_prob(estimated_params, original_data, series_idx, lot_index)
            predicted_probs[(series_idx, lot_index)] = p
    
    bootstrap_params = []
    bootstrap_attempts = 0
    success_count = 0
    
    while success_count < n_bootstrap and bootstrap_attempts < n_bootstrap * 2:
        new_data = []
        for series_idx, series in enumerate(original_data):
            new_series = []
            for lot_index, obs in enumerate(series):
                n_trials = int(obs[0] + obs[1])
                p_safe = predicted_probs[(series_idx, lot_index)]
                n_safe_b = binom.rvs(n_trials, p_safe)
                n_risk_b = n_trials - n_safe_b
                new_series.append([n_safe_b, n_risk_b])
            new_data.append(new_series)
        
        try:
            params_b, _ = estimate_parameters(new_data, initial_guesses=[estimated_params])
            if params_b is not None:
                bootstrap_params.append(params_b)
                success_count += 1
        except:
            pass
        
        bootstrap_attempts += 1
    
    if not bootstrap_params:
        print("Warning: Bootstrap failed")
        return None
    
    bootstrap_matrix = np.array(bootstrap_params)
    std_errors = np.std(bootstrap_matrix, axis=0, ddof=1)
    lower_pct = 100 * alpha / 2
    upper_pct = 100 * (1 - alpha / 2)
    ci_sigma = np.percentile(bootstrap_matrix[:, 0], [lower_pct, upper_pct])
    ci_lambda = np.percentile(bootstrap_matrix[:, 1], [lower_pct, upper_pct])
    ci_gamma = np.percentile(bootstrap_matrix[:, 2], [lower_pct, upper_pct])
    
    return {
        'sigma': {
            'estimate': estimated_params[0],
            'std_error': std_errors[0],
            f'ci_{int((1-alpha)*100)}%': (ci_sigma[0], ci_sigma[1])
        },
        'lambda': {
            'estimate': estimated_params[1],
            'std_error': std_errors[1],
            f'ci_{int((1-alpha)*100)}%': (ci_lambda[0], ci_lambda[1])
        },
        'gamma': {
            'estimate': estimated_params[2],
            'std_error': std_errors[2],
            f'ci_{int((1-alpha)*100)}%': (ci_gamma[0], ci_gamma[1])
        }
    }

def perform_MLE(data, model_name, show_predictions=False):
    global probs
    if "Qwen" in model_name and "7B" in model_name:
        probs = [
            [30, 70, 10, 90],
            [88, 12, 70, 30],
            [50, 50, 50, 50]
        ] 
    elif "Qwen" in model_name and "14B" in model_name:
        probs = [
            [30, 70, 10, 90],
            [90, 10, 70, 30],
            [50, 50, 50, 50]
        ] 
    elif "Qwen" in model_name and "32B" in model_name:
        probs = [
            [29, 71, 10, 90],
            [82, 18, 70, 30],
            [50, 50, 50, 50]
        ] 
    elif "Llama" in model_name and "8B" in model_name:
        probs = [
            [32, 68, 10, 90],
            [74, 26, 70, 30],
            [50, 50, 50, 50]
        ] 
    elif "Mistral" in model_name and "7B" in model_name:
        probs = [
            [30, 70, 10, 90],
            [78, 22, 70, 30],
            [50, 50, 50, 50]
        ] 

    estimated_params, opt_result = estimate_parameters(data)
    total_samples = data[0][0][0] + data[0][0][1]
    print(f"Total Samples: {total_samples}")
    
    if estimated_params is not None:
        alpha_est, lambda_est, gamma_est = estimated_params
        print(f"Estimated Parameters:")
        print(f"sigma = {alpha_est:.3f}")
        print(f"lambda = {lambda_est:.3f}")
        print(f"gamma = {gamma_est:.3f}")
        
        fit_metrics = calculate_goodness_of_fit(estimated_params, data)
        print("\nGoodness of Fit:")
        print(f"Log-Likelihood: {fit_metrics['log_likelihood']:.2f}")
        print(f"Null Log-Likelihood: {fit_metrics['null_log_likelihood']:.2f}")
        print(f"McFadden R²: {fit_metrics['mcfadden_r2']:.3f}")

        confidence_intervals = compute_confidence_intervals(estimated_params, opt_result, data)
        if confidence_intervals:
            print("\nConfidence Intervals:")
            for param, info in confidence_intervals.items():
                print(f"{param}: {info['estimate']:.3f} ± {info['std_error']:.3f}")
                print(f"    {list(info.keys())[2]}: ({info[list(info.keys())[2]][0]:.3f}, {info[list(info.keys())[2]][1]:.3f})")
        
        mae, null_mae = report_predictions(estimated_params, data, show_predictions=show_predictions)
        print(f"MAE: {mae:.3f}")
        print(f"NULL_MAE: {null_mae:.3f}")
        return {
            "model": model_name,
            "samples": int(total_samples),
            "parameters": [round(para, 3) for para in estimated_params],
            "95%CI": [
                [round(confidence_intervals['sigma']['ci_95%'][0], 3), round(confidence_intervals['sigma']['ci_95%'][1], 3)],
                [round(confidence_intervals['lambda']['ci_95%'][0], 3), round(confidence_intervals['lambda']['ci_95%'][1], 3)],
                [round(confidence_intervals['gamma']['ci_95%'][0], 3), round(confidence_intervals['gamma']['ci_95%'][1], 3)]
            ],
            "sd": [
                round(confidence_intervals['sigma']['std_error'], 3),
                round(confidence_intervals['lambda']['std_error'], 3),
                round(confidence_intervals['gamma']['std_error'], 3)
            ],
            "McFadden R square": round(fit_metrics['mcfadden_r2'], 3),
            "MAE": round(mae, 3)
        }
    else:
        print("Estimation failed")
        return None

if __name__ == "__main__":
    # Example
    data = [[[251.0, 5.0], [247.0, 9.0], [216.0, 40.0], [200.0, 56.0], [166.0, 90.0], [147.0, 109.0], [130.0, 126.0], [99.0, 157.0], [100.0, 156.0], [82.0, 174.0], [76.0, 180.0], [65.0, 191.0], [60.0, 196.0], [58.0, 198.0]], [[96.0, 160.0], [92.0, 164.0], [70.0, 186.0], [56.0, 200.0], [57.0, 199.0], [34.0, 222.0], [41.0, 215.0], [27.0, 229.0], [21.0, 235.0], [22.0, 234.0], [2.0, 254.0], [11.0, 245.0], [0.0, 256.0], [0.0, 256.0]], [[256.0, 0.0], [193.0, 63.0], [135.0, 121.0], [58.0, 198.0], [38.0, 218.0], [16.0, 240.0], [17.0, 239.0]]]
    result = perform_MLE(data, model_name="Qwen32B", show_predictions=True)
    with open("result.json", 'w') as f:
        json.dump(result, f, indent=4)