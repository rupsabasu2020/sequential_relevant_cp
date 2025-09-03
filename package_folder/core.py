import numpy as np
from multiprocessing import Pool#, get_context
import concurrent.futures
import matplotlib.pyplot as plt
import pandas as pd
import sys
import math


sys.setrecursionlimit(10000)
plt.rcParams['axes.grid'] = True


""" 
This module contains functions for detecting change points in time series data using various statistical methods.
It includes implementations of the CUSUM algorithm, iterative global change-point detection, Brownian motion generation,
and parallel simulations for statistical testing.
Functions:
    s(k, bar_k, N):
        Generate a set of values for computing gamma in the CUSUM algorithm.
    cusum_gamma(N, k, l_set, data, beta):
        Compute the CUSUM statistic for change detection.
    glob_fund_iterative(data, r_run, k_bar, k_inspect, theta_hat, beta=0.1, C_cp=4, min_dist=1, N=None, cusum_vals=[], delay_times={}):
        Iterative function to detect change-points in data.
    init_global_iterative(data, N, beta=0.9, C_cp=0.05, min_dist=10):
    brownian_motion_array(t: float, multiplier, variance=1):
        Generate the value of Brownian motion at a specific time point.
    sign_funcDiff(mu_0, mu_j):
    extremals(data, theta_list, k, Delta, N):
        Generate the extremal sets for the unified LN range computation.
    monitoring_index_choice(data, theta_set, N, delay_times, upto_which_k, x_upper=None, delta_val=0.1, variance=1):
        Automatically choose values of k for which unified_LN_range is run.
    run_simulation(bf_function, data, theta_set, N, x_upper, delta_val, delay_times, upto_which_k, variance):
        Submit function for the parallel computation of the bf_function.
    parallel_simulations(bf_function, data, theta_set, N, x_upper, upto_which_k, num_simulations=100, delta_val=0.1, delay_times=None, variance=1, level=0.95):
    L_N_2(x, N, prev_theta, W_val):
    L_N_1(data, theta_list, N, k, W_val, Delta=0.1):
        Compute the maximum value of a specific function over a range of indices.
    unified_LN_range(data, theta_set, N, k, delay_time, results, x_upper=None, delta_val=0.1, brownian_motion=None):
        Compute the unified L_N range for a given dataset, theta set, and parameters.
    scanning_stat(data, N, theta_set, Delta=0.1, total_datalength=None, delay_times={}):
    generate_data_with_change_points(length, num_cp=0, loc=None, mean_shifts=None, std_dev=1):
    wrapper_func(N, total_data, delta, num_monte_carlos, data=None, level=0.95, beta_val=0.1):
        Compute rejection rates by generating data with change points and running parallel simulations.
    rejection_count(test_stat, asymptotics):
    compute_sigma_hat(X, N):
        Compute the estimator \hat{\sigma}^2.
    subset_data_from_date_and_party(df, start_date, party):
        Filter the dataframe to include only rows starting from the specified start date and only the specified political party.

"""

# -----------code for algorithm 1 for locating changes  -----------------#

def s(k, bar_k, N):
    """
    inputs: 
        k: current time step
        bar_k: variable value that controls how far back in the past we look
        N: training period length
    outputs:
        set_l: set of values of l which choose data subsets to compute gamma
    """
    k_bar = bar_k #// N
    upper_limit = int((k - k_bar + 1) // 2)
    set_l = set(range(1,upper_limit +2))
    return set_l



def cusum_gamma(N, k, l_set, data, beta):
    """
    Main CUSUM statistic for change detection
    Inputs:
        data: univariate data (numpy array or list)
        l_set: set of values of l (set or list)
        N: initial time step
        k: current time step
        beta: parameter controlling the scaling
    Output:
        The maximum computed value over the set of l
    """
    result = []
    k = k - N
    for l in l_set:
        w_lk = np.sqrt(N) / (((N + k) ** (1 - beta))*(l** beta) *  np.log(2 + (k / N)))
        term1 = np.sum(data[N + k - l: N + k+1])
        term2 = np.sum(data[N + k - 2 * l: N + k - l+1])
        res = w_lk * np.abs(term2 - term1)
        result.append(res)

    max_result = max(result)   # taking the max over l
    index_max = np.argmax(result)
    return max_result, index_max





def glob_fund_iterative(data, r_run, k_bar, k_inspect, theta_hat, beta=0.1, C_cp=4, min_dist=1, N=None, cusum_vals=[], delay_times={}):
    """
    Iterative version of the glob_fund function to detect change-points in data.

    Parameters:
        data (list or array-like): The input data in which change-points are to be detected.
        r_run (int): The current index being processed.
        k_bar (int): The starting index for the change-point detection.
        k_inspect (int): The index up to which the data is inspected.
        theta_hat (set): A set to store detected change-points.
        beta (float, optional): A parameter for the change-point detection algorithm. Default is 0.1.
        C_cp (float, optional): A constant used in the change-point detection criterion. Default is 4.
        min_dist (int, optional): The minimum distance between change-points. Default is 1.
        N (int, optional): A parameter used in the computation of gamma. Default is None.
        cusum_vals (list, optional): Stores the computed gamma values at each step.
        delay_times (dict, optional): Stores the delay times for detected change-points.

    Returns:
        tuple: (sorted theta_hat, cusum_vals, delay_times)
    """
    while r_run < k_inspect:  # Replace recursion with a loop
        set_l = s(r_run, k_bar, N)  # Generate set and compute gamma
        gamma_value, index_max = cusum_gamma(N, r_run, set_l, data, beta)
        cusum_vals.append(gamma_value)
        if gamma_value >= C_cp * np.log(N):  # Change-point detected
            Struktur_Br = r_run - index_max
            theta_hat.add(Struktur_Br)    #r_run - index_max)  # Add detected change point
            k_bar = r_run  # Update k_bar
            delay_times[Struktur_Br] = r_run 
            r_run += 1 + min_dist  # Skip ahead by min_dist after detection
        else:
            r_run += 1  # Increment to check the next point

    return sorted(theta_hat), cusum_vals, delay_times


def init_global_iterative(data, N, beta=0.9, C_cp=0.05, min_dist=10):
    """
    Initialize global change-point detection using the iterative approach.
    """
    k_bar = N  # Start at the beginning of the data
    k_inspect = len(data)  #k_bar + 4*min_dist  # Initial inspection window
    theta_hat = set()  # Store detected change points
    delay_time = {}
    r_running = k_bar + 1
    res = glob_fund_iterative(data, r_running, k_bar, k_inspect, theta_hat, beta, C_cp, min_dist, N, delay_times= delay_time)
    return res



#---------------------------------------------------------------------------------#
#-------------------Code for algorithm 2------------------------------------------#
#---------------------------------------------------------------------------------#


###################################################################################
#--------------------------------Helper functions---------------------------------#
###################################################################################



def brownian_motion_array(t: float, multiplier, variance=1):
    """
    Generate the value of Brownian motion at a specific time point on the real line.

    Parameters:
        t (float): The time point at which to evaluate the Brownian motion (t >= 0). Can also be an array of time points.
        initial_value (float): The initial value of the Brownian motion, in our case it should we the value whenn we start monitoring.
        seed (int, optional): Seed for reproducibility of randomness.

    Returns:
        float: The value of the Brownian motion at upto time t.
    """
    normals= np.random.normal(loc=0, scale=variance, size= t*multiplier)
    W_value = np.cumsum(normals)
    return W_value

def sign_funcDiff(mu_0, mu_j):
    """
    Generate the sign of the difference between historical mean and j-th mean.
    """
    return np.sign(mu_0- mu_j)

def rel_size_delta(data, theta_list, k, Delta, N, varying_delta = False):
    """
    Compute the relevant size of delta.
    """
    theta_list = sorted(theta_list)   # sort the theta values
    mu_0 = np.mean(data[:N])
    mu_j = [np.mean(data[theta_list[i]:theta_list[i+1]]) for i in range(len(theta_list)-1) if theta_list[i+1] <= k]
    #abs_diff = np.abs(mu_0 - mu_j)
    absolut_diffs = np.abs(mu_0 - np.array(mu_j))
    if varying_delta== True:
        Delta_Size = np.max(absolut_diffs)
    else:
        Delta_Size = Delta
    return Delta_Size, theta_list, mu_0, mu_j


def extremals(data, theta_list, k, Delta, N, varying_delta = False):
    """
    Generate the extremal sets for the unified LN range computation. (required for L_N_1)
    Inputs:
        mu_0: the mean of the data uptil N (training period)
        mu_j: the mean of the data between theta_j and theta_j+1
    """
    theta_list = sorted(theta_list)   # sort the theta values
    mu_0 = np.mean(data[:N])
    mu_j = [np.mean(data[theta_list[i]:theta_list[i+1]]) for i in range(len(theta_list)-1) if theta_list[i+1] <= k]
    abs_diff = np.abs(mu_0 - mu_j)
    Delta_Size, theta_list, mu_0, mu_j = rel_size_delta(data, theta_list, k, Delta, N, varying_delta)
    abs_diff = np.abs(mu_0 - mu_j)
    valid_indices = np.where(abs_diff> Delta-np.log(N)/np.sqrt(N))

    signs = sign_funcDiff(mu_0, mu_j)
    valid_indices_dict = {(theta_list[i], theta_list[i+1]): idx for i, idx in enumerate(valid_indices[0])}  # valid_indices[0] because of the structure of the output of np.where
    return valid_indices_dict, signs, mu_0, mu_j, None


#---------------------------------------------------------------------------------#
#-------------------required for simulations--------------------------------------#
#---------------------------------------------------------------------------------#

def monitoring_index_choice(data, theta_set,  N,  delay_times, upto_which_k,  x_upper=None, delta_val = 0.1, variance = 1, delta_variations = False, delta_dict= {}):
    """
    This function is required for the simulations. Automatically chooses values of k for which unified_LN_range is run.
    Run unified_LN_range over sub-intervals of times demarcated by theta_set.

    !!! This function is used for simulations only. !!! (way to set an upper limit to the monitoring preiod after each theta.)
    !! In real world application, use unified_LN_range directly for incoming k. !!
    
    Parameters:
        data (array-like): The dataset being analyzed.
        theta_set (list): A list of change points.
        N (int): The total number of observations.
        x_upper (int): The upper bound of the range for k (default is N * 20).
        delay_times: dictionary of delay times (k) for each change point.
    
    Returns:
        dict: Results of unified_LN_range for each value of k.
    """

    res= {}

    brownian_motion = brownian_motion_array(t=x_upper//N, multiplier=N*100, variance= 1)  # rescaling x_upper//N for computation of correct brownian motion necessary!

    for k in upto_which_k:
        result = unified_LN_range(data, theta_set, N, k, delay_times, results= res, x_upper=x_upper, delta_val = delta_val, brownian_motion= brownian_motion, vary_delta=delta_variations)
    
    return result




def run_simulation(bf_function, data, theta_set, N, x_upper, delta_val, delay_times, upto_which_k, variance, delta_variations, delta_dict):   
    """
    A submit function for the parallel computation of the bf_function.
    """
    np.random.seed(None) 
    return bf_function(data, theta_set, N, delay_times, upto_which_k, x_upper, delta_val, variance, delta_variations, delta_dict)


def parallel_simulations(bf_function, data, theta_set, N, x_upper, upto_which_k, num_simulations=100, delta_val = 0.1, delay_times = None, variance = 1, level = 0.95, delta_variations = False, delta_dict = {}):  
    """
    Perform multiple simulations of bf_function in parallel.

    Parameters:
        bf_function (callable): The function to be executed in parallel. 
                                Should match the signature of `bf.run_unified_LN_half_theta`.
        data (object): Data to be passed to the bf_function.
        theta_set (object): Theta set to be passed to the bf_function.
        N (int): The parameter N to be passed to the bf_function.
        x_upper (float): The x_upper parameter for bf_function.
        num_simulations (int): Number of simulations to run.

    Returns:
        np.ndarray: Array of simulation results.
    """
    results = []
    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = [
            executor.submit(run_simulation, bf_function, data, theta_set, N, x_upper, delta_val, delay_times, upto_which_k, variance, delta_variations, delta_dict)
            for _ in range(num_simulations)
        ]
        for future in concurrent.futures.as_completed(futures):
            results.append(future.result())


    res_array = np.array(results)
    # Extract unique keys
    keys_res = res_array[0].keys()

    
    quantiles = {key: np.quantile([entry[key] for entry in res_array], level) for key in keys_res} # Compute 95% quantiles

    return quantiles #np.array(results)


###################################################################################
#--------------------------------Main asymptotics---------------------------------#
###################################################################################


def L_N_2(x, N, prev_theta, W_val): 
    """
    Asymptotic computation when there are no further change points in the future.

    Parameters:
        x (int): The current value of x.
        N (int): The current index N.
        prev_theta (int): The previous theta value.
        W_val (array-like): An array of W values.

    Returns:
        float: The computed value based on the given parameters.
    """
    if x <= prev_theta:
        x = prev_theta + 2


    stheta = np.linspace(prev_theta, x, num=x - prev_theta, dtype=int)
    main1 = np.array([(x-theta_val)*np.abs(W_val[N]) for theta_val in stheta])/(N)

    main2 = np.array([(W_val[x]- W_val[theta_val]) for theta_val in stheta])

    main_fNR = np.abs(main1- main2)*np.sqrt(N)
    main_fDR =  x 
    main_f = np.max(main_fNR/main_fDR) # sup over x between theta_i and theta_i+1
    L_N2_dict = main_f
    return L_N2_dict


def L_N_1(data, theta_list, N, k, W_val, Delta=0.1, delta_variations = False):
    """
    Computes the maximum value of a specific function over a range of indices.
    Parameters:
        data (array-like): Input data used for computation.
        theta_list (list): List of theta values.
        N (int): An integer parameter used in the computation.
        k (int): An integer parameter used in the computation.
        W_val (array-like): Array of values used in the computation.
    Returns:
        dict: A dictionary containing the maximum value of the computed function.
    
    """
    store_res = {}
    if len(theta_list) <= 1:
        pass
    else:

        valid_idx, s_i, _, _, _ = extremals(data, theta_list, k, Delta, N= N, varying_delta=delta_variations)

        theta_consider = np.asarray(theta_list)
        store_mainf = {}
        for i in range(len(theta_consider)-1):
            
            x_val = np.linspace(theta_consider[i], theta_consider[i+1], num=theta_consider[i+1] - theta_consider[i] + 1, dtype=int)
        
            main1 = (x_val-theta_consider[i])*np.abs(W_val[N])/N    #s_i[i]*
            main2 = (W_val[x_val] - W_val[theta_consider[i]])  #brownian_motion_at_time(t=theta_consider[i], initial_value=N))
            main_fNr = s_i[i]*(main1 - main2)*np.sqrt(N)
            main_fDr = x_val   
            main_f = np.max(main_fNr/main_fDr) # sup over x between theta_i and theta_i+1
            store_mainf[theta_consider[i], theta_consider[i+1]] = main_f

        filtered_mainf = {key: store_mainf[key] for key in valid_idx.keys() if key in store_mainf}   # takes the max over valid indices
        max_main_f = max(filtered_mainf.values()) if filtered_mainf else float('-inf')
        store_res['max'] = max_main_f

    return store_res #store_mainf





def unified_LN_range(data, theta_set, N, k, delay_time, results, x_upper=None, delta_val = 0.1, brownian_motion= None, vary_delta = False):
    """

    Computes the unified L_N range for a given dataset, theta set, and parameters.

    Parameters:
        data (array-like): The input data for the computation.
        theta_set (list): A list of theta values.
        N (int): The initial value for the computation.
        k (int): The current value for which the computation is being performed.
        x_upper (int, optional): The upper limit for the range of x values. Defaults to None.

    Returns:
        dict: A dictionary containing the results of the computation for the given k value.

    Notes:
    - If x_upper is not provided, it defaults to N * 30.
    - The function computes the Brownian motion at specified time points given by range_x_upper.
    - The theta_set is sorted and used to determine the previous theta value.
    - Depending on the value of k relative to the theta_set, different computations are performed.
    - The results dictionary contains the maximum L2 value for k greater than the previous theta value.
    """

    if x_upper is None:
        x_upper = N * 30    # set some value that is very large


    theta_set = sorted(theta_set)
    #print("unif1", theta_set)

    if not theta_set or k < min(theta_set): # if k is less than the minimum theta value or if theta_set is empty
        # no L1 no extremal sets
        prev_theta = N
        prev_delay = N+1
        next_val_k = min([key for key in delay_time.keys() if key > k], default=x_upper)
        k_monitor = np.linspace(prev_delay, next_val_k, num=next_val_k - prev_theta + 1, dtype=int) # Generate k_monitor values
        ll = [L_N_2(ik, N, prev_theta, W_val=brownian_motion) for ik in k_monitor]    # list of asymptotic values over k_monitor window
        #results.update({prev_theta: np.max(ll)})
        results.update({prev_delay: np.max(ll)}) # update results with the maximum value
    else:
        #print("case2", theta_set)

        prev_theta = max(theta for theta in theta_set if theta <= k)
        prev_delay = max([delay_time[delay] for delay in delay_time.keys() if delay <= k], default=N)
        updated_theta_set = [theta for theta in theta_set if theta <= prev_theta] # includes all change locations smaller than k
        #print("updatedCase2", updated_theta_set)
        if len(updated_theta_set) <= 1:
            # no L1 no extremal sets
            next_val_k = x_upper #- prev_theta
            k_monitor = np.linspace(prev_delay, next_val_k, num=next_val_k - prev_theta + 1, dtype=int)
            ll = [L_N_2(ik, N, prev_theta, W_val=brownian_motion) for ik in k_monitor]
            #results.update({prev_theta: np.max(ll)})
            results.update({prev_delay: np.max(ll)}) # update results with the maximum value

        else:
            # here L1 and hence extremals
            #next_val_k = min([key for key in delay_time.keys() if key > k], default=x_upper) # !!!! version on Feb11,25
            next_val_k = min([delay_time[key] for key in delay_time.keys() if key > k], default=x_upper)
            #delta_size = {v: delta_dict[k] for k, v in delay_time.items()}### unnecessary?
            L1 = L_N_1(data, updated_theta_set, N, k, W_val=brownian_motion, Delta=delta_val, delta_variations=vary_delta)
            k_monitor = np.linspace(prev_delay, next_val_k, num=next_val_k - prev_theta + 1, dtype=int)

            ll_o = [L_N_2(ik, N, prev_theta, W_val=brownian_motion) for ik in k_monitor]
            L2 = np.max(ll_o)     # outer sup of L_N_2
            max_value = max(max(L1.values()), L2)
            #results.update({prev_theta: max_value})
            results.update({prev_delay: max_value}) # update results with the maximum value

    return results



#---------------------------------------------------------------------------------#
#---------------------scanning statistic CP testing------------------------------------------#
#---------------------------------------------------------------------------------#


def scanning_stat(data, N, theta_set, delta, total_datalength= None, delay_times = {}, varying_delta = True):
    """
    Calculate the scanning statistic for change point testing.
    Parameters:
        data (list or numpy array): The input data sequence.
        N (int): The initial segment length.
        theta_set (set): A set of candidate change points.
        k (int): The current time point.
        Delta (float): The threshold parameter.
    Returns:
        dict: A dictionary with the key as the current time point `k` and the value as the calculated scanning statistic.
    """
    # choose half_theta_values to have arbitrary theta points

    upto_which_k = []

    if not theta_set:
        upto_which_k.append(total_datalength - N)
    else:
        
        total_scanlength = total_datalength - int(theta_set[-1]) #if total_datalength else 25
        upto_which_k.append(N)
        #delay_times_values = list(delay_times.values())
        delay_times_values = list(delay_times.keys())   # modification April 3
        upto_which_k.extend(delay_times_values) # append delay times
        upto_which_k.append(upto_which_k[-1] + total_scanlength) # append append the end of the data


    psi_0 = np.mean(data[:N])  # mean of training set
    gamma = {}  # Initialize gamma to store all 
    gamma_signed = {}  # Initialize gamma to store all

    for k_theta in upto_which_k:
        prev_monitor_k = max([delay_times[key] for key in delay_times.keys() if key <= k_theta], default=N) # if no delay times, set to N+1
        next_val_k = min([delay_times[key] for key in delay_times.keys() if key > k_theta], default= total_datalength)#max([val2 for val2 in delay_times.values() if val2 > k_theta], default=total_datalength)
        #---------------previous working version with monitoring exactly at theta -----------------------#
        # if delay_times and any(key <= k_theta for key in delay_times.keys()):
        
        #     #prev_monitor_k = max([key for key in delay_times.keys() if key <= k_theta]) 
        #     prev_monitor_k = max([delay_times[key] for key in delay_times.keys() if key <= k_theta], default=N) # if no delay times, set to N+1
        #     print("k_theta", k_theta, "prev_monitor_k", prev_monitor_k)
        #     next_val_k = min([key2 for key2 in delay_times.keys() if key2 > k_theta], default=total_datalength)
        # else:
        #     prev_monitor_k = N + 1 # initial monitoring after historical data
        #     next_val_k = total_datalength



        k_monitor = np.linspace(prev_monitor_k, next_val_k, num= next_val_k+1 - prev_monitor_k) # Generate k_monitor values
        psi_1_values = [np.mean(data[prev_monitor_k:int(k_monitor_val)]) for k_monitor_val in k_monitor]

        if varying_delta== True:
            Delta = 0
            multiplier = np.sqrt(N) * (N+ k_monitor - prev_monitor_k) /(k_monitor)
            gamma_signs = multiplier * ((psi_0 - psi_1_values) - Delta)    # not doing N+k because k= N+k in our case
            gamma_signed.update(dict(zip(map(int, k_monitor), gamma_signs)))
        else:
            Delta = delta
        factor_multiple = np.sqrt(N) * (N+ k_monitor - prev_monitor_k) /(k_monitor)
        gamma_values = factor_multiple * (np.abs(psi_0 - psi_1_values) - Delta)    # not doing N+k because k= N+k in our case

        gamma.update(dict(zip(map(int, k_monitor), gamma_values)))
    return gamma, upto_which_k , gamma_signed  #, delta_store
    


################################################################################################
#------------------------- Simulations --------------------------------------------------------#
################################################################################################


#---------------------------------------------------------------------------------#


def generate_data_with_change_points(length, num_cp=0, loc=None, mean_shifts=None, std_dev=1, seed=None):
    """
    Generate a data array consisting of random normal variables with structural breaks (change points).

    :param length: Total length of the data array.
    :param num_cp: Number of change points to introduce (default is 0, no change points).
    :param loc: List of locations (indices) for the change points.
    :param mean_shifts: List of mean values after each change point.
    :param std_dev: Standard deviation of the normal distribution (default is 1).
    :return: Generated data array with the specified structural breaks.
    """
    if seed is not None:
        np.random.seed(seed)   # <-- seed numpy
    if num_cp == 0:
        # No change points, generate data with mean 0
        return np.random.normal(loc=0, scale=std_dev, size=length)
    
    if not loc or not mean_shifts or len(loc) != num_cp or len(mean_shifts) != num_cp + 1:
        raise ValueError("Invalid inputs: loc must have num_cp indices, and mean_shifts must have num_cp + 1 values.")

    data = []
    start = 0
    for i, change_point in enumerate(loc):
        # Generate data for the segment before the change point
        segment = np.random.normal(loc=mean_shifts[i], scale=std_dev, size=change_point - start)
        data.extend(segment)
        start = change_point
    
    # Generate data for the final segment after the last change point
    data.extend(np.random.normal(loc=mean_shifts[-1], scale=std_dev, size=length - start))
    
    return np.array(data)


#--------------wrapper function----------------------#
def wrapper_func(N, total_data, delta_fixed, num_monte_carlos, data = None, level = 0.95, beta_val = 0.45, delta_variations=False):  #, cp_locs, jump_valsd
    """
    Computes rejection rates in the end by generating data with change points, initializing global iterative parameters,
    calculating scanning statistics, and running parallel simulations.

    Parameters:
    N (int): The sample size or number of observations.
    total_data (int): The total length of the data to be generated.
    cp_locs (list): List of change point locations.
    jump_vals (list): List of mean shift values at the change points.
    delta (float): The threshold parameter for the scanning statistic.
    num_monte_carlos (int): The number of Monte Carlo simulations to run.

    Returns:
    tuple: A tuple containing:
        - test_stat (float): The calculated test statistic.
        - asymptotics (list): Results from the parallel simulations.
        - theta_set (list): The set of initialized parameters.
    """
    long_run = compute_sigma_hat(data, N)
    theta_set, cusum, r_run = init_global_iterative(data, N, beta=beta_val, C_cp=np.sqrt(long_run), min_dist=20)
    test_stat, upto_ktime, gamma_signed = scanning_stat(data, N, theta_set, delta_fixed, total_datalength=total_data, delay_times= r_run, varying_delta=delta_variations)
    asymptotics = parallel_simulations(monitoring_index_choice, data, theta_set, N, x_upper=total_data, num_simulations=num_monte_carlos, delta_val=delta_fixed, delay_times= r_run, upto_which_k= upto_ktime, variance= long_run, level=level, delta_variations= delta_variations, delta_dict = 0)

    # if delta_variations == True: 
    #     asymptotics = parallel_simulations(monitoring_index_choice, data, theta_set, N, x_upper=total_data, num_simulations=num_monte_carlos, delta_val=delta_fixed, delay_times= r_run, upto_which_k= upto_ktime, variance= long_run, level=level, delta_variations= delta_variations)
    # else:
    #     asymptotics = parallel_simulations(monitoring_index_choice, data, theta_set, N, x_upper=total_data, num_simulations=num_monte_carlos, delta_val=delta_fixed, delay_times= r_run, upto_which_k= upto_ktime, variance= long_run, level=level)
    return test_stat, asymptotics, theta_set, gamma_signed, r_run , long_run# ,  data

def wrapper_func_simulations(N, total_data, delta, num_monte_carlos, number_changes, cp_locs, jump_vals, level = 0.95, beta_val = 0.1):
    """
    Wrapper function to run simulations for change point detection.
    Parameters:
        N (int): The window size for the change point detection algorithm.
        total_data (int): The total length of the data to be generated.
        delta (float): The threshold parameter for the scanning statistic.
        num_monte_carlos (int): The number of Monte Carlo simulations to run.
        number_changes (int): The number of change points to introduce in the data.
        cp_locs (list): The locations of the change points in the data.
        jump_vals (list): The mean shifts at the change points.
        level (float, optional): The significance level for the asymptotic test. Default is 0.95.
        beta_val (float, optional): The beta parameter for the iterative algorithm. Default is 0.1.
        C_cp (float, optional): The constant for change point detection. Default is 1.
    Returns:
    tuple: A tuple containing:
            - test_stat (float): The test statistic value.
            - asymptotics (list): The results of the parallel simulations.
            - theta_set (list): The set of estimated parameters.
            - data (numpy.ndarray): The generated data with change points.
    """    
    data = generate_data_with_change_points(total_data, num_cp=number_changes, loc=cp_locs, mean_shifts=jump_vals, std_dev=1)
    # plt.plot(data)
    # plt.show()
    long_run = compute_sigma_hat(data, N)
    theta_set, cusum, r_run = init_global_iterative(data, N, beta=beta_val, C_cp=np.sqrt(long_run)/2, min_dist=1)

    test_stat, upto_ktime = scanning_stat(data, N, theta_set, Delta=delta, total_datalength=total_data, delay_times= r_run)

    asymptotics = parallel_simulations(monitoring_index_choice, data, theta_set, N, x_upper=total_data, num_simulations=num_monte_carlos, delta_val=delta, delay_times= r_run, upto_which_k= upto_ktime, variance= long_run, level=level)

    return test_stat, asymptotics, theta_set,  data



def rejection_count(test_stat, asymptotics):
    """
    Calculate the rejection count for given test statistics and asymptotic values.
    Args:
        test_stat (dict): A dictionary where keys are theta values and values are test statistics.
        asymptotics (dict): A dictionary where keys are theta values and values are asymptotic critical values.
    Returns:
        tuple: A tuple containing:
            - rejection_perTheta (dict): A dictionary where keys are theta values and values are 1 if the test statistic 
              is greater than or equal to the asymptotic critical value, otherwise 0.
            - global_rejection (list): A list containing a single element, 1 if any of the values in rejection_perTheta 
              are 1, otherwise 0.
    """

    sorted_asymptotics_keys = sorted(asymptotics.keys())
    rejection_perTheta={}

    for key1, stat_value in test_stat.items():
        # Find the largest key2 in asymptotics that is less than or equal to key1
        valid_key2 = max((key2 for key2 in sorted_asymptotics_keys if key2 <= key1), default=None)
        if valid_key2 is not None:
            # Compare test_stat[key1] with asymptotics[valid_key2]
            rejection_perTheta[valid_key2] = 1 if stat_value >= asymptotics[valid_key2] else 0
        else:
            # If no valid key2 is found, assume no rejection
            rejection_perTheta[valid_key2] = 0
    global_rejection =[ 1 if any(value == 1 for value in rejection_perTheta.values()) else 0]
    return rejection_perTheta, global_rejection



#--------------------Additional functions data analysis----------------------#


def compute_sigma_hat(X, N):
    """
    Computes the estimator \hat{\sigma}^2.

    Parameters:
    X : array-like
        Input array of observations X_{i,n}.
    N : int
        The total number of observations.

    Returns:
    sigma_hat : float
        The estimated value of \hat{\sigma}^2.
    """
    m_N = int(N**(1/3))  # Set m_N = N^(1/3)
    S = np.cumsum(X)  # Compute cumulative sums S_j,k 
    num_blocks = int(N // m_N)   # Compute the number of blocks

    # Compute the \hat{\sigma}^2 estimator
    sigma_hat = 0
    for j in range(1, num_blocks):
        block_start = (j - 1) * m_N
        block_middle = j * m_N
        block_end = (j + 1) * m_N

        term = ((S[block_middle - 1] - S[block_start - 1] if block_start > 0 else S[block_middle - 1]) -
                (S[block_end - 1] - S[block_middle - 1])) ** 2
        sigma_hat += term / (2 * m_N)

    sigma_hat /= (num_blocks - 1)
    return sigma_hat



def subset_data_from_date_and_party(df, start_date, party, end_date=None):
    """
    Filters the dataframe to include only rows starting from the specified start date
    and only the specified political party.

    Parameters:
    df : pandas.DataFrame
        The input dataframe with a 'Datum' column and political party columns.
    start_date : str
        The start date in '%d.%m.%Y' format.
    party : str
        The name of the political party to filter by.

    Returns:
    pandas.DataFrame
        The filtered dataframe.
    """
    # Convert 'Datum' column to datetime if not already
    # df['Datum'] = pd.to_datetime(df['Datum'], format='%d.%m.%Y', errors='coerce')

    # # Convert start_date to datetime
    # start_date = pd.to_datetime(start_date, format='%d.%m.%Y')

    # # Filter the dataframe by date and party
    # filtered_df = df[df['Datum'] >= start_date][['Datum', party]]

    # return filtered_df
    df['Datum'] = pd.to_datetime(df['Datum'], format='%d.%m.%Y', errors='coerce')

    # Convert start_date and end_date to datetime
    start_date = pd.to_datetime(start_date, format='%d.%m.%Y')
    end_date = pd.to_datetime(end_date, format='%d.%m.%Y')

    # Filter by date range and select the relevant party
    filtered_df = df[(df['Datum'] >= start_date) & (df['Datum'] <= end_date)][['Datum', party]]

    return filtered_df



#-------------------------------------------------------------------------------------------------#
#------------------------------WIP----------------------------------------------------------------#
#-------------------------------------------------------------------------------------------------#


def get_first_exceeding_key(test_statistic, asymptotics):
    sorted_ts_keys = sorted(test_statistic.keys())
    sorted_asym_keys = sorted(asymptotics.keys())
    keys_upcross = {}
    for i in range(len(sorted_asym_keys)):
        start = sorted_asym_keys[i]
        end = sorted_asym_keys[i + 1] if i + 1 < len(sorted_asym_keys) else float('inf')
        threshold = asymptotics[start]
    
        for hat_k in sorted_ts_keys:
            if start <= hat_k < end:
                val = test_statistic[hat_k]
                if val is not None and not (val != val):  # exclude NaN
                    if val > threshold:
                        
                        numerator = val - threshold
                        keys_upcross[start] = hat_k, numerator
                        break
    return keys_upcross  # return the first matching key



def compute_delta(keys_upcross, factor_multiple):
    get_deltaSize = {}
    
    for start, (hat_k, numerator) in keys_upcross.items():
        delta_A = numerator / factor_multiple
        
        # Replace inf and -inf with 0 in the array
        if isinstance(delta_A, np.ndarray):
            delta_A = np.where(np.isinf(delta_A), 0, delta_A)
        
        get_deltaSize[hat_k] = delta_A
    
    # If you're computing max per array, this is fine:
    #max_deltaSize = {key: np.max(values) for key, values in get_deltaSize.items()}
    return get_deltaSize #max_deltaSize
