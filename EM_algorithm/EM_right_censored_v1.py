"""This code uses the Expectation Maximization Algorithm to fit a distribution 
(limited to the family of exponential distributions for this code) to right-censored data,
in order to predict the tail-end of the distribution for the censored data 
and preserve analytic continuity of the underlying overall distribution.

This is done here primarily in the context of survival analysis where the data contains a
time (numerical with censored values) and event (1/0) column.
"""

import numpy as np
import matplotlib.pyplot as plt

class EM:
    
    def __init__(self, table, time, event, dist="expon", max_iter=100, tol=1e-6):
        """
        table : pd.DataFrame
        time : numerical column containing right-censored (None/NaN) values
        event : binary column indicating occurence (1) or non-occurence (0) of event
        dist : the underlying distribution for which EM algorithm is tested.
                Choose from among the following - [expon, invg, logn, pareto]
        max_iter : max no. of iterations allowed
        tol : convergence criteria for stopping the algorithm
        """
        self.table = table
        self.time = time
        self.event = event
        self.dist = dist
        self.max_iter = max_iter
        self.tol = tol
        
        
    def em_algorithm(self, plot_dist = True):
        """
        plot_llc : plots the iterative Log-likelihood convergence of the EM-algorithm
        plot_dist : plots the revised distribution with the censoring threshold
        fits the EM algorithm based on the distribution passed.
        Returns the estimated parameter(s) of the distribution and the list of log-likelihood values from each iteration.
        """
        df = self.table
        t = self.time
        e = self.event
        uncensored_data = df[t][df[e] == True]
        log_likelihood = []
        
        if self.dist == "expon":
            
            lambda_est = 1/np.nanmean(uncensored_data)
            
            for i in range(self.max_iter):
                
                # E-step : Calculate expected complete data log-likelihood
                censored = df[e] == False
                complete_data_times = df[t].copy()
                if np.sum(censored) > 0:
                    # For censored data, assume they are greater than the max uncensored time
                    max_uncensored_time = np.nanmax(uncensored_data)
                    expected_tail_values = max_uncensored_time + np.random.exponential(scale=1/lambda_est, size=np.sum(censored))
                    complete_data_times[censored] = expected_tail_values
                    
                # M-step : Maximize the expected complete data log-likelihood
                new_lambda_est = 1/np.mean(complete_data_times)
                
                # Convergence check
                log_likelihood.append(np.sum(df[e]*np.log(lambda_est) - lambda_est*complete_data_times))
                if np.abs(new_lambda_est - lambda_est) < tol:
                    break
                lambda_est = new_lambda_est
                
            if plot_dist:
                from scipy.stats import expon
                
                # Define the distribution
                dist1 = expon(scale=1/lambda_est)
                
                # Generate values for plotting
                x = np.linspace(0,100,1000)
                y = dist1.pdf(x)
                
                # Plot the histogram of the uncensored data
                plt.figure(figsize=(12,6))
                plt.hist(uncensored_data, bins=int(max(uncensored_data)), density=True, alpha=0.6, color='g', label='Uncensored Data Histogram')
                
                # Plot the PDF of the distribution
                plt.plot(x, y, label=f'Exponential Distribution (λ={lambda_est:.4f})', lw=2)
                
                # Mark the censoring threshold
                plt.axvline(max(uncensored_data), color='r', linestyle='--', label=f'Censoring Threshold (x={max(uncensored_data):.2f})')
                
                # Add labels and legend
                plt.xlabel('Value')
                plt.ylabel('Probability Density')
                plt.title('Exponential Distribution with Censoring Threshold')
                plt.legend()
                plt.grid(True)
                plt.savefig("EM_Exponential_with_censoring.png")
            
            return lambda_est, log_likelihood
        
        if self.dist == "invg":
            
            from scipy.stats import invgauss
            mu_est = np.nanmean(uncensored_data)
            lambda_est = 1/np.var(uncensored_data - mu_est)
            
            for i in range(self.max_iter):
                
                # E-step:
                censored = df[e] == False
                complete_data_times = df[t].copy()
                if np.sum(censored) > 0:
                    max_censored_time = np.nanmax(uncensored_data)
                    expected_tail_values = max_censored_time + invgauss.rvs(mu_est, scale=1/lambda_est, size=np.sum(censored))
                    complete_data_times[censored] = expected_tail_values
                    
                # M-step:
                new_mu_est = np.mean(complete_data_times)
                new_lambda_est = 1/np.var(complete_data_times - new_mu_est)
                
                # Calculate Log-likelihood
                ll = np.sum(np.log(invgauss.pdf(complete_data_times, new_mu_est, scale=1/new_lambda_est)))
                log_likelihood.append(ll)
                
                # Convergence check
                if len(log_likelihood) > 1 and np.abs(log_likelihood[-1] - log_likelihood[-2]) < tol:
                    break
                    
                mu_est, lambda_est = new_mu_est, new_lambda_est
                
            if plot_dist:
                
                # Define the distribution
                dist1 = invgauss(mu_est, scale=1/lambda_est)
                
                # Generate values for plotting
                x = np.linspace(0,100,1000)
                y = dist1.pdf(x)
                
                # Plot the histogram of the uncensored data
                plt.figure(figsize=(12,6))
                plt.hist(uncensored_data, bins=int(max(uncensored_data)), density=True, alpha=0.6, color='g', label='Uncensored Data Histogram')
                
                # Plot the PDF of the distribution
                plt.plot(x, y, label=f'Inverse Gaussian Distribution (μ={mu_est:.4f}, λ={lambda_est:.4f})', lw=2)
                
                # Mark the censoring threshold
                plt.axvline(max(uncensored_data), color='r', linestyle='--', label=f'Censoring Threshold (x={max(uncensored_data):.2f})')
                
                # Add labels and legend
                plt.xlabel('Value')
                plt.ylabel('Probability Density')
                plt.title('Inverse Gaussian Distribution with Censoring Threshold')
                plt.legend()
                plt.grid(True)
                plt.savefig("EM_Inverse_Gaussian_with_censoring.png")
                
            return mu_est, lambda_est, log_likelihood
        
        if self.dist == "logn":
            
            from scipy.stats import lognorm
            mu_est = np.nanmean(np.log(uncensored_data))
            sigma_est = np.nanstd(np.log(uncensored_data))
            
            for i in range(self.max_iter):
                
                # E-step:
                censored = df[e] == False
                complete_data_times = df[t].copy()
                if np.sum(censored) > 0:
                    max_censored_time = np.nanmax(uncensored_data)
                    expected_tail_values = max_censored_time*np.exp(np.random.normal(mu_est, sigma_est, size=np.sum(censored)))
                    complete_data_times[censored] = expected_tail_values
                    
                # M-step:
                log_complete_data_times = np.log(complete_data_times)
                new_mu_est = np.nanmean(log_complete_data_times)
                new_sigma_est = np.nanstd(log_complete_data_times)
                
                if new_sigma_est <= 0:
                    new_sigma_est = 1e-6
                
                # Calculate Log-likelihood
                ll = np.sum(lognorm.logpdf(complete_data_times, s=new_sigma_est, scale=np.exp(new_mu_est)))
                log_likelihood.append(ll)
                
                # Convergence check
                if len(log_likelihood) > 1 and np.abs(log_likelihood[-1] - log_likelihood[-2]) < tol:
                    break
                    
                mu_est, sigma_est = new_mu_est, new_sigma_est
                
            if plot_dist:
                
                # Define the distribution
                dist1 = lognorm(s=sigma_est, scale=np.exp(mu_est))
                
                # Generate values for plotting
                x = np.linspace(0,np.nanmax(df[t]),1000)
                y = dist1.pdf(x)
                
                # Plot the histogram of the uncensored data
                plt.figure(figsize=(12,6))
                plt.hist(uncensored_data, bins=int(max(uncensored_data)), density=True, alpha=0.6, color='g', label='Uncensored Data Histogram')
                
                # Plot the PDF of the distribution
                plt.plot(x, y, label=f'Log-normal Distribution (μ={mu_est:.4f}, σ={sigma_est:.4f})', lw=2)
                
                # Mark the censoring threshold
                plt.axvline(max(uncensored_data), color='r', linestyle='--', label=f'Censoring Threshold (x={max(uncensored_data):.2f})')
                
                # Add labels and legend
                plt.xlabel('Value')
                plt.ylabel('Probability Density')
                plt.title('Log-normal Distribution with Censoring Threshold')
                plt.legend()
                plt.grid(True)
                plt.savefig("EM_Log_Normal_with_censoring.png")
                
            return mu_est, sigma_est, log_likelihood
        
        if self.dist == "pareto":
            
            from scipy.stats import pareto
            xm_est = np.nanmin(uncensored_data)
            alpha_est = 1 + len(uncensored_data)/np.sum(np.log(uncensored_data/xm_est))
            
            for i in range(self.max_iter):
                
                # E-step:
                censored = df[e] == False
                complete_data_times = df[t].copy()
                if np.sum(censored) > 0:
                    max_censored_time = np.nanmax(uncensored_data)
                    expected_tail_values = max_censored_time*(1+np.random.pareto(alpha_est, size=np.sum(censored)))
                    complete_data_times[censored] = expected_tail_values
                    
                # M-step:
                xm_est = np.nanmin(complete_data_times)
                alpha_est = 1 + len(complete_data_times)/np.sum(np.log(complete_data_times/xm_est))
                
                if alpha_est <= 0:
                    alpha_est = 1e-6
                
                # Calculate Log-likelihood
                ll = np.sum(pareto.logpdf(complete_data_times, alpha_est, scale=xm_est))
                log_likelihood.append(ll)
                
                # Convergence check
                if len(log_likelihood) > 1 and np.abs(log_likelihood[-1] - log_likelihood[-2]) < tol:
                    break
                
            if plot_dist:
                
                # Define the distribution
                dist1 = pareto(alpha_est, scale=xm_est)
                
                # Generate values for plotting
                x = np.linspace(1,np.nanmax(df[t]),1000)
                y = dist1.pdf(x)
                
                # Plot the histogram of the uncensored data
                plt.figure(figsize=(12,6))
                plt.hist(uncensored_data, bins=int(max(uncensored_data)), density=True, alpha=0.6, color='g', label='Uncensored Data Histogram')
                
                # Plot the PDF of the distribution
                plt.plot(x, y, label=f'Pareto Distribution (Xm={xm_est:.4f}, α={alpha_est:.4f})', lw=2)
                
                # Mark the censoring threshold
                plt.axvline(max(uncensored_data), color='r', linestyle='--', label=f'Censoring Threshold (x={max(uncensored_data):.2f})')
                
                # Add labels and legend
                plt.xlabel('Value')
                plt.ylabel('Probability Density')
                plt.title('Pareto Distribution with Censoring Threshold')
                plt.legend()
                plt.grid(True)
                plt.savefig("EM_Pareto_with_censoring.png")
                
            return xm_est, alpha_est, log_likelihood
        
    
    def log_likelihood_convergence_plot(self):
        ll = em_algorithm(plot_dist=False)[-1]
        plt.figure(figsize=(12,6))
        plt.plot(ll, marker='o')
        plt.xlabel('Iteration')
        plt.ylabel('Log-Likelihood')
        plt.title('EM Algorithm Log-Likelihood Convergence')
        plt.savefig("log_likelihood_convergence_plot.png")
