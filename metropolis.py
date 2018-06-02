'''
A simple metropolis sampler class
'''

import numpy as np
np.random.seed(0)

def metropolis(n_samples,fun_calcLLK,fun_verify,data,m_ini,prior_bounds,prop_cov,LLK_data=False,verbose=False):

    ''' 
    Metropolis algorithm 
    Args:
        * n_samples: Number of samples
        * fun_calcLLK: Function to calculate Log-likelihood
        * fun_verify: Function to verify model
        * data: Data
        * m_ini: Initial model
        * prior_bounds: Parameter bounds (uniform priors)
        * prop_cov: Covariance of the (gaussian) proposal PDF
        * LLK_data: Use False (default) if the calcLLK function only returns log-likelihood values
                    Use True if the calcLLK function returns log-LLK + one extra parameter (returned at the end of metropolis)
        * verbose: (default: False) verbose mode
    '''

    # Initiate model chain
    M   = np.zeros((n_samples,m_ini.size))
    LLK = np.zeros((n_samples,))
    M[0,:] = m_ini # Initialize the chain
    if LLK_data:
        LLK[0],LLK_p = fun_calcLLK(m_ini,data)
        LLK_d = [LLK_p]
    else:
        LLK[0] = fun_calcLLK(m_ini,data) 

    # Proposal mean
    prop_mean = np.zeros(m_ini.shape)

    # Iteration loop
    count = 0 # Number of accepted models
    verbose_count = int(n_samples/10)
    for i in range(1,n_samples): 

        # Verbose
        if verbose and not i%verbose_count:
            print('%d%%'%(int(100*i/n_samples)))
        # Random walk
        Mnew = M[i-1,:].copy()
        Mnew += np.random.multivariate_normal(prop_mean,prop_cov)

        # Check if model sample is within prior
        valid = fun_verify(Mnew,prior_bounds)
        if not valid:
            M[i,:] = M[i-1,:].copy()
            LLK[i] = LLK[i-1]
            continue

        # LLK of new model        
        if LLK_data:
            logLLKn,LLK_p = fun_calcLLK(Mnew, data)
        else:
            logLLKn = fun_calcLLK(Mnew, data)        
        dLLK    = logLLKn - LLK[i-1]
        
        # Metropolis acceptance/rejection
        accept = 0
        U = np.log(np.random.rand())
        if U < dLLK:
            accept=1        
        else:
            accept=0

        # Update model
        if accept:
            M[i,:]  = Mnew.copy()
            LLK[i]  = logLLKn
            if LLK_data:
                LLK_d.append(LLK_p)
            count += 1
        else:
            M[i,:] = M[i-1,:].copy()
            LLK[i] = LLK[i-1]
            if LLK_data:            
                LLK_d.append(LLK_d[-1])

    if LLK_data:
        return M,LLK,LLK_d,count
    else:
        return M,LLK,count
