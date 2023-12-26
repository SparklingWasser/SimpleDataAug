
import numpy as np

def noisy(noise_type, X, var_par=0):
    
    if noise_type == "gauss":
        
        mean = 0
        var = 10**var_par
        sigma = var**0.5
        gauss = np.random.normal(mean, sigma, X.shape)
        noisy = X + gauss
        return noisy
    
    elif noise_type == "s&p":
        s_vs_p = 0.5    # ratio between salt and pepper
        amount = 0.1    # ratio of data to be added by noise
        out = np.copy(X)    
        # Salt mode
        num_salt = np.ceil(amount * X.size * s_vs_p)
        coords = [np.random.randint(0, i - 1, int(num_salt)) for i in X.shape]
        out[coords] = 1     # salt noise

        # Pepper mode (same as in salt mode)
        num_pepper = np.ceil(amount* X.size * (1. - s_vs_p))
        coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in X.shape]
        out[coords] = 0     # pepper noise
        return out
    
    elif noise_type =="speckle":
        gauss = np.random.randn(*X.shape) 
        noisy = X * gauss + gauss         # speckle model = original data*multiplicative noise + additive noise
        
        return noisy
    