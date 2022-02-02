import numpy as np
import matplotlib.pyplot as plt
import scipy.stats

def inv_sigmoid(y):
    return np.log(y/(1-y))


def samplingCLT(mu = 0,sigma = 1):
    no_of_samples = 2000
    y_s = np.random.uniform(0,1,size = (no_of_samples,3))
    x_s = np.mean(inv_sigmoid(y_s),axis = 1)
    x_s = x_s*sigma
    # Checking for normal #
    #n_bins = 20
    #mu,sigma = scipy.stats.norm.fit(x_s)
    #print("Mean:",mu)
    #print("Sigma:",sigma)
    #_,bins,_ = plt.hist(x_s,bins = n_bins,density = 1, alpha = 0.5)
    #best_fit_curve = scipy.stats.norm.pdf(bins,mu,sigma)
    #plt.plot(bins,best_fit_curve)
    #plt.show()
    return x_s
    
def samplingNormal(mu = 0,sigma = 1):
    no_of_samples = 2000
    x_normal = np.random.normal(mu,sigma,size = no_of_samples)
    #n_bins = 20
    #mu,sigma = scipy.stats.norm.fit(x_normal)
    #print("Mean:",mu)
    #print("Sigma:",sigma)
    #_,bins,_ = plt.hist(x_normal,bins = n_bins,density = 1, alpha = 0.5)
    #best_fit_curve = scipy.stats.norm.pdf(bins,mu,sigma)
    #plt.plot(bins,best_fit_curve)
    #plt.show()
    return x_normal    



def samplingMVN(sigma_matrix,mu_matrix):
    lambda_,gamma_ = np.linalg.eig(sigma_matrix)
    dimensions = lambda_.shape[0]
    x_n_1d = samplingNormal()
    x_n_md = x_n_1d.reshape((-1,dimensions))
    x_n_md = (x_n_md*lambda_**0.5)@np.transpose(gamma_) + mu_matrix
    ## Verification ##
    mu_cap = np.mean(x_n_md,axis = 0)
    sigma_cap = np.transpose(x_n_md - mu_cap)@(x_n_md - mu_cap)
    sigma_cap = sigma_cap/(x_n_md.shape[0] - 1)
    print("Mu Cap:")
    print(mu_cap)
    print("Sigma Cap:")
    print(sigma_cap)
    return x_n_md
    




def samplingGMM(pi,sigma_1,mu_1,sigma_2,mu_2,sigma_3,mu_3,no_of_samples):
    
    freq_pi = np.zeros(pi.shape[0])
    for i in range(no_of_samples):
        p = np.random.uniform(0,1)
        if p < pi[0]:
            freq_pi[0] += 1
        elif p < pi[0] + pi[1]:
            freq_pi[1] += 1
        else:
            freq_pi[2] += 1
    print("Frequency Array:",freq_pi)
    x_n_GMM = np.zeros((no_of_samples,mu_1.shape[0]))
    x_n_md_1 = samplingMVN(sigma_1,mu_1)
    x_n_md_2 = samplingMVN(sigma_2,mu_2)
    x_n_md_3 = samplingMVN(sigma_3,mu_3)
    x_n_GMM[:int(freq_pi[0]),:] = x_n_md_1[:int(freq_pi[0]),:]
    x_n_GMM[int(freq_pi[0]):int(freq_pi[0] + freq_pi[1]),:] = x_n_md_2[:int(freq_pi[1]),:]
    x_n_GMM[int(freq_pi[0] + freq_pi[1]):no_of_samples,:] = x_n_md_3[:int(freq_pi[2]),:]
    
    return x_n_GMM

mu_1 = np.array([1, 12])
mu_2 = np.array([5, 0]) 
mu_3 = np.array([-8, 3])

sigma_1 = np.array([[3, 1], [1, 3]]) 
sigma_2 = np.array([[1, 0], [0, 1]]) 
sigma_3 = np.array([[10, 1], [1, 0.3]])

pi = np.array([0.3, 0.2, 0.5])

x_n_GMM = samplingGMM(pi,sigma_1,mu_1,sigma_2,mu_2,sigma_3,mu_3,1000)

print(x_n_GMM.shape)
x_axis_GMM = x_n_GMM[:,0]
y_axis_GMM = x_n_GMM[:,1]
print(x_axis_GMM.shape)
print(y_axis_GMM.shape)
plt.scatter(x_axis_GMM,y_axis_GMM)
plt.show()
    

##inititalize
def initializeParam(k):
    k = 3
    dimensions = 2
    pi_cap = np.zeros(k)
    cum_sum = 0
    for i in range(k):
       if i < k -1 :
            pi_cap[i] = np.random.uniform(0,1-cum_sum)
       else:
            pi_cap[i] = 1 - cum_sum
       cum_sum += pi_cap[i]
    print(pi_cap)
    
    
    mu_cap = np.array([[0.1,5,7],[2,3,10]])
    sigma_cap = np.zeros((k,dimensions,dimensions))
    identity_matrix = np.array([[1,0],[0,1]])
    for i in range(k):
        sigma_cap[i] = identity_matrix
        
    print(mu_cap)
    print(sigma_cap)
    
    return pi_cap,mu_cap,sigma_cap

def E_Step(x_s,k,mu_cap,sigma_cap,pi_cap):
    p_matrix = np.zeros((x_s.shape[0],k))
    r_matrix = np.zeros((x_s.shape[0],k))
    for j in range(p_matrix.shape[1]):
        mu = mu_cap[:,j]
        sigma = sigma_cap[j]
        for i in range(p_matrix.shape[0]):
            x = x_s[i]
            p_matrix[i,j] = scipy.stats.multivariate_normal.pdf(x,mean = mu, cov = sigma)*pi_cap[j]
    
    for i in range(r_matrix.shape[0]):
        conditional_sum = np.sum(p_matrix[i])
        for j in range(r_matrix.shape[1]):
            r_matrix[i,j] = p_matrix[i,j]/conditional_sum
    
    return r_matrix,p_matrix


def M_Step(x_s,k,mu_cap,sigma_cap,pi_cap,r_matrix):
    for j in range(k):
        sum_r_matrix_x_s = np.zeros(mu_cap.shape[0])
        sum_r_matrix = 0
        sum_r_matrix_xmu = np.zeros((mu_cap.shape[0],mu_cap.shape[0]))
        for i in range(x_s.shape[0]):
            sum_r_matrix_x_s += r_matrix[i,j]*x_s[i]
            sum_r_matrix += r_matrix[i,j]
        mu_cap[:,j] = sum_r_matrix_x_s/sum_r_matrix
        for i in range(x_s.shape[0]):
            sum_r_matrix_xmu += r_matrix[i,j] * ((x_s[i] - mu_cap[:,j]).reshape(2,1)@(x_s[i] - mu_cap[:,j]).reshape(1,2))
        sigma_cap[j] = sum_r_matrix_xmu/sum_r_matrix
        #print(sigma_cap[j])
        #print("Determinant of j:",np.Determinant(sigma_cap[j]))
        pi_cap[j] = sum_r_matrix/np.sum(r_matrix)
    
    return mu_cap,sigma_cap,pi_cap


def EMforGMM(x_s,k,max_iter):
    pi_cap,mu_cap,sigma_cap = initializeParam(k)
    for i in range(max_iter):
        #print("Iteration:",i)
        r_matrix,p_matrix = E_Step(x_s,k,mu_cap,sigma_cap,pi_cap)
        mu_cap,sigma_cap,pi_cap = M_Step(x_s,k,mu_cap,sigma_cap,pi_cap,r_matrix)
    print("Mu Cap:",mu_cap)
    print("Sigma Cap:",sigma_cap)
    print("Pi Cap:",pi_cap)

    return mu_cap,sigma_cap,pi_cap

print(x_n_GMM.shape)
mu_cap,sigma_cap,pi_cap = EMforGMM(x_n_GMM,3,20)






                    
    
    
    

