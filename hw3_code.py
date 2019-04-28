import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from numpy.linalg import inv
%matplotlib inline
from scipy.stats import multivariate_normal as mvn

mu1 = [0,0]
cov1 = [[1,0],[0,1]]
mu2 = [3,0]
cov2 = [[1,0],[0,1]]
mu3 = [0,3]
cov3 = [[1,0],[0,1]]
pi = [.2,.5,.3]

n= []
for i in range(500):
    g = np.random.choice(np.arange(1,4), p = pi)
    if g ==1: 
        mu,var=mu1,cov1
    elif g==2:
        mu,var = mu2,cov2
    else:
        mu,var = mu3,cov3
    n.extend(np.random.multivariate_normal(mu, var, 1))                       

n=np.array([np.array(xi) for xi in n])

def kmeans(n,k, plotit = False, f= None, idx = None):
    init_u = n[np.random.randint(500, size =k),:]
    dists = {}
    cost_arr = []
    c= np.zeros(500)
    for i in range(20):
        for j in range(k):
            dists[j] = np.sqrt(((n - init_u[j])**2).sum(axis=1))
        for index,val in np.ndenumerate(c):
            c[index] = 0
            tmp= dists[0][index]
            for j in range(k):
                if j ==0: continue
                if tmp > dists[j][index]:
                    c[index] = j
                    tmp= dists[j][index]
        for j in range(k):
            init_u[j] = n[np.where(c==j)].mean(axis=0)
        cost =0
        for j in range(k):
            cost += dists[j][np.where(c==j)].sum()
        cost_arr.append(cost)
    if plotit: 
        plt.plot(cost_arr, label = "K = " + str(k))
        plt.xlabel("Iteration Step")
        plt.legend()
    
    return c
        
####PLOTTING FOR 1.1 
f = plt.figure(figsize=(5,7))
v = 0 
for k in [2,3,4,5]:
    kmeans(n,k,plotit=True,f=f, idx =v)
    v+= 1
plt.xlabel("Iteration Step")
plt.ylabel("Cost, L2 Norm")
plt.title("Cost vs Iteration Step")


c3 = kmeans(n,3)
c5 = kmeans(n,5)

###1.b PLOTTING FOR k=3
plt.scatter(n[c3==0,0],n[c3==0,1],label = 'Cluster=0')
plt.scatter(n[c3==1,0],n[c3==1,1],label = 'Cluster=1')
plt.scatter(n[c3==2,0],n[c3==2,1],label = 'Cluster=2')
plt.legend()
plt.title('K=3 Cluster Assignment')

###1.b PLOTTING FOR k=5
plt.scatter(n[c5==0,0],n[c5==0,1],label = 'Cluster=0')
plt.scatter(n[c5==1,0],n[c5==1,1],label = 'Cluster=1')
plt.scatter(n[c5==2,0],n[c5==2,1],label = 'Cluster=2')
plt.scatter(n[c5==3,0],n[c5==3,1],label = 'Cluster=3')
plt.scatter(n[c5==4,0],n[c5==4,1],label = 'Cluster=4')
plt.legend()
plt.title('K=5 Cluster Assignment')


##########NUBMER TWO ###############
X_train = pd.read_csv("data/hw3-data/Prob2_Xtrain.csv", header=None)
X_test = pd.read_csv("data/hw3-data/Prob2_Xtest.csv", header=None)
y_train = pd.read_csv("data/hw3-data/Prob2_ytrain.csv", header=None)
y_test = pd.read_csv("data/hw3-data/Prob2_ytest.csv", header=None)

X_train0 = X_train[y_train[0]==0]
y_train0 = y_train[y_train[0]==0]
X_train1 = X_train[y_train[0]==1]
y_train1 = y_train[y_train[0]==1]

def normalize(x):
    return x / np.sum(x)

def em_gmm(X, pis, mus, cov, max_iter=30):
    rows= X.shape[0]
    k = len(pis)
    loglike = []
    for i in range(max_iter):
        
        ws = np.zeros((rows, k))
        for j in range(k):
            ws[:, j] = mvn(mus[j], cov[j]).pdf(X) * pis[j]
        ws = np.apply_along_axis(lambda x:x/np.sum(x), 1, ws)

        n_sum = np.sum(ws, axis=0)
        pis = n_sum/rows
        mus = np.transpose(np.dot(np.transpose(X), ws) / n_sum)

        for j in range(k):
            phis = ws[:,j]
            nk = n_sum[j]
            sigma = np.dot(((X - mus[j]).T * phis),(X - mus[j]))/ nk
            cov[j] = sigma

        tmp_ll=  0
        for pi, mu, sigma in zip(pis, mus, cov):
            tmp_ll += pi*mvn(mu, sigma).pdf(X)
        loglike.append(np.log(tmp_ll).sum())
        
    return (pis, mus, cov, loglike)


def predict(x, p, best0, best1, k):
    prob0 = 0
    prob1 = 0
    pis0 = best0[0]
    mus0 = best0[1]
    sigmas0 = best0[2]
    pis1 = best1[0]
    mus1 = best1[1]
    sigmas1 = best1[2]
    for i in range(k):
        prob1 +=  mvn(mus1[i], sigmas1[i]).pdf(x) * pis1[i] 
        prob0 +=  mvn(mus0[i], sigmas0[i]).pdf(x) * pis0[i]
    prob0 =  p *prob0
    prob1 =  (1 - p) * prob1
    if prob0 > prob1:
        return 0
    else:
        return 1

k = 3
X = X_train0
y = y_train0
mu = np.mean(X)
cov= np.cov(X, rowvar =False)
outs =[]
for i in range(10):  
    mus = mvn.rvs(mu, cov, size=k)
    covs = []
    for j in range(k):
        covs.append(cov.copy())
    pis = np.repeat(1/k, k)
    out = em_gmm(X, pis, mus, covs, max_iter=30)
    outs.append(out)
    plt.plot(range(5,31), out[3][4:], label= "Run"+str(i+1))
plt.legend()
plt.title("Objective Function for 10 Different Runs and 5-30 Iterations, Class = 0")
plt.xlabel("Iteration Step")
plt.ylabel("Log marginal Objective Function")

k = 3
X = X_train1
y = y_train1
mu = np.mean(X)
cov= np.cov(X, rowvar =False)
outs_1 =[]
for i in range(10):  
    mus = mvn.rvs(mu, cov, size=k)
    covs = []
    for j in range(k):
        covs.append(cov.copy())
    pis = np.repeat(1/k, k)
    out = em_gmm(X, pis, mus, covs, max_iter=30)
    outs_1.append(out)
    plt.plot(range(5,31), out[3][4:], label= "Run"+str(i+1))
plt.legend()
plt.title("Objective Function for 10 Different Runs and 5-30 Iterations, Class = 1")
plt.xlabel("Iteration Step")
plt.ylabel("Log marginal Objective Function")

best1 =  outs_1[1]
best0 =  outs[3]
p = y_train1.shape[0] / y_train.shape[0]
preds = X_test.apply(lambda x: predict(x, p, best0, best1, k), axis=1)
mat = [[0,0],[0,0]]
for i in range(len(preds)):
    pred = preds[i]
    actual = y_test[0].values[i]
    mat[pred][actual] += 1
print(mat[0], '\n', mat[1])
print("accuracy: ", (mat[0][0] + mat[1][1]) / 460)


### PART 2.B 

##k=1 
k = 1
X = X_train0
y = y_train0
mu = np.mean(X)
cov= np.cov(X, rowvar =False)
outs =[]
for i in range(10):  
    mus = mvn.rvs(mu, cov, size=k)
    mus = [mus]
    covs = []
    for j in range(k):
        covs.append(cov.copy())
    pis = np.repeat(1/k, k)
    out = em_gmm(X, pis, mus, covs, max_iter=30)
    outs.append(out)

k = 1
X = X_train1
y = y_train1
mu = np.mean(X)
cov= np.cov(X, rowvar =False)
outs_1 =[]
for i in range(10):  
    mus = mvn.rvs(mu, cov, size=k)
    mus = [mus]
    covs = []
    for j in range(k):
        covs.append(cov.copy())
    pis = np.repeat(1/k, k)
    out = em_gmm(X, pis, mus, covs, max_iter=30)
    outs_1.append(out)
    plt.plot(range(5,31), out[3][4:], label= "Run"+str(i+1))
plt.legend()
plt.title("Objective Function for 10 Different Runs and 5-30 Iterations, Class = 1")
plt.xlabel("Iteration Step")
plt.ylabel("Log marginal Objective Function")


best1 =  outs_1[0]
best0 =  outs[0]
p = y_train1.shape[0] / y_train.shape[0]
preds = X_test.apply(lambda x: predict(x, p, best0, best1, k), axis=1)
mat = [[0,0],[0,0]]
for i in range(len(preds)):
    pred = preds[i]
    actual = y_test[0].values[i]
    mat[pred][actual] += 1
print(mat[0], '\n', mat[1])
print("accuracy: ", (mat[0][0] + mat[1][1]) / 460)

### k =2



k = 2
X = X_train0
y = y_train0
mu = np.mean(X)
cov= np.cov(X, rowvar =False)
outs =[]
for i in range(10):  
    mus = mvn.rvs(mu, cov, size=k)
    covs = []
    for j in range(k):
        covs.append(cov.copy())
    pis = np.repeat(1/k, k)
    out = em_gmm(X, pis, mus, covs, max_iter=30)
    outs.append(out)
    plt.plot(range(5,31), out[3][4:], label= "Run"+str(i+1))
plt.legend()
plt.title("Objective Function for 10 Different Runs and 5-30 Iterations, Class = 1")
plt.xlabel("Iteration Step")
plt.ylabel("Log marginal Objective Function")

k = 2
X = X_train1
y = y_train1
mu = np.mean(X)
cov= np.cov(X, rowvar =False)
outs_1 =[]
for i in range(10):  
    mus = mvn.rvs(mu, cov, size=k)
    covs = []
    for j in range(k):
        covs.append(cov.copy())
    pis = np.repeat(1/k, k)
    out = em_gmm(X, pis, mus, covs, max_iter=30)
    outs_1.append(out)
    plt.plot(range(5,31), out[3][4:], label= "Run"+str(i+1))
plt.legend()
plt.title("Objective Function for 10 Different Runs and 5-30 Iterations, Class = 1")
plt.xlabel("Iteration Step")
plt.ylabel("Log marginal Objective Function")

best1 =  outs_1[8]
best0 =  outs[6]
p = y_train1.shape[0] / y_train.shape[0]
preds = X_test.apply(lambda x: predict(x, p, best0, best1, k), axis=1)
mat = [[0,0],[0,0]]
for i in range(len(preds)):
    pred = preds[i]
    actual = y_test[0].values[i]
    mat[pred][actual] += 1
print(mat[0], '\n', mat[1])
print("accuracy: ", (mat[0][0] + mat[1][1]) / 460)


#### K=3 

k = 3
X = X_train0
y = y_train0
mu = np.mean(X)
cov= np.cov(X, rowvar =False)
outs =[]
for i in range(10):  
    mus = mvn.rvs(mu, cov, size=k)
    covs = []
    for j in range(k):
        covs.append(cov.copy())
    pis = np.repeat(1/k, k)
    out = em_gmm(X, pis, mus, covs, max_iter=30)
    outs.append(out)
    plt.plot(range(5,31), out[3][4:], label= "Run"+str(i+1))
plt.legend()
plt.title("Objective Function for 10 Different Runs and 5-30 Iterations, Class = 1")
plt.xlabel("Iteration Step")
plt.ylabel("Log marginal Objective Function")

k = 3
X = X_train1
y = y_train1
mu = np.mean(X)
cov= np.cov(X, rowvar =False)
outs_1 =[]
for i in range(10):  
    mus = mvn.rvs(mu, cov, size=k)
    covs = []
    for j in range(k):
        covs.append(cov.copy())
    pis = np.repeat(1/k, k)
    out = em_gmm(X, pis, mus, covs, max_iter=30)
    outs_1.append(out)
    plt.plot(range(5,31), out[3][4:], label= "Run"+str(i+1))
plt.legend()
plt.title("Objective Function for 10 Different Runs and 5-30 Iterations, Class = 1")
plt.xlabel("Iteration Step")
plt.ylabel("Log marginal Objective Function")


best1 =  outs_1[7]
best0 =  outs[2]
p = y_train1.shape[0] / y_train.shape[0]
preds = X_test.apply(lambda x: predict(x, p, best0, best1, k), axis=1)
mat = [[0,0],[0,0]]
for i in range(len(preds)):
    pred = preds[i]
    actual = y_test[0].values[i]
    mat[pred][actual] += 1
print(mat[0], '\n', mat[1])
print("accuracy: ", (mat[0][0] + mat[1][1]) / 460)


### k= 4

k = 4
X = X_train0
y = y_train0
mu = np.mean(X)
cov= np.cov(X, rowvar =False)
outs =[]
for i in range(10):  
    mus = mvn.rvs(mu, cov, size=k)
    covs = []
    for j in range(k):
        covs.append(cov.copy())
    pis = np.repeat(1/k, k)
    out = em_gmm(X, pis, mus, covs, max_iter=30)
    outs.append(out)
    plt.plot(range(5,31), out[3][4:], label= "Run"+str(i+1))
plt.legend()
plt.title("Objective Function for 10 Different Runs and 5-30 Iterations, Class = 1")
plt.xlabel("Iteration Step")
plt.ylabel("Log marginal Objective Function")


k = 4
X = X_train1
y = y_train1
mu = np.mean(X)
cov= np.cov(X, rowvar =False)
outs_1 =[]
for i in range(10):  
    mus = mvn.rvs(mu, cov, size=k)
    covs = []
    for j in range(k):
        covs.append(cov.copy())
    pis = np.repeat(1/k, k)
    out = em_gmm(X, pis, mus, covs, max_iter=30)
    outs_1.append(out)
    plt.plot(range(5,31), out[3][4:], label= "Run"+str(i+1))
plt.legend()
plt.title("Objective Function for 10 Different Runs and 5-30 Iterations, Class = 1")
plt.xlabel("Iteration Step")
plt.ylabel("Log marginal Objective Function")


best1 =  outs_1[8]
best0 =  outs[6]
p = y_train1.shape[0] / y_train.shape[0]
preds = X_test.apply(lambda x: predict(x, p, best0, best1, k), axis=1)
mat = [[0,0],[0,0]]
for i in range(len(preds)):
    pred = preds[i]
    actual = y_test[0].values[i]
    mat[pred][actual] += 1
print(mat[0], '\n', mat[1])
print("accuracy: ", (mat[0][0] + mat[1][1]) / 460)



### PROBLEM 3####


### 3.a)

df = pd.read_csv('data/hw3-data/Prob3_ratings.csv', header= None)
df_test = pd.read_csv('data/hw3-data/Prob3_ratings_test.csv', header= None)
df.columns = ['uid','movie_id','rating']
df_test.columns = ['uid','movie_id','rating']
M= np.zeros((df['uid'].max(),df['movie_id'].max()))
M_test = np.zeros((df['uid'].max(),df['movie_id'].max()))

for row in df.iterrows():
    M[int(row[1]['uid']-1), int(row[1]['movie_id']-1)] = row[1]['rating']

for row in df_test.iterrows():
    M_test[int(row[1]['uid']-1), int(row[1]['movie_id']-1)] = row[1]['rating']

test_idx = np.where(M_test!=0)
train_idx = np.where(M!=0)
n_test = len(M[test_idx])
n_train = len(M[train_idx])

rmse_test_out= []
rmse_train_out = []
obj_out = [] 
U_out = []
V_out = []

for runs in range(10):
    obj_arr = [] 
    rmse_test_arr = []
    rmse_train_arr = []
    U= np.random.multivariate_normal(np.zeros(10),np.identity(10),df['uid'].max())
    V= np.random.multivariate_normal(np.zeros(10),np.identity(10),df['movie_id'].max())
    for iteration in range(100):
        for ui in range(M.shape[0]):
            idx = np.where(M[ui] != 0)
            out= np.zeros((10,10))
            outm = np.zeros((10,1))
            for i in idx[0]:
                ro = V[i].reshape(10,1)
                out += ro.dot(ro.T)
                outm += M[ui][i]*ro
            U[ui] = inv(np.identity(10)*.25 + out).dot(outm).flatten()
            l2 = ro.dot(ro.T)
        for vj in range(M.shape[1]):
            idx = np.where(M[:,vj] != 0)
            out= np.zeros((10,10))
            outm = np.zeros((10,1))
            for i in idx[0]:
                ro = U[i].reshape(10,1)
                out += ro.dot(ro.T)
                outm += M[:,vj][i]*ro
            V[vj] = inv(np.identity(10)*.25 + out).dot(outm).flatten()
        
        L1 = 0
        L2 = 0
        L3 = 0
        for i,j in zip(train_idx[0],train_idx[1]):
            L1 += (M[i,j] - U[i].dot(V[j]))**2
        for i in range(M.shape[0]):
            L2 += U[i].dot(U[i])
        for j in range(M.shape[1]):
            L3 += V[j].dot(V[j])
        
        obj = -2*L1 -1/2*L2 - 1/2*L3 
        
        preds = U.dot(V.T)
        rmse_test = np.sqrt(((M_test[test_idx]- preds[test_idx])**2).sum()/n_test)
        rmse_train = np.sqrt(((M[train_idx]- preds[train_idx])**2).sum()/n_train)
        rmse_test_arr.append(rmse_test)
        rmse_train_arr.append(rmse_train)
        
        if iteration>0:
            obj_arr.append(obj)
    U_out.append(U)
    V_out.append(V)
    rmse_test_out.append(rmse_test_arr)
    rmse_train_out.append(rmse_train_arr)
    obj_out.append(obj_arr)

###PLOTS
for idx, val in enumerate(obj_out):
    plt.plot(range(1,100), obj_out[idx], label= "Run: "+ str(idx+1))
plt.title("Log Joint Likelihood for 10 trials across iterations 2-100")
plt.xlabel("Iteration Step")
plt.ylabel("Log Joint Likelihood")
plt.legend()


test_rmse_fin =[]
obj_fn_fin = []
for t,o in zip(rmse_test_out, obj_out):
    test_rmse_fin.append(t[99])
    obj_fn_fin.append(o[98])

report_obj = pd.DataFrame(
    { 
     'joint_log_likelihood': obj_fn_fin,
     'test_set_RMSE': test_rmse_fin
    })

report_obj.index=(range(1,11))

report_obj.sort_values(by = ['joint_log_likelihood'],ascending=False)

with open('data/hw3-data/Prob3_movies.txt') as f:
    lines = f.readlines()

V=V_out[9]

v_star = V[49]
v_my_fair = V[484]
v_good = V[181]
dist_star=[]
dist_fair=[]
dist_good = []
for v in V:
    dist_star.append(np.sqrt(((v - v_star)**2).sum()))
    dist_fair.append(np.sqrt(((v - v_my_fair)**2).sum()))
    dist_good.append(np.sqrt(((v - v_good)**2).sum()))

star_idx=np.argsort(dist_star)[:11]
fair_idx= np.argsort(dist_fair)[:11]
good_idx= np.argsort(dist_good)[:11]

report = pd.DataFrame(
    {'star_wars_nearest_movies': [lines[i] for i in star_idx if i >0],
     'star_wars_distances': [dist_star[i] for i in star_idx if i >0],
     'my_fair_lady_nearest_movies': [lines[i] for i in fair_idx if i >0],
     'my_fair_lady_distances': [dist_fair[i] for i in fair_idx if i >0],
     'good_fellas_nearest_movies': [lines[i] for i in good_idx if i >0],
     'good_fellas_distances': [dist_good[i] for i in good_idx if i >0]
    })

report=report[['star_wars_nearest_movies','star_wars_distances','my_fair_lady_nearest_movies','my_fair_lady_distances','good_fellas_nearest_movies','good_fellas_distances']]
report[report.index>0]


