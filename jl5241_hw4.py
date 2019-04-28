import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse.linalg as sp_linalg


####Load data
df = pd.read_csv('data/hw4_data/CFB2018_scores.csv', header = None)
teams = pd.read_csv('data/hw4_data/TeamNames.txt', header = None)

teams.columns = ['team_name']

df.columns = ['a_idx','a_pts','b_idx','b_pts']

M = np.zeros((767,767))

for row in df.iterrows():
    row= row[1]
    team_a = row['a_idx'] -1 
    team_b = row['b_idx'] -1 
    a_pts = row['a_pts']
    b_pts = row['b_pts']
    a_ind = 0
    b_ind =0 
    if a_pts > b_pts:
        a_ind = 1 
    else:
        b_ind =1 
    M[team_a][team_a] = M[team_a][team_a] + a_ind + a_pts/(a_pts+b_pts)
    M[team_b][team_b] = M[team_b][team_b] + b_ind + b_pts/(a_pts+b_pts)
    M[team_a][team_b] = M[team_a][team_b] + b_ind + b_pts/(a_pts+b_pts)
    M[team_b][team_a] = M[team_b][team_a] + a_ind + a_pts/(a_pts+b_pts)   
    

r_sums = M.sum(axis=1)
M= M / r_sums[:,np.newaxis]

wo = np.full((1,767),1/767)

eigval, eigvec = sp_linalg.eigs(M.T,k=1,which="LM")
ev1 = eigvec[:,0]
winf = ev1/ev1.sum()

###Run algorithm
wt = wo
out = []
norm_out = []
for i in range(10000):
    wt = np.matmul(wt,M)
    norm_out.append(np.linalg.norm((winf-wt), ord =1))
    if i in [9,99,999,9999]:
        rank = teams.iloc[np.argsort(-wt)[0]]
        out.append([wt,rank])

def create_report(out, which):
    df = pd.concat([out[which][1][0:25]['team_name'].reset_index(drop=True),pd.Series(out[which][0][:,:25][0])],axis=1)
    df.columns = ['team_name','w']
    print(df)


###Create reports
create_report(out,0)

create_report(out,1)

create_report(out,2)

create_report(out,3)


###PLot l1 norm
plt.plot(norm_out)
plt.ylabel('l1_norm')
plt.xlabel('iteration')
plt.title('l1_norm of Wt vs Winf after Each Iteration')
plt.show()



### PROBLEM 2 ####
X = np.zeros((3012, 8447))


##Make X Matrix
i = 0
f = open('data/hw4_data/nyt_data.txt')
for line in f.readlines():
    for w in line.split(','):
        cnt = w.split(':')
        X[int(cnt[0])-1][i] = int(cnt[1])
    i+=1

##Vocabs
j =  0
vocabs=[]
f = open('data/hw4_data/nyt_vocab.dat')
for line in f.readlines():
    vocabs.append(line.split('\n')[0])

##Scale
def normalize(v):
    return v / np.sum(v)

W = np.random.rand(3012,25) + 1
H = np.random.rand(25, 8447) + 1
out_arr = []
for i in range(100):
    
    ##Update H
    purp = X / (W.dot(H) + 1e-100)
    pink = np.apply_along_axis(normalize, 1, W.T)
    H *= pink.dot(purp)
    
    ##Update W 
    purp = X / (W.dot(H) + 1e-100)
    teal = np.apply_along_axis(normalize, 0, H.T)
    W *= purp.dot(teal)
    
    ##Calc Divergence
    out_arr.append(-np.sum(X * np.log(W.dot(H)+ 1e-100) - W.dot(H)))

###PLOT
plt.plot(out_arr)
plt.ylabel('Divergence')
plt.xlabel('Iteration')
plt.title('Divergence vs Iteration')
plt.show()

W_norm = np.apply_along_axis(scale, 0, W+ 1e-10)


###Compile report
report = []
for column in W_norm.T:
    top25= (-column).argsort()[:10]
    arr_out = []
    for word in top25:
        arr_out.append((vocabs[word] +" " "%.4f" % round(column[word],4)))
    report.append(arr_out)

matrix = [[0 for i in range(5)] for i in range(5)]

for idx,arr in enumerate(report):
    matrix[int(np.floor(idx/5))][idx%5] = arr


##Generate Lists
for i in matrix[4][0]:
    print(i)