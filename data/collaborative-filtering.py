import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv('data/hw3-data/Prob3_ratings.csv', header= None)
df.columns = ['uid','movie_id','rating']
M= np.zeros((df['uid'].max(),df['movie_id'].max()))

for row in df.iterrows():
    M[int(row[1]['uid']-1), int(row[1]['movie_id']-1)] = row[1]['rating']

U = np.random.normal(0, 1,df.uid.max())
V = np.random.normal(0, 1,df.movie_id.max())


def matrix_factorization(M,U,V, K, steps=100, alpha=0.0002, beta=0.02):
    V = V.T
    for step in range(steps):
        for i in range(len(M)):
            for j in range(len(M[i])):
                if M[i][j] > 0:
                    eij = M[i][j] - np.dot(U[i,:],V[:,j])
                    for k in range(K):
                        U[i][k] = U[i][k] + alpha * (2 * eij * V[k][j] - beta * U[i][k])
                        V[k][j] = V[k][j] + alpha * (2 * eij * U[i][k] - beta * V[k][j])
        eM = np.dot(U,V)
        e = 0
        for i in range(len(M)):
            for j in range(len(M[i])):
                if M[i][j] > 0:
                    e = e + pow(M[i][j] - np.dot(U[i,:],V[:,j]), 2)
                    for k in range(K):
                        e = e + (beta/2) * (pow(U[i][k],2) + pow(V[k][j],2))
        if e < 0.001:
            break
    return U, V.T


nP, nQ = matrix_factorization(M, U, V, 10)
nR = np.dot(nP, nQ.T)
