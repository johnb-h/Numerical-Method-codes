#This file contains different iterative methods for system of linear equations
#1.Jacobi Iterative Method
#2.Gaus-seidel Iterative Method
#3.Successive Over Relaxation Method

#import required libraries
from matplotlib.pylab import *
from pylab import *
import numpy as np
from math import *

#Note SR stands for successive relaxation 
###################################
###Iteration matrices and methods###
###################################

#Jacobi Iterative method

#Iteration formula
def Jacobi(x, B_J, B, D):
    return(np.dot(B_J, x)+np.dot(np.linalg.inv(D), B))#returns next iteration using formula
    
#Jacobi matrix
def B_J(D, L, U):
    return(np.dot(np.linalg.inv(D),(L+U)))
    
###################################

#Successive relaxation method
#note setting omega = 1 yeilds Gauss-Siedel Method

#Iteration formula
def B_SR(D, L, U, omega):
    return(np.dot(np.linalg.inv(D-(omega*L)),((1-omega)*D)+(omega*U)))

#SR matrix
def SR(x, B, D, L, B_SR, omega):
    return(np.dot(B_SR, x)+np.dot(np.linalg.inv(D-(omega*L)),omega*B))

###################################
#Functions used in analysis of program 
###################################

#Spectral Radius function
def spec_rad(B, JAC):#Inputs a matrix as parameter and JAC if wanting eigenvalues of Jacobian outputted
    eigs = np.linalg.eigvals(B)#calculates all eigenvalues and stores in an array eigs
    if JAC == 1:
        print("\nEigenvalues for the Jacobi matrix are:",eigs)
    k = 0#creates a dummy variable k for indexing        
    while k < np.size(eigs):#loops for the size of the array eigs
        eigs[k] = abs(eigs[k])#calculates the absolute of all eigenvalues
        k = k+1#increases k by 1 so can move onto next element of array
    return(abs(max(eigs)))#returns the max absolute eigenvalue, ie the spectral radius

#Infinity norm for ||v1-v2||
def error(v1, v2, dim):#takes parameters of 2 vectors and the dimension of the vectors
    j = 0#dummy variable to index elements
    err_arr = np.array([])#creates an empty array 
    while j < dim:#loops for the size of the vecotrs so all elements are compared 
        err_arr = np.append(err_arr, abs(v1[j]-v2[j]))#calcualtes the difference between two corresponding elements
        j = j+1#increase index by 1 to move to next element of the vectors
    return(max(err_arr))#finds the infinity norm by taking max of err_arr 

#Consistently ordered tester (To prove NOT consistently ordered if suspected)
def independance(D,L,U):
    a = 1#a = 1 is Jacobi iteration matrix
    C1 = np.array(a*(np.dot(np.linalg.inv(D),L))+((1/a)*np.dot(np.linalg.inv(D),U)))#first matrix to find eigvals from
    a = 100
    C2 = np.array(a*(np.dot(np.linalg.inv(D),L))+((1/a)*np.dot(np.linalg.inv(D),U)))#first matrix to find eigvals from
    print("Eigenvalues to test if A is consistently ordered: \n","For a value of alpha = 1(Jacobi), eigenvalues are: ",set(eigvals(C1)), "\n\nFor a value of alpha = 100, eigenvalues are: \n",set(eigvals(C2)))#output for use in analysis
    
#finds numerically the optimum value of omega to minimize the spectral radius(fastest rate of convergence )
def omegafind(D,L,U):
    #To find the optimum omega(the one yeilding smallest spectral radius)
    n = 1000#number of different omega values to chose from 
    omega = np.linspace(0.001, 1.999, n)#creates a linear space of omegas, note the range is(0,2) since diagonal is non 0 so omega must be between 0 and 2(see notes)
    specrads = zeros(n)#creates an array of 0 called specrads to store the calculated spectral radius' in later
    for i in range(n):#loops through all the omega values
        specrads[i] = spec_rad( B_SR(D, L, U, omega[i]),0)#calculates the spectral radius for a given successive relaxation matrix corresponding to a specfic value of omega
    
    #Graph plotted to be used in justification of value 
    fig1, ax1 = plt.subplots()
    ax1.plot(omega, specrads)#plots graph of spectral radius vs corresponding value of omega
    ax1.set_xlabel("Omega Value")#labels the x axis
    ax1.set_ylabel("Spectral radius")#labels the y axis
    ax1.set_title("Spectral radius for Successive Relaxation iterative matrix vs omega")#titles the graph
    
    omega_maxim = omega[np.argmin(specrads)]#assigs omega_maxim to the value of omega which yeilds the smallest spectral radius 
    return(omega_maxim)#returns the value of the optimised omega
