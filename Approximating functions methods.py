#This file contains different methods for approximating a function
#Methods:
#1.Interpolating polynomial with equally spaced points sampled
#2.Interpolating polynomial with chebyshev points sampled
#3.Cubic spline
##################################



##################################

#Import required libraries
from pylab import *
from sympy import solve, Matrix, diff
from sympy.abc import z
import sympy as sy
import numpy as np
from math import *

##################################
###functions###
##################################

#Define a function to find the coefficients a_k
def inter_poly_coeff(x, n , g):#takes parameters x(vector of points at which g is sampled at), n, g (a function)
    M = np.array([])#creates an empty array M, this will store x_i to a power in order as seen in notes
    b = np.array([])#creates an empty array b to store value of g at x_i
    for i in range(n+1):#loops n+1 times to account for the polynomial having degree n, hence sstarting at 0 to n+1
        b = np.append(b, g(x[i]))#add value of g at x_i to b
        row = np.array([[]])#create an empty array row to store values of x_i at different exponents
        for j in range(n+1):#loop as before to find the values of x_i to different exponents from 0 to n
            row = np.append(row, (x[i])**(n-j))#add value of x_i at certain power to row, note (n-j) to account for order in matrix 
        M = np.append(M, row, axis=0)#Add row to the matrix M
    M = np.reshape(M, (n+1,n+1))#reshape the matrix M to a (n+1)*(n+1) as is flattened in loop above 
    return(np.dot(np.linalg.inv(M), b))#return (M**-1)*b which gives the coefficients of the interpolating polynomial

#defines a function to find the coefficients of the splines
def cubic_spline_coeff(x, n, g, h):
    b = np.array([])#creates an empty array b to store required values at x_i
    for i in range(n+1):#loops n+1 times to account for the polynomial having degree n, hence starting at 0 to n+1
        if (i==0 or i==n):#if first or final entry :
            b = np.append(b, (1/(h**3))*g(x[i]))#appends the value of b at i ==0 and i==n
        else:#for everything not first or last entry 
            b = np.append(b, (1/(h**3))*(6*g(x[i])))#appends the new value of b to the array
    x = np.dot(np.linalg.inv(cubic_matrix(n)), b)#calculates the coefficients ai for i = 0,1,...n
    a_min1 = [(2*x[0]-x[1])]#finds the a_-1 coefficient 
    a_n1 = [2*(x[n]-x[n-1])]#finds the a_n+1 coefficient 
    x = np.append(x, a_n1)#appends a_n+1 to coefficients array
    x = np.insert(x, 0, a_min1)#append a_-1 to coefficients array
    return(x)#return coefficients     

#function which acts as the interpolating polynomial 
def inter_poly(n, coeff, x):#takes parameters degree of polynomial, coefficients of polynomial, x(point at which polynomial is evaluated at)
    p = 0#creates a variable p, will act as the polynomial value at the given value of x
    for i in range(n+1):#loops n+1 times to account for a polynomial degree n having n+1 terms
        p = p + (coeff[n-i]*(x**i))#enumerate the ith value of x and times by coefficient and add to p
    return(p)#returns the value of the polynomial at the inputted value of x

#function which acts as the cubic spline
def cubic_spline(n, coeff, x, h, a):
    s = 0#creates dummy variable to store the value of the spline at
    for i in range(n+3):#loops for all the coefficients of the splines
        s = s + (coeff[i]*(B_0((x-((i-1)*h)), h, a)))#calculates a_i*B_i
    return(s)#returns value of spline at x

#function acts as the cubic B-spline B_0
def B_0(x, h, a):
    #for different values of x changes the contribution from each cubic spline base
    if (((x-a)<=-(2*h)) or ((x-a)>=(2*h))):
        return(0)
    elif(-(2*h)<=(x-a)<=-h):
        return((1/6)*(((2*h)+(x-a))**3))
    elif(-h<=(x-a)<=0):
        return((2*(h**3)/3)-((1/2)*((x-a)**2)*((2*h)+(x-a))))
    elif(0<=(x-a)<=h):
        return((2*(h**3)/3)-((1/2)*((x-a)**2)*((2*h)-(x-a))))
    elif(h<=(x-a)<=(2*h)):
        return((1/6)*(((2*h)-(x-a))**3))

#function used to print the polynomial found
def polyprint(n, coeff):
    poly = str()#creates an empty str which will store the polynomial 
    for i in range(n+1):#loops through all the interger exponentials of x upto x^n
        if i==0 and coeff[n-i] == 0:#
            poly = poly
        elif i == 0:
            poly = poly+str(coeff[n-i])
        else:
            poly = poly+"+"+str("("+str(coeff[n-i])+"*x^"+str(i)+")")
    return(poly)#returns the complete polynomial string 

#function to find chebyshev zeros
def chebyshev(n, a, b):#takes degree of polynomial and start and end points as parameters
    x = np.array([])#creates an enmpty array to stores points in 
    for i in range(n+1):#loops through n+1 times to find the n+1 points
        x = np.append(x, ((a+b)/2)+(((b-a)/2)*cos((((2*i)+1)/(2*(n+1)))*pi)))#finds the chebyshev nodes, adjusted for start and end condition
    return(x)#returns chebyshev nodes

#calculates the matrix for use with the cubic splines
def cubic_matrix(n):
    M = np.array([])
    n = n+1
    row = np.array([])
    for i in range(n):
        row = np.array([])
        if (i==0):
            row = np.append([1], zeros(n-1))
        elif (i==(n-1)):
            row = np.append(zeros(n-1), [1])
        else:
            row = np.append(zeros(i-1), np.append(np.array([1,4,1]),zeros(n-(i+2))))
        M = np.append(M, row)
    M = np.reshape(M, (n,n))    
    return(M)

#finds the maximum products for use in the maximum theoreitcal error
#finds the max of (x-x_0)(x-x_1)...(x-x_n)
def max_prod(pts, x):
    products = ([])#creates a variable to store all the possible products in
    for i in range(size(x)):#loop for all points in x 
        product = 1#creates a variable product which will store the product at a value of x, note = 1 so product can be found
        for j in range(size(pts)):#loop for the number of points sampled
            product = product*(x[i]-pts[j])#find the value of the bracket(x-x_i)
        products = append(products, abs(product))#appends the product at a certain value of x to all the products found
    return(max(products))

#finds the maximum of the |nth derrivative of a function| (=f^(n)=d^n(f(x))/dx^n) over a linearspace of points x
def max_fn(f, x, n):
    f_n = f(z).diff(z, n)
    results = ([])
    for i in range(size(x)):
        results = append(results, abs(f_n.evalf(subs={z:x[i]})))
    return(max(results))

def max_theoretical_error(f, x, n, pts):
    return(max_prod(pts,x)*max_fn(f,x,n+1)*(1/factorial(n+1)))        

#function which calculates the actual maximum error for a given interpolating polynomial(poly)
def max_error(poly, func, x, n, coeff):
    xindex = 0#creates a varibale to index, used to say which value of x yeilds maximum error
    error = 0#creates a variable which will store the maximum error, set to 0 as all point will have an error of at least 0
    for i in range(size(x)):#loop for all points in the linspace x
        abs_error = abs(func(x[i])-poly(n, coeff, x[i]))#calculates the absolute error at a point x
        if abs_error > error:#if the error found is larger than the largest error of all points before 
            error = abs_error#set new error found to be the largest error
            xindex = i #changes xindex to the new index for the value of x associated to the current largest error
        else:
            error = error#max error remains the same
    return(x[xindex], error)#returns the value of x which leads the maximum error and that maximum error in a coordinate form
