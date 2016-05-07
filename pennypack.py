#!/usr/bin/env python
# -*- coding: utf-8 -*-

# using R.L. Grahan and N.J.A. Sloane's equations

# COMMENTS {{{1
# REVIEW/PREFACE {{{2
# Suppose the pennies have diameter d, let P₁,...,P₂ be their centers and
# P̄=n⁻¹ΣPᵢ is their centroid. Then the problem is to choose
# points P₁,...,P₂ so as to satisfy:
# (1)     ‖Pᵢ- Pⱼ‖ ≥ d    ;   for i,j=1...,n ; i≠j
#
# and so that the 'second moment':
# (2)     𝑈 = 1/d² Σ(i=1 to n)(‖Pᵢ- P̄‖²)
#                                   centroid P̄ = n⁻¹ΣPᵢ
# is minimized, where ‖ ‖ is the Euclidean distance.
#   Euclidean distance: ‖q-p‖ = √((q-p)*(q-p)) = √((q₁-p₁)²+(q₂-p₂)²)
#
# let 𝑈(n) be the minimal value of 𝑈
# a set of points 𝒫 ={P₁,...,Pₙ} satisfying (1) is 
# called an n-point packing, and is optimal if it attains 𝑈(n)

# Greedy Algorithm
# A sequence of packings 𝒫 ₁, 𝒫 ₂, 𝒫 ₃... is produced by the
# greedy algorithm if:
# (a) 𝒫 ₁ contains a single point, and
# (b) for n=2,3,...,
#     𝒫 ₙ = 𝒫 ₙ₋₁ ∪ {Pₙ}
#           minimizes over 𝑈 all choices of Pₙ satisfying (1)
#
# remember: greedy algorithms are where if you're minimizing cost,
#     the greedy algorithm chooses the element with the least cost,
#     and if you're trying to maximize a cost, it picks the element
#     with the greatest cost, per iteration.

# PLAN {{{2
# quick reference:
# (1)     ‖Pᵢ- Pⱼ‖ ≥ d    ;   for i,j=1...,n ; i≠j
#   Euclidean distance: ‖q-p‖ = √((q-p)*(q-p)) = √((q₁-p₁)²+(q₂-p₂)²)
# (2)     𝑈 = 1/d² Σ(i=1 to n)(‖Pᵢ- P̄‖²)
#                                   centroid P̄ = n⁻¹ΣPᵢ
#
# so.... plan of attack...
# 1) create the first packing, 𝒫 ₁, by choosing some point
#
# 2) create the subsequent packings for n=2,3,...
#     so for the next iteration n,
#       we must pick the next point Pₙ such that:
#
#       i ) Pₙ must satisfy (1)    (for all n really)
#       ii) the packing of n, 𝒫 ₙ = 𝒫 ₙ₋₁ ∪ {Pₙ},
#                 ie: 𝒫 ₙ is equal to the union of the existing
#                     packing, 𝒫 ₙ₋₁, union'd with the new point Pₙ
#             will minimize 𝑈 over all Pₙ already in the set
#                 ie: 𝑈 is the sum of all the euclidean distances from
#                     each point P to the centroid of the blob P̄,
#                     minimizing this means that we are adding a new
#                     point Pₙ such that the centroid P̄ does not move
#                     very far. iow, addings circles like a flower
#
# 3) repeat 2 until the boundary condition is met
#
# quick ver of plan
# must pick new point such that (1) that also minimizes (2)

#}}}1

# CODE {{{1

# HEADERS AND CONSTANTS {{{2
# import
import numpy as np
from scipy.optimize import minimize
import timeit


# CONSTANTS
ALLOCATEARRAYELEMENTS = 100000



# VARIABLES {{{2
# initial guess
x0 = np.array([5,5])
# diameter, keep constant
Ø = 5
# list of points, P₁,...,Pₙ
# preallocate space, as numpy arrays copy entire array when appending
#𝒫 = np.zeros(shape=(ALLOCATEARRAYELEMENTS,2))
𝒫 = np.zeros(shape=(5,2))
# n, the current point number, starts at 1!!! NOT ZERO INDEX!!!
n = 1
# the centroid of the packing
P̄ = np.array([0,0])

# FUNCTIONS {{{2
# centroid function
f_P̄ = lambda: 𝒫 [1:n].sum()/n

# optimization is where we try to find the best values for the
#   objective function, since we are trying to find the best Pₙ such
#   that (2) is minimal, (2) is the objective function
#   sometimes also called the cost fuction. The solution that produces
#   a minimum (or maximum) for the objective function is called the optimal solution
# note: I simplified it algebraically and with vectorizations
#   to greatly save time
def 𝑈 (newP):
  print('%-38.38s' % newP, end=''); print('    ', end='')
  global 𝒫 ,P̄   # must use global to modify
  𝒫 [n] = newP  # 'temporarily' add new point P to the list to compute
                #   ie: 𝒫 ₙ = 𝒫 ₙ₋₁ ∪ {Pₙ}
  # evaluate new centroid
  P̄=f_P̄()
  print('@: ', end='')
  print(P̄)
  print(𝒫 [1:n])
  return ((𝒫 [1:n]-P̄)**2).sum()/n**2
#U = lambda p, nn: ((p-P̄)**2).sum()/nn**2

# Euclidean distance function minus diameter, the function (1), check if ≥ 0
#D = lambda p, q: np.sqrt(((q-p)**2).sum())-Ø
def D(newP):
  print(' D', end='')
  print('%-36.36s' % newP, end=''); print('    ', end='')
  # iterate through the list of points 𝒫
  for p in 𝒫 [1:n-1]:
    #print(p)
    # check that the new point is not overlapping with any existing points
    if np.sqrt(((newP-p)**2).sum())-Ø < 0:
      # distance is less than diameter, return failure
      #print("NOPE")
      #print(𝒫 [1:n])
      print('!  ', end=''); print(np.sqrt(((newP-p)**2).sum())-Ø)
      return -1
  print('.')
  # no overlaps, return success
  return 0

def construct_jacobian(func, epsilon):
  def jac(x, *args):
    x0  = np.asfarray(x)
    f0  = np.atleast_1d(func(*((x0,)+args)))
    jac = np.zeros([len(x0),len(f0)])
    dx  = np.zeros(len(x0))
    for i in range(len(x0)):
      dx[i]   = epsilon
      jac[i]  = (func(*((x0+dx,)+args)) - f0)/epsilon
      dx[i]   = 0.0

    return jac.transpose()
  return jac

# OPTIMIZATION CONSTRAINTS {{{2
# the constraints
cons = ({'type': 'eq', 'fun': D})
# the bounds, new point must stay within the bounding shape
bnds = ((2, None), (2, None))


# MAIN {{{2


# evaluate new centroid
P̄=f_P̄()
# place first point, P₁
𝒫 [n] = np.array([0,0])


print(𝒫 [1:n])
print()

n += 1

for limit in range(3):
  print(n, end=''); print(', ', end=''); print(P̄)
  # find new point for maximum packing
  res = minimize(𝑈, x0, method='SLSQP', bounds=bnds, constraints=cons,
            jac=construct_jacobian(𝑈, 1e-4),
            options={'disp': True}) #, 'eps': 1e0})
  #res = minimize(𝑈, x0, method='SLSQP', bounds=bnds, constraints=cons,
  #          options={'disp': True}) #, 'eps': 1e0})
  #minimize(𝑈, np.array([1, 1]), method='SLSQP', bounds=bnds, constraints=cons)

  # increment n for finding next new point in the next iteration
  print(𝒫 [1:n]); print()
  n += 1

print("\nEND")
print(𝒫)
print()
print(res)


#}}}1


# notes to self...
# line 175, the prints, they aren't printing what I expect them to?
# try fix that and see where it takes you, the 𝒫 [1:n] business
# seem a bit off now
