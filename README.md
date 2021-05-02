# Non-convex Exact Community Recovery in Stochastic Block Model
This folder contains the MATLAB source codes for the implementation of all the experiments in the paper

"A Nearly-Linear Time Algorithm for Exact Community Recovery in Stochastic Block Model" (ICML 2020)
By Peng Wang, Zirui Zhou, Anthony Man-Cho So.

* Contact: Peng Wang
* If you have any questions, please feel free to contact "wp19940121@gmail.com".

=============================================================================

This package contains 2 experimental tests to output the results in the paper:

* In the folder named phase-transition, we conduct the expriment of phase transition to test recovery performance of our approach GPM and compare it with SDP-based approach in Amini et al. (2018), the manifold optimization (MFO) based approach in Bandeira et al. (2016), and the spectral clustering (SC) approach in Abbe et al. (2017).
  - phase_transition.m: Output the recovery performance and running time of above methods
  - GPM.m: Implement our approach by PM + GPM
  - manifold_GD: Implement MFO based approach by manifold gradient descent (MGD) in Bandeira et al. (2016)
  - sdp_admm1: Implement SDP-based approach by alternating direction method of multipliers (ADMM) in Amini et al. (2018)
  - SDP_solver.m: Implement SDP-based approach by CVX in Hajek et al. (2016) 

* In the folder named convergence-performance, we conduct the experiments of convergence performance to test the number of iterations needed by our approach
GPM to exactly identify the underlying communities. For comparison, we also test the convergence performance of MGD, which is an iterative algorithm that has similar periteration
cost to our method.
  - convergence_rate_iterdist.m: Output the convergence performance of our methods and MGD
