# SLBFGS
A structured L-BFGS method for inverse problems


Many inverse problems are phrased as unconstrained optimization problems of the form  
$$J(x) = D(x) + S(x)$$
where $D$ represents a data-fidelity term and $S$ a regularizer.

Often, the Hessian of the fidelity term is computationally unavailable, while the Hessian of the regularizer is available and allows for cheap
matrix-vector products. We propose an L-BFGS method that takes advantage of this structure.

Numerical results show that the new method outperforms other structured L-BFGS methods and classical L-BFGS on non-convex real-life problems from medical image
registration.

