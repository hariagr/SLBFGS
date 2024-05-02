# SLBFGS
TULIP and ROSE, structured L-BFGS methods for inverse problems

Many inverse problems are phrased as unconstrained optimization problems of the form  
$$J(x) = D(x) + S(x)$$
where $D$ represents a data-fidelity term and $S$ a regularizer.

Often, the Hessian of the fidelity term is computationally expensive, while the Hessian of the regularizer is available and allows for cheap
matrix-vector products. We propose two L-BFGS methods that take advantage of this structure.

The proposed methods outperform other structured L-BFGS methods and classical L-BFGS on non-convex real-life problems from medical image
registration.

# Publications:
Mannel, F., Om Aggrawal, H., & Modersitzki, J. (2024). A structured L-BFGS method and its application to inverse problems. In Inverse Problems (Vol. 40, Issue 4, p. 045022). IOP Publishing. https://doi.org/10.1088/1361-6420/ad2c31
