% A demo for Structured L-BFGS (sLBFGS) method to solve an unconstrained optimization 
% problem of the form
% minimize_x D(x) + S(x) 
% where D is a data-fidelity and S a regularization term.
% sLBFGS assume that the Hessian of S is known.

% setup a quadratic problem
n = 100;
alpha = 1e-2; % regularization parameter
c = ones(n,1);
fctn = @(x) objfun(x, c, alpha);

% initial estimate
y0 = zeros(n,1);

% Hessian initialization methods 
% (unstructured-Hy, structured-scalar, structured-diagonal)
initM = {'Hy','adap','diag-dg-cu-bz'};

% minimize objective function
His = {};
for k = 1:numel(initM)
    [yc,His{k}] = slBFGS(fctn,y0,'initMethod',initM{k},'xref',c);
end

%% display convergence plots
fig = figure; 

for k = 1:numel(initM)
    subplot(1,3,1); semilogy(His{k}.his(2:end,8)); hold on; legend(initM); xlabel('iterations'); title('||x_k-x*||'); grid minor;
    subplot(1,3,2); semilogy(His{k}.his(2:end,2)); hold on; legend(initM); xlabel('iterations'); title('J(x_k)'); grid minor;
    subplot(1,3,3); semilogy(His{k}.his(2:end,4)); hold on; legend(initM); xlabel('iterations'); title('||\nabla J(x_k)||'); grid minor;
end

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% quadratic function
function [Jc,dJ,dD,mf_d2S] = objfun(x, c, alpha)

n = numel(x);

% laplacian matrix
d2S = alpha*full(spdiags(ones(n,1)*[-1,2,-1],-1:1,n,n));

% data-fidelity Hessian matrix
d2D = diag(exp(-(1:n)));

% objective function
Jc = 0.5*(x - c)'*(d2D + d2S)*(x-c);

% gradients
dD = d2D*(x-c);
dS = d2S*(x-c);
dJ = dD + dS; dJ = dJ';

% in-line function that calculates d2S*y 
mf_d2S = @(y) d2S*y;
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
