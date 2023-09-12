

% setup a quadratic problem
n = 100;
c = rand(n,1);
fctn = @(x) objfun(x, c);

% initial estimate
y0 = rand(n,1);

% minimize objective function
[yc,His] = slBFGS(fctn,y0,'initMethod','adap');

% compute performance matric
yerror = norm(yc - c)/norm(yc);
runtime = 1;


function [Jc,dJ,dD,d2S] = objfun(x, c)

para = []; Jc = []; dJ = []; d2S = [];
if isempty(x)
    Jc = 'test';
    return
end
n = numel(x);

d2S = 1e-3*eye(n);
D = diag(exp(-(1:n)));
Jc = 0.5*(x - c)'*(D + d2S)*(x-c);
dD = D*(x-c);
dS = d2S*(x-c);
dJ = dD + dS; dJ = dJ';
end


