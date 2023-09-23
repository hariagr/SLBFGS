%==============================================================================
%
% function [yc,His] = slBFGS(fctn,yc,varargin)
%
% structured limited BFGS optimizer
%
% Input:
% ------
%   fctn        function handle
%   yc          starting guess 
%   varargin    optional parameter, see below
%
% Output:
% -------
%   yc          numerical optimizer (current iterate)
%   His         iteration history
%
% Reference: 
% 
% For any questions, 
% contact: hariom85@gmail.com, mannel@mic.uni-luebeck.com
%==============================================================================

function [yc,His] = slBFGS(fctn,yc,varargin)

% set and overwrites default parameters
p = set_parser;
parse(p,varargin{:});
p = p.Results;

% cautious update parameters
cs = 1e-9;
c0 = 1e-6; c1 = 1e-6; c2 = 1;
C0 = 1/c0; C1 = 1/c1; C2 = 1/c2;

% -- initialize  ---------------------------------------------------------                                        
zBFGS = [];     % memory for BFGS gradient directions
sBFGS = [];     % memory for BFGS directions
nBFGS = 0;     % counter for number of limited BFGS directions
iter  = 0; 
LSiter = 0;
% evaluate objective function for starting values and plots
[Jc,dJ,dD,d2S] = fctn(yc); 
yold = 0*yc; Jold = Jc; dDold = dD; yref = p.xref;

% initial estimate of tau for initializing Hessian
H0 = d2S;
tau = norm(dD);

% for display iteration history
hisStr    = {'iter','J','Jold-J','|\nabla J|','|dy|','LS','tau','|dy*|'};
his        = zeros(p.maxIter+2,8);
his(1,1:3) = [-1,Jold,Jold-Jc];
his(2,:)   = [0,Jc,Jold-Jc,norm(dJ),norm(yc-yold),LSiter,tau,norm(yc-yref)];

% initial output
fprintf('%4s %-12s %-12s %-12s %-12s %4s %-12s %-12s %-12s %-12s %-12s %4s %4s\n%s\n',...
  hisStr{:},char(ones(1,120)*'-'));
dispHis = @(var) ...
  fprintf('%4d %-12.4e %-12.3e %-12.3e %-12.3e %4d %-12.3e %-12.3e\n',var);
dispHis(his(1,:));
% -- end initMethod   ------------------------------------------------

% ==============================================================================
% MAIN LOOP
% ==============================================================================
while 1,
  % check stopping rules
  STOP = check_stopping_rules(iter,Jold,Jc,yold,yc,dJ,p);
  if all(STOP(1:3)) || any(STOP(4:5)), break;  end;
  
  iter = iter + 1;
  
  % update at most maxLBFGS BFGS directions
  if iter > 1,
    zz = (dJ - dJold)';
    ss =  yc - yold;
    
    if zz'*ss > cs*(ss'*ss)
      start = 2-(nBFGS<p.maxLBFGS);
      zBFGS = [zBFGS(:,start:end),zz];
      sBFGS = [sBFGS(:,start:end),ss];
      nBFGS = min(p.maxLBFGS,nBFGS+1);
    else
      warning('skip update');
    end;
  end;
  
  % initialize Hessian
  if iter > 1
      p.Jold = Jold; p.Jc = Jc; p.LSiter = LSiter; p.iter = iter;

      % cautious update
      cumin = min([c0, c1*norm(dJ)^c2]);
      cumax = max([C0, C1*norm(dJ)^-C2]);

      % calculate tau
      yy = zz - d2S*ss;
      tau = cal_tau(p,zz,yy,ss,cumin,cumax);
  end
  if ~ismember(p.initMethod,["Hs","Hy"])
    H0 = d2S + tau*speye(size(d2S));
  else
    H0 = tau*speye(size(d2S));
  end

  % compute search direction using recursive and limited BFGS
  [dy] = bfgsrec(p.LSsolver,nBFGS,sBFGS,zBFGS,H0,-dJ',p);

  % perform linesearch
  [t,yt,LSiter] = lineSearch(fctn,yc,dy,Jc,dJ,p);

  % update variables
  yold = yc; Jold = Jc; dJold = dJ; yc = yt; dDold = dD;
  [Jc,dJ,dD,d2S] = fctn(yc);  % evaluate objective function
  
  % iteration history
  his(iter+2,:) = [iter,Jc,Jold-Jc,norm(dJ),norm(yc-yold),LSiter,tau,norm(yc - yref)];
  if p.dispHist
    dispHis(his(iter+1,:));
  end

end;%while; % end of iteration loop
% ==============================================================================

% clean up
His.str = hisStr;
His.his = his(1:iter+2,:);
fprintf('STOPPING:\n');
fprintf('%d[ %-10s=%16.8e <= %-25s=%16.8e]\n',STOP(4),...
  'norm(dJ)',norm(dJ),'eps',1e3*eps);
fprintf('%d[ %-10s=  %-14d >= %-25s=  %-14d]\n',STOP(5),...
  'iter',iter,'maxIter',p.maxIter);

%==============================================================================
function[d] = bfgsrec(solver,n,S,Z,H,d,p)

if n == 0
    switch solver
        case 'minres'
            [d,flag,relres,iterPCG,resvec] = minres(Afctn,d,p.tolLS,p.maxIterLS,PC,[]);
        case 'backslash'
            d = H\d;
        otherwise,
          error('nyi - solver %s', solver)
    end
else
  alpha           = (S(:,n)'*d)/(Z(:,n)'*S(:,n));
  d               = d - alpha*Z(:,n);
  [d]             = bfgsrec(solver,n-1,S,Z,H,d,p);
  d               = d + (alpha - (Z(:,n)'*d)/(Z(:,n)'*S(:,n)))*S(:,n);
end;
%==============================================================================
function p = set_parser()

p = inputParser;

% parameter initMethod -----------------------------------------------
addParameter(p,'maxIter',5000); % maximum number of iterations
addParameter(p,'tolJ',1e-9); % for stopping, objective function
addParameter(p,'tolY',1e-9); %   - " -     , current value
addParameter(p,'tolG',1e-9); %   - " -     , norm of gradient
addParameter(p,'maxLBFGS',5);               % maximum number of BFGS vectors
addParameter(p,'dispHist',1);
addParameter(p,'initMethod','Bg');
addParameter(p,'LSsolver','backslash');
addParameter(p,'tolLS',1e-2); % linear solve tolerance
addParameter(p,'maxIterLS',50); % linear solve maximum iterations

addParameter(p,'LSmaxIter',10); % maximum number of line search iterations
addParameter(p,'LSreduction',1e-4); % minimal reduction in line search
addParameter(p,'lineSearch',@Armijo); % minimal reduction in line search

addParameter(p,'ws',0); % initialize weights for adaptive scheme
addParameter(p,'wg',0); % initialize weights for adaptive scheme
addParameter(p,'wz',0); % initialize weights for adaptive scheme

addOptional(p,'xref',0); % true value of unknown parameter

%==============================================================================
function STOP = check_stopping_rules(iter,Jold,Jc,yold,yc,dJ,p)

%STOP(1) = (iter>0) && abs(Jold-Jc)  <= p.tolJ*(1+abs(Jc));
%STOP(2) = (iter>0) && (norm(yc-yold) <= p.tolY*(1+norm(yc)));
%STOP(3) = norm(dJ) <= p.tolG*(1+abs(Jc));

STOP(4) = norm(dJ) <= 1e-13;
STOP(5) = (iter >= p.maxIter);
%==============================================================================
function [tau,p] = cal_tau(p,zz,yy,ss,cumin,cumax)

switch p.initMethod
    case 'Hs'
        tau = (ss'*zz)/(ss'*ss);
    case 'Hy'
        tau = (zz'*zz)/(ss'*zz);
    case 'Bs'
        tau = (ss'*yy)/(ss'*ss);
    case 'Bg'
        tau = sqrt((yy'*yy)/(ss'*ss));
    case 'Bz'
        tau = (yy'*yy)/(ss'*yy);
    case 'Bu'
        mu = 0.5*((ss'*ss + yy'*yy) - sqrt((ss'*ss - yy'*yy)^2 + 4*(ss'*yy)^2));
        V1  = [ss'*yy; mu - ss'*ss];
        tau = -V1(1)/V1(2);
    case 'adap'
        ws = p.ws; wg = p.wg; wz = p.wz;
        Jold = p.Jold; Jc = p.Jc; LSiter = p.LSiter;

        Bz = max((yy'*yy)/(ss'*yy),cumin);
        Bs = max((ss'*yy)/(ss'*ss),cumin);
        Bg = max(sqrt((yy'*yy)/(ss'*ss)),cumin);

        if (ss'*yy) < 0
            tau = Bg;
        else
            if p.iter == 2 || (ws == 0 && wg == 0 && wz == 0)
                ws = 0.75; wg = 1 - ws; wz = 0;
            else
                if abs(Jold-Jc) <= 1e-4*(abs(Jold))
                    tc1 = 1/10;
                elseif abs(Jold-Jc) <= 1e-3*(abs(Jold))
                    tc1 = 1/20;
                else
                    tc1 = 1/40;
                end
                tc2 = 1/100;

                if wz == 0.0 && wg < 1
                    ws = max(ws - tc1*LSiter,0);
                    wg = 1 - ws; wz = 0.0;
                elseif wz > 0.0 || wg >= 1
                    wg = max(wg - tc2*LSiter,0.1);
                    wz = 1 - wg; ws = 0.0;
                end
            end
            tau = exp(ws*log(Bs) + wg*log(Bg) + wz*log(Bz));
            
            p.ws = ws; p.wg = wg; p.wz = wz;
        end
    otherwise
        error('%s initialization for Hessian is not defined',p.initMethod)
end
tau = max(cumin,min(tau,cumax));
%==============================================================================
function [t,yt,LSiter] = lineSearch(fctn,yc,dy,Jc,dJ,p)

if isequal(p.lineSearch,@Wolfe)
    wolfe = ncg('defaults'); % the standard parameters
    wolfe.LineSearch_ftol = 1e-04;
    wolfe.LineSearch_gtol = 0.9; % 0.9
    wolfe.LineSearch_maxfev = 3000;
    wolfe.LineSearch_stpmax = 2; % 2
    wolfe.LineSearch_stpmin = 0;
    wolfe.LineSearch_xtol = 1e-06;

    fcn = @(x) wolfefunc(x,fctn);
    [yt,~,~,t,info,LSiter] = cvsrch(fcn,yc,Jc,dJ',1,dy,wolfe);
else
    [t,yt,LSiter] = Armijo(fctn,yc,dy,Jc,dJ,...
                    'LSmaxIter',p.LSmaxIter,'LSreduction',p.LSreduction);
end
if (t == 0),
    warning('break! line-search failed');
end; % break if line-search fails
%==============================================================================
function [J, dJ] = wolfefunc(x,objfctn)
[J,~,dJ] = objfctn(x);
dJ = dJ';
%==============================================================================




