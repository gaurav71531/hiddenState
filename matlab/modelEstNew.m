function [Aout,Bout,order,u,relErr] = modelEstNew(varargin)

% input format:
% sensInd: sensors index needs to be selected out of total state dimensions
%          could be set as [1:length(data)] if all states need to be
%          selected for modeling
%
% numInp: unknown input dimensions (must be less than length(sensInd) )
% 
% data: input data of size NxD, where 'N' is number of samples and 'D' is
% state dimension
% 
% silentFlag: for toggling script intermediary status outputs

% sample Input:
% 
% data: matrix of size NxD as explained above
% say, N = 500 and D = 64 and,
% select states from index 1->32 and,
% we wish to have number of unknown inputs to be 16

% [Aout, B, order, u, relErr] = modelEst('sensInd', 1:32, 'numInp', 16,
% 'data', data, 'silentFlag', 0);
% 


% output:
% Aout: estimated spatial coupling matrix
% B: input coupling matrix
% order: fractional order vector of size length(sensInd)
% u: estimated unknown inputs delayed by one step
% relErr: relative error vector between data and predicted model with numStep set
% in the script. First error is without taking unknown inputs and second
% error is with taking unknown inputs into consideration

global infit
global numCh
% addpath(genpath(pwd));

nargin = length(varargin);
patInpEdfStr = [];
silentFlag = 1;
data = [];
if nargin > 1
    for i = 1:2:nargin-1
        switch varargin{i}
            case 'sensInd'
                sensInd = varargin{i+1};
            case 'numInp'
                numInp = varargin{i+1};
            case 'data'
                data = varargin{i+1};
            case 'edfStr'
                patInpEdfStr = varargin{i+1};
            case 'silentFlag'
                silentFlag = varargin{i+1};
            case 'cvForLambda'
                cvForLambda = varargin{i+1};
            case 'lambdaRange'
                lambdaRange = varargin{i+1};
        end
    end
    if ~isempty(patInpEdfStr)
        [~,data] = edfread(patInpEdfStr);
    elseif isempty(data)
        patInpStr = 'S001R03_edfm.mat';
        matObj = matfile(patInpStr);
        data = matObj.record;
    end 
    try
        if cvForLambda>0
            MException('MATLAB:variableNotExist', 'cvForLambda');
            if isempty(lambdaRange)
                lambdaRange = logspace(-2,1,10);
            end
        end
    catch ME
        switch ME.identifier
            case 'MATLAB:UndefinedFunction'
                messageStr = extractBetween(ME.message,'''','''');
                if strcmp(messageStr{1}, 'cvForLambda')
                    cvForLambda = 0;
                end
        end
    end
    
    calledFromOutside = 1;
else
    sensInd = 1:64;
    numInp = 32;
    patInpStr = 'S001R03_edfm.mat';
    matObj = matfile(patInpStr);
    data = matObj.record;
    silentFlag = 0;
    calledFromOutside = 0;
    cvForLambda = 0;
    if cvForLambda
        lambdaRange = logspace(-2,1,10);
    end
end

% sensInd = 1:4;
% numInp = 2;
numCh = length(sensInd);
% % % K = 1000;
% K = 500;
% sampleID = 1:K;
% % % sampleID = [0:K-1]+5000;
% sampleID = [0:K-1]+8900;

% % % X = data(sensInd,sampleID);

X = data;

% % % temp
K = size(data,2);

%  center data
X = bsxfun(@minus, X, mean(X,2));

order = zeros(numCh,1);
infit = 20;

yCascade = zeros(numCh, K); % [y[0], y[1], ..., y[K-1]]

for i = 1:numCh
    order(i) = WT_estimator_v3(X(i,:),1);
    yCascade(i,:) = getFractionalExpan(X(i,1:K),order(i),infit);
end

niter = 50;
A = cell(niter,1);
B = cell(niter,1);

[A{1},mse] = performLeastSq(yCascade, X(:,1:K), 0, 'woInp');

B_1 = zeros(size(A{1},1),size(A{1},2));
B_1(abs(A{1})>0.01) = A{1}(abs(A{1})>0.01);
[~,r] = qr(B_1);
colInd = find(abs(diag(r))>1e-7);
if length(colInd) < numInp
    B_1 = [eye(numInp);zeros(numCh-numInp,numInp)];
else
    colInd = colInd(1:numInp);
    B_1 = B_1(:,colInd);
end

% B = [eye(numInp);zeros(numCh-numInp,numInp)];
if rank(B_1) < numInp
    error('rank deficient B');
end
B{1} = B_1;

tic

if cvForLambda
    lambdaUse = performLambdaCV(lambdaRange, A, B, order, yCascade, X, silentFlag);
else
%     lambdaUse = 0.5;
    lambdaUse = 4;
end

niter = 50;
mseIter = zeros(niter+1,1);
mseIter(1) = mse;
[Aout, Bout, u, mseIter] = performEM(A, B, yCascade, X, lambdaUse, niter, mseIter, silentFlag);

% predict values for models, k-step prediction
T = K;
% numStep = 100;
numStep = 1;
% chUse = 10;
chUse = 1;
% without inputs
xPred = predictValues(X, order, T, numStep, A{1}, Bout, zeros(size(Bout,2),K));
% mean squared error across all channels 
relErr1 = sqrt(sum(sum((xPred-X).^2))/T/numCh);

% with inputs
xPred = predictValues(X, order, T, numStep, Aout, Bout, u);
relErr2 = sqrt(sum(sum((xPred-X).^2))/T/numCh);
relErr = [relErr1,relErr2];


function lambdaUse = performLambdaCV(lambdaRange, A, B, order, yCascade, X, silentFlag)

[numCh, K] = size(X);
niter = 50;
T = K;
numStep = 1;

relErr2 = zeros(length(lambdaRange),1);
for i = 1:length(lambdaRange)
%     ATemp = A{1};
%     BTemp = B{1};
    lambda = lambdaRange(i);
    [Aout, Bout, u, ~] = performEM(A, B, yCascade, X, lambda, niter, zeros(1,niter), 1);
    
    xPred = predictValues(X, order, T, numStep, Aout, Bout, u);
    relErr2(i) = sqrt(sum(sum((xPred-X).^2))/T/numCh);
end
[~, indUse] = min(relErr2);
lambdaUse = lambaRange(indUse);
if ~silentFlag
    fprintf('best lambda after CV = %f\n', lambdaUse);
end


function [Aout, Bout, u, mseIter] = performEM(A, B, yCascade, X, lambda, niter, mseIter, silentFlag)

[numCh, K] = size(X);
u = zeros(size(B{1},2), K);
if ~silentFlag,fprintf('before iteration, mse = %f\n', mseIter(1));end

for iterInd = 1:niter
    for kInd = 2:K
        yUse = yCascade(:,kInd) - A{iterInd}*X(:,kInd-1);
        u(:,kInd) = getLassoSoln(B{iterInd}, yUse, lambda);
%         options.verbose = 0;
%         u(:,kInd) = L1General2_BBST(@(x) squaredLoss(B{iterInd}, yUse, x), zeros(size(B{iterInd},2),1),lambda*ones(size(B{iterInd},2),1),options);
    end
    zInd = find(mean(abs(u),2)<1e-6);
    if ~isempty(zInd)
%         all u's are 0 for these dimensions
        u(zInd,:) = [];
    end
    [Ahat,mseIter(iterInd+1)] = performLeastSq(yCascade, X(:,1:K), u, 'wInp');
    A{iterInd+1} = Ahat(:,1:numCh);
    B{iterInd+1} = Ahat(:,numCh+1:end);
        
    if ~silentFlag,fprintf('iter ind = %d, mse = %f\n', iterInd, mseIter(iterInd+1));end
end
Aout = A{end};
Bout = B{end};


function [xPred] = predictValues(X, order, T, numStep, A, B, u)

global infit
global numCh

TSteps =  ceil(T/numStep);

xPred = zeros(numCh,T);
xPred(:,1:numStep) = X(:,1:numStep) + B*u(:,1:numStep);

for i = 2:TSteps
    XTemp = zeros(numCh,T);
    XTemp(:,1:(i-1)*numStep) = X(:,1:(i-1)*numStep);
    for stepInd = 1:numStep
        for chInd = 1:numCh
            alpha = order(chInd);
            if ceil(alpha) ~= alpha
                trailLen = min(infit,(i-1)*numStep+stepInd-1);
                j = 1:trailLen;
                preFact = gamma(-alpha+j)./(gamma(-alpha).*gamma(j+1));
                XTemp(chInd,(i-1)*numStep + stepInd) = XTemp(chInd,(i-1)*numStep + stepInd) ...
                    - XTemp(chInd,(i-1)*numStep + stepInd-j)*preFact';
            end
        end
        XTemp(:,(i-1)*numStep + stepInd) = XTemp(:,(i-1)*numStep + stepInd) ...
            + A*XTemp(:,(i-1)*numStep + stepInd-1) ...
            + B*u(:,(i-1)*numStep + stepInd)+ randn(numCh,1);
%         u is already arranged in one step behind order, so no need for
%         '-1' in the index
    end
    xPred(:,(i-1)*numStep+1:i*numStep) = XTemp(:,(i-1)*numStep+1:i*numStep);
end
% relErr = sqrt(mean(abs(xPred(chUse,:) - X(chUse,:)).^2));


function out = getLassoSoln(A, b, lambda)


% method :
% 1: CVX
% 2: lasso MATLAB
% 3: ADMM

method = 3;
switch method
    
    case 1
        cvx_begin quiet
            variable uGG(32)
            minimize(sum_square(A*uGG - b) + lambda*norm(uGG,1))
        cvx_end
        out = uGG;
    case 2
        [out, fInfo] = lasso(A, b, 'Lambda', lambda);

% problem: it doesn't allow to remove the intercept
        
    case 3
%         downloaded from: https://web.stanford.edu/~boyd/papers/admm/lasso/lasso.html 
        % QUIET = 1;
        MAX_ITER = 100;
        ABSTOL   = 1e-4;
        RELTOL   = 1e-2;

        [m, n] = size(A);
        Atb = A'*b;
        % lambda = 1;
        rho = 1/lambda;
        alpha = 1;

        x = zeros(n,1);
        z = zeros(n,1);
        u = zeros(n,1);

        [L, U] = factor(A, rho);

        for k = 1:MAX_ITER

            % x-update
            q = Atb + rho*(z - u);    % temporary value
            if( m >= n )    % if skinny
               x = U \ (L \ q);
            else            % if fat
               x = q/rho - (A'*(U \ ( L \ (A*q) )))/rho^2;
            end

            % z-update with relaxation
            zold = z;
            x_hat = alpha*x + (1 - alpha)*zold;
            z = shrinkage(x_hat + u, lambda/rho);

            % u-update
            u = u + (x_hat - z);

            % diagnostics, reporting, termination checks
            history.objval(k)  = objective(A, b, lambda, x, z);

            history.r_norm(k)  = norm(x - z);
            history.s_norm(k)  = norm(-rho*(z - zold));

            history.eps_pri(k) = sqrt(n)*ABSTOL + RELTOL*max(norm(x), norm(-z));
            history.eps_dual(k)= sqrt(n)*ABSTOL + RELTOL*norm(rho*u);

        %     if ~QUIET
        %         fprintf('%3d\t%10.4f\t%10.4f\t%10.4f\t%10.4f\t%10.2f\n', k, ...
        %             history.r_norm(k), history.eps_pri(k), ...
        %             history.s_norm(k), history.eps_dual(k), history.objval(k));
        %     end

            if (history.r_norm(k) < history.eps_pri(k) && ...
               history.s_norm(k) < history.eps_dual(k))
                 break;
            end

        end
        out = z;
end


function p = objective(A, b, lambda, x, z)

p = ( 1/2*sum((A*x - b).^2) + lambda*norm(z,1) );


function [p,g] = squaredLoss(A, b, x)

p = 1/2*norm(A*x-b)^2;
g = A'*(A*x-b);


function z = shrinkage(x, kappa)
    
z = max( 0, x - kappa ) - max( 0, -x - kappa );

function [L, U] = factor(A, rho)
    
[m, n] = size(A);
if ( m >= n )    % if skinny
   L = chol( A'*A + rho*speye(n), 'lower' );
else            % if fat
   L = chol( speye(m) + 1/rho*(A*A'), 'lower' );
end
% % % if ( m >= n )    % if skinny
% % %    L = chol( A'*A + rho*eye(n), 'lower' );
% % % else            % if fat
% % %    L = chol( eye(m) + 1/rho*(A*A'), 'lower' );
% % % end

% force matlab to recognize the upper / lower triangular structure
L = sparse(L);
U = sparse(L');
% % % U = L';


function [A,mse] = performLeastSq(yCascade, X, u, flag)

numCh = size(X,1);

if strcmp(flag, 'woInp')
    XUse = [zeros(1,numCh);X(:,1:end-1)'];
    % The data may not be zero mean, so take care of intercept as well
    % XUse = [ones(size(XUse,1),1),XUse];

    A = zeros(numCh,numCh);
    mse = zeros(numCh,1);
    for i = 1:numCh
        A(i,:) = regress(yCascade(i,:)', XUse);
        mse(i) = norm(yCascade(i,:)' - XUse*A(i,:)',2)^2 / size(yCascade,2);
    end
    mse = mean(mse);
elseif strcmp(flag, 'wInp')
    XUse = [[zeros(1,numCh);X(:,1:end-1)'], u'];
    A = zeros(numCh, size(XUse,2));
    mse = zeros(numCh, 1);
    for i = 1:numCh
        A(i,:) = regress(yCascade(i,:)', XUse);
        mse(i) = norm(yCascade(i,:)' - XUse*A(i,:)',2)^2 / size(yCascade,2);
    end
    mse = mean(mse);
end


function y = getFractionalExpan(x, alpha, infit)

l = length(x);
j = 0:infit;
preFactVec = gamma(-alpha+j)./(gamma(-alpha).*gamma(j+1));

y = conv(x, preFactVec);
y = y(1:l);
