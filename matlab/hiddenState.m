function [relErrOut, outFn] = hiddenState(matObj, oInd, hInd)


% [X, A, order] = getTestData();
dataComplete = matObj.X;
Aorig = matObj.A;
order = matObj.order;

% % % % % % Complete Data
% % % % % [numCh, K] = size(dataComplete);
% % % % % 
% % % % % global infit 
% % % % % infit = 20;
% % % % % 
% % % % % yCascade = zeros(numCh, K); % [y[0], y[1], ..., y[K-1]]
% % % % % 
% % % % % for i = 1:numCh
% % % % %     yCascade(i,:) = getFractionalExpan(dataComplete(i,1:K),order(i),infit);
% % % % % end
% % % % % 
% % % % % [A_Complete,~,~] = performLeastSq(yCascade, dataComplete(:,1:K));


niter = 100;
infit = 20;

perm = [oInd,hInd];
invPerm(perm) = 1:length(oInd)+length(hInd);
X = dataComplete(oInd,:);

PMat = zeros(length(oInd)+length(hInd));
for i = 1:size(PMat,1)
    PMat(i,invPerm(i)) = 1;
end

N = size(X,2);
sizeX = size(X,1);
sizeZ = 1;
orderX = order(oInd);
orderZ = order(hInd);

OrdMatX = zeros(sizeX, sizeX, infit);
OrdMatZ = zeros(sizeZ, sizeZ, infit);


for j = 1:infit
    preFact = gamma(-orderX+j)./(gamma(-orderX).*gamma(j+1));
    OrdMatX(:,:,j) = diag(preFact);
end

for j = 1:infit
    preFact = gamma(-orderZ+j)./(gamma(-orderZ).*gamma(j+1));
    OrdMatZ(:,:,j) = diag(preFact);
end

zHat = cell(1,niter);
zTilda = cell(1,niter);

PHat = cell(1,niter);
PTilda = cell(1,niter);

K = cell(1,niter);

A11 = cell(1,niter);
A12 = cell(1,niter);
A21 = cell(1,niter);
A22 = cell(1,niter);
Arec = cell(1,niter);

% B1 = cell(1,niter);
% B2 = cell(1,niter);

Sigma_1 = cell(1,niter);
Sigma_2 = cell(1,niter);

x0 = zeros(sizeX,1);
z0 = zeros(sizeZ,1);

xHat = zeros(size(X));
for i = 1:sizeX
%     assuming x0 = 0
    xHat(i,:) = getFractionalExpan(X(i,:),orderX(i),infit);
end


PHat_0 = eye(sizeZ) * 100;

[A11{1}, Sigma_1{1},~] = performLeastSq(xHat, X);
% A21{1} = randn(sizeZ, sizeX);
% A12{1} = randn(sizeX, sizeZ);
% A22{1} = randn(sizeZ,sizeZ);
% A21{1} = zeros(sizeZ, sizeX);
% A12{1} = zeros(sizeX, sizeZ);
% A22{1} = -0.5;
% Sigma_1{1} =eye(sizeX);
Sigma_2{1} = eye(sizeZ);

A12{1} = 2*rand(sizeX, sizeZ)-1;
A21{1} = 2*rand(sizeZ, sizeX)-1;
A22{1} = 2*rand(sizeZ, sizeZ)-1;




[zHatTemp, zTildaTemp, PHatTemp, PTildaTemp, KTemp] = ...
        getKalmanFilteringSoln(xHat, X, OrdMatZ, infit, Aorig(oInd,oInd), Aorig(oInd,hInd), Aorig(hInd,oInd),...
        Aorig(hInd,hInd), Sigma_1{1}, Sigma_2{1}, PHat_0, x0, z0);

fprintf('Correct Aorig, mean zHat error = %f\n', sqrt(mean((dataComplete(hInd,1:end-1)-zHatTemp).^2)));
% figure;plot(dataComplete(hInd,:));hold on;plot(zHatTemp);

% A11{1} = Aorig(oInd,oInd);
% A12{1} = Aorig(oInd,hInd);
% A21{1} = Aorig(hInd, oInd);
% A22{1} = Aorig(hInd, hInd);

QVal = zeros(niter,1);
HVal = zeros(niter,1);
FVal = zeros(niter,1);

for iterInd = 2:niter
    
    [zHat{iterInd}, zTilda{iterInd}, PHat{iterInd}, PTilda{iterInd}, K{iterInd}] = ...
        getKalmanFilteringSoln(xHat, X, OrdMatZ, infit, A11{iterInd-1}, A12{iterInd-1}, A21{iterInd-1},...
        A22{iterInd-1}, Sigma_1{iterInd-1}, Sigma_2{iterInd-1}, PHat_0, x0, z0);
    
    fprintf('mean zHat error = %f\n', sqrt(mean((dataComplete(hInd,1:end-1)-zHat{iterInd}).^2)));
    
    x_nm1Mat = [x0,X(:,1:end-1)];
    z_nm1HatMat = [z0,zHat{iterInd}];
    PHatSumMat = PHat_0;
    for i = 1:size(PHat{iterInd},3)
        PHatSumMat = PHatSumMat + PHat{iterInd}(:,:,i);
    end
    colMat1 = [x_nm1Mat*xHat' ; z_nm1HatMat*xHat'];
    tempMat1 = [x_nm1Mat*x_nm1Mat', x_nm1Mat * z_nm1HatMat';...
                z_nm1HatMat*x_nm1Mat', PHatSumMat + z_nm1HatMat*z_nm1HatMat'];
    solTemp1 = tempMat1 \ colMat1;
    A11{iterInd} = solTemp1(1:sizeX,:)';
    A12{iterInd} = solTemp1(sizeX+1:sizeX+sizeZ,:)';
    
    Sigma_1{iterInd} = 1/N*(xHat*xHat' - A11{iterInd}*(x_nm1Mat*xHat') ...
        - A12{iterInd}*(z_nm1HatMat*xHat'));
    
    Sigma_1{iterInd} = trace(Sigma_1{iterInd})/2*eye(2);
    
    zoMat = zeros(sizeZ, N-1);
    for i = 1:sizeZ
%         assuming z0 = 0;
        zoMat(i,:) = getFractionalExpan(zHat{iterInd}(i,:),orderZ(i),infit);
    end
    
    PHatSum_Nm1 = PHatSumMat - PHat{iterInd}(:,:,end);
    colMat2 = [x_nm1Mat(:,1:end-1)*zoMat';PHatSum_Nm1*OrdMatZ(:,:,1)' + z_nm1HatMat(:,1:end-1)*zoMat'];
    tempMat2 = [x_nm1Mat(:,1:end-1)*x_nm1Mat(:,1:end-1)', x_nm1Mat(:,1:end-1)*z_nm1HatMat(:,1:end-1)';...
        z_nm1HatMat(:,1:end-1)*x_nm1Mat(:,1:end-1)', PHatSum_Nm1 + z_nm1HatMat(:,1:end-1)*z_nm1HatMat(:,1:end-1)'];
    solTemp2 = tempMat2 \ colMat2;
    
    A21{iterInd} = solTemp2(1:sizeX,:)';
    A22{iterInd} = solTemp2(sizeX+1:sizeX+sizeZ,:)';
    
    Sigma_2{iterInd} = -A21{iterInd} * (x_nm1Mat(:,1:end-1) * zoMat')...
        -A22{iterInd}*(PHatSum_Nm1*OrdMatZ(:,:,1)' + z_nm1HatMat(:,1:end-1)*zoMat');
    
    matSumAggr = PHatSumMat - PHat_0;
    for j = 1:infit
        matSum = PHatSum_Nm1;
        for i = 1:j-1
            matSum = matSum - PHat{iterInd}(:,:,end-i);
        end
        matSumAggr = matSumAggr + OrdMatZ(:,:,j) * matSum * OrdMatZ(:,:,j)';
    end
    matSumAggr = matSumAggr + zoMat*zoMat';
    Sigma_2{iterInd} = Sigma_2{iterInd} + matSumAggr;
    Sigma_2{iterInd} = Sigma_2{iterInd}/(N-1);
    
    QVal(iterInd) = -N/2*log(det(Sigma_1{iterInd})) - (N-1)/2*log(det(Sigma_2{iterInd}))...
        -1/2*sizeX*N -1/2*sizeZ*(N-1);

    for i = 1:N-1
        HVal(iterInd) = HVal(iterInd) + 1/2*log(det(PHat{iterInd}(:,:,i)));
    end
    FVal(iterInd) = QVal(iterInd) + HVal(iterInd);
    Arec{iterInd} = [A11{iterInd},A12{iterInd};A21{iterInd},A22{iterInd}];
    fprintf('iteration  = %d, FVal = %f\n', iterInd, FVal(iterInd));
    
end

AHat = [A11{end},A12{end};A21{end},A22{end}];

numStep = 5;

[A_Incomplete,~,~] = performLeastSq(xHat, X);

xPred_wInCompleteData = predictValues(X, orderX, N, numStep, A_Incomplete, zeros(sizeX,sizeZ), zeros(sizeZ,N));
relErr_wInCompleteData = sqrt(sum((xPred_wInCompleteData-X).^2,2)/N)./sqrt(sum(X.^2,2)/N);

% relErr_wInCompleteData1 = (sum(abs(xPred_wInCompleteData-X),2)/N)./(sum(abs(X),2)/N);

relErr_wInCompleteData1 = sqrt(mean((abs(X(:,2:end)-xPred_wInCompleteData(:,2:end))./abs(X(:,2:end))).^2,2));



xPred_wAlg = predictValues(X, orderX, N, numStep, A11{end}, A12{end}, [z0,zHat{end}]);
relErr_wAlg = sqrt(sum((xPred_wAlg-X).^2,2)/N)./sqrt(sum(X.^2,2)/N);

relErr_wAlg1 = sqrt(mean((abs(X(:,2:end)-xPred_wAlg(:,2:end))./abs(X(:,2:end))).^2,2));

% relErrOut = [mean(relErr_wInCompleteData), mean(relErr_wAlg)];
relErrOut = [{relErr_wInCompleteData}, {relErr_wAlg}];

outFn.predictValues = @predictValues;
outFn.getFractionalExpan = @getFractionalExpan;
outFn.performLeastSq = @performLeastSq;

% fprintf('r1 = %f, r2 = %f\n', relErr_wInCompleteData, relErr_wAlg);

function [xPred] = predictValues(X, order, T, numStep, A11, A12, z)

global infit
% global numCh
sizeX = size(X,1);

TSteps =  ceil(T/numStep);

xPred = zeros(sizeX,T);
xPred(:,1:numStep) = X(:,1:numStep) + A12*z(:,1:numStep);

for i = 2:TSteps
    XTemp = zeros(sizeX,T);
    XTemp(:,1:(i-1)*numStep) = X(:,1:(i-1)*numStep);
    for stepInd = 1:numStep
        for chInd = 1:sizeX
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
            + A11*XTemp(:,(i-1)*numStep + stepInd-1) ...
            + A12*z(:,(i-1)*numStep + stepInd);
%         z is already arranged in one step behind order, so no need for
%         '-1' in the index
    end
    xPred(:,(i-1)*numStep+1:i*numStep) = XTemp(:,(i-1)*numStep+1:i*numStep);
end
% relErr = sqrt(mean(abs(xPred(chUse,:) - X(chUse,:)).^2));


function [zHat, zTilda, PHat, PTilda, K] = getKalmanFilteringSoln...
    (xHat, X, OrdMatZ, infit, A11, A12, A21, A22, Sigma_1, Sigma_2, PHat_0, x0, z0)

N = size(X,2);
n = size(X,1);
m = size(A22,1);
% N-1 samples for hidden vars
zHat = zeros(m,N-1);
zTilda = zeros(m,N-1);
PHat = zeros(m,m,N-1);
PTilda = zeros(m,m,N-1);
K = zeros(m,n,N-1);
y = xHat(:,2:end) - A11*X(:,1:end-1);

PTilda(:,:,1) = (A22 - OrdMatZ(:,:,1)) * PHat_0 * (A22 - OrdMatZ(:,:,1))' + Sigma_2;
K(:,:,1) = PTilda(:,:,1) * A12' / (Sigma_1 + A12 * PTilda(:,:,1) * A12');
PHat(:,:,1) = PTilda(:,:,1) - K(:,:,1)*A12*PTilda(:,:,1);
zTilda(:,1) = A22*z0 + A21*x0;
zHat(:,1) = zTilda(:,1) + K(:,:,1)*(y(:,1) - A12*zTilda(:,1));

for n = 2:N-1
    PTilda(:,:,n) = (A22 - OrdMatZ(:,:,1))*PHat(:,:,n-1)*(A22-OrdMatZ(:,:,1))' + Sigma_2;
    for j = 2:infit
        if n-j<1,break;end
        PTilda(:,:,n) = PTilda(:,:,n) + OrdMatZ(:,:,j) * PHat(:,:,n-j) * OrdMatZ(:,:,j)';
    end
    if n-j == 0
        PTilda(:,:,n) = PTilda(:,:,n) + OrdMatZ(:,:,j) * PHat_0 * OrdMatZ(:,:,j)';
    end
    K(:,:,n) = PTilda(:,:,n) * A12' / (Sigma_1 + A12 * PTilda(:,:,n) * A12');
    PHat(:,:,n) = PTilda(:,:,n) - K(:,:,n)*A12*PTilda(:,:,n);
    
    zTilda(:,n) = A22*zHat(:,n-1) + A21*X(:,n-1);
    for j = 1:infit
        if n-j<1,break;end
        zTilda(:,n) = zTilda(:,n) - OrdMatZ(:,:,j)*zHat(:,n-j);
    end
    if n-j == 0
        zTilda(:,n) = zTilda(:,n) - OrdMatZ(:,:,j)*z0;
    end
    
    zHat(:,n) = zTilda(:,n) + K(:,:,n)*(y(:,n) - A12*zTilda(:,n));
end

function [x] = genFracData(A,alpha,Sigma_1,Sigma_2)

n = size(A,1);
initState = zeros(n,1);
noiseCov = blkdiag(Sigma_1,Sigma_2);

K = 1000;

x = zeros(n,K);
x(:,1) = initState;

infit = 20;

for k = 2:K
    x(:,k) = A * x(:,k-1) + mvnrnd(zeros(n,1), noiseCov)';
    for j = 1:n
        x(j,k) = x(j,k) - getFracRemainder(x(j,1:k-1), alpha(j), infit);
    end
end


function [x,A,alpha] = getTestData()

n = 2;
m = 1;
initState = zeros(n+m,1);
initCov = eye(n+m);

noiseCov = eye(n+m);


% alpha = [0.7, 1.2];
% A = [0,0.1;-0.01,-0.02];
% alpha = [0.7 1.2 0.2];
% A = [0,0.1,0.2;-0.01,-0.02,1;0.01,-0.03,0.5];

% % alpha = [0.7 1.1 0.8];
% % A = [0,0.1,0.2;-0.01,-0.02,0.3;0.01,-0.03,-0.05];

alpha = [0.7 0.9 1.1];
A = [0,0.1,0;-0.01,-0.02,-0.03;0.01,-0.05,-0.03];

% alpha = [0.5;0.6;0.6];
% A = [-0.73,-0.5,0.3;-0.2,-0.6,-0.3;0.03,-0.5,-0.6];

K = 1000;

x = zeros(n+m,K);
x(:,1) = initState;

infit = 20;

for k = 2:K
    x(:,k) = A * x(:,k-1) + mvnrnd(zeros(n+m,1), noiseCov)';
    for j = 1:n+m
        x(j,k) = x(j,k) - getFracRemainder(x(j,1:k-1), alpha(j), infit);
    end
end

figure;
plot(1:length(x(1,:)), x(1,:), 1:length(x(2,:)), x(2,:), 1:length(x(3,:)), x(3,:));
grid
    

function y = getFracRemainder(x, alpha, infit)

l = length(x);
j = 0:infit;
preFactVec = gamma(-alpha+j)./(gamma(-alpha).*gamma(j+1));
preFactVec = preFactVec(2:end);

if length(preFactVec) > l
    y = preFactVec(1:l) * x(end:-1:1)';
else
    y = preFactVec * x(end:-1:end-length(preFactVec)+1)';
    
end
% y = conv(x, preFactVec);
% y = y(1:l);


function y = getFractionalExpan(x, alpha, infit)

l = length(x);
j = 0:infit;
preFactVec = gamma(-alpha+j)./(gamma(-alpha).*gamma(j+1));

y = conv(x, preFactVec);
y = y(1:l);


function [A,Sigma,mse] = performLeastSq(yCascade, X)

numCh = size(X,1);

XUse = [zeros(1,numCh);X(:,1:end-1)'];
% The data may not be zero mean, so take care of intercept as well
% XUse = [ones(size(XUse,1),1),XUse];

A = zeros(numCh,numCh);
mse = zeros(numCh,1);
for i = 1:numCh
    A(i,:) = regress(yCascade(i,:)', XUse);
    mse(i) = norm(yCascade(i,:)' - XUse*A(i,:)',2)^2 / size(yCascade,2);
end
Sigma = (yCascade * yCascade' - A*XUse' * yCascade')/size(X,2);
mse = mean(mse);
