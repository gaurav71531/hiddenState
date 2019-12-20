function relErrOut = latentFracUU(dataComplete, order, numInp, oInd, hInd, baseInd)


% [X, A, order] = getTestData();
% matObj = matfile('testData1.mat');
% dataComplete = matObj.X;
% Aorig = matObj.A;
% order = matObj.order;

% % % % % sensInd = 1:64;
% % % % % numInp = 32;
% % % % % patInpStr = 'S001R03_edfm.mat';
% % % % % matObj = matfile(patInpStr);
% % % % % dataComplete = matObj.record;
% % % % % 
% % % % % K = 1000;
% % % % % sampleID = [0:K-1]+5000;
% % % % % dataComplete = dataComplete(sensInd,sampleID);
% % % % % 
% % % % % %  center data
% % % % % dataComplete = bsxfun(@minus, dataComplete, mean(dataComplete,2));
% % % % % 
% % % % % [AOrig,BOrig,order,u,~] = modelEstNew();
% matObj = matfile('eegOrigParam.mat');
% matObj = matfile('eegOrigParam1.mat');
% dataComplete = matObj.data;
% AOrig = matObj.AOrig;
% BOrig = matObj.BOrig;
% order = matObj.order;
% uOrig = matObj.uOrig;
% uOrig = uOrig(:,2:end); % original indexing is from u0 to u_N-1

infit = 20;
% numInp = 32;
% numInp = 5;

% oInd = 1:60;
% hInd = 61:64;


niter = 15;
perm = [oInd,hInd];
invPerm(perm) = 1:length(oInd)+length(hInd);
X = dataComplete(oInd,:);
% PMat = eye(3);

PMat = zeros(length(oInd)+length(hInd));
for i = 1:size(PMat,1)
    PMat(i,invPerm(i)) = 1;
end

N = size(X,2);
sizeX = length(oInd);
sizeZ = length(hInd);
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

B1 = cell(1,niter);
B2 = cell(1,niter);

Sigma_1 = cell(1,niter);
Sigma_2 = cell(1,niter);

x0 = zeros(sizeX,1);
z0 = zeros(sizeZ,1);
u0 = zeros(numInp,1);

xHat = zeros(size(X));
for i = 1:sizeX
%     assuming x0 = 0
    xHat(i,:) = getFractionalExpan(X(i,:),orderX(i),infit);
end

% [A_Incomplete,~,~] = performLeastSq(xHat, X);

PHat_0 = eye(sizeZ) * 100;

[A11{1}, B_Incomplete,~,u_Incomplete,~] = modelEstNew('sensInd',1:length(oInd),...
    'numInp', numInp, 'data', X, 'silentFlag', 1);

if size(B_Incomplete,2) < numInp
    BTemp = 2*rand(sizeX,numInp)-1;
    BTemp(:,1:size(B_Incomplete,2)) = B_Incomplete;
    B1{1} = BTemp;
else
    B1{1} = B_Incomplete;
end

A21{1} = 2*rand(sizeZ, sizeX)-1;
A12{1} = 2*rand(sizeX, sizeZ)-1;
A22{1} = 2*rand(sizeZ,sizeZ)-1;
% % A21{1} = zeros(sizeZ, sizeX);
% % A12{1} = zeros(sizeX, sizeZ);
% % A22{1} = -0.5;
Sigma_1{1} = eye(sizeX);
Sigma_2{1} = eye(sizeZ);

B2{1} = 2*rand(sizeZ,numInp)-1;

% e = -rand(sizeX+sizeZ,1);
% [U,~,~] = svd(randn(sizeX+sizeZ));
% Ainit = U*diag(e)*U';
% matObj1 = matfile('initParam_testData.mat');
% Ainit =matObj1.Ainit;
% A12{1} = Ainit(1:sizeX,sizeX+1:sizeX+sizeZ);
% A21{1} = Ainit(sizeX+1:sizeX+sizeZ, 1:sizeX);
% A22{1} = Ainit(sizeX+1:sizeX+sizeZ,sizeX+1:sizeX+sizeZ);


% [zHatTemp, ~, ~, ~, ~] = ...
%         getKalmanFilteringSoln(xHat, X, OrdMatZ, infit, AOrig(oInd,oInd), AOrig(oInd,hInd), AOrig(hInd,oInd),...
%         AOrig(hInd,hInd), BOrig(oInd,:), uOrig, Sigma_1{1}, Sigma_2{1}, PHat_0, x0, z0);

% fprintf('Correct Aorig, mean zHat error = %f\n', sqrt(mean((dataComplete(hInd(1),1:end-1)-zHatTemp(1,:)).^2)));
% figure;plot(dataComplete(hInd,:));hold on;plot(zHatTemp);

% A11{1} = Aorig(oInd,oInd);
% A12{1} = Aorig(oInd,hInd);
% A21{1} = Aorig(hInd, oInd);
% A22{1} = Aorig(hInd, hInd);

QVal = zeros(niter,1);
HVal = zeros(niter,1);
FVal = zeros(niter,1);

u = cell(1,niter);
u{1} = zeros(numInp,N-1);
lambdaBase = 4;

x_nm1Mat = [x0,X(:,1:end-1)];

for iterInd = 2:niter
    
    [zHat{iterInd}, zTilda{iterInd}, PHat{iterInd}, PTilda{iterInd}, K{iterInd}] = ...
        getKalmanFilteringSoln(xHat, X, OrdMatZ, infit, A11{iterInd-1}, A12{iterInd-1}, A21{iterInd-1},...
        A22{iterInd-1}, B1{iterInd-1}, u{iterInd-1}, Sigma_1{iterInd-1}, Sigma_2{iterInd-1}, PHat_0, x0, z0);
    
%     fprintf('mean zHat error = %f\n', sqrt(mean((dataComplete(hInd,1:end-1)-zHat{iterInd}).^2)));
%     figure;plot(dataComplete(hInd,:));hold on;plot(zHat{iterInd});

    zoMat = zeros(sizeZ, N-1);
    for i = 1:sizeZ
%         assuming z0 = 0;
        zoMat(i,:) = getFractionalExpan(zHat{iterInd}(i,:),orderZ(i),infit);
    end
    z_nm1HatMat = [z0,zHat{iterInd}];

    u{iterInd} = zeros(numInp,N-1);
    for k = 1:N-2
        AUse = [B1{iterInd-1}/sqrt(Sigma_1{iterInd-1}(1,1));...
            B2{iterInd-1}/sqrt(Sigma_2{iterInd-1}(1,1))];
        b1 = xHat(:,k+1) - A11{iterInd-1}*X(:,k) - A12{iterInd-1}*zHat{iterInd}(:,k);
        b2 = zoMat(:,k+1) - A21{iterInd-1}*X(:,k) - A22{iterInd-1}*zHat{iterInd}(:,k);
        b = [b1/sqrt(Sigma_1{iterInd-1}(1,1));...
            b2/sqrt(Sigma_2{iterInd-1}(1,1))];
            
        lambda = lambdaBase/mean([sqrt(Sigma_1{iterInd-1}(1,1)),sqrt(Sigma_2{iterInd-1}(1,1))]);
        u{iterInd}(:,k) = getLassoModSoln(AUse, b, lambda);
        
%         options.verbose = 0;
%         b1 = xHat(:,k+1) - A11{iterInd-1}*X(:,k) - A12{iterInd-1}*zHat{iterInd}(:,k);
%         b2 = zoMat(:,k+1) - A21{iterInd-1}*X(:,k) - A22{iterInd-1}*zHat{iterInd}(:,k);
%         u{iterInd}(:,k) = L1General2_BBST(...
%             @(x) mahalanobisDist2(B1{iterInd-1}, Sigma_1{iterInd-1}, b1,...
%             B2{iterInd-1}, Sigma_2{iterInd-1}, b2, x),...
%             zeros(numInp,1),lambda*ones(numInp,1),options);
    end
    b1 = xHat(:,N) - A11{iterInd-1}*X(:,N-1) - A12{iterInd-1}*zHat{iterInd}(:,N-1);
%     u{iterInd}(:,N-1) = L1General2_BBST(...
%             @(x) mahalanobisDist1(B1{iterInd-1}, Sigma_1{iterInd-1}, b1, x),...
%             zeros(numInp,1),lambda*ones(numInp,1),options);
    AUse = B1{iterInd-1}/sqrt(Sigma_1{iterInd-1}(1,1));
    b = b1/sqrt(Sigma_1{iterInd-1}(1,1));
    lambda = lambdaBase/sqrt(Sigma_1{iterInd-1}(1,1));
    u{iterInd}(:,N-1) = getLassoModSoln(AUse, b, lambda);
        
    u_nm1 = [u0,u{iterInd}];
    
    PHatSumMat = PHat_0;
    for i = 1:size(PHat{iterInd},3)
        PHatSumMat = PHatSumMat + PHat{iterInd}(:,:,i);
    end
    colMat1 = [x_nm1Mat*xHat' ; z_nm1HatMat*xHat' ; u_nm1*xHat'];
    tempMat1 = [x_nm1Mat*x_nm1Mat', x_nm1Mat * z_nm1HatMat', x_nm1Mat*u_nm1';...
                z_nm1HatMat*x_nm1Mat', PHatSumMat + z_nm1HatMat*z_nm1HatMat', z_nm1HatMat*u_nm1';...
                u_nm1*x_nm1Mat', u_nm1*z_nm1HatMat', u_nm1*u_nm1'];
    solTemp1 = tempMat1 \ colMat1;
    A11{iterInd} = solTemp1(1:sizeX,:)';
    A12{iterInd} = solTemp1(sizeX+1:sizeX+sizeZ,:)';
    B1{iterInd} = solTemp1(sizeX+sizeZ+1:end,:)';
    
    Sigma_1{iterInd} = 1/N*(xHat*xHat' - A11{iterInd}*(x_nm1Mat*xHat') ...
        - A12{iterInd}*(z_nm1HatMat*xHat') - B1{iterInd}*(u_nm1*xHat'));
    
    Sigma_1{iterInd} = 1/sizeX*trace(Sigma_1{iterInd})*eye(sizeX);
    PHatSum_Nm1 = PHatSumMat - PHat{iterInd}(:,:,end);
    colMat2 = [x_nm1Mat(:,1:end-1)*zoMat';...
        PHatSum_Nm1*OrdMatZ(:,:,1)' + z_nm1HatMat(:,1:end-1)*zoMat';...
        u_nm1(:,1:end-1)*zoMat'];
    tempMat2 = [x_nm1Mat(:,1:end-1)*x_nm1Mat(:,1:end-1)', x_nm1Mat(:,1:end-1)*z_nm1HatMat(:,1:end-1)', x_nm1Mat(:,1:end-1)*u_nm1(:,1:end-1)';...
        z_nm1HatMat(:,1:end-1)*x_nm1Mat(:,1:end-1)', PHatSum_Nm1 + z_nm1HatMat(:,1:end-1)*z_nm1HatMat(:,1:end-1)', z_nm1HatMat(:,1:end-1)*u_nm1(:,1:end-1)';...
        u_nm1(:,1:end-1)*x_nm1Mat(:,1:end-1)', u_nm1(:,1:end-1)*z_nm1HatMat(:,1:end-1)', u_nm1(:,1:end-1)*u_nm1(:,1:end-1)'];
    solTemp2 = tempMat2 \ colMat2;
    
    A21{iterInd} = solTemp2(1:sizeX,:)';
    A22{iterInd} = solTemp2(sizeX+1:sizeX+sizeZ,:)';
    B2{iterInd} = solTemp2(sizeX+sizeZ+1:end,:)';
    
    Sigma_2{iterInd} = -A21{iterInd} * (x_nm1Mat(:,1:end-1) * zoMat')...
        -A22{iterInd}*(PHatSum_Nm1*OrdMatZ(:,:,1)' + z_nm1HatMat(:,1:end-1)*zoMat')...
        -B2{iterInd}*(u_nm1(:,1:end-1)*zoMat');
    
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
    
    Sigma_2{iterInd} = 1/sizeZ*trace(Sigma_2{iterInd})*eye(sizeZ);
    
%     matTemp1 = xHat*xHat' - 2*A11{iterInd}*(x_nm1Mat*xHat') ...
%         -2*A12{iterInd}*(z_nm1HatMat*xHat') + A11{iterInd} * (x_nm1Mat*x_nm1Mat')*A11{iterInd}' + ...
%         2*A12{iterInd}*(z_nm1HatMat*x_nm1Mat')*A11{iterInd}' ...
%         + A12{iterInd} * (PHatSumMat + z_nm1HatMat*z_nm1HatMat') * A12{iterInd}'...
%         -2*B1{iterInd}*(u_nm1*xHat') + 2*A11{iterInd}*(x_nm1Mat*u_nm1')*B1{iterInd}'...
%         +2*A12{iterInd}*(z_nm1HatMat*u_nm1')*B1{iterInd}'...
%         +B1{iterInd}*(u_nm1*u_nm1')*B1{iterInd}';
%     
%     matTemp2 = matSumAggr...
%         - 2*A21{iterInd}*(x_nm1Mat(:,1:end-1)*zoMat') + A21{iterInd} * (x_nm1Mat(:,1:end-1)*x_nm1Mat(:,1:end-1)') * A21{iterInd}' ...
%         + 2*A22{iterInd} * (z_nm1HatMat(:,1:end-1)*x_nm1Mat(:,1:end-1)') * A21{iterInd}' ...
%         - 2*A22{iterInd}*(PHatSum_Nm1*OrdMatZ(:,:,1)' + z_nm1HatMat(:,1:end-1)*zoMat')...
%         + A22{iterInd} * (PHatSum_Nm1 + z_nm1HatMat(:,1:end-1)*z_nm1HatMat(:,1:end-1)') * A22{iterInd}'...
%         -2*B2{iterInd}*(u_nm1(:,1:end-1)*zoMat') + 2*A21{iterInd}*(x_nm1Mat(:,1:end-1)*u_nm1(:,1:end-1)')*B2{iterInd}'...
%         +2*A22{iterInd}*(z_nm1HatMat(:,1:end-1)*u_nm1(:,1:end-1)')*B2{iterInd}'...
%         +B2{iterInd}*(u_nm1(:,1:end-1)*u_nm1(:,1:end-1)')*B2{iterInd}';
    
    QVal(iterInd) = -N/2*log(det(Sigma_1{iterInd})) - (N-1)/2*log(det(Sigma_2{iterInd}))...
        -1/2*sizeX*N -1/2*sizeZ*(N-1);
%         -1/2*trace(Sigma_1{iterInd}\matTemp1) - 1/2*trace(Sigma_2{iterInd}\matTemp2);
%     fprintf('t1=%f, t2=%f\n', trace(Sigma_1{iterInd}\matTemp1)/N, trace(Sigma_2{iterInd}\matTemp2)/(N-1));

    for i = 1:N-1
        HVal(iterInd) = HVal(iterInd) + 1/2*log(det(PHat{iterInd}(:,:,i)));
    end
    FVal(iterInd) = QVal(iterInd) + HVal(iterInd);
    Arec{iterInd} = [A11{iterInd},A12{iterInd};A21{iterInd},A22{iterInd}];
    fprintf('iteration  = %d, FVal = %f\n', iterInd, FVal(iterInd));
end
% figure;plot(FVal(2:end));grid;

% AHat = [A11{end},A12{end};A21{end},A22{end}];
% AHat = PMat * AHat * PMat';

numStep=5;
% baseInd = 1:10;

% [A_Incomplete,B_Incomplete,~,u_Incomplete,~] = modelEstNew('sensInd',1:length(oInd), 'numInp', numInp, 'data', X);
% [A_Incomplete,B_Incomplete,~] = performLeastSq(xHat, X);

xPred_wInCompleteData = predictValues(X, orderX, N, numStep, A11{1}, zeros(sizeX,sizeZ), zeros(sizeZ,N), B_Incomplete, u_Incomplete);
relErr_wInCompleteData = sqrt(sum((xPred_wInCompleteData(baseInd,:)-X(baseInd,:)).^2,2)/N)./sqrt(sum(X(baseInd,:).^2,2)/N);



xPred_wAlg = predictValues(X, orderX, N, numStep, A11{end}, A12{end}, [z0,zHat{end}], B1{end}, [u0,u{end}]);
relErr_wAlg = sqrt(sum((xPred_wAlg(baseInd,:)-X(baseInd,:)).^2,2)/N)./sqrt(sum(X(baseInd,:).^2,2)/N);

relErrOut = [mean(relErr_wInCompleteData), mean(relErr_wAlg)];
fprintf('rwInComplete=%f, rwAlg=%f\n', relErrOut(1), relErrOut(2));


% xGen = genFracData(AHat,order,Sigma_1{end},Sigma_2{end});
% figure;plot(dataComplete(1,:));hold on;plot(xGen(1,:));grid;
% figure;plot(dataComplete(2,:));hold on;plot(xGen(2,:));grid;

% figure;
% plot(1:length(dataComplete(1,:)), dataComplete(1,:), 1:length(dataComplete(2,:)), dataComplete(2,:), 1:length(dataComplete(3,:)), dataComplete(3,:));
% grid

function [xPred] = predictValues(X, order, T, numStep, A11, A12, z, B1, u)

global infit
% global numCh
sizeX = size(X,1);

TSteps =  ceil(T/numStep);

xPred = zeros(sizeX,T);
xPred(:,1:numStep) = X(:,1:numStep) + A12*z(:,1:numStep) + B1*u(:,1:numStep);

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
            + A12*z(:,(i-1)*numStep + stepInd) ...
            + B1*u(:,(i-1)*numStep + stepInd);
%         z and u are already arranged in one step behind order, so no need for
%         '-1' in the index
    end
    xPred(:,(i-1)*numStep+1:i*numStep) = XTemp(:,(i-1)*numStep+1:i*numStep);
end
% relErr = sqrt(mean(abs(xPred(chUse,:) - X(chUse,:)).^2));


function [zHat, zTilda, PHat, PTilda, K] = getKalmanFilteringSoln...
    (xHat, X, OrdMatZ, infit, A11, A12, A21, A22, B1, u, Sigma_1, Sigma_2, PHat_0, x0, z0)

N = size(X,2);
n = size(X,1);
m = size(A22,1);
% N-1 samples for hidden vars
zHat = zeros(m,N-1);
zTilda = zeros(m,N-1);
PHat = zeros(m,m,N-1);
PTilda = zeros(m,m,N-1);
K = zeros(m,n,N-1);
y = xHat(:,2:end) - A11*X(:,1:end-1) - B1*u;

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

% alpha = [0.7 1.1 0.8];
% A = [0,0.1,0.2;-0.01,-0.02,0.3;0.01,-0.03,-0.05];

alpha = [0.5;0.6;0.6];
A = [-0.73,-0.5,0.3;-0.2,-0.6,-0.3;0.03,-0.5,-0.6];

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


function out = getLassoModSoln(A, b, lambda)


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


function [p,g] = mahalanobisDist1(A, Sigma, b, x)

y = A*x-b;
p = 1/2*(y'/Sigma*y);
g = A'/Sigma*y;

function [p,g] = mahalanobisDist2(A1, Sigma_1, b1, A2, Sigma_2, b2, x)

y1 = A1*x-b1;
y2 = A2*x-b2;
p = 1/2*(y1'/Sigma_1*y1 + y2'/Sigma_2*y2);
g = A1'/Sigma_1*y1 + A2'/Sigma_2*y2;


function z = shrinkage(x, kappa)
    
z = max( 0, x - kappa ) - max( 0, -x - kappa );

function [L, U] = factor(A, rho)
    
[m, n] = size(A);
if ( m >= n )    % if skinny
   L = chol( A'*A + rho*speye(n), 'lower' );
else            % if fat
   L = chol( speye(m) + 1/rho*(A*A'), 'lower' );
end

% force matlab to recognize the upper / lower triangular structure
L = sparse(L);
U = sparse(L');
