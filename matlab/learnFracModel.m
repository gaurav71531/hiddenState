function learnFracModel()

% [X, y, A, C, order] = getTestData();
matObj = matfile('fracModelData.mat');
xData = matObj.X;
Aorig = matObj.A;
order = matObj.order;
Corig = matObj.C;
y = matObj.y;

infit = 20;
n = size(xData,1);
m = size(y,1);
N = size(y,2);
x0 = zeros(n,1);
OrdMatX = zeros(n,n,infit);
for j = 1:infit
    preFact = gamma(-order+j)./(gamma(-order).*gamma(j+1));
    OrdMatX(:,:,j) = diag(preFact);
end
PHat_0 = eye(n)*100;


[xHatTemp, xTildaTemp, PHatTemp, PTildaTemp, KTemp] = getKalmanFilteringSoln...
    (y, OrdMatX, infit, Aorig, Corig, eye(n), eye(m), PHat_0, x0);

% figure;plot(xData(1,:));hold on;plot(xHatTemp(1,:));
% figure;plot(xData(2,:));hold on;plot(xHatTemp(2,:));

niter = 70;
A = cell(1,niter);
C = cell(1,niter);
Sigma_1 = cell(1,niter);
Sigma_2 = cell(1,niter);

xHat = cell(1,niter);
xTilda = cell(1,niter);
PHat = cell(1,niter);
PTilda = cell(1,niter);
K = cell(1,niter);

e = [-0.5, -0.9];
[U,~,~] = svd(randn(n));
A{1} = U*diag(e)*U';
C{1} = randn(m,n);
% matObj = matfile('initTestParamFracModel.mat');
% ATemp = matObj.A;
% CTemp = matObj.C;
% A{1} = ATemp{1};
% C{1} = CTemp{1};
% C{1} = Corig;
% A{1} = Aorig;
Sigma_1{1} = eye(n);
Sigma_2{1} = eye(m);

% xHat{1} = x0;
QVal = zeros(niter,1);
HVal = zeros(niter,1);
FVal = zeros(niter,1);

[xHat{1}, ~, PHat{1}, ~, ~] = getKalmanFilteringSoln...
    (y, OrdMatX, infit, A{1}, C{1}, Sigma_1{1}, Sigma_2{1}, PHat_0, x0);

xoMat = zeros(n,N);
for i = 1:length(order)
    xoMat(i,:) = getFractionalExpan(xHat{1}(i,:), order(i), infit);
end
xHat_nm1 = [x0,xHat{1}(:,1:end-1)];
PHatSum = zeros(n);
for i = 1:size(PHat{1},3)
    PHatSum = PHatSum + PHat{1}(:,:,i);
end
PHatSum_nm1 = PHat_0 + PHatSum - PHat{1}(:,:,end);

matAggreg = PHatSum;
for j = 1:infit
    matUse = PHatSum_nm1;
    for i = 1:j-1
        matUse = matUse - PHat{1}(:,:,end-i);
    end
    matAggreg = matAggreg + OrdMatX(:,:,j)*matUse*OrdMatX(:,:,j)';
end

matTemp1 = matAggreg + xoMat*xoMat' - 2*A{1}*(PHatSum_nm1*OrdMatX(:,:,1) + xHat_nm1*xoMat')...
    +A{1}*(PHatSum_nm1 + xHat_nm1*xHat_nm1')*A{1}';

matTemp2 = y*y' - 2*C{1}*(xHat{1}*y') + C{1}*(PHatSum + xHat{1}*xHat{1}')*C{1}';

QVal(1) = -N/2*log(det(Sigma_1{1})) - N/2*log(det(Sigma_2{1})) ...
        - 1/2*trace(Sigma_1{1}\matTemp1)...
        - 1/2*trace(Sigma_2{1}\matTemp2);    
for i = 1:N
    HVal(1) = HVal(1) + 1/2*log(det(PHat{1}(:,:,i)));
end
FVal(1) = QVal(1) + HVal(1);
fprintf('iteration  = %d, Qval = %f\n', 1, FVal(1));


for iterInd = 2:niter
    [xHat{iterInd}, xTilda{iterInd}, PHat{iterInd}, PTilda{iterInd}, K{iterInd}] = getKalmanFilteringSoln...
    (y, OrdMatX, infit, A{iterInd-1}, C{iterInd-1}, Sigma_1{iterInd-1}, Sigma_2{iterInd-1}, PHat_0, x0);

    fprintf('mean xHat(1) error = %f\n', sqrt(mean((xData(1,:)-xHat{iterInd}(1,:)).^2)));
    
    xoMat = zeros(n,N);
    for i = 1:length(order)
        xoMat(i,:) = getFractionalExpan(xHat{iterInd}(i,:), order(i), infit);
    end
    xHat_nm1 = [x0,xHat{iterInd}(:,1:end-1)];
    PHatSum = zeros(n);
    for i = 1:size(PHat{iterInd},3)
        PHatSum = PHatSum + PHat{iterInd}(:,:,i);
    end
    PHatSum_nm1 = PHat_0 + PHatSum - PHat{iterInd}(:,:,end);
    
    tempMat1 = OrdMatX(:,:,1)*PHatSum_nm1 + xoMat*xHat_nm1';
    tempMat2 = PHatSum_nm1 + xHat_nm1*xHat_nm1';
    A{iterInd} = tempMat1/tempMat2;
    
    C{iterInd} = (y*xHat{iterInd}')/(PHatSum + xHat{iterInd}*xHat{iterInd}');
    
    matAggreg = PHatSum;
    for j = 1:infit
        matUse = PHatSum_nm1;
        for i = 1:j-1
            matUse = matUse - PHat{iterInd}(:,:,end-i);
        end
        matAggreg = matAggreg + OrdMatX(:,:,j)*matUse*OrdMatX(:,:,j)';
    end
    Sigma_1{iterInd} = (matAggreg + xoMat*xoMat' ...
        - A{iterInd}*(PHatSum_nm1*OrdMatX(:,:,1)' + xHat_nm1*xoMat'))/N;
    Sigma_2{iterInd} = (y*y' - C{iterInd}*xHat{iterInd}*y')/N;
    
%     matTemp1 = matAggreg + xoMat*xoMat' - 2*A{iterInd}*(PHatSum_nm1*OrdMatX(:,:,1) + xHat_nm1*xoMat')...
%         +A{iterInd}*tempMat2*A{iterInd}';
%     
%     matTemp2 = y*y' - 2*C{iterInd}*(xHat{iterInd}*y') + C{iterInd}*(PHatSum + xHat{iterInd}*xHat{iterInd}')*C{iterInd}';

    ee = eig(Sigma_1{iterInd});
    Sigma_1{iterInd} = diag(ee);
%     Sigma_1{iterInd} = diag(diag(Sigma_1{iterInd}));
    
    QVal(iterInd) = -N/2*log(det(Sigma_1{iterInd})) - N/2*log(det(Sigma_2{iterInd}))...
        -1/2*n*N -1/2*m*N;
%         - 1/2*trace(Sigma_1{iterInd}\matTemp1)...
%         - 1/2*trace(Sigma_2{iterInd}\matTemp2);
    
    for i = 1:N
        HVal(iterInd) = HVal(iterInd) + 1/2*log(det(PHat{iterInd}(:,:,i)));
    end
    FVal(iterInd) = QVal(iterInd) + HVal(iterInd);
    fprintf('iteration  = %d, FVal = %f\n', iterInd, FVal(iterInd));
end
AHat = A{end};
figure;plot(FVal(2:end));grid;
% subplot(3,1,1);plot(QVal(1:end));grid;
% subplot(3,1,2);plot(HVal(1:end));grid;
% subplot(3,1,3);plot(FVal(1:end));grid;
figure;plot(xData(1,:));hold on;plot(xHat{end}(1,:));
figure;plot(xData(2,:));hold on;plot(xHat{end}(2,:));
grid;




function [xHat, xTilda, PHat, PTilda, K] = getKalmanFilteringSoln...
    (y, OrdMatX, infit, A, C, Sigma_1, Sigma_2, PHat_0, x0)

N = size(y,2);
n = size(A,1);
m = size(y,1);

xHat = zeros(n,N);
xTilda = zeros(n,N);
PHat = zeros(n,n,N);
PTilda = zeros(n,n,N);
K = zeros(n,m,N);
% y = xHat(:,2:end) - A11*X(:,1:end-1);

PTilda(:,:,1) = (A - OrdMatX(:,:,1)) * PHat_0 * (A - OrdMatX(:,:,1))' + Sigma_1;
K(:,:,1) = PTilda(:,:,1) * C' / (Sigma_2 + C * PTilda(:,:,1) * C');
PHat(:,:,1) = PTilda(:,:,1) - K(:,:,1)*C*PTilda(:,:,1);
xTilda(:,1) = A*x0;
xHat(:,1) = xTilda(:,1) + K(:,:,1)*(y(:,1) - C*xTilda(:,1));

for n = 2:N
    PTilda(:,:,n) = (A - OrdMatX(:,:,1))*PHat(:,:,n-1)*(A-OrdMatX(:,:,1))' + Sigma_1;
    for j = 2:infit
        if n-j<1,break;end
        PTilda(:,:,n) = PTilda(:,:,n) + OrdMatX(:,:,j) * PHat(:,:,n-j) * OrdMatX(:,:,j)';
    end
    if n-j == 0
        PTilda(:,:,n) = PTilda(:,:,n) + OrdMatX(:,:,j) * PHat_0 * OrdMatX(:,:,j)';
    end
    K(:,:,n) = PTilda(:,:,n) * C' / (Sigma_2 + C * PTilda(:,:,n) * C');
    PHat(:,:,n) = PTilda(:,:,n) - K(:,:,n)*C*PTilda(:,:,n);
    
    xTilda(:,n) = A*xHat(:,n-1);
    for j = 1:infit
        if n-j<1,break;end
        xTilda(:,n) = xTilda(:,n) - OrdMatX(:,:,j)*xHat(:,n-j);
    end
    if n-j == 0
        xTilda(:,n) = xTilda(:,n) - OrdMatX(:,:,j)*x0;
    end
    
    xHat(:,n) = xTilda(:,n) + K(:,:,n)*(y(:,n) - C*xTilda(:,n));
end


function [x,y,A,C,alpha] = getTestData()

n = 2;
m = 1;
initState = zeros(n,1);
initCov = eye(n);

noiseCov = eye(n);


% alpha = [0.7, 1.2];
% A = [0,0.1;-0.01,-0.02];
C = [0.5 1];
% alpha = [0.7 1.2 0.2];
% A = [0,0.1,0.2;-0.01,-0.02,1;0.01,-0.03,0.5];

% alpha = [0.7 1.1 0.8];
% A = [0,0.1,0.2;-0.01,-0.02,0.3;0.01,-0.03,-0.05];

alpha = [0.5;0.6];
A = [-0.73,-0.5;-0.2,-0.6];

K = 1000;

x = zeros(n,K);
y = zeros(m,K);
x(:,1) = initState;
y(:,1) = C*x(:,1) + mvnrnd(zeros(m,1), eye(m));

infit = 20;

for k = 2:K
    x(:,k) = A * x(:,k-1) + mvnrnd(zeros(n,1), noiseCov)';
    for j = 1:n
        x(j,k) = x(j,k) - getFracRemainder(x(j,1:k-1), alpha(j), infit);
    end
    y(:,k) = C*x(:,k) + mvnrnd(zeros(m,1), eye(m));
end

figure;
plot(1:length(x(1,:)), x(1,:), 1:length(x(2,:)), x(2,:),1:length(y), y);
grid
% figure;
% plot(1:length(y), y);
% grid;

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

function y = getFractionalExpan(x, alpha, infit)

l = length(x);
j = 0:infit;
preFactVec = gamma(-alpha+j)./(gamma(-alpha).*gamma(j+1));

y = conv(x, preFactVec);
y = y(1:l);