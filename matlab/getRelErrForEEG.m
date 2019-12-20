clear

matObj = matfile('S001R05T2_bothFeet.mat');
data = matObj.data;
data = data(:,1:650);
chSel = [1:21];
% chSel = [1:12, 56:64];

data = bsxfun(@minus, data, mean(data,2));
data = data(chSel,:);
chTotal = length(chSel);

order = zeros(1,chTotal);
for i = 1:chTotal
    order(i) = WT_estimator_v3(data(i,:),1);
end
baseInd = ceil(length(chSel)/2)+1;
numInp = ceil(baseInd/2);
indTuple = {1:baseInd,setdiff(1:chTotal,1:baseInd)};
for i = baseInd+1:chTotal-1
    indTuple = [indTuple;{1:i, setdiff(1:chTotal,1:i)}];
end

relErr = zeros(size(indTuple,1),2);
for i = 1:size(indTuple,1)
    relErr(i,:) = latentFracUU(data, order, numInp, indTuple{i,1}, indTuple{i,2}, 1:baseInd);
end