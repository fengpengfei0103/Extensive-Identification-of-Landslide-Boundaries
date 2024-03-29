%from 费南多
%email：571428374@qq.com
%TIME:2023-10-18

clc
clear
%%
%Load Data

dataFolder  = fullfile('C:\Users\NFZC\Desktop\DeeplabV3\test01');

imageDir = fullfile(dataFolder,'trainingImage256');
labelDir = fullfile(dataFolder,'trainingLabel256');

testimageDir = fullfile(dataFolder,'testImage256');
testlabelDir = fullfile(dataFolder,'testLabel256');
%创建包含图像的 ImageDatastore
imds = imageDatastore(imageDir);

testimds = imageDatastore(testimageDir);
% 创建一个包含基本真实像素标签的 PixelLabelDatastore。
% 该数据集有两个类别: "landsilde" and "background".
classNames = ["landsilde","background"];
% classNames = ["255","0"];
labelIDs   = [255 0];

pxds = pixelLabelDatastore(labelDir,classNames,labelIDs);

testpxds = pixelLabelDatastore(testlabelDir,classNames,labelIDs);
% 使用 combine 函数将图像数据存储和像素标签数据存储合并为一个 CombinedDatastore 对象。
% 合并后的数据存储可保持底层数据存储中一对图像之间的奇偶性。
cds = combine(imds,pxds);
% 调整训练图像的大小
% imageSize = [256 256];

% tds = transform(cds, @(data)preprocessTrainingData(data,imageSize));
tds = cds;
dsTrain = tds;
testcds = combine(testimds,testpxds);
% testtds = transform(testcds, @(data)preprocessTrainingData(data,imageSize));
testtds = testcds;
dsValidation = testtds;
%%
%构建网络
imageSize = [256 256];
classNames = ["landsilde","background"];
numClasses = numel(classNames);
% network = 'resnet18';
% lgraph = deeplabv3plusLayers(imageSize,numClasses,network, ...
%              'DownsamplingFactor',16);
lgraph = deeplabv3plusLayers(imageSize,numClasses,'resnet18');

analyzeNetwork(lgraph)
%%
%设置训练选项
opts = trainingOptions("adam",...
    "ExecutionEnvironment","auto",...
    "InitialLearnRate",0.01,...
    "MaxEpochs",20,...
    "MiniBatchSize",10,...
    "Shuffle","every-epoch",...
    "Plots","training-progress",...
    "ValidationData",dsValidation);

%%
%训练网络
[net, traininfo] = trainNetwork(dsTrain,lgraph,opts);
% save('net_adam_resnet18_0.0001_20_9616.mat','net')
%%
%测试

% 使用测试数据和经过训练的网络进行预测。
% 使用 分割测试图像。使用该函数在图像上显示标签。semanticseglabeloverlay
loaded_model = load('net_adam_resnet18_0.0001_20_9616.mat');
net = loaded_model.net;

imgTest = imread('C:\Users\NFZC\Desktop\DeeplabV3\test01\testImage256\zj096.png');
[testSeg, score, allScores]= semanticseg(imgTest,net);
testImageSeg = labeloverlay(imgTest,testSeg);
% 显示结果。
figure
% imshow(testImageSeg)
imshow(testImageSeg,'border','tight','initialmagnification','fit');
axis normal;

pxdsTruth = testpxds;
% Run semantic segmentation on all of the test images with a batch size of 4.
% You can increase the batch size to increase throughput based on your systems memory resources.
tempdirR = 'C:\Users\NFZC\Desktop\DeeplabV3\test01\RESULTS';
pxdsResults = semanticseg(testimds,net,'MiniBatchSize',1,'WriteLocation',tempdirR);
% Compare the results against the ground truth.
metrics = evaluateSemanticSegmentation(pxdsResults,pxdsTruth)

 % imgT01 = imread('C:\Users\NFZC\Desktop\毕设\test01\RESULTS\R50_0.001_200\semanticsegOutput\pixelLabel_008_qxg095.png')

% I = imread('triangleTest.jpg');
% 
% % Resize the test image by a factor equal to the input image size divided 
% % by 32 so that the triangles in the test image are roughly equal to the size of the triangles during training.
% I = imresize(I,'Scale',imageSize./32);
% 
% % Segment the image.
% C = semanticseg(I,net);
% % 显示结果。
% B = labeloverlay(I,C);
% figure
% imshow(B)



% testSeg = semanticseg(imgTest,net);
% testImageSeg = labeloverlay(imgTest,testSeg);
% % 显示结果。
% figure
% imshow(testImageSeg)

%%
%Supporting Functions
function data = preprocessTrainingData(data, imageSize)
% Resize the training image and associated pixel label image.
data{1} = imresize(data{1},imageSize);
data{2} = imresize(data{2},imageSize);

% Convert grayscale input image into RGB for use with ResNet-18, which
% requires RGB image input.
data{1} = repmat(data{1},1,1,3);
end