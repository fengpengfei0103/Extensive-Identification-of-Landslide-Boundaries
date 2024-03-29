%from 费南多
%email：571428374@qq.com
%TIME:2023-10-19
%UNet Segmentation Network
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
labelIDs   = [255 0];

pxds = pixelLabelDatastore(labelDir,classNames,labelIDs);

testpxds = pixelLabelDatastore(testlabelDir,classNames,labelIDs);
% 使用 combine 函数将图像数据存储和像素标签数据存储合并为一个 CombinedDatastore 对象。
% 合并后的数据存储可保持底层数据存储中一对图像之间的奇偶性。
cds = combine(imds,pxds);
dsTrain = cds;
testcds = combine(testimds,testpxds);
dsValidation = testcds;
%%
%构建网络
imageSize = [256 256 3];
numClasses = 2;
% lgraph = unetLayers(imageSize, numClasses);
encoderDepth = 4;
lgraph = unetLayers(imageSize,numClasses,'EncoderDepth',encoderDepth);

analyzeNetwork(lgraph)
%%
%设置训练选项
% options = trainingOptions('sgdm', ...
%     'InitialLearnRate',0.01, ...
%     'MaxEpochs',2, ...
%     "Plots","training-progress",...
%     'VerboseFrequency',10);

opts = trainingOptions("sgdm",...
    "ExecutionEnvironment","auto",...
    "InitialLearnRate",0.001,...
    "MaxEpochs",20,...
    "MiniBatchSize",10,...
    "Shuffle","every-epoch",...
    "Plots","training-progress",...
    "ValidationData",dsValidation);
%%
%训练网络
[net, traininfo] = trainNetwork(dsTrain,lgraph,opts);

%%
%测试
% 使用测试数据和经过训练的网络进行预测。
% 使用 分割测试图像。使用该函数在图像上显示标签。semanticseglabeloverlay
imgTest = imread('C:\Users\NFZC\Desktop\DeeplabV3\test01\testImage256\qxg095.png');
testSeg = semanticseg(imgTest,net);
testImageSeg = labeloverlay(imgTest,testSeg);
% 显示结果。
figure
imshow(testImageSeg)
% imgTest = imread('triangleTest.jpg');
% testSeg = semanticseg(imgTest,net);
% testImageSeg = labeloverlay(imgTest,testSeg);
% % 显示结果。
% figure
% imshow(testImageSeg)