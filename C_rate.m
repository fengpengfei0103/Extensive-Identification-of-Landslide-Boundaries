% TPR，FPR，TNR

% load image
SEG = imread('C:\Users\NFZC\Desktop\毕设\test01\testLabel256\qxg095.png');
GT = imread('C:\Users\NFZC\Desktop\毕设\test01\testLabel256\qxg095.png');
% binarize(0~255 to 0~1)
SEG = imbinarize(SEG, 0.3);
GT = imbinarize(GT, 0.1);

rate = Cal_RATE(SEG, GT);

% 计算TPR,FPR,TNP
function rate = Cal_RATE(SEG, GT)
    [rows, cols] = size(SEG);
    total_area = rows * cols;
    
    % 统计标签GT、分割结果SEG中像素值为1的像素个数
    % 初始化
    label_area = 0; % 标签图像的面积
    seg_area = 0;   % 分割结果的面积
    intersection_area = 0; % 相交区域面积

    % 计算各部分的面积
    for i = 1: rows
        for j = 1: cols
            % 均被分为前景
            if GT(i, j)==1 && SEG(i, j)==1
                intersection_area = intersection_area + 1;
                label_area = label_area + 1;
                seg_area = seg_area + 1;
            % 误分割为背景（false negtive）
            elseif GT(i, j)==1 && SEG(i, j)~=1
                label_area = label_area + 1;
            % 误分割为前景（false positive）
            elseif GT(i, j)~=1 && SEG(i, j)==1
                seg_area = seg_area + 1;
            end
        end
    end
    
    % true positive rate
    tpr = double(intersection_area) / double(label_area);
    fprintf("TPR = %f\n", tpr);
    % false positive rate
    fpr = double(seg_area - intersection_area) / double(total_area - label_area);
    fprintf("FPR = %f\n", fpr);
    % true negtive rate
    tnr = 1 - fpr;
    fprintf("TNR = %f\n", tnr);
    
    rate = [tpr, fpr, tnr];
    
end
