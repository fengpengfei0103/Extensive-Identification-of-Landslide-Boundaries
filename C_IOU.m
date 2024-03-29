% IOU

% 加载图像
SEG = imread('C:\Users\NFZC\Desktop\毕设\test01\testLabel256\qxg095.png');
GT = imread('C:\Users\NFZC\Desktop\毕设\test01\testLabel256\qxg095.png');
% binarize(0~255 to 0~1)
% SEG = imbinarize(SEG, 0.3);
% GT = imbinarize(GT, 0.1);

iou = Cal_IOU(SEG, GT);
[iou1,yl] = Calc_IOU(SEG, GT)
% 计算IOU
function iou = Cal_IOU(SEG, GT)
    [rows, cols] = size(SEG);
    
    % 统计标签GT、分割结果SEG中像素值为1的像素个数
    % 初始化
    label_area = 0; % 标签图像的面积（即总像素个数）
    seg_area = 0;   % 分割结果的面积
    intersection_area = 0; % 相交区域面积
    combine_area = 0;      % 两个区域联合面积

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
    % fprintf("intersection_area = %f\n",intersection_area);
    % fprintf("label_area = %f\n",label_area);
    % fprintf("seg_area = %f\n",seg_area);
    
    combine_area = combine_area + label_area + seg_area - intersection_area;
    % fprintf("combine_area = %f\n",combine_area);
    
    % 计算iou
    iou = double(intersection_area) / double(combine_area);
    fprintf('IOU = %f\n', iou);
    % figure(1)
    % subplot(1,2,1)
    % imshow(SEG);title('SEG')
    % subplot(1,2,2)
    % imshow(GT);title('GT')
end
