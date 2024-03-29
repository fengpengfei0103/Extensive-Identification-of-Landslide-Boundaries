% 函数功能：计算两张图像的IOU
% lable_img：输入的图像A，只包含0和255像素值的图像
% res_img:输入的图像B，也同样只包含0和255像素值的图像
% 返回值：iou两张图像的iou；yl假阳性率
function [iou,yl] = Calc_IOU(lable_img, res_img)
[rows, cols] = size(lable_img);

% 计算总面积
%total_area = rows * cols;

% 统计lable_img、res_img中255像素值的个数
lable_area = 0; % 标记出来的面积
res_area = 0;   % 分割出来结果的面积
intersection_area = 0; % 相交区域的面积
combine_area = 0;      % 两个区域联合的面积

% 开始计算各部分的面积
for i = 1: 1: rows
    for j = 1: 1: cols
        if lable_img(i, j)==255 && res_img(i, j)==255
            intersection_area = intersection_area + 1;
            lable_area = lable_area + 1;
            res_area = res_area + 1;
        elseif lable_img(i, j)==255 && res_img(i, j)~=255
            lable_area = lable_area + 1;
        elseif lable_img(i, j)~=255 && res_img(i, j)==255
            res_area = res_area + 1;
        end
    end
end
combine_area = combine_area + lable_area + res_area - intersection_area;

% 得到IOU
iou = double(intersection_area) / double(combine_area);
fprintf('IOU: %f\n', iou);
% 得到假阳性率
yl = double(res_area - intersection_area) / double(combine_area);
fprintf('假阳性率为：%f\n', yl);

end