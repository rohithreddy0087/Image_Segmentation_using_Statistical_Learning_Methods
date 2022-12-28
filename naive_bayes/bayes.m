%% Reading Inputs
training_data = load('TrainingSamplesDCT_8');
fg = training_data.TrainsampleDCT_FG;
bg = training_data.TrainsampleDCT_BG;
img = imread('cheetah.bmp');
zig_zag_pat = uint8(readmatrix('Zig-Zag Pattern.txt'));
gt = imread('cheetah_mask.bmp');

%% Computing prior and class distribution probabilty
fg_size = size(fg);
bg_size = size(bg);
py_cheetah = fg_size(1)/(fg_size(1) + bg_size(1));
py_grass = bg_size(1)/(fg_size(1) + bg_size(1));
fprintf("Prior probability of cheetah is %f \n",py_cheetah);
fprintf("Prior probability of grass is %f \n",py_grass);
bins = 64;
[Mfg,Xfg] = max(abs(fg(:,2:end)),[],2);
Xfg = Xfg+1;
[hist_fg,edges_fg] = histcounts(Xfg,bins);

[Mbg,Xbg] = max(abs(bg(:,2:end)),[],2);
Xbg = Xbg+1;
[hist_bg,edges_bg] = histcounts(Xbg,bins);

p_fg = zeros(1,fg_size(2));
p_bg = zeros(1,bg_size(2));
for p=1:1:fg_size(2)
    p_fg(p) = p_x_given_y(p,hist_fg,edges_fg);
    p_bg(p) = p_x_given_y(p,hist_bg,edges_bg);
end

figure(1)
histogram(Xfg,bins,'Normalization', 'probability');
title('Histogram of P_{X|Y}(x|cheetah)');
xlabel('Features');
ylabel('Probabilities');
figure(2)
histogram(Xbg,bins,'Normalization', 'probability');
title('Histogram of P_{X|Y}(x|grass)');
xlabel('Features');
ylabel('Probabilities');

%% Testing the model on image
img = im2double(img);
[row,col] = size(img);
A = {};
ci = 1;
cj = 1;
binary_mask = [];
for i=1:1:row
    cj = 1;
    for j=1:1:col
        if j+8 < col && i+8 < row
            tmp = img(i:i+7,j:j+7);
        elseif j+8 > col && i+8 < row
            tmp = img(i:i+7,j:end);
        elseif j+8 < col && i+8 > row
            tmp = img(i:end,j:j+7);
        else
            tmp = img(i:end,j:end);
        end
        dct_array = dct2(tmp,[8,8]);
        dct_vec = zig_zag_converter(dct_array,zig_zag_pat);
        feature = get_second_largest(dct_vec);
        y = map_rule(feature,py_cheetah,py_grass,p_fg,p_bg);
        binary_mask(ci,cj) = y;
        cj = cj + 1;
    end
    ci = ci + 1;
end
figure(3)
imagesc(binary_mask);
colormap(gray(255));
title('Segmented cheetah using Minimum Probabilty of error rule nbins = 9');

%% Error calculation
count = sum(sum(xor(binary_mask,gt)));
error_prob = 100*count/(row*col);
fprintf("Probability of error is %f \n",error_prob*0.01);

%% Optimal Decision Function
function cls = map_rule(x,py_cheetah,py_grass,p_fg,p_bg)
    p_cheetah_given_x = p_fg(x)*py_cheetah;
    p_grass_given_x = p_bg(x)*py_grass;
    if p_grass_given_x > p_cheetah_given_x
        cls = 0;
    else
        cls = 1;
    end
%     [m,cls] = max([p_bg(x)*py_grass,p_fg(x)*py_cheetah]);
end

%% Class distribution function
function p = p_x_given_y(x,histcount,edges)
    dim = size(edges);
    count = 1;
    p = 0;
    for i=1:1:dim(2)-1
        if x <= edges(i+1) && x >= edges(i)  
            p = histcount(count)/sum(histcount);
        end
        count = count + 1;
    end
end

%% Helper functions for testing the image
function vec = zig_zag_converter(array,zig_zag_pat)
    [row,col] = size(array);
    vec = zeros([1,row*col]);
    for i=1:1:row
        for j=1:1:col
            vec(zig_zag_pat(i,j)+1) = array(i,j);
        end
    end
end

function x = get_second_largest(vec)
    [m,x] = max(abs(vec(1,2:end)));
    x=x+1;
end