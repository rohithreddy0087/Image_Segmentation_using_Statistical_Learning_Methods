%% Reading Inputs
training_data = load('TrainingSamplesDCT_8_new');
fg = training_data.TrainsampleDCT_FG;
bg = training_data.TrainsampleDCT_BG;
img = imread('cheetah.bmp');
zig_zag_pat = uint8(readmatrix('Zig-Zag Pattern.txt'));
gt = imread('cheetah_mask.bmp');

%% Computing Prior probabilties
fg_size = size(fg);
bg_size = size(bg);
py_cheetah = fg_size(1)/(fg_size(1) + bg_size(1));
py_grass = bg_size(1)/(fg_size(1) + bg_size(1));
fprintf("Prior probability of cheetah is %f \n",py_cheetah);
fprintf("Prior probability of grass is %f \n",py_grass);

%% Segmenting given image using 64 features
features = (1:1:64);
[mean_fg,var_fg,cov_fg] = get_gauss_params(fg,features);
[mean_bg,var_bg,cov_bg] = get_gauss_params(bg,features);
% plot_features(features,mean_fg,mean_bg,var_fg,var_bg,'Features')
binary_mask = generate_mask(img,features,mean_fg,cov_fg,mean_bg,cov_bg,py_cheetah,py_grass,zig_zag_pat);
figure()
imagesc(binary_mask);
colormap(gray(255));
title('Segmented cheetah using 64 features');
error_prob = calculate_error(binary_mask,gt);
fprintf("Probability of error using 64 features is %f \n",error_prob);

%% Segmenting given image using best 8 features
best_features = [1,2,3,4,5,6,7,8]; %% 9,14
[mean_fg,var_fg,cov_fg] = get_gauss_params(fg,best_features);
[mean_bg,var_bg,cov_bg] = get_gauss_params(bg,best_features);
plot_features(best_features,mean_fg,mean_bg,var_fg,var_bg,'Best Features')
binary_mask = generate_mask(img,best_features,mean_fg,cov_fg,mean_bg,cov_bg,py_cheetah,py_grass,zig_zag_pat);
figure()
imagesc(binary_mask);
colormap(gray(255));
title('Segmented cheetah using best 8 features');
error_prob = calculate_error(binary_mask,gt);
fprintf("Probability of error using best 8 features is %f \n",error_prob);

%% Plotting worst features
worst_features = [64,63,59,62,60,58,61,50]; %% 57, 49
[mean_fg,var_fg,cov_fg] = get_gauss_params(fg,worst_features);
[mean_bg,var_bg,cov_bg] = get_gauss_params(bg,worst_features);
plot_features(worst_features,mean_fg,mean_bg,var_fg,var_bg,'Worst Features')

%% Generating binary mask or segmenting the given image using set of features
function binary_mask = generate_mask(img,features,mean_fg,cov_fg,mean_bg,cov_bg,py_cheetah,py_grass,zig_zag_pat)
    img = im2double(img);
    [row,col] = size(img);
    binary_mask = zeros(row,col);
    for i=1:1:row
        for j=1:1:col
            if j+8 < col && i+8 < row
                tmp = img(i:i+7,j:j+7);
                dct_array = dct2(tmp,[8,8]);
                inp = zig_zag_converter(dct_array,zig_zag_pat,features);
                y = BDR(inp,mean_fg,mean_bg,cov_fg,cov_bg,py_cheetah,py_grass);
            else
                y = 0;
            end
            binary_mask(i,j) = y;
        end
    end
end

%%  Bayes decision rule for multivariate gaussian
function cls = BDR(features,mean_fg,mean_bg,cov_fg,cov_bg,py_fg,py_bg)
    y_fg = decision(features,mean_fg,cov_fg,py_fg);
    y_bg = decision(features,mean_bg,cov_bg,py_bg);
    if y_fg<y_bg
        cls = 1;
    else
        cls = 0;
    end
end

function val = decision(features,mean,cov,py)
    [row,col] = size(features);
    t = features-mean;
    d = (t/cov)*transpose(t);
    a = col*log(2*pi)+log(det(cov))-2*log(py);
    val = d + a;
end

%% Estimating parameters from training set for a given set of features
function [m,var,covar] = get_gauss_params(T,features)
    X_size = size(features);
    T_size = size(T);
    X = zeros(T_size(1),X_size(2));
    for j=1:1:X_size(2)
        X(:,j) = T(:,features(j));
    end
    m = mean(X);
    var = zeros(1,X_size(2));
    for j=1:1:X_size(2)
        t = X(:,j)-m(j);
        var(1,j) =  sqrt(mean(t.^2));
    end
    covar = cov(X);
end

%% Plots class ditribution curves for given features
function plot_features(features,mean_fg,mean_bg,var_fg,var_bg,ti)
    fg_size = size(features);
    areas = size(features);
    for k=0:1:(fg_size(2)/4)-1
        f = figure();
        f.Position = [0 0 2000 2000];
        hold on
        count = 1;
        for i=4*k+1:1:4*(k+1)
            f = features(i);
            u_fg = mean_fg(i);
            u_bg = mean_bg(i);
            s_fg = var_fg(i);
            s_bg = var_bg(i);
            subplot(2,2,count)
            x1 = linspace(u_fg-3*s_fg,u_fg+3*s_fg,10000);
            x2 = linspace(u_bg-3*s_bg,u_bg+3*s_bg,10000);
            y1 = gauss_distribution(u_fg,s_fg,x1);
            y2 = gauss_distribution(u_bg,s_bg,x2);
            y_d = [y2(y2<y1) y1(y1<y2)]; 
            areas(i)  = trapz(y_d);
%             fprintf("%d %d \n",f,area_int);
            plot(x1,y1,x2,y2);
            t = sprintf("Feature %d",f);
            title(t)
            legend('cheetah','grass')
            count = count + 1;
        end
        sgtitle(ti)
    end
    hold off
%     [B I] = maxk(areas,10);
%     disp(B);
%     disp(I);
%     [B I] = mink(areas,10);
%     disp(B);
%     disp(I);
end

%%  Gaussian Distribution N(mean,sigma)
function v = gauss_distribution(u,sig,x)
    v = 1/(sig*sqrt(2*pi))*exp(-0.5*((x-u)/sig).^2);
end

%% Error calculation
function error = calculate_error(mask,gt)
    [row,col] = size(mask);
    count = sum(sum(xor(mask,gt)));
    error = count/(row*col);
end

%% Zig Zag converter
function ret = zig_zag_converter(array,zig_zag_pat,features)
    [row,col] = size(array);
    fg_size = size(features);
    vec = zeros([1,row*col]);
    ret = zeros(fg_size);
    for i=1:1:row
        for j=1:1:col
            vec(zig_zag_pat(i,j)+1) = array(i,j);
        end
    end
    for i=1:1:fg_size(2)
        ret(i) = vec(features(i));
    end
end