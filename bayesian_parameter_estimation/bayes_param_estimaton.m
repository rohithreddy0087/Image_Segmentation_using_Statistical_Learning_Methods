%% Reading Inputs
d1.bg = load('TrainingSamplesDCT_subsets_8').D1_BG;
d1.fg = load('TrainingSamplesDCT_subsets_8').D1_FG;

d2.bg = load('TrainingSamplesDCT_subsets_8').D2_BG;
d2.fg = load('TrainingSamplesDCT_subsets_8').D2_FG;

d3.bg = load('TrainingSamplesDCT_subsets_8').D3_BG;
d3.fg = load('TrainingSamplesDCT_subsets_8').D3_FG;

d4.bg = load('TrainingSamplesDCT_subsets_8').D4_BG;
d4.fg = load('TrainingSamplesDCT_subsets_8').D4_FG;
datasets = [d1,d2,d3,d4];

alpha = load('Alpha').alpha;

prior1.w0 = load('Prior_1').W0;
prior1.mu0_FG = load('Prior_1').mu0_FG;
prior1.mu0_BG = load('Prior_1').mu0_BG;

prior2.w0 = load('Prior_2').W0;
prior2.mu0_FG = load('Prior_2').mu0_FG;
prior2.mu0_BG = load('Prior_2').mu0_BG;
prior = [prior1,prior2];
img = imread('cheetah.bmp');
zig_zag_pat = uint8(readmatrix('Zig-Zag Pattern.txt'));
gt = imread('cheetah_mask.bmp');

%% Computing error for each dataset and each alpha

error_prob_pred = zeros(length(prior),length(datasets),length(alpha));
error_prob_ML = zeros(length(prior),length(datasets),length(alpha));
error_prob_MAP = zeros(length(prior),length(datasets),length(alpha));
for p=1:1:length(prior)
    fprintf("Prior %d\n",p);
    pri = prior(p);
    mu0_fg = pri.mu0_FG;
    mu0_bg = pri.mu0_BG;
    w0 = pri.w0;
    for d=1:1:length(datasets)
        data = datasets(d);
        fprintf("==== Dataset %d\n",d);
        [n_fg,~] = size(data.fg);
        [n_bg,num_features] = size(data.bg);
        sigma_fg = cov(data.fg);
        sigma_bg = cov(data.bg);
        mu_ml_fg = mean(data.fg);
        mu_ml_bg = mean(data.bg);
        py_cheetah = n_fg/(n_fg + n_bg);
        py_grass = n_bg/(n_fg + n_bg);
        for a=1:1:length(alpha)
            fprintf("======== Alpha %d\n",a);
            sigma0 = alpha(a)*diag(w0);
            [mu_n_fg,sigma_n_fg] = compute_mu_n(mu_ml_fg,mu0_fg,sigma_fg,sigma0,n_fg);
            [mu_n_bg,sigma_n_bg] = compute_mu_n(mu_ml_bg,mu0_bg,sigma_bg,sigma0,n_bg);
            cov_fg = sigma_fg + sigma_n_fg;
            cov_bg = sigma_bg + sigma_n_bg;
            binary_mask_predictive = generate_mask(img,mu_n_fg,cov_fg,mu_n_bg,cov_bg,py_cheetah,py_grass,zig_zag_pat);
            error_prob_pred(p,d,a) = calculate_error(binary_mask_predictive,gt);
            fprintf("=========== Error Predctive %f\n",error_prob_pred(p,d,a));
            binary_mask_ML = generate_mask(img,mu_ml_fg,sigma_fg,mu_ml_bg,sigma_bg,py_cheetah,py_grass,zig_zag_pat);
            error_prob_ML(p,d,a) = calculate_error(binary_mask_ML,gt);
            fprintf("=========== Error ML %f\n",error_prob_ML(p,d,a));
            binary_mask_MAP = generate_mask(img,mu_n_fg,sigma_fg,mu_n_bg,sigma_bg,py_cheetah,py_grass,zig_zag_pat);
            error_prob_MAP(p,d,a) = calculate_error(binary_mask_MAP,gt);
            fprintf("=========== Error MAP %f\n",error_prob_MAP(p,d,a));
        end
    end
end

plot_curves(alpha,error_prob_pred,error_prob_ML,error_prob_MAP)
%% Generating binary mask or segmenting the given image using set of features

function binary_mask = generate_mask(img,mean_fg,cov_fg,mean_bg,cov_bg,py_cheetah,py_grass,zig_zag_pat)
    img = im2double(img);
    [row,col] = size(img);
    binary_mask = zeros(row,col);
    features = (1:1:64);
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
    if y_fg>y_bg
        cls = 0;
    else
        cls = 1;
    end
end

function val = decision(features,mean,cov,py)
    [row,col] = size(features);
    t = features-mean;
    d = (t/cov)*transpose(t);
    a = col*log(2*pi)+log(det(cov))-2*log(py);
    val = d + a;
end

function [mu_n,sigma_n] = compute_mu_n(mu_ml,mu0,sigma,sigma0,n)
    E = sigma./n;
    comm_term = inv(sigma0+E);
    first_term = sigma0*comm_term*transpose(mu_ml);
    second_term = E*comm_term*transpose(mu0);
    mu_n = transpose(first_term + second_term);
    sigma_n = sigma0*comm_term*E;
end
%% Plots class ditribution curves for given features
function plot_curves(alpha,error_prob_pred,error_prob_ML,error_prob_MAP)
    [priors,datasets,len_a] = size(error_prob_pred);
    for p=1:1:priors
        for d=1:1:datasets
            f = figure();
            f.Position = [0 0 2000 2000];
            pred = error_prob_pred(p,d,1:end);
            ml = error_prob_ML(p,d,1:end);
            map = error_prob_MAP(p,d,1:end);
            plot(alpha,reshape(pred,1,len_a))
            hold on
            plot(alpha,reshape(ml,1,len_a))
            plot(alpha,reshape(map,1,len_a))
            hold off
            set(gca,'XScale', 'log');
            t = sprintf("Prior %d Dataset %d",p,d);
            title(t)
            legend('Predictive','ML','MAP')
            save = sprintf("Prior %d Dataset %d.png",p,d);
            saveas(f,save)
        end
    end
    hold off
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