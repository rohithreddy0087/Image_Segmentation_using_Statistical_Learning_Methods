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

%% Run EM on for all dimensions and compute error probabilty
C = 8;
[pi_fg,mean_fg,cov_fg] = expectation_maximization(C,fg);
[pi_bg,mean_bg,cov_bg] = expectation_maximization(C,bg);

dimensions = [1,2,4,8,16,24,32,40,48,56,64];
for dim = 1:length(dimensions)
    k = dimensions(dim);
    features = (1:1:k);
    binary_mask = generate_mask(img,features,mean_fg,cov_fg,mean_bg,cov_bg,pi_fg,pi_bg,zig_zag_pat);
    error(dim) = calculate_error(binary_mask,gt);
end


%% Expectation Maximization
function [pi_em,mu_em,sigma_em] = expectation_maximization(C, train_data)
    EMLimit = 1000;
    samples = size(train_data,1);
    dim = size(train_data,2);
    
    pi_em = randi(1, C);         
    pi_em = pi_em / sum(pi_em); 

    % Initialize mu_c by choosing C random observations in the sample data.
    mu_em = train_data(randi([1 samples],1,C),:);

    % Initialize sigma_c by creating an identity matrix of random values.
    sigma_em = zeros(dim,dim,C);
    for i =1:C
        sigma_em(:,:,i) = (rand(1,dim)).*eye(dim);
    end   

    gaussian_prob = zeros(samples,C);
    for i = 1:EMLimit
        % ---------- E-step ----------
        % Compute hIJ by Gaussian gaussian_prob for P_Z|X using mu, sigma and pi.
        for j = 1:C
            gaussian_prob(:,j) = mvnpdf(train_data,mu_em(j,:),sigma_em(:,:,j))*pi_em(j);    
        end
        hIJ = gaussian_prob./sum(gaussian_prob,2);
        
        % log-likelihood of the resulting gaussian_prob data.
        logLikelihood(i) = sum(log(sum(gaussian_prob,2)));

        % ---------- M-step ----------
        pi_em = sum(hIJ)/samples;
        mu_em = (hIJ'*train_data)./sum(hIJ)';
        for j = 1:C
            sigma_em(:,:,j) = diag(diag(((train_data-mu_em(j,:))'.*hIJ(:,j)'* ... 
                (train_data-mu_em(j,:))./sum(hIJ(:,j),1))+0.0000001));
        end

        % If likelihood hasn't changed by more than .1% between iteration stop.
        if i > 1
            if abs(logLikelihood(i) - logLikelihood(i-1)) < 0.001
                break; 
            end
        end
    end
end
%% Generating binary mask or segmenting the given image using set of features
function binary_mask = generate_mask(img,features,mean_fg,cov_fg,mean_bg,cov_bg,pi_fg,pi_bg,zig_zag_pat)
    img = im2double(img);
    [row,col] = size(img);
    binary_mask = zeros(row,col);
    for i=1:1:row
        for j=1:1:col
            if j+8 < col && i+8 < row
                tmp = img(i:i+7,j:j+7);
                dct_array = dct2(tmp,[8,8]);
                inp = zig_zag_converter(dct_array,zig_zag_pat,features);
                y = BDR(inp,mean_fg,mean_bg,cov_fg,cov_bg,pi_fg,pi_bg);
            else
                y = 0;
            end
            binary_mask(i,j) = y;
        end
    end
end

%%  Bayes decision rule for multivariate gaussian
function cls = BDR(features,mean_fg,mean_bg,cov_fg,cov_bg,pi_fg,pi_bg)
    y_fg = decision(features,mean_fg,cov_fg,pi_fg);
    y_bg = decision(features,mean_bg,cov_bg,pi_bg);
    if y_fg>y_bg
        cls = 1;
    else
        cls = 0;
    end
end

function val = decision(features,mean,cov,pi)
    val = 0;
    k = length(features);
    for y = 1:size(mean,1)
        val = val + mvnpdf(features,mean(y,1:k),cov(1:k,1:k,y))*pi(y);
    end
end

%%  Gaussian Distribution N(mean,sigma)
function v = gauss_distribution(mu,sigma,x)
    [~,col] = size(x);
    v = (1/sqrt((2*pi).^col).*det(sigma))*exp(-0.5.*(transpose(x-mu)/sig)*(x-mu));
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