# Image_Segmentation_using_Statistical_Learning_Methods
Image segmentation of a chettah image using different statistical learning algortihms

This project aims to segment the following cheetah image into its two components, cheetah (foreground) and grass (background).

![cheetah](naive_bayes/cheetah.bmp)

The goal is to obtain a mask image as follows:

![cheetah](naive_bayes/cheetah_mask.bmp)

## Discrete Cosine Transform(DCT)
To formulate this as a pattern recognition problem an observation space of 8x8 image blocks is used, i.e. each image as a collection of 8x8 blocks.
For each block DCT is computed as an array of 8 x 8 frequency coeffcients. Since cheetah and the grass have different
textures, with different frequency decompositions, the two classes can be better represented and seperated in the
frequency domain. Each 8 x 8 array is then converted into a 64 dimensional vector because it is easier
to work with vectors than with arrays. The file [Zig-Zag Pattern.txt](naive_bayes\Zig-Zag Pattern.txt) contains the position (in the 1D vector) of each coeffcient in the 8 x 8 array.

## Training data
Each method contains a training set of vectors obtained from a similar image (stored as a matrix, each row is a training vector) for each
of the classes. There are two matrices, TrainsampleDCT BG and TrainsampleDCT FG for foreground
and background samples in each training set respectively. All training sets are .mat files.
1. Naive bayes - [Training set](naive_bayes\TrainingSamplesDCT_8.mat)
2. Maximum Likelihood estimation - [Training set](maximum_likelihood_estimation\TrainingSamplesDCT_8_new.mat)
3. Bayes Parameter Estimation - [Training set](bayesian_parameter_estimation\TrainingSamplesDCT_subsets_8.mat), [Priors](bayesian_parameter_estimation\Prior_1.mat)
4. Expectation Maximization - [Trainin set](expectation_maximization\TrainingSamplesDCT_8_new.mat)

## Naive bayes
#### Reasonable estimates for prior probabilities
Prior probabilities are estimated using the training set, where the number of samples for each class are given. So a reasonable estimate for probability of a
class would be the fraction of samples of that respective class in the whole training set.
1. Prior probability of cheetah is 
	$$
	P_{y}(cheetah) = \frac{number\ of\ training\ samples\ in\ foreground}{Total\ number\ of\ samples} \\
	               = \frac{250}{250+1053} = 0.1919
	$$
2. Prior probability of grass is 
	$$
	P_{y}(grass) = \frac{number\ of\ training\ samples\ in\ background}{Total\ number\ of\ samples} \\
	               = \frac{1053}{250+1053} = 0.8081
	$$

#### Histograms of $P_{X|Y}(x|cheetah)$ and $P_{X|Y}(x|grass)$
From each training sample, feature is extracted by taking the second maximum element in the given 64 length DCT vector. All the features of a class for the training set are stored in the respective feature array and using the histogram of this array we compute $P_{X|Y}(x|given\_class)$. We use 64 bins to plot all the histograms. Each bar in histogram has edges, using which we can find out the position of a given feature in the histogram and later using this position we can find the frequency of the feature in the training set. Dividing this frequency with total number of samples for that class gives $P_{X|Y}(x|given\_class)$.
![Histogram cheetah](naive_bayes\results\b_cheetah.png)
![Histogram grass](naive_bayes\results\b_grass.png)

#### Segmentation of Cheetah
1. Using a $8\times8$ window, we slide through each pixel for the given image.
2. DCT for this $8\times8$ window is computed and then is converted into a feature vector using zig-zag sequence scanning.
3. Feature is taken as the second maximum element from the computed feature vector.
4. $P_{Y|X}(cheetah|x)$ and $P_{Y|X}(grass|x)$ are computed using Bayes rule.
	        $$P_{Y|X}(cheetah|x) = P_{X|Y}(x|cheetah)P_{y}(cheetah)$$
	        $$P_{Y|X}(grass|x) = P_{X|Y}(x|grass)P_{y}(grass)$$
5. From bayes decision rule,
            Y= 
                \begin{cases}
                    1,& \text{if } P_{Y|X}(cheetah|x) \geq P_{Y|X}(grass|x)\\
                    0,              & \text{otherwise}
                \end{cases}
6. Repeating above, class is assigned for each pixel and a segmentation mask is created.

#### Probabilty of Error
Using the given ground truth image, we perform logical xor operation between ground truth image and the obtained segmented mask. Logical xor gives all the pixel locations where pixels don't match, taking sum of all such pixels gives the count of error pixels.

$$
Probabilty\ of\ error = \frac{Count\ of\ Error\ pixels}{Total\ number\ of\ pixels} 
$$	
$$Probabilty\ of\ error =  0.1699$$

#### Final Output
![output](naive_bayes\results\out.png)

