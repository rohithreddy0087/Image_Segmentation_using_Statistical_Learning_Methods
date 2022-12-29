## Expectation Maximization for Mixtures
To perform EM we must first randomly initialize our 3 parameters, $\pi$, $\mu$ and $\Sigma$. To initialize $\pi$ we create C values where all the values are less than 1 and they all sum up to 1. To initialize $\mu$ we choose C random observations from the sample data. To initialize $\Sigma$ we create an identity matrix of random values less than 1.

Following equations are obtained from EM:
$$h_{ij} = P_{Z|X}(e_j|x_i; \Psi^{(n)}) = { G(x_i,\mu_j^{(n)},\Sigma_j^{(n)})\pi_j^{(n)} \over \sum_{k=1}G(x_i,\mu_k^{(n)},\Sigma_k^{(n)})\pi_k^{(n)} }$$
$$\mu_j^{(n+1)} = {\sum_{i}h_{ij}x_i \over \sum_{i}h_{ij}}$$
$$\sigma_j^{2(n+1)} = {\sum_{i}h_{ij}(x_i - \mu_j)^2 \over \sum_{i}h_{ij}}$$
$$\pi_j^{(n+1)} = {1 \over n}\sum_{i}h_{ij}$$


#### Segmentation of Cheetah
1. Using a $8\times8$ window, we slide through each pixel for the given image.
2. DCT for this $8\times8$ window is computed and then is converted into a feature vector using zig-zag sequence scanning.
3. Estimate mean, covariance and priors using EM quations.
4. Compute likelihood using the above obtained parameters.
5. Repeating above steps for each pixel, class is assigned and a segmentation mask is created.
#### Final Output
##### Probability of error is 0.0467
![em](expectation_maximization/results/em.png)