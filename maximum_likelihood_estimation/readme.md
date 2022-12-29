# Maximum Likelihood estimation
\item Histogram estimates for prior probabilities:\\
    From the training set, the histogram plot for the two classes is obtained as follows:
        \begin{center}
            	\includegraphics[height=11cm]{a1.png} \\
        \end{center}
    This histogram plot can be viewed as a multinomial distribution. The observations have a probabilty distribution $P_k = \pi_k$ where $k = \{1,2\}$. Using the Maximum Likelihood estimate we obtain $\pi_k = \frac{c_j}{n}$ where $c = \{c_1,c_2\}$ are the number of times that the observed value is $k$\\
    In this problem $c_1 = 250$, $c_1 = 1053$ and $n = c_1 + c_2 = 1303$
        \begin{enumerate}[(1)]
        	\item Prior probability of cheetah is 
        	$$
        	P_{y}(cheetah) = P_1 = \pi_1 = \frac{c_1}{n}
        	               = \frac{250}{1303} = 0.1919
        	$$
        	\item Prior probability of grass is 
        	$$
        	P_{y}(grass) = P_2 = \pi_2 = \frac{c_2}{n}
        	               = \frac{1053}{1303} = 0.1919
        	$$
        \end{enumerate}
    The prior probabilities are the same as the ones that are obtained last week, this is because for the given training set which has only two classes a reasonable estimate for the prior probabilities is equal to the fraction of samples of that respective class in the whole training set. This is equal to the mathematical formula obtained using ML estimate for the multinomial distribution i.e., $\pi_k = \frac{c_j}{n}$, where $c_j$ represents the number of samples of a respective class and $n$ represents the total number of samples in the dataset. 
    
    \item Class conditional Densities $P_{X|Y}(x|cheetah)$ and $P_{X|Y}(x|grass)$ \\
    ML estimates for multivariate Gaussian distribution are:
    $$\mu_i =  \frac{1}{n}\sum_{j}x_j^{(i)}$$
    $$\Sigma_i = \frac{1}{n}\sum_{j}(x_j^{(i)}-\mu_i)(x_j^{(i)}-\mu_i)^\top$$
    Hence from the given training data, for each class we estimate $\mu$ and $\Sigma$ using the above formulas. Since there are 64 features for each sample, the obtained dimensions of $\mu$ and $\Sigma$ are $64\times1$ and $64\times64$ respectively.\\
    Using the $\mu$ and the diagonal values of $\Sigma$, marginal densities for each feature
    can be plotted. Denoting feature vectors as $X = \{X_1,X_2,......,X_64\}$ and the marginal densities for the two classes as $P_{X_k|Y}(x_k|cheetah)$ and $P_{X_k|Y}(x_k|grass)$ where $k = 1,2,3,...,64$. After plotting marginal densities for the two classes for all features, by visual inspection i.e., the overlap area and the mean difference between two distributions we can roughly determine the best and worst possible eight features. The plots of the best and worst 8 features obtained from visual inspection are as follows:
    \begin{center}
        \includegraphics[height=10cm,width =15cm]{bf1.png} \\
        \includegraphics[height=10cm,width =15cm]{bf2.png} \\
    \end{center}
     \begin{center}
        \includegraphics[height=11cm,width =15cm]{wf1.png} \\
        \includegraphics[height=11cm,width =15cm]{wf2.png} \\
    \end{center}