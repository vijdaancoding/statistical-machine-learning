# Maximum Likelihood Estimation

## MLE and Normal Distribution 

## The Theory behind 1 data point 

Suppose we choose 1 data point among a sample. The distribution we are going to use is the Normal Distribution - keep in mind that MLE works with any distribution. 

The Normal Distribution has the following equation 
$$\frac{1}{\sqrt{2\pi\sigma^2}} * e^{-(x-\mu)^2/2\sigma^2}$$

Seeing the equation we realize that it has 2 unknown variables and 1 variable for denoting a data point. Therefore, it's likelihood function would look something like this: 
$$L(\mu, \sigma | x)$$
Our goal is to find the Maximum Likelihood, to do this we can substitute $x, \mu,$ and $\sigma$ in the equation. By keeping 2 variables constant and changing the 3rd we are able to get a graph through which we can know the maximum likelihood for both $\mu$ and $\sigma$. 

![image](https://github.com/vijdaancoding/statistical-machine-learning/assets/131896316/fb5ea852-504e-4362-9437-16263aa35b94)


![image](https://github.com/vijdaancoding/statistical-machine-learning/assets/131896316/5b25e829-ecc5-499d-881d-bd483a6c7c49)


## The Theory behind 2 data points 

What if we wanted the to find the MLE for 2 data points? The likelihood function for two data points would look something like this: 
$$L(\mu = 28, \sigma = 2 | x_1 = 32, x_2 = 34)$$
The thing is, these data points are **independent**. Therefore, it is reasonable to say that the equation above can also just be written as: 
$$L(\mu = 28, \sigma = 2 | x_1 = 32) * L(\mu = 28, \sigma = 2 | x_2 = 34)$$
If for multiple data points we can just multiply them then a general equation for when having data point of 2 or more would just look like: 
$$L(\mu, \sigma | x) = \prod_{i = 1}^{n}\frac{1}{\sqrt{2\pi\sigma^2}}e^{-(x_i - \mu)^2/2\sigma^2 } $$
### Finding the Optimal $\mu$ and $\sigma$

Finding the optimal mean and standard deviation can be pretty straight-forward but math heavy. We try to find the derivative and equal it to 0 since the optimal values are the peak values. However, before finding their derivative we apply a log transformation to make the process easier. 

#### Step 1: Apply the Log Transformation 

The equation we are applying a log transformation on is the same we saw above: 
$$\frac{1}{\sqrt{2\pi\sigma^2}}e^{-(x_1 - \mu)^2/2\sigma^2 } \cdot \frac{1}{\sqrt{2\pi\sigma^2}}e^{-(x_2 - \mu)^2/2\sigma^2 } \cdot \ldots \cdot \frac{1}{\sqrt{2\pi\sigma^2}}e^{-(x_n - \mu)^2/2\sigma^2 }$$
By applying a log on this we are able to change the multiplication signs into addition
$$\log{(\frac{1}{\sqrt{2\pi\sigma^2}}e^{-(x_1 - \mu)^2/2\sigma^2 } )  }+...+ \log{(\frac{1}{\sqrt{2\pi\sigma^2}}e^{-(x_n - \mu)^2/2\sigma^2 } )}$$
By breaking each term down the entire equation can be simplified into 
$$-\frac{n}{2}\ln{(2\pi)} - n\ln{(\sigma)} - \frac{1}{2\sigma^2} \sum_{i=1}^{n}{(x_i - \mu)^2}$$
#### Step 2: Derivation w.r.t $\mu$

The derivation of the log likelihood function looks something like this 
$$\dfrac{\mathrm{d}}{\mathrm{d}\mu} \ln{[L(\sigma, \mu | x_1, ..., x_n)]} = \frac{1}{\sigma^2}[(x_1+...+x_n) - n\mu]$$
#### Step 3: Derivation w.r.t $\sigma$

The derivation of the log likelihood function w.r.t to $\sigma$ would look like this 
$$\dfrac{\mathrm{d}}{\mathrm{d}\sigma} \ln{[L(\sigma, \mu | x_1, ..., x_n)]} = -\frac{n}{\sigma} + \frac{1}{\sigma^3}[(x_1 - \mu)^2 + ...+ (x_n - \mu)^2]$$
#### Step 4: Solving by equaling to 0

Now that we have the derivatives, by equaling them to 0 we can find the peak likelihood. 

For the mean we obtain the following equation
$$\mu = \frac{(x_1+...+x_n)}{n}$$
For standard deviation we obtain the following equation 
$$\sigma = \sqrt{\frac{(x_1 - \mu)^2+...+(x_n - \mu)^2}{n}}$$

## MLE and Linear Regression 

When using MLE to solve a linear regression problem we use the following model equation 
$$Y = \beta_0 + \beta_1x + \epsilon$$
Where: 
$Y$ is the response variable 
$x$ is the independent variable 
$\beta_0$ is the intercept 
$\beta_1$ is the gradient 
$\epsilon$ is the error term with a normally distributed mean of 0 and a constant variance 

The likelihood function of the linear regression equation comes out to be as 
$$L(\beta_0, \beta_1 | Y, X) = \prod_{i = 1}^{n}f(y_i|\beta_0, \beta_1, x_i)$$
#### Step 1: Log-Likelihood Function 

As done before we apply a log transformation to simply the differentiation 
$$l_n{(\beta_0, \beta_1, \sigma^2 | Y, X)} = \sum_{i=1}^{n}\ln{f(y_i | \beta_0, \beta_1, x_1)}$$
#### Step 2: Partial Derivation w.r.t to $\beta_0$ and $\beta_1$
$$\frac{dl_n}{d\beta_0} = \sum_{i=1}^{n}\frac{y_i - (\beta_0 + \beta_1x_i)}{\sigma^2}$$

$$\frac{dl_n}{d\beta_1} = \sum_{i=1}^{n}\frac{y_i - (\beta_0 + \beta_1x_i)}{\sigma^2}$$
#### Step 3: Set Derivatives to 0

$$
\beta_0 = \frac{\sum_{i=1}^{n} y_i - \beta_1\sum_{i=1}^{n} x_i}{n}
$$

$$
\beta_1 = \frac{\sum_{i=1}^{n}(x_i - \overline{x})(y_i - \overline{y})}{\sum_{i=1}^{n}(x_i - \overline{x})^2}
$$


#### Step 4: Estimate Variance 
$$\sigma^2 = \frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y_i})^2$$

## MLE and Poisson Distribution 

The Poisson distribution formula is 
$$P(X = x) = \frac{e^{-\lambda}\lambda^x}{x!}$$
By taking its logarithm we get 
$$ln{P(x)} = -\lambda + x ln\lambda - ln(x!)$$
After taking the logarithm we maximize the equation, meaning we take its $\prod$ 
$$lnP(X) = -n\lambda + ln\lambda\sum_{i=1}^{n}X_i + C$$
where $C = -ln(x!)$

After taking its derivative we get
$$\hat{\lambda} = \frac{1}{n}\sum_{i=1}^{n}X_i = \overline{X}$$
The equation states that after taking the derivative of the logarithmic function it equals the mean. Hence, we can state that when using MLE in Poisson distribution the mean is the MLE. 

-- -- 
# References 

https://www.youtube.com/watch?v=Dn6b9fCIUpM

https://www.youtube.com/watch?v=phyLFA437PM

https://www.youtube.com/watch?v=bhTIpGtWtzQ&list=PLLTSM0eKjC2cYVUoex9WZpTEYyvw5buRc&index=2

