# Poisson Regression 

## Definition 

The Poisson Distribution is a **discrete** distribution that can calculate the probability of a given number of **independent event** occurring in a **fixed time interval**. 
$$P(X = x) = \frac{e^{-\lambda}\lambda^x}{x!}$$
## Applications of Poisson Distribution

- Epidemiology
- Finance
- Engineering
- Social Sciences 
- Marketing

## Shape of the Poisson Distribution 

- Unlike the binomial distribution the Poisson distribution cannot be negative, hence its shape gets cut off as it gets closer to the y-axis
- The greater the mean the lower the peak as it has more spread 
- The variance function is always equal to the mean in Poisson distribution 

## When to use Poisson Regression 

- When the variance should increase or decrease as mean increases or decreases 
- When the data should not be negative 

## The Poisson Regression as a GLM 

- The Probability Function: The Poisson distribution 

The initial formula of the Poisson Distribution looks something like this: 
							$$y = e^{b_0 + b_1x_1 + b_2x_2 + ... + b_nx_n}$$

- Linear predictor ($\eta$): It is the original formula in the form a linear function 
				$$\eta = b_0 + b_1x_1 + b_2x_2+...+b_nx_n$$

- Link Function: It is through the link function that our original formula can be converted into a linear predictor. The Poisson distribution uses the log link. Hence, the Poisson Regression formula as a GLM looks like: 
				$$log(y) = b_0 + b_1x_1 + b_2x_2 +...+ b_nx_n$$

-- -- 
# References 

https://www.youtube.com/watch?v=Obpz_Uvo2rQ&list=PLLTSM0eKjC2cYVUoex9WZpTEYyvw5buRc&index=8

