## A/B test review

1. Central Limit Theorem

The sampling distribution of the sample mean approaches a normal distrbution as the sample size gets larger (but not infinite) no matter what the shape of the population distribution is.

X1, X2, ..., Xn are independent, identically distributed (IID), random variables, Xi has finite mean and variance

```math
\bar{X}_n = \frac{1}{n} \sum_{i=1}^n X_i \textasciitilde N(\mu, \sigma^2)
```

2. Law of Large Number

As the number of trials in a random experiment increases, the observed average outcome converges towards the expected value or true probability of the event.

3. Null hypothesis H0

There is no significant difference or effect between the groups or conditions being compared.

```math
diff = \mu_A - \mu_B = 0
```

4. p-value

Given the null hypothesis is true, the probability of observing data as extreme as or more extreme than the observed data. A smaller p-value means a higher chance of rejecting the null hypothesis.

Q: How would you describe what a 'p-value' is to a non-technical person? 

The best way to describe the p-value in simple terms is with an example. In practice, if the p-value is less than the alpha, say of 0.05, then we're saying that there's a probability of less than 5% that the result could have happened by chance. Similarly, a p-value of 0.05 is the same as saying "5% of the time, we would see this by chance."

- alpha is the prob of reject H0 when H0 is true == FP.

```math
\alpha = P(\text{reject} H_0 | H_0)
```

- beta is the prob of not reject H0 when H1 is true == FN.

```math
\beta = P(\text{not reject} H_0 | H_1)
```

- power is the prob of reject H0 when H1 is true == 1-beta

```math
\text{power} = P(\text{reject} H_0 | H_1)
```

- confidence level is the prob of not reject Ho when Ho is true = 1 - alpha

5. confidence interval

A CI gives us a range of values that is likely to contain the unknown population parameter. 95% is a commonly used confidence level which means that in repeated sampling 95% of the confidence intervals include the parameter.