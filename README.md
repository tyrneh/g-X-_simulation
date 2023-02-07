# g(X) simulation

- In this exercise, we want to find the probability of occurance of extreme tail-risk events for 6 'archetypes'. To find the probability of extreme realizations, we'll need to find the probability density function (pdf) of each 'archetype', which is a random variable (RV)
- This archetype RV we're interested in, is itself is a combination of 6 underlying component RVs ('subarchetypes'). We don't have information on the pdf of any of these 'subarchetypes'

1. In the first (and flawed) approach, we place the distributional assumption that the 6 subarchetypes are normally distributed. We use historical data on the 6 underlying RVs to calculate their variance and covariance, from which we can simulate the combined archetype by generating realizations of each subarchetype from the normal distribution

2. In the second approach, we don't place any distributional assumptions. We use bootstrap to generate the empirical pdf of the 6 subarchetypes, and then use these empirical distributions to generate realizations to simulate the pdf of the combined RV

Comparing approach 1 and 2, we see that by assuming the underlying RVs are normally distributed, we severely underestimate the probability of left-tail events. 

