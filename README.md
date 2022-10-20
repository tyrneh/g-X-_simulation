# g-X-_simulation

- In this exercise, we want to find the probability of occurance of extreme tail-risk events for 6 'archetypes'. To do so, we'll need to find the probability density function (pdf) of each 'archetype', which is a random variable (RV). 
- This RV we're interested in itself is a combination of 6 underlying RVs ('subarchetypes'). We don't have information on the pdf of any of these 'subarchetypes'. 
1. In the first (and flawed) approach, we place the distributional assumption that the 6 underlying RVs are normally distributed. We use historical data on the 6 underlying RVs to calculate their variance and covariance, in order to simulate the pdf of the combined RV. 
2. In the second approach, we don't place any distributional assumptions. We use bootstrap to generate the empirical pdf of the 6 underlying RVs, and then simulate the pdf of the combined RV. 

Comparing approach 1 and 2, we see that by assuming the underlying RVs are normally distributed, we severely underestimate the probability of left-tail events. 

