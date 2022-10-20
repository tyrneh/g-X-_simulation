#!/usr/bin/env python
# coding: utf-8

# # Assumptions:

# In[38]:


# set probabilities:
BELOW_RATE_OF_RETURN = -0.25
NEG_OPER_PROFIT = -0.5

# set the number of samples you want to generate
NUM_SAMPLES = 10000


# # Start model

# In[39]:


import numpy as np
from scipy.linalg import eigh, cholesky
from scipy.stats import norm
from scipy.spatial import distance
import seaborn as sns 

import pandas as pd
import matplotlib.pyplot as plt

from pylab import plot, show, axis, subplot, xlabel, ylabel, grid

from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"


# # ---- First, we want to generate simulations from the 6 underlying RV, $ x_i $,  for i = 1,...,6, where $x_i$ are NORMALLY DISTRIBUTED with known mean and variance and are correlated with each other ----

# In[40]:


# read in covariance matrix as a dataframe
df_cov = pd.read_excel("Variance Matrices.xlsx", sheet_name = "cov", index_col = 0)
df_cov

# convert cov dataframe to an array 
cov_array = np.array(df_cov)
cov_array


# In[41]:


# generate random samples using mean = 0 and cov matrix
simulations = np.random.multivariate_normal(mean=np.zeros(6), cov=cov_array, size=NUM_SAMPLES)

df_simulations = pd.DataFrame(simulations)
df_simulations.rename(columns = {0:'x1',1:'x2',2:'x3',3:'x4',4:'x5',5:'x6'}, inplace=True)
df_simulations


# #### We have generated simulations of our 6 underlying RVs.
# #### Let's sanity check that the simulated observations make sense compared to the actual observations

# In[42]:


### sanity check: compute cov matrix from simulation. ###

# Measure the euclidean distance between simulated cov matrix and empirical cov matrix
# https://medium.com/swlh/euclidean-distance-matrix-4c3e1378d87f

# if Euclidean distance is sufficiently far from 0, then your simulations messed up somewhere

def EDM(A,B):
    p1 = np.sum(A**2, axis=1)[:, np.newaxis]
    p2 = np.sum(B**2, axis=1)
    p3 = -2 * np.dot(A,B.T)
    return np.round(np.sqrt(p1+p2+p3),2)

# create covariance array from simulations
simulations_cov_array = np.array(df_simulations.cov())

# calculate euclidean distance
EDM(simulations_cov_array,cov_array) # distance seems sufficiently close to 0


# In[43]:


# can manually check if cov matrix of simulations looks like cov matrix of data
df_simulations.cov()


# In[44]:


# some more sanity checks - let's plot some variables
fig, axs = plt.subplots(2,2)
fig.suptitle('Plotting some x_i correlations')

#x1 and x2 are indeed positively correlated
axs[0,0].scatter(df_simulations['x1'], df_simulations['x2'])
axs[0,0].set_title('x1 and x2 scatter')

# x1 and x3 are indeed negatively correlated
axs[0,1].scatter(df_simulations['x1'], df_simulations['x3'])
axs[0,1].set_title('x1 and x3 scatter')

# x1 and x4 are indeed weakly correlated
axs[1,0].scatter(df_simulations['x1'], df_simulations['x4'])
axs[1,0].set_title('x1 and x4 scatter')

# x3 and x5 are indeed strongly correlated
axs[1,1].scatter(df_simulations['x3'], df_simulations['x5'])
axs[1,1].set_title('x3 and x5 scatter')


# ## So we have created our simulations of $ x_i $ for i from 1 to 6. 
# ## Now we can generate subarchetype simulations: $ g_j(X) $ = weighted sum of $ x_i $, for j subarchetypes (j = 1,...,6)

# In[45]:


# import weights 
# read in weights matrix as a dataframe
df_weights = pd.read_excel("Variance Matrices.xlsx", sheet_name = "weights_subarchetype", index_col = 0)

#replace NA with 0:
df_weights.fillna(0, inplace=True)
df_weights


# ### 1) Simulate the first subarchetype: "Medium grid-sourced electrolytic producer: Transport-focused"

# In[46]:


# simulate g(X) for "Medium grid-sourced electrolytic producer: Transport-focused"

g_subarchetype1_simulations = (
 df_weights["subarchetype1"][0]*df_simulations['x1'] #weight of x1 * observations of x1
 +df_weights["subarchetype1"][1]*df_simulations['x2'] #weight of x2 * observations of x2
 +df_weights["subarchetype1"][2]*df_simulations['x3'] #weight of x3 * observations of x3
 +df_weights["subarchetype1"][3]*df_simulations['x4'] #weight of x4 * observations of x4
 +df_weights["subarchetype1"][4]*df_simulations['x5'] #weight of x5 * observations of x5
 +df_weights["subarchetype1"][5]*df_simulations['x6'] #weight of x6 * observations of x6
)

g_subarchetype1_simulations


# In[47]:


g_subarchetype1_simulations.hist()


# #### Compute the left-sided probability of g distribution, for subarchetype 1

# In[161]:


# let's create the empirical distribution of g archetype 1 
sns.histplot(data=g_subarchetype1_simulations, kde=True, stat='probability')
plt.title("Subarchetype 1 Simulated Distribution - Normal x_i assumption")


plt.savefig('g_subarchetype1_distribution-NORMAL_UNDERLYING.png')


# In[49]:


# left sided area of PDF is equal to the cumulative probability 
# convert to CDF to extract the probability at any given demand value

sns.histplot(data=g_subarchetype1_simulations, kde=True, stat='probability', cumulative = True)


# In[50]:


# get data from CDF
df_g_subarchetype1_probabilities = pd.DataFrame(
    np.transpose(
        sns.histplot(data=g_subarchetype1_simulations, kde=True, stat='probability', cumulative = True)
             .get_lines()[0].get_data()
            )
)

df_g_subarchetype1_probabilities.rename(columns={0:'Left-Sided Demand Deviation',1:'Probability'}, inplace=True)
df_g_subarchetype1_probabilities


# In[51]:


# create function to find the closest value to any specified probability

def find_neighbours(value, df, colname):
    exactmatch = df[df[colname] == value]
    if not exactmatch.empty:
        return exactmatch.index
    else:
        lowerneighbour_ind = df[df[colname] < value][colname].idxmax()
        upperneighbour_ind = df[df[colname] > value][colname].idxmin()
        return [lowerneighbour_ind, upperneighbour_ind] 


# In[52]:


# return the probability of the demand deviation
df_g_subarchetype1_probabilities.loc[
    find_neighbours(
        NEG_OPER_PROFIT,df_g_subarchetype1_probabilities,'Left-Sided Demand Deviation')[1]
]

# note that index 1 of the above object gives probability
df_g_subarchetype1_probabilities.loc[
    find_neighbours(
        NEG_OPER_PROFIT,df_g_subarchetype1_probabilities,'Left-Sided Demand Deviation')[1]
][1]


# In[53]:


# create dataframe to store results
df_subarchetype_results = pd.DataFrame(
    {
        "Subarchetype":[
                        "1: Medium grid-sourced electrolytic producer: Transport-focused"
                        ,"2: Medium grid-sourced electrolytic producer: Variety of end-uses"
                        ,"3: CCUS-enabled producer serving a large industrial cluster: Primary off-taker"
                        ,"4: CCUS-enabled producer serving a large industrial cluster: Industry-specific mix"
                        ,"5: CCUS-enabled producer serving a large industrial cluster: Many off-takers"
                        ,"6: Small co-developed renewable electrolytic sites for assorted uses"
                        ],
        "Below Rate of Return":np.nan,
        "Negative Operating Profit":np.nan
    }
)

df_subarchetype_results


# In[54]:


# store results for subarchetype 1

df_subarchetype_results.at[0,'Below Rate of Return']=df_g_subarchetype1_probabilities.loc[
    find_neighbours(
        BELOW_RATE_OF_RETURN,df_g_subarchetype1_probabilities,'Left-Sided Demand Deviation')[1]
][1]

df_subarchetype_results.at[0,'Negative Operating Profit']=df_g_subarchetype1_probabilities.loc[
    find_neighbours(
        NEG_OPER_PROFIT,df_g_subarchetype1_probabilities,'Left-Sided Demand Deviation')[1]
][1]

df_subarchetype_results


# ### 2) Simulate the second subarchetype: "Medium grid-sourced electrolytic producer: Variety of end-uses"

# In[55]:


# simulate g(X) for "Medium grid-sourced electrolytic producer: Variety of end-uses"
g_subarchetype2_simulations = (
 df_weights["subarchetype2"][0]*df_simulations['x1'] #weight of x1 * observations of x1
 +df_weights["subarchetype2"][1]*df_simulations['x2'] #weight of x2 * observations of x2
 +df_weights["subarchetype2"][2]*df_simulations['x3'] #weight of x3 * observations of x3
 +df_weights["subarchetype2"][3]*df_simulations['x4'] #weight of x4 * observations of x4
 +df_weights["subarchetype2"][4]*df_simulations['x5'] #weight of x5 * observations of x5
 +df_weights["subarchetype2"][5]*df_simulations['x6'] #weight of x6 * observations of x6
)

g_subarchetype2_simulations


# In[56]:


g_subarchetype2_simulations.hist()


# #### Compute the left-sided probability of g distribution, for subarchetype 2

# In[162]:


# let's create the empirical distribution of g archetype 2
sns.histplot(data=g_subarchetype2_simulations, kde=True, stat='probability')
plt.title("Subarchetype 2 Simulated Distribution - Normal x_i assumption")


plt.savefig('g_subarchetype2_distribution-NORMAL_UNDERLYING.png')


# In[58]:


# get data from CDF
df_g_subarchetype2_probabilities = pd.DataFrame(
    np.transpose(
        sns.histplot(data=g_subarchetype2_simulations, kde=True, stat='probability', cumulative = True)
             .get_lines()[0].get_data()
            )
)

df_g_subarchetype2_probabilities.rename(columns={0:'Left-Sided Demand Deviation',1:'Probability'}, inplace=True)
df_g_subarchetype2_probabilities


# In[59]:


# store results for subarchetype 2

df_subarchetype_results.at[1,'Below Rate of Return']=df_g_subarchetype2_probabilities.loc[
    find_neighbours(
        BELOW_RATE_OF_RETURN,df_g_subarchetype2_probabilities,'Left-Sided Demand Deviation')[1]
][1]

df_subarchetype_results.at[1,'Negative Operating Profit']=df_g_subarchetype2_probabilities.loc[
    find_neighbours(
        NEG_OPER_PROFIT,df_g_subarchetype2_probabilities,'Left-Sided Demand Deviation')[1]
][1]

df_subarchetype_results


# ### 3) Simulate the third subarchetype: "CCUS-enabled producer serving a large industrial cluster: Primary off-taker"

# In[60]:


# simulate g(X) for "CCUS-enabled producer serving a large industrial cluster: Primary off-taker"
g_subarchetype3_simulations = (
 df_weights["subarchetype3"][0]*df_simulations['x1'] #weight of x1 * observations of x1
 +df_weights["subarchetype3"][1]*df_simulations['x2'] #weight of x2 * observations of x2
 +df_weights["subarchetype3"][2]*df_simulations['x3'] #weight of x3 * observations of x3
 +df_weights["subarchetype3"][3]*df_simulations['x4'] #weight of x4 * observations of x4
 +df_weights["subarchetype3"][4]*df_simulations['x5'] #weight of x5 * observations of x5
 +df_weights["subarchetype3"][5]*df_simulations['x6'] #weight of x6 * observations of x6
)

g_subarchetype3_simulations

g_subarchetype3_simulations.hist()


# #### Compute the left-sided probability of g distribution, for subarchetype 3

# In[163]:


# let's create the empirical distribution of g archetype 3
sns.histplot(data=g_subarchetype3_simulations, kde=True, stat='probability')
plt.title("Subarchetype 3 Simulated Distribution - Normal x_i assumption")


plt.savefig('g_subarchetype3_distribution-NORMAL_UNDERLYING.png')


# In[62]:


# get data from CDF
df_g_subarchetype3_probabilities = pd.DataFrame(
    np.transpose(
        sns.histplot(data=g_subarchetype3_simulations, kde=True, stat='probability', cumulative = True)
             .get_lines()[0].get_data()
            )
)

df_g_subarchetype3_probabilities.rename(columns={0:'Left-Sided Demand Deviation',1:'Probability'}, inplace=True)
df_g_subarchetype3_probabilities


# In[63]:


# store results for subarchetype 3

df_subarchetype_results.at[2,'Below Rate of Return']=df_g_subarchetype3_probabilities.loc[
    find_neighbours(
        BELOW_RATE_OF_RETURN,df_g_subarchetype3_probabilities,'Left-Sided Demand Deviation')[1]
][1]

df_subarchetype_results.at[2,'Negative Operating Profit']=df_g_subarchetype3_probabilities.loc[
    find_neighbours(
        NEG_OPER_PROFIT,df_g_subarchetype3_probabilities,'Left-Sided Demand Deviation')[1]
][1]

df_subarchetype_results


# ### 4) Simulate the fourth subarchetype: "CCUS-enabled producer serving a large industrial cluster: Industry-specific mix"

# In[64]:


# simulate g(X) for "CCUS-enabled producer serving a large industrial cluster: Industry-specific mix"
g_subarchetype4_simulations = (
 df_weights["subarchetype4"][0]*df_simulations['x1'] #weight of x1 * observations of x1
 +df_weights["subarchetype4"][1]*df_simulations['x2'] #weight of x2 * observations of x2
 +df_weights["subarchetype4"][2]*df_simulations['x3'] #weight of x3 * observations of x3
 +df_weights["subarchetype4"][3]*df_simulations['x4'] #weight of x4 * observations of x4
 +df_weights["subarchetype4"][4]*df_simulations['x5'] #weight of x5 * observations of x5
 +df_weights["subarchetype4"][5]*df_simulations['x6'] #weight of x6 * observations of x6
)

g_subarchetype4_simulations

g_subarchetype4_simulations.hist()


# #### Compute the left-sided probability of g distribution, for subarchetype 4

# In[164]:


# let's create the empirical distribution of g archetype 4
sns.histplot(data=g_subarchetype4_simulations, kde=True, stat='probability')
plt.title("Subarchetype 4 Simulated Distribution - Normal x_i assumption")


plt.savefig('g_subarchetype4_distribution-NORMAL_UNDERLYING.png')


# In[66]:


# get data from CDF
df_g_subarchetype4_probabilities = pd.DataFrame(
    np.transpose(
        sns.histplot(data=g_subarchetype4_simulations, kde=True, stat='probability', cumulative = True)
             .get_lines()[0].get_data()
            )
)

df_g_subarchetype4_probabilities.rename(columns={0:'Left-Sided Demand Deviation',1:'Probability'}, inplace=True)
df_g_subarchetype4_probabilities

# note that there is 0 probability of -0.5 demand deviation


# In[67]:


# store results for subarchetype 4

df_subarchetype_results.at[3,'Below Rate of Return']=df_g_subarchetype4_probabilities.loc[
    find_neighbours(
        BELOW_RATE_OF_RETURN,df_g_subarchetype4_probabilities,'Left-Sided Demand Deviation')[1]
][1]

# if 0 probability, there will be an error. If so, then manually set probability to 0
try:
    df_subarchetype_results.at[3,'Negative Operating Profit']=df_g_subarchetype4_probabilities.loc[
    find_neighbours(
        NEG_OPER_PROFIT,df_g_subarchetype4_probabilities,'Left-Sided Demand Deviation')[1]
][1]

except ValueError:
    df_subarchetype_results.at[3,'Negative Operating Profit']=0
    
df_subarchetype_results


# ### 5) Simulate the fifth subarchetype: "CCUS-enabled producer serving a large industrial cluster: Many off-takers"

# In[68]:


# simulate g(X) for "CCUS-enabled producer serving a large industrial cluster: Many off-takers"
g_subarchetype5_simulations = (
 df_weights["subarchetype5"][0]*df_simulations['x1'] #weight of x1 * observations of x1
 +df_weights["subarchetype5"][1]*df_simulations['x2'] #weight of x2 * observations of x2
 +df_weights["subarchetype5"][2]*df_simulations['x3'] #weight of x3 * observations of x3
 +df_weights["subarchetype5"][3]*df_simulations['x4'] #weight of x4 * observations of x4
 +df_weights["subarchetype5"][4]*df_simulations['x5'] #weight of x5 * observations of x5
 +df_weights["subarchetype5"][5]*df_simulations['x6'] #weight of x6 * observations of x6
)

g_subarchetype5_simulations

g_subarchetype5_simulations.hist()


# #### Compute the left-sided probability of g distribution, for subarchetype 5

# In[165]:


# let's create the empirical distribution of g archetype 5
sns.histplot(data=g_subarchetype5_simulations, kde=True, stat='probability')
plt.title("Subarchetype 5 Simulated Distribution - Normal x_i assumption")


plt.savefig('g_subarchetype5_distribution-NORMAL_UNDERLYING.png')


# In[70]:


# get data from CDF
df_g_subarchetype5_probabilities = pd.DataFrame(
    np.transpose(
        sns.histplot(data=g_subarchetype5_simulations, kde=True, stat='probability', cumulative = True)
             .get_lines()[0].get_data()
            )
)

df_g_subarchetype5_probabilities.rename(columns={0:'Left-Sided Demand Deviation',1:'Probability'}, inplace=True)
df_g_subarchetype5_probabilities

# note that there is 0 probability of -0.5 demand deviation


# In[71]:


# store results for subarchetype 5

df_subarchetype_results.at[4,'Below Rate of Return']=df_g_subarchetype5_probabilities.loc[
    find_neighbours(
        BELOW_RATE_OF_RETURN,df_g_subarchetype5_probabilities,'Left-Sided Demand Deviation')[1]
][1]

# if 0 probability, there will be an error. If so, then manually set probability to 0
try:
    df_subarchetype_results.at[4,'Negative Operating Profit']=df_g_subarchetype5_probabilities.loc[
    find_neighbours(
        NEG_OPER_PROFIT,df_g_subarchetype5_probabilities,'Left-Sided Demand Deviation')[1]
][1]

except ValueError:
    df_subarchetype_results.at[4,'Negative Operating Profit']=0
    
df_subarchetype_results


# ### 6) Simulate the sixth subarchetype: "Small co-developed renewable electrolytic sites for assorted uses"

# In[72]:


# simulate g(X) for "Small co-developed renewable electrolytic sites for assorted uses"
g_subarchetype6_simulations = (
 df_weights["subarchetype6"][0]*df_simulations['x1'] #weight of x1 * observations of x1
 +df_weights["subarchetype6"][1]*df_simulations['x2'] #weight of x2 * observations of x2
 +df_weights["subarchetype6"][2]*df_simulations['x3'] #weight of x3 * observations of x3
 +df_weights["subarchetype6"][3]*df_simulations['x4'] #weight of x4 * observations of x4
 +df_weights["subarchetype6"][4]*df_simulations['x5'] #weight of x5 * observations of x5
 +df_weights["subarchetype6"][5]*df_simulations['x6'] #weight of x6 * observations of x6
)

g_subarchetype6_simulations

g_subarchetype6_simulations.hist()


# #### Compute the left-sided probability of g distribution, for subarchetype 6

# In[166]:


# let's create the empirical distribution of g archetype 6
sns.histplot(data=g_subarchetype6_simulations, kde=True, stat='probability')
plt.title("Subarchetype 6 Simulated Distribution - Normal x_i assumption")


plt.savefig('g_subarchetype6_distribution-NORMAL_UNDERLYING.png')


# In[74]:


# get data from CDF
df_g_subarchetype6_probabilities = pd.DataFrame(
    np.transpose(
        sns.histplot(data=g_subarchetype6_simulations, kde=True, stat='probability', cumulative = True)
             .get_lines()[0].get_data()
            )
)

df_g_subarchetype6_probabilities.rename(columns={0:'Left-Sided Demand Deviation',1:'Probability'}, inplace=True)
df_g_subarchetype6_probabilities

# note that there is 0 probability of -0.5 demand deviation


# In[75]:


# store results for subarchetype 6

df_subarchetype_results.at[5,'Below Rate of Return']=df_g_subarchetype6_probabilities.loc[
    find_neighbours(
        BELOW_RATE_OF_RETURN,df_g_subarchetype6_probabilities,'Left-Sided Demand Deviation')[1]
][1]

# if 0 probability, there will be an error. If so, then manually set probability to 0
try:
    df_subarchetype_results.at[5,'Negative Operating Profit']=df_g_subarchetype6_probabilities.loc[
    find_neighbours(
        NEG_OPER_PROFIT,df_g_subarchetype6_probabilities,'Left-Sided Demand Deviation')[1]
][1]

except ValueError:
    df_subarchetype_results.at[5,'Negative Operating Profit']=0
    
df_subarchetype_results


# ### STORE RESULTS

# In[76]:


# save to csv

df_subarchetype_results.to_csv('subarchetype_results-normal_underlying.csv')


# # ---------------------------------------------------------------------------------------------------------------

# # ---- In the second step, we want to generate simulations from the 6 underlying RV, $ x_i $,  for i = 1,...,6, where simulations will come from the bootstrap, which does not place distributional assumptions on the underyling data generating process ----

# In[77]:


from scipy.stats import bootstrap
from sklearn.utils import resample


# In[78]:


# import raw data to bootstrap with
df_bootstrap_input = pd.read_excel("Variance Matrices.xlsx", sheet_name = "bootstrap_input")
df_bootstrap_input


# In[79]:


# create bootstrap samples for x1

boot_x1 = resample(df_bootstrap_input['x1'].dropna(), replace=True, n_samples=NUM_SAMPLES, random_state=1)
boot_x1

sns.histplot(boot_x1, kde=True)


# In[80]:


# create bootstrap samples for x2

boot_x2 = resample(df_bootstrap_input['x2'].dropna(), replace=True, n_samples=NUM_SAMPLES, random_state=1)
boot_x2

sns.histplot(boot_x2, kde=True)


# In[81]:


# create bootstrap samples for x3

boot_x3 = resample(df_bootstrap_input['x3'].dropna(), replace=True, n_samples=NUM_SAMPLES, random_state=1)
boot_x3

sns.histplot(boot_x3, kde=True)


# In[82]:


# create bootstrap samples for x4

boot_x4 = resample(df_bootstrap_input['x4'].dropna(), replace=True, n_samples=NUM_SAMPLES, random_state=1)
boot_x4

sns.histplot(boot_x4, kde=True)


# In[83]:


# create bootstrap samples for x5

boot_x5 = resample(df_bootstrap_input['x5'].dropna(), replace=True, n_samples=NUM_SAMPLES, random_state=1)
boot_x5

sns.histplot(boot_x5, kde=True)


# In[84]:


# create bootstrap samples for x6

boot_x6 = resample(df_bootstrap_input['x6'].dropna(), replace=True, n_samples=NUM_SAMPLES, random_state=1)
boot_x6

sns.histplot(boot_x6, kde=True)


# In[108]:


bootstrap_array = np.array([
    boot_x1,
    boot_x2,
    boot_x3,
    boot_x4,
    boot_x5,
    boot_x6
])
bootstrap_array


# In[112]:


# create a dataframe of all x_i bootstrap simulations

df_bootstraps = pd.DataFrame(np.transpose(bootstrap_array))
df_bootstraps.rename(columns = {0:'x1',1:'x2',2:'x3',3:'x4',4:'x5',5:'x6'}, inplace=True)
df_bootstraps


# ## So we have created our bootstrap simulations of $ x_i $ for i from 1 to 6. 
# ## Now we can generate subarchetype simulations: $ g_j(X) $ = weighted sum of $ x_i $, for j subarchetypes (j = 1,...,6)

# In[135]:


# create dataframe to store bootstrap_res
df_subarchetype_bootstrap_res = pd.DataFrame(
    {
        "Subarchetype":[
                        "1: Medium grid-sourced electrolytic producer: Transport-focused"
                        ,"2: Medium grid-sourced electrolytic producer: Variety of end-uses"
                        ,"3: CCUS-enabled producer serving a large industrial cluster: Primary off-taker"
                        ,"4: CCUS-enabled producer serving a large industrial cluster: Industry-specific mix"
                        ,"5: CCUS-enabled producer serving a large industrial cluster: Many off-takers"
                        ,"6: Small co-developed renewable electrolytic sites for assorted uses"
                        ],
        "Below Rate of Return":np.nan,
        "Negative Operating Profit":np.nan
    }
)

df_subarchetype_bootstrap_res


# ### 1) Simulate the first subarchetype: "Medium grid-sourced electrolytic producer: Transport-focused"

# In[134]:



# In[ ]:


# simulate g(X) for "Medium grid-sourced electrolytic producer: Transport-focused"

g_subarchetype1_bootstraps = (
 df_weights["subarchetype1"][0]*df_bootstraps['x1'] #weight of x1 * observations of x1
 +df_weights["subarchetype1"][1]*df_bootstraps['x2'] #weight of x2 * observations of x2
 +df_weights["subarchetype1"][2]*df_bootstraps['x3'] #weight of x3 * observations of x3
 +df_weights["subarchetype1"][3]*df_bootstraps['x4'] #weight of x4 * observations of x4
 +df_weights["subarchetype1"][4]*df_bootstraps['x5'] #weight of x5 * observations of x5
 +df_weights["subarchetype1"][5]*df_bootstraps['x6'] #weight of x6 * observations of x6
)

g_subarchetype1_bootstraps


# #### Compute the left-sided probability of g distribution, for subarchetype 1

# In[ ]:


# let's create the empirical distribution of g archetype 1 
sns.histplot(data=g_subarchetype1_bootstraps, kde=True, stat='probability')


# In[167]:


# instead of cutting off the distribution at the lowest empirical value, let's smooth it out

sns.kdeplot(data=g_subarchetype1_bootstraps, shade = True)
plt.title("Subarchetype 1 Simulated Distribution - Bootstrapped x_i")

plt.savefig('g_subarchetype1_distribution-BOOTSTRAP_UNDERLYING.png')


# In[151]:


# get data from CDF
kde_g1 = sns.kdeplot(data=g_subarchetype1_bootstraps, cumulative=True)

line = kde_g1.lines[0]
x, y = line.get_data()

df_g_subarchetype1_bootstrap_probs = pd.DataFrame(
    {
        'Left-Sided Demand Deviation':x,
        'Probability':y
    }
)

df_g_subarchetype1_bootstrap_probs


# In[152]:


# store bootstrap_res for subarchetype 1

df_subarchetype_bootstrap_res.at[0,'Below Rate of Return']=df_g_subarchetype1_bootstrap_probs.loc[
    find_neighbours(
        BELOW_RATE_OF_RETURN,df_g_subarchetype1_bootstrap_probs,'Left-Sided Demand Deviation')[1]
][1]

try:
    df_subarchetype_bootstrap_res.at[0,'Negative Operating Profit']=df_g_subarchetype1_bootstrap_probs.loc[
        find_neighbours(
            NEG_OPER_PROFIT,df_g_subarchetype1_bootstrap_probs,'Left-Sided Demand Deviation')[1]
    ][1]
except ValueError:
    df_subarchetype_bootstrap_res.at[0,'Negative Operating Profit']=0
    
df_subarchetype_bootstrap_res


#  ### 2) Simulate the second subarchetype: "Medium grid-sourced electrolytic producer: Variety of end-uses"

# In[168]:



# In[ ]:


# simulate g(X) for "Medium grid-sourced electrolytic producer: Variety of end-uses"
g_subarchetype2_bootstraps = (
 df_weights["subarchetype2"][0]*df_bootstraps['x1'] #weight of x1 * observations of x1
 +df_weights["subarchetype2"][1]*df_bootstraps['x2'] #weight of x2 * observations of x2
 +df_weights["subarchetype2"][2]*df_bootstraps['x3'] #weight of x3 * observations of x3
 +df_weights["subarchetype2"][3]*df_bootstraps['x4'] #weight of x4 * observations of x4
 +df_weights["subarchetype2"][4]*df_bootstraps['x5'] #weight of x5 * observations of x5
 +df_weights["subarchetype2"][5]*df_bootstraps['x6'] #weight of x6 * observations of x6
)

g_subarchetype2_bootstraps


# #### Compute the left-sided probability of g distribution, for subarchetype 2

# In[ ]:


# let's create the empirical distribution of g archetype 2
sns.histplot(data=g_subarchetype2_bootstraps, kde=True, stat='probability')


# In[169]:


# instead of cutting off the distribution at the lowest empirical value, let's smooth it out

sns.kdeplot(data=g_subarchetype2_bootstraps, shade = True)
plt.title("Subarchetype 2 Simulated Distribution - Bootstrapped x_i")

plt.savefig('g_subarchetype2_distribution-BOOTSTRAP_UNDERLYING.png')


# In[182]:


# get data from CDF
kde_g2 = sns.kdeplot(data=g_subarchetype2_bootstraps, cumulative=True)

line = kde_g2.lines[0]
x, y = line.get_data()

df_g_subarchetype2_bootstrap_probs = pd.DataFrame(
    {
        'Left-Sided Demand Deviation':x,
        'Probability':y
    }
)

df_g_subarchetype2_bootstrap_probs


# In[183]:


# store bootstrap_res for subarchetype 2

df_subarchetype_bootstrap_res.at[1,'Below Rate of Return']=df_g_subarchetype2_bootstrap_probs.loc[
    find_neighbours(
        BELOW_RATE_OF_RETURN,df_g_subarchetype2_bootstrap_probs,'Left-Sided Demand Deviation')[1]
][1]

try:
    df_subarchetype_bootstrap_res.at[1,'Negative Operating Profit']=df_g_subarchetype2_bootstrap_probs.loc[
        find_neighbours(
            NEG_OPER_PROFIT,df_g_subarchetype2_bootstrap_probs,'Left-Sided Demand Deviation')[1]
    ][1]
    
except ValueError:
    df_subarchetype_bootstrap_res.at[1,'Negative Operating Profit']=0

df_subarchetype_bootstrap_res


# ### 3) Simulate the third subarchetype: "CCUS-enabled producer serving a large industrial cluster: Primary off-taker"

# In[179]:




# In[ ]:


# simulate g(X) for "CCUS-enabled producer serving a large industrial cluster: Primary off-taker"
g_subarchetype3_bootstraps = (
 df_weights["subarchetype3"][0]*df_bootstraps['x1'] #weight of x1 * observations of x1
 +df_weights["subarchetype3"][1]*df_bootstraps['x2'] #weight of x2 * observations of x2
 +df_weights["subarchetype3"][2]*df_bootstraps['x3'] #weight of x3 * observations of x3
 +df_weights["subarchetype3"][3]*df_bootstraps['x4'] #weight of x4 * observations of x4
 +df_weights["subarchetype3"][4]*df_bootstraps['x5'] #weight of x5 * observations of x5
 +df_weights["subarchetype3"][5]*df_bootstraps['x6'] #weight of x6 * observations of x6
)

g_subarchetype3_bootstraps

# #### Compute the left-sided probability of g distribution, for subarchetype 3

# In[ ]:


# let's create the empirical distribution of g archetype 3
sns.histplot(data=g_subarchetype3_bootstraps, kde=True, stat='probability')


# In[187]:


# instead of cutting off the distribution at the lowest empirical value, let's smooth it out

sns.kdeplot(data=g_subarchetype3_bootstraps, shade = True)
plt.title("Subarchetype 3 Simulated Distribution - Bootstrapped x_i")

plt.savefig('g_subarchetype3_distribution-BOOTSTRAP_UNDERLYING.png')


# In[184]:


# get data from CDF
kde_g3 = sns.kdeplot(data=g_subarchetype3_bootstraps, cumulative=True)

line = kde_g3.lines[0]
x, y = line.get_data()

df_g_subarchetype3_bootstrap_probs = pd.DataFrame(
    {
        'Left-Sided Demand Deviation':x,
        'Probability':y
    }
)

df_g_subarchetype3_bootstrap_probs


# In[185]:



# store bootstrap_res for subarchetype 3

df_subarchetype_bootstrap_res.at[2,'Below Rate of Return']=df_g_subarchetype3_bootstrap_probs.loc[
    find_neighbours(
        BELOW_RATE_OF_RETURN,df_g_subarchetype3_bootstrap_probs,'Left-Sided Demand Deviation')[1]
][1]

try:
    df_subarchetype_bootstrap_res.at[2,'Negative Operating Profit']=df_g_subarchetype3_bootstrap_probs.loc[
        find_neighbours(
            NEG_OPER_PROFIT,df_g_subarchetype3_bootstrap_probs,'Left-Sided Demand Deviation')[1]
    ][1]
except ValueError:
    df_subarchetype_bootstrap_res.at[2,'Negative Operating Profit']=0

df_subarchetype_bootstrap_res


# ### 4) Simulate the fourth subarchetype: "CCUS-enabled producer serving a large industrial cluster: Industry-specific mix"

# In[186]:




# simulate g(X) for "CCUS-enabled producer serving a large industrial cluster: Industry-specific mix"
g_subarchetype4_bootstraps = (
 df_weights["subarchetype4"][0]*df_bootstraps['x1'] #weight of x1 * observations of x1
 +df_weights["subarchetype4"][1]*df_bootstraps['x2'] #weight of x2 * observations of x2
 +df_weights["subarchetype4"][2]*df_bootstraps['x3'] #weight of x3 * observations of x3
 +df_weights["subarchetype4"][3]*df_bootstraps['x4'] #weight of x4 * observations of x4
 +df_weights["subarchetype4"][4]*df_bootstraps['x5'] #weight of x5 * observations of x5
 +df_weights["subarchetype4"][5]*df_bootstraps['x6'] #weight of x6 * observations of x6
)

g_subarchetype4_bootstraps


# #### Compute the left-sided probability of g distribution, for subarchetype 4

# In[ ]:


# let's create the empirical distribution of g archetype 4
sns.histplot(data=g_subarchetype4_bootstraps, kde=True, stat='probability')


# In[188]:


# instead of cutting off the distribution at the lowest empirical value, let's smooth it out

sns.kdeplot(data=g_subarchetype4_bootstraps, shade = True)
plt.title("Subarchetype 4 Simulated Distribution - Bootstrapped x_i")

plt.savefig('g_subarchetype4_distribution-BOOTSTRAP_UNDERLYING.png')


# In[189]:


# get data from CDF
kde_g4 = sns.kdeplot(data=g_subarchetype4_bootstraps, cumulative=True)

line = kde_g4.lines[0]
x, y = line.get_data()

df_g_subarchetype4_bootstrap_probs = pd.DataFrame(
    {
        'Left-Sided Demand Deviation':x,
        'Probability':y
    }
)

df_g_subarchetype4_bootstrap_probs


# In[190]:



# store bootstrap_res for subarchetype 4

df_subarchetype_bootstrap_res.at[3,'Below Rate of Return']=df_g_subarchetype4_bootstrap_probs.loc[
    find_neighbours(
        BELOW_RATE_OF_RETURN,df_g_subarchetype4_bootstrap_probs,'Left-Sided Demand Deviation')[1]
][1]

# if 0 probability, there will be an error. If so, then manually set probability to 0
try:
    df_subarchetype_bootstrap_res.at[3,'Negative Operating Profit']=df_g_subarchetype4_bootstrap_probs.loc[
    find_neighbours(
        NEG_OPER_PROFIT,df_g_subarchetype4_bootstrap_probs,'Left-Sided Demand Deviation')[1]
][1]

except ValueError:
    df_subarchetype_bootstrap_res.at[3,'Negative Operating Profit']=0
    
df_subarchetype_bootstrap_res


# ### 5) Simulate the fifth subarchetype: "CCUS-enabled producer serving a large industrial cluster: Many off-takers"

# In[191]:



# simulate g(X) for "CCUS-enabled producer serving a large industrial cluster: Many off-takers"
g_subarchetype5_bootstraps = (
 df_weights["subarchetype5"][0]*df_bootstraps['x1'] #weight of x1 * observations of x1
 +df_weights["subarchetype5"][1]*df_bootstraps['x2'] #weight of x2 * observations of x2
 +df_weights["subarchetype5"][2]*df_bootstraps['x3'] #weight of x3 * observations of x3
 +df_weights["subarchetype5"][3]*df_bootstraps['x4'] #weight of x4 * observations of x4
 +df_weights["subarchetype5"][4]*df_bootstraps['x5'] #weight of x5 * observations of x5
 +df_weights["subarchetype5"][5]*df_bootstraps['x6'] #weight of x6 * observations of x6
)

g_subarchetype5_bootstraps


# #### Compute the left-sided probability of g distribution, for subarchetype 5

# In[ ]:


# let's create the empirical distribution of g archetype 5
sns.histplot(data=g_subarchetype5_bootstraps, kde=True, stat='probability')


# In[192]:


# instead of cutting off the distribution at the lowest empirical value, let's smooth it out

sns.kdeplot(data=g_subarchetype5_bootstraps, shade = True)
plt.title("Subarchetype 5 Simulated Distribution - Bootstrapped x_i")

plt.savefig('g_subarchetype5_distribution-BOOTSTRAP_UNDERLYING.png')


# In[193]:


# get data from CDF
kde_g5 = sns.kdeplot(data=g_subarchetype5_bootstraps, cumulative=True)

line = kde_g5.lines[0]
x, y = line.get_data()

df_g_subarchetype5_bootstrap_probs = pd.DataFrame(
    {
        'Left-Sided Demand Deviation':x,
        'Probability':y
    }
)

df_g_subarchetype5_bootstrap_probs


# In[194]:



# store bootstrap_res for subarchetype 5

df_subarchetype_bootstrap_res.at[4,'Below Rate of Return']=df_g_subarchetype5_bootstrap_probs.loc[
    find_neighbours(
        BELOW_RATE_OF_RETURN,df_g_subarchetype5_bootstrap_probs,'Left-Sided Demand Deviation')[1]
][1]

# if 0 probability, there will be an error. If so, then manually set probability to 0
try:
    df_subarchetype_bootstrap_res.at[4,'Negative Operating Profit']=df_g_subarchetype5_bootstrap_probs.loc[
    find_neighbours(
        NEG_OPER_PROFIT,df_g_subarchetype5_bootstrap_probs,'Left-Sided Demand Deviation')[1]
][1]

except ValueError:
    df_subarchetype_bootstrap_res.at[4,'Negative Operating Profit']=0
    
df_subarchetype_bootstrap_res


# ### 6) Simulate the sixth subarchetype: "Small co-developed renewable electrolytic sites for assorted uses"

# In[195]:



# simulate g(X) for "Small co-developed renewable electrolytic sites for assorted uses"
g_subarchetype6_bootstraps = (
 df_weights["subarchetype6"][0]*df_bootstraps['x1'] #weight of x1 * observations of x1
 +df_weights["subarchetype6"][1]*df_bootstraps['x2'] #weight of x2 * observations of x2
 +df_weights["subarchetype6"][2]*df_bootstraps['x3'] #weight of x3 * observations of x3
 +df_weights["subarchetype6"][3]*df_bootstraps['x4'] #weight of x4 * observations of x4
 +df_weights["subarchetype6"][4]*df_bootstraps['x5'] #weight of x5 * observations of x5
 +df_weights["subarchetype6"][5]*df_bootstraps['x6'] #weight of x6 * observations of x6
)

g_subarchetype6_bootstraps


# #### Compute the left-sided probability of g distribution, for subarchetype 6

# In[ ]:


# let's create the empirical distribution of g archetype 6
sns.histplot(data=g_subarchetype6_bootstraps, kde=True, stat='probability')


# In[196]:


# instead of cutting off the distribution at the lowest empirical value, let's smooth it out

sns.kdeplot(data=g_subarchetype6_bootstraps, shade = True)
plt.title("Subarchetype 6 Simulated Distribution - Bootstrapped x_i")

plt.savefig('g_subarchetype6_distribution-BOOTSTRAP_UNDERLYING.png')


# In[197]:


# get data from CDF
kde_g6 = sns.kdeplot(data=g_subarchetype6_bootstraps, cumulative=True)

line = kde_g6.lines[0]
x, y = line.get_data()

df_g_subarchetype6_bootstrap_probs = pd.DataFrame(
    {
        'Left-Sided Demand Deviation':x,
        'Probability':y
    }
)

df_g_subarchetype6_bootstrap_probs


# In[198]:



# store bootstrap_res for subarchetype 6

df_subarchetype_bootstrap_res.at[5,'Below Rate of Return']=df_g_subarchetype6_bootstrap_probs.loc[
    find_neighbours(
        BELOW_RATE_OF_RETURN,df_g_subarchetype6_bootstrap_probs,'Left-Sided Demand Deviation')[1]
][1]

# if 0 probability, there will be an error. If so, then manually set probability to 0
try:
    df_subarchetype_bootstrap_res.at[5,'Negative Operating Profit']=df_g_subarchetype6_bootstrap_probs.loc[
    find_neighbours(
        NEG_OPER_PROFIT,df_g_subarchetype6_bootstrap_probs,'Left-Sided Demand Deviation')[1]
][1]

except ValueError:
    df_subarchetype_bootstrap_res.at[5,'Negative Operating Profit']=0
    
df_subarchetype_bootstrap_res


# ### STORE RESULTS

# In[199]:


# save to csv

df_subarchetype_bootstrap_res.to_csv('subarchetype_results-bootstrapped_underlying.csv')

