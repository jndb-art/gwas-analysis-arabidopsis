#!/usr/bin/env python
# coding: utf-8

# ## Project Goal
# This notebook links Arabidopsis dry-weight phenotype data to genotype data and performs GWAS analysis step by step.

# In[1]:


"""
In dit project haal ik een phenotype uit https://aragwas.1001genomes.org/#/
Ik link dit aan het genotype van arabidopsis thialiana (gevonden in https://aragwas.1001genomes.org/#/)
"""

# ## 1) Import Libraries and Load Phenotype Data
# Load all required packages and read the phenotype table used in downstream analysis.

# In[2]:


import h5py
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import statsmodels.api as sm
import limix
print(limix.__version__)



# In[3]:


DATA_PATH = "data"

pheno = pd.read_csv(f"{DATA_PATH}/plant_dry_weight.csv")
print(pheno.head())

# ### Select Relevant Phenotype Columns
# Keep only accession IDs and the phenotype value used for GWAS.

# In[4]:


pheno = pheno[['accession_id', 'phenotype_value']]
print(pheno.head())

# ## 2) Inspect Genotype HDF5 Structure
# Open the genotype file and inspect available keys/datasets.

# In[5]:


file_path = f"{DATA_PATH}/genotype.hdf5"

with h5py.File(file_path, "r") as f:
    print("Keys in file:")
    print(list(f.keys()))

# In[6]:


with h5py.File(file_path, "r") as f:
    for key in f.keys():
        print(key, f[key])

# ### Read Accessions, Positions, and SNP Subset
# Load accession names, SNP positions, and a subset of SNPs for analysis.

# In[6]:


with h5py.File("data/genotype.hdf5", "r") as f:
    accessions = [a.decode() for a in f["accessions"][:]]
    positions = f["positions"][:]
    
    # selecteer bijvoorbeeld eerste 50,000 SNPs
    snp_subset = f["snps"][:50000, :]

# ### Build Genotype Matrix
# Convert SNP array to a sample-by-SNP DataFrame.

# In[7]:


geno = pd.DataFrame(
    snp_subset.T, 
    index=accessions
)

# In[8]:


print(geno.head())

# ## 3) Clean and Harmonize Sample IDs
# Set accession IDs as indices, fix datatypes, and remove duplicates.

# In[9]:


# Zet phenotype index op accession_name
pheno = pheno.set_index("accession_id")
print(pheno.index[:5])
print(geno.index[:5])
pheno.index.duplicated().sum() #twee duplicates in pheno!


pheno.index = pheno.index.astype(int)
geno.index = geno.index.astype(int)
pheno = pheno[~pheno.index.duplicated(keep="first")]  #removed duplicates
print(pheno.index[:5])
print(geno.index[:5])


# ### Keep Only Overlapping Accessions
# Restrict phenotype and genotype tables to shared accessions in identical order.

# In[10]:


# Neem alleen overlappende accessions
common = pheno.index.intersection(geno.index)
print("Aantal overlap:", len(common))

# Sorteer de IDs zodat beide datasets exact dezelfde volgorde hebben
common = common.sort_values()

# Subset beide datasets
pheno = pheno.loc[common]
geno = geno.loc[common]

print("common:", common.shape)
print("pheno:", pheno.shape)
print("geno:", geno.shape)

# In[11]:


print(pheno.index[:10])
print(geno.index[:10])

# In[12]:


"""
Interpretatie:

Je phenotype file bevat 422 accessions

Je genotype subset bevat 2029 accessions totaal, maar

Slechts 420 komen in beide voor (zijn accessories met zowel een phenotype en een genotype)
"""

# ## 4) SNP Quality Control
# Handle missing values and remove SNPs with high missingness.

# In[13]:


# Vul eventueel missing data met NaN
geno = geno.replace(-1, np.nan)  # afhankelijk van je HDF5 encoding

# Missingness filter: verwijder SNPs met >10% missing
geno = geno.loc[:, geno.isna().mean() < 0.1]

# ### Minor Allele Frequency (MAF) Filtering
# Keep informative SNPs by excluding rare variants below the MAF threshold.

# In[14]:


# MAF filter: verwijder SNPs met MAF < 0.05

#geno.mean = gemiddelde genotypische waarde per SNP -> delen door 2 geeft frequentie van het alternatieve allel
allele_freq = geno.mean(axis=0) / 2  # 0,1,2 encoding â†’ freq alt allele

#allele_freq is frequentie van alternatieve allel, 1-allele_freq is frequentie van referentie allel
#minimum van beide is Minor Allele Frequency (MAF)
maf = np.minimum(allele_freq, 1 - allele_freq)

#filter op MAF â‰¥ 0.05
geno = geno.loc[:, maf > 0.05]

# In[15]:


print("Aantal SNPs na filtering:", geno.shape[1])

# ## 5) PCA for Population Structure
# Compute principal components from genotype data and append PCs to phenotype data.

# In[16]:


# Vul missing values voor PCA
geno_filled = geno.fillna(geno.mean())
print(geno_filled.head())

# PCA uitvoeren
pca = PCA(n_components=10) #5 PC's although we only plot first 10 PCs
pcs = pca.fit_transform(geno_filled)

# DataFrame met PC scores
pcs_df = pd.DataFrame(pcs, index=geno.index, columns=[f'PC{i+1}' for i in range(pcs.shape[1])])
print(pcs_df.head())

# Voeg PC1 tot PC10 toe aan phenotype dataframe
pheno = pheno.join(pcs_df)

# Plot PCA (PC1 vs PC2)
plt.figure(figsize=(8,6))
plt.scatter(pheno['PC1'], pheno['PC2'])
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('PCA - Population Structure')
plt.show()

# In[17]:


print(pheno.head())

# ### Evaluate PCA Explained Variance
# Check variance captured by each PC and cumulative trend.

# In[18]:


#Check variance explained by each PC
explained_var = pca.explained_variance_ratio_
for i, var in enumerate(explained_var):
    print(f"PC{i+1} explains {var*100:.2f}% of total variance")

# Optioneel: cumulative variance plot
plt.figure(figsize=(6,4))
plt.plot(np.cumsum(explained_var)*100, marker='o')
plt.xlabel('Number of PCs')
plt.ylabel('Cumulative variance explained (%)')
plt.title('Variance explained by PCs')
plt.grid(True)
plt.show()

# In[19]:


"""
is there not a lot of variation lost by doing PCA?
You have tens of thousands of SNPs (~50k in your subset). In genomics, itâ€™s common for the first 2â€“5 PCs to explain 20â€“40% of variance. 
GWAS pipelines usually donâ€™t need to capture 100% of variance.
Goal of PCs in GWAS: not to explain all variance, but to capture population structure / ancestry to avoid false positives.
The remaining ~65% of SNP variance is mostly random variation, technical noise, or very subtle structure, which GWAS will deal with via residuals.
"""

# In[20]:


print(geno.head())
print(geno.shape)
print(pheno.shape)

# ## 6) Initial GWAS Using OLS
# Run SNP-wise linear models with phenotype as response and PCs as covariates.

# In[21]:


phenotype_col = 'phenotype_value'
p_values = []

for snp in geno.columns:
    X = pheno[['PC1','PC2']].copy()
    X[snp] = geno[snp]
    X = sm.add_constant(X)
    y = pheno[phenotype_col]
    
    try:
        model = sm.OLS(y, X).fit()
        p_values.append(model.pvalues[snp])
    except:
        p_values.append(np.nan)

# Manhattan plot
plt.figure(figsize=(12,4))
plt.scatter(range(len(p_values)), -np.log10(p_values), s=2)
plt.xlabel('SNP index')
plt.ylabel('-log10(p-value)')
plt.title('GWAS Manhattan Plot')
plt.show()

# QQ plot
from statsmodels.graphics.gofplots import qqplot
qqplot(-np.log10([p for p in p_values if not np.isnan(p)]), line='45')
plt.title('QQ plot GWAS')
plt.show()

# In[22]:


"""
clearly inflated because alll points are above the diagonal 
"""

# ### Quantify Genomic Inflation
# Calculate lambda GC to assess potential p-value inflation.

# In[23]:


from scipy.stats import chi2
import numpy as np

# Remove nan p-values
pvals_clean = np.array(p_values)
pvals_clean = pvals_clean[~np.isnan(pvals_clean)]

# Convert p-values to chi-square statistics
chi2_stats = chi2.ppf(1 - pvals_clean, df=1)

# Calculate lambda
lambda_gc = np.median(chi2_stats) / 0.456

print("Genomic inflation factor (lambda):", lambda_gc)

# In[24]:


"""
lambda is 3.96 -> way too high -> add more PCs
"""

# ## 7) Re-run OLS GWAS With More PCs
# Increase the number of PC covariates to reduce structure-related inflation.

# In[25]:


phenotype_col = 'phenotype_value'
p_values = []

for snp in geno.columns:
    X = pheno[['PC1','PC2', 'PC3','PC4', 'PC5']].copy()
    X[snp] = geno[snp]
    X = sm.add_constant(X)
    y = pheno[phenotype_col]
    
    try:
        model = sm.OLS(y, X).fit()
        p_values.append(model.pvalues[snp])
    except:
        p_values.append(np.nan)

# Manhattan plot
plt.figure(figsize=(12,4))
plt.scatter(range(len(p_values)), -np.log10(p_values), s=2)
plt.xlabel('SNP index')
plt.ylabel('-log10(p-value)')
plt.title('GWAS Manhattan Plot')
plt.show()

# QQ plot
from statsmodels.graphics.gofplots import qqplot
qqplot(-np.log10([p for p in p_values if not np.isnan(p)]), line='45')
plt.title('QQ plot GWAS')
plt.show()

# ### OLS GWAS With 10 PCs
# Repeat association testing with a larger covariate set.

# In[26]:


phenotype_col = 'phenotype_value'
p_values = []

for snp in geno.columns:
    X = pheno[['PC1','PC2', 'PC3','PC4', 'PC5', 'PC6','PC7', 'PC8','PC9', 'PC10']].copy()
    X[snp] = geno[snp]
    X = sm.add_constant(X)
    y = pheno[phenotype_col]
    
    try:
        model = sm.OLS(y, X).fit()
        p_values.append(model.pvalues[snp])
    except:
        p_values.append(np.nan)

# Manhattan plot
plt.figure(figsize=(12,4))
plt.scatter(range(len(p_values)), -np.log10(p_values), s=2)
plt.xlabel('SNP index')
plt.ylabel('-log10(p-value)')
plt.title('GWAS Manhattan Plot')
plt.show()

# QQ plot
from statsmodels.graphics.gofplots import qqplot
qqplot(-np.log10([p for p in p_values if not np.isnan(p)]), line='45')
plt.title('QQ plot GWAS')
plt.show()

# In[28]:


phenotype_col = 'phenotype_value'
p_values = []


# ## 8) Mixed Model Preparation (LIMIX)
# Prepare phenotype, genotype, kinship, and covariates for mixed-model GWAS.

# In[30]:


#let's use the limix library for Mixed Linear Model

#prepare the data
# phenotype vector
y = pheno[phenotype_col].values


# genotype matrix
G = geno.values

# In[31]:


print(y[0:5,])
print(G[0:5, 0:5])

# In[32]:


from limix.stats import linear_kinship

K = linear_kinship(G)

# In[33]:


y = y.reshape(-1,1)

# In[34]:


# covariates (PCs)
PCs = pheno[[f'PC{i}' for i in range(1,6)]].values
covs = np.column_stack([
    np.ones(len(y)),  # intercept
    PCs
])

print(covs[0:5, 0:5])

# In[35]:


print(G.shape)      # SNP matrix (samples Ã— SNPs)
print(y.shape)      # phenotype (samples Ã— 1)
print(K.shape)      # kinship (samples Ã— samples)
print(covs.shape)   # covariates (samples Ã— covariates)

# In[36]:


print(pd.__version__)

# ### LIMIX Compatibility Patch and Scan
# Apply compatibility fix (if needed), then run mixed-model association scan.

# In[37]:


import sys
import types
import pandas as pd

# create a fake module 'pandas.core.index'
mod = types.SimpleNamespace()

# use the public InvalidIndexError from pandas.errors
from pandas.errors import InvalidIndexError
mod.InvalidIndexError = InvalidIndexError

# insert it into sys.modules so Python uses it whenever something does
# 'from pandas.core.index import InvalidIndexError'
sys.modules['pandas.core.index'] = mod

# Now import LIMIX
from limix.qtl import scan


# ### Extract LIMIX P-values
# Collect p-values from LIMIX GWAS output for interpretation and plotting.

# In[38]:


result = scan(
    G=G,
    Y=y,
    K=K,
    M=covs
)

pvals = result.stats["pv20"]

# In[ ]:




# In[ ]:



