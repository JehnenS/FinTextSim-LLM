# ADD FEATURE_CREATION, RAG + FINAL RESULTS AND IMPLICATIONS AFTER EVERYTHING IS IN PLACE --> UPDATED RAG-TEXT-FEATURES

# FinTextSim-LLM: Improving Corporate Performance Prediction and Investment Decisions with Scalable LLM-Enhanced Financial Topic Modeling
Repository containing the code for FinTextSim-LLM as outlined in Jehnen et al. (2025) (https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5365002) 
FinTextSim-LLM is a a scalable and interpretable framework for extracting topic-specific sentiment from 10-K filings. 
Our approach combines FinTextSim, a domain-specific sentence-transformer, with a lightweight, retrieval-augmented generation setup. 
Over 86% of sentences are classified via efficient embedding-based similarity. Only ambiguous cases, identified through entropy and silhouette score, are passed to an LLM, guided by topic definitions and document context. 
This hybrid design injects domain-knowledge, improves efficiency and enhances transparency, addressing key concerns about LLM deployment in financial settings.

Empirically, FinTextSim-LLM achieves \textcolor{red}{98.2\%} coverage, substantially outperforming embedding-clustering and LLM-based topic modeling methods.
When integrated into predictive models,  FinTextSim-LLM's topic-specific sentiment significantly enhances forecasts of changes in diluted earnings per share, outperforming financial-only models, and exceeding the effect of document-level sentiment by up to \textcolor{red}{49\%}.

Importantly, in a downstream hedge portfolio task, models incorporating topic-level sentiment generate statistically significant excess returns, capturing persistent underreaction to narrative disclosures. 
Text-only features alone recover a substantial share of the return-relevant information, while combining sentiment with financial variables yields the strongest and most robust portfolio performance across multiple time horizons. 
These findings highlight that qualitative disclosures encode durable, forward-looking information that complements quantitative fundamentals.

Overall, FinTextSim-LLM provides a transparent, scalable, and domain-aligned approach for integrating LLM-derived textual signals into financial prediction, offering both improved forecasting performance and economically meaningful investment insights.
The processing of the code is organized into several modules.


## Data Origin
We source our data from the Notre Dame Software Repository for Accounting and Finance in text-file format, which underwent a 'Stage One Parse' to remove all HTML tags. 
The data can be found at: https://sraf.nd.edu/data/stage-one-10-x-parse-data/. 
Financial data is sourced from FinancialModelingPrep (https://site.financialmodelingprep.com/)


## Main Components

### documents
- extract 10-K reports from S&P 500 companies
- isolation of the MDA&A section (Item 7 and Item 7A)
- detection of outlier documents
- tokenization into sentences and text cleaning
- perform necessary preprocessing steps

### modalities
- extraction of modalities (sentiment)
- embedding generation

### preprocess
- preprocessing for ML approaches

### topic_assignment
- pseudo-label assignment for each datapoint
- assignment based on cosine similarity to the closest topic centroid from FinTextSim's labeled dataset

### evaluation
- evaluation of cluster properties of FinTextSim-LLM's pseudo label assignments
- calculation of entropy and silhouette score for each datapoint

### rag
- running of the RAG module for ambiguous datapoints identified by entropy and silhouette score

### feature_creation
- create financial and textual features for the ML tasks

### ml
- creation and evaluation of
-   binary prediction of normalized diluted EPS movement
-   hedge portfolio


## Results

### Topic Modeling with FinTextSim-LLM


### Corporate Performance Prediction


### Hedge Portfolio


## Implications

## Citation
To cite the FinTextSim paper, please use the following bibtex reference:

**Jehnen, S., Ordieres-Meré, J., & Villalba-Díez, J. (2025).**  
*FinTextSim-LLM: Improving Corporate Performance Prediction Through Scalable, Llm-Enhanced Financial Topic Modeling and Aspect-Based Sentiment.*  
https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5365002  

```bibtex
@article{jehnen2025,
  title={FinTextSim-LLM: Improving Corporate Performance Prediction Through Scalable, Llm-Enhanced Financial Topic Modeling and Aspect-Based Sentiment},
  author={Jehnen, Simon and Ordieres-Mer{\'e}, Joaqu{\'\i}n and Villalba-D{\'\i}ez, Javier},
  journal={SSRN},
  year={2025}
}
