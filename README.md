# FinTextSim-LLM: Improving Corporate Performance Prediction and Investment Decisions with Scalable LLM-Enhanced Financial Topic Modeling
Repository containing the code for FinTextSim-LLM as outlined in Jehnen et al. (2025) (https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5365002) 
FinTextSim-LLM is a a scalable and interpretable framework for extracting topic-specific sentiment from 10-K filings. 
Our approach combines FinTextSim, a domain-specific sentence-transformer, with a lightweight, retrieval-augmented generation setup. 
88% of sentences are classified via efficient embedding-based similarity. Only ambiguous cases, identified through entropy and silhouette score, are passed to an LLM, guided by topic definitions and document context. 
This hybrid design injects domain-knowledge, improves efficiency and enhances transparency, addressing key concerns about LLM deployment in financial settings.

Empirically, FinTextSim-LLM achieves 99.1% coverage, substantially outperforming embedding-clustering and LLM-based topic modeling methods.
When integrated into predictive models of diluted earnings-per-share direction, topic-specific sentiment yields systematic improvements in probability calibration.
Importantly, in a downstream hedge-portfolio evaluation, models incorporating topic-level sentiment generate economically large and statistically significant excess returns of up to 13%, capturing persistent market underreaction. Text-only strategies recover up to 84% of the perfect-foresight benchmark. Overall, combining sentiment with financial variables yields the strongest and most robust portfolio performance across multiple time horizons.
These findings indicate that qualitative disclosures encode durable, forward-looking information that complements quantitative fundamentals.

Overall, FinTextSim-LLM provides a transparent, scalable, and domain-aligned approach for integrating LLM-derived textual signals into financial prediction, offering both improved forecasting performance and economically meaningful investment insights.

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

### evaluation
- evaluation of cluster properties of FinTextSim-LLM's pseudo label assignments
- calculation of entropy and silhouette score for each datapoint
- topic quality assessment

### feature_creation
- creation of textual, financial and economic features
- extract financial targets

### ml
- creation and evaluation of
-   binary prediction of normalized diluted EPS movement
-   hedge portfolio

### modalities
- extraction of modalities (sentiment)
- embedding generation

### preprocess
- preprocessing for ML approaches

### rag
- running of the RAG module for ambiguous datapoints identified by entropy and silhouette score

### topic_assignment
- pseudo-label assignment for each datapoint
- assignment based on cosine similarity to the closest topic centroid from FinTextSim's labeled dataset




## Results

### Topic Modeling with FinTextSim-LLM

![entropy_ssc_acl_modern_bert](https://github.com/user-attachments/assets/2df1563f-d686-4d60-85b2-53da6eb6e393)

![knn_rag_topic_representations](https://github.com/user-attachments/assets/a7cd5f6f-7adb-4432-ae16-55e515683f14)


### Corporate Performance Prediction

| Model                | Intratopic Similarity ↑ | Intertopic Similarity ↓ | Outliers within BERTopic ↓ |
|-----------------------|--------------------------|---------------------------|------------------|
| FinTextSim            | **0.998**                | **-0.075**                  | **240,823**             |
| all-MiniLM-L6-v2 (AM) | 0.584                     | 0.563                      | 781,965             |
| all-mpnet-base-v2     | 0.614                     | 0.625                      | 784,225             |
| distil-RoBERTa      | 0.773                    | 0.883                        | 1,332,620          |


### Hedge Portfolio

#### t = 0

| Model                | Intratopic Similarity ↑ | Intertopic Similarity ↓ | Outliers within BERTopic ↓ |
|-----------------------|--------------------------|---------------------------|------------------|
| FinTextSim            | **0.998**                | **-0.075**                  | **240,823**             |
| all-MiniLM-L6-v2 (AM) | 0.584                     | 0.563                      | 781,965             |
| all-mpnet-base-v2     | 0.614                     | 0.625                      | 784,225             |
| distil-RoBERTa      | 0.773                    | 0.883                        | 1,332,620          |


#### t = 1

| Model                | Intratopic Similarity ↑ | Intertopic Similarity ↓ | Outliers within BERTopic ↓ |
|-----------------------|--------------------------|---------------------------|------------------|
| FinTextSim            | **0.998**                | **-0.075**                  | **240,823**             |
| all-MiniLM-L6-v2 (AM) | 0.584                     | 0.563                      | 781,965             |
| all-mpnet-base-v2     | 0.614                     | 0.625                      | 784,225             |
| distil-RoBERTa      | 0.773                    | 0.883                        | 1,332,620          |



#### t = 2

| Model                | Intratopic Similarity ↑ | Intertopic Similarity ↓ | Outliers within BERTopic ↓ |
|-----------------------|--------------------------|---------------------------|------------------|
| FinTextSim            | **0.998**                | **-0.075**                  | **240,823**             |
| all-MiniLM-L6-v2 (AM) | 0.584                     | 0.563                      | 781,965             |
| all-mpnet-base-v2     | 0.614                     | 0.625                      | 784,225             |
| distil-RoBERTa      | 0.773                    | 0.883                        | 1,332,620          |




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
