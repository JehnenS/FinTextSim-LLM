# FinTextSim-LLM: Improving Corporate Performance Prediction and Investment Decisions with Scalable LLM-Enhanced Financial Topic Modeling
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

Entropy and Silhouette Score exhibit a strong negative correlation, suggesting that areas of higher uncertainty are concentrated in regions where the clusters are less clearly separated. 
By applying SSC and entropy thresholds of 0.5 each, we identify 642,169 datapoints as ambiguous. 
This corresponds to 12% of the total dataset, which is subsequently routed to the RAG-LLM module for further refinement.
Of these flagged datapoints, 45,355 are ultimately classified as noise. 
This selective handling yields an overall coverage of 99.1%, providing a significant improvement over embedding-clustering and other LLM-based topic modeling methods. 

![knn_rag_topic_representations](https://github.com/user-attachments/assets/a7cd5f6f-7adb-4432-ae16-55e515683f14)

Besides quantitative improvements, qualitative investigation of the topic representations further highlight FinTextSim-LLM's ability to extract meaningful financial themes from financial disclosures.

### Corporate Performance Prediction
| Model  | Accuracy ↑  | F1 Score ↑  | ROC-AUC ↑   |
| ------ | ----------- | ----------- | ----------- |
| LR–F0  | 44.79       | 50.00       | 50.16       |
| LR–F1  | 72.46       | 63.18       | **73.39**   |
| LR–F2  | 72.52       | 62.90       | 73.19       |
| LR–F3  | **72.57**   | **63.28**   | 73.32       |
| RF–F0  | 48.71       | 65.51       | 54.25       |
| RF–F1  | **76.73**   | ***70.58*** | **78.66**   |
| RF–F2  | **76.73**   | ***70.58*** | **78.66**   |
| RF–F3  | **76.73**   | ***70.58*** | **78.66**   |
| XGB–F0 | 53.03       | 63.08       | 58.96       |
| XGB–F1 | 76.58       | 69.66       | ***80.47*** |
| XGB–F2 | 76.61       | 69.51       | 80.42       |
| XGB–F3 | ***76.80*** | **69.70**   | 80.43       |

RF achieves the strongest overall predictive performance, particularly with feature sets F1–F3, while XGBoost delivers the best single model–feature combination in ROC-AUC. Logistic Regression improves steadily with richer feature sets but remains weaker overall.

### Hedge Portfolio

#### t = 0
| Model  | Excess Return ↑ | t-Statistic ↑ | p-Value ↓   |
| ------ | --------------- | ------------- | ----------- |
| LR–F0  | 0.0578          | 1.93          | 0.054       |
| LR–F1  | 0.1279          | 2.95          | ***0.003*** |
| LR–F2  | 0.1294          | 2.97          | ***0.003*** |
| LR–F3  | ***0.1324***    | ***3.03***    | ***0.003*** |
| RF–F0  | **0.0774**      | **2.23**      | **0.026**   |
| RF–F1  | 0.0621          | 1.66          | 0.098       |
| RF–F2  | 0.0621          | 1.66          | 0.098       |
| RF–F3  | 0.0621          | 1.66          | 0.098       |
| XGB–F0 | 0.0593          | 1.81          | 0.071       |
| XGB–F1 | 0.0844          | 2.01          | 0.044       |
| XGB–F2 | 0.0862          | 2.05          | 0.041       |
| XGB–F3 | **0.0905**      | **2.14**      | **0.033**   |

At portfolio formation, strategies augmented with topic-specific sentiment achieve economically large and highly significant excess returns, peaking at 13.2% for Logistic Regression with F3. Models using richer textual representations consistently outperform finance-only baselines, indicating immediate market underreaction to narrative disclosures.


#### t = 1
| Model  | Excess Return ↑ | t-Statistic ↑ | p-Value ↓   |
| ------ | --------------- | ------------- | ----------- |
| LR–F0  | 0.0496          | ***2.57***    | ***0.010*** |
| LR–F1  | 0.0490          | 2.02          | 0.046       |
| LR–F2  | 0.0497          | 2.00          | 0.044       |
| LR–F3  | **0.0525**      | 2.12          | 0.034       |
| RF–F0  | **0.0475**      | **2.26**      | **0.024**   |
| RF–F1  | 0.0374          | 1.71          | 0.088       |
| RF–F2  | 0.0374          | 1.71          | 0.088       |
| RF–F3  | 0.0374          | 1.71          | 0.088       |
| XGB–F0 | 0.0420          | 2.07          | 0.038       |
| XGB–F1 | 0.0518          | 2.17          | 0.031       |
| XGB–F2 | 0.0537          | 2.23          | 0.026       |
| XGB–F3 | ***0.0542***    | **2.25**      | **0.025**   |

One year after formation, portfolios remain profitable and statistically significant. 
Both LR and XGB achieve the strongest performance with F3.
XGBoost-F3 delivers the strongest performance. This persistence suggests that topic-specific sentiment contains durable forward-looking information not fully incorporated by markets.



#### t = 2
| Model  | Excess Return ↑ | t-Statistic ↑ | p-Value ↓   |
| ------ | --------------- | ------------- | ----------- |
| LR–F0  | **0.0157**      | **0.88**      | **0.379**   |
| LR–F1  | -0.0017         | -0.09         | 0.930       |
| LR–F2  | -0.0017         | -0.09         | 0.931       |
| LR–F3  | -0.0046         | -0.23         | 0.815       |
| RF–F0  | -0.0051         | -0.29         | 0.775       |
| RF–F1  | **0.0061**      | **0.33**      | **0.743**   |
| RF–F2  | **0.0061**      | **0.33**      | **0.743**   |
| RF–F3  | **0.0061**      | **0.33**      | **0.743**   |
| XGB–F0 | -0.0101         | -0.56         | 0.578       |
| XGB–F1 | 0.0240          | 1.23          | 0.219       |
| XGB–F2 | ***0.0282***    | ***1.44***    | ***0.150*** |
| XGB–F3 | 0.0261          | 1.33          | 0.184       |

Two years after formation, return magnitudes decline and statistical significance weakens.
Yet, XGBoost with topic-level sentiment still delivers the highest performance. 
The decay pattern is consistent with gradual information diffusion rather than immediate price adjustment.

## Implications
The results demonstrate that topic-specific sentiment extracted from corporate disclosures provides economically meaningful and persistent signals for financial prediction. 
Integrating FinTextSim-LLM outputs with traditional fundamentals improves classification accuracy, probability calibration, and downstream portfolio performance. 
Hedge strategies based on enriched textual features generate large and statistically significant excess returns at formation and remain profitable over subsequent horizons, consistent with slow market incorporation of narrative information. 
These findings suggest that qualitative disclosures encode durable, forward-looking signals that complement accounting data and can be deployed in scalable, transparent forecasting systems for real-world financial applications.
