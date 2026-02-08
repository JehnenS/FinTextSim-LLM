import pandas as pd
import numpy as np
import pickle

import pandas as pd
import re

swade_features = [
    #value
    "ratios_priceEarningsRatio", #price earnings ratio --> in Swade, other way around earnings to price
    "ratios_priceToSalesRatio", #price to sales ratio --> in Swade, other way around Sales to price
    "km_dividendYield", #dividend yield
    "ratios_priceBookValueRatio", #price to book value ratio --> in Swade, other way around book-value to price
    #growth
    "is_growthRevenue", #growth revenue
    "is_growthEPS", #growth EPS
    #profitability
    "gp_to_assets", 
    "net_income_to_equity", 
    "net_income_to_assets", #profitability
    #momentum
    "medium_term_price_momentum",
    #other
    "km_marketCap", #market capitalization
    "short_term_price_reversal", 
    "volatility", 
    "market_leverage", 
    "share_turnover",
]


dups = [
    "km_operatingCashFlowPerShare",  # also in ratios_operatingCashFlowPerShare
    "km_freeCashFlowPerShare",       # also in ratios_freeCashFlowPerShare
    "km_cashPerShare",               # also in ratios_cashPerShare

    "km_peRatio",                    # also in ratios_priceEarningsRatio
    "km_priceToSalesRatio",          # also in ratios_priceToSalesRatio / ratios_priceSalesRatio
    "km_pbRatio",                    # also in ratios_priceBookValueRatio / ratios_priceToBookRatio
    "km_ptbRatio",                   # also in ratios_priceToBookRatio (redundant naming)

    "km_currentRatio",               # also in ratios_currentRatio
    "km_interestCoverage",           # also in ratios_interestCoverage
    "km_dividendYield",              # also in ratios_dividendYield
    "km_payoutRatio",                # also in ratios_payoutRatio / ratios_dividendPayoutRatio
    #fg_ duplicates to drop
    "fg_revenueGrowth",                 # duplicate of is_growthRevenue
    "fg_grossProfitGrowth",             # duplicate of is_growthGrossProfit
    "fg_operatingIncomeGrowth",         # duplicate of is_growthOperatingIncome
    "fg_netIncomeGrowth",               # duplicate of is_growthNetIncome
    "fg_epsgrowth",                     # duplicate of is_growthEPS
    "fg_epsdilutedGrowth",              # duplicate of is_growthEPSDiluted
    "fg_weightedAverageSharesGrowth",   # duplicate of is_growthWeightedAverageShsOut
    "fg_weightedAverageSharesDilutedGrowth", # duplicate of is_growthWeightedAverageShsOutDil

    "fg_operatingCashFlowGrowth",       # duplicate of cf_growthOperatingCashFlow
    "fg_freeCashFlowGrowth",            # duplicate of cf_growthFreeCashFlow

    "fg_receivablesGrowth",             # duplicate of bs_growthNetReceivables
    "fg_inventoryGrowth",               # duplicate of bs_growthInventory
    "fg_assetGrowth",                   # duplicate of bs_growthTotalAssets
    "fg_debtGrowth",                    # duplicate of bs_growthTotalDebt

    "fg_rdexpenseGrowth",               # duplicate of is_growthResearchAndDevelopmentExpenses
    "fg_sgaexpensesGrowth",             # duplicate of is_growthSellingAndMarketingExpenses
]
print(f"Number of duplicate features: {len(dups)}")


    

def safe_divide(numer, denom):
    """
    Safe division by replacing denominators equal to 0 with NaN --> values of 0 does not make sense --> NA
    """
    denom = denom.replace(0, np.nan)
    return numer / denom

def extract_topic_sentiments(df: pd.DataFrame) -> dict:
    """
    Extracts topic-level sentiment scores from wide-format DataFrame. --> input for transformer later on 

    Returns:
        dict mapping (doc_id, topic_id) → sentiment score (float)
    """
    topic_sentiments = {}

    # Regex to match columns like 'mean_sentiment_topic_3' --> also captures topic_-1
    topic_pattern = re.compile(r"mean_sentiment_topic_(-?\d+)")

    for _, row in df.iterrows():
        doc_id = int(row["doc_id"])
        for col in df.columns:
            match = topic_pattern.match(col)
            if match:
                topic_id = int(match.group(1))
                sentiment_score = row[col]
                if pd.notna(sentiment_score):
                    topic_sentiments[(doc_id, topic_id)] = float(sentiment_score)

    return topic_sentiments

def extract_doc_level_features(df: pd.DataFrame, fin_features:list[str]) -> dict:
    """
    Extracts document-level features from wide-format DataFrame. --> input for transformer later on 

    Returns:
        dict mapping (doc_id, topic_id) → sentiment score (float)
    """
    doc_features = {}

    for _, row in df.iterrows():#iterate over dataframe
        doc_id = int(row["doc_id"]) #extract document-id
        doc_results = []
        for col in df.columns: #iterate over each column
            if col in fin_features: #check if the column is in the relevant financial features
                value = row[col] #extract the value
                if pd.notna(value):
                    doc_results.append(float(value))
                else:
                    doc_results.append(float(0.0)) #impute 0 when there is no value
            doc_features[doc_id] = doc_results

    return doc_features



#--------------create the features


#add the missing cik-ticker combinations
add_cik_ticker_mapping = {
    "0000008868": "AVP", #Avon Products
    "0000014693": "BF-A", #not present?
    "0000020520": "FYBR", #not present?
    "0000023082": "CSC", #computer science corp
    "0000029915": "DOW", #dow chemical
    "0000050104": "ANDV", #andeavor llc
    "0000098246": "TIF", #Tiffany and co
    "0000203527": "VAR", #varian medical systems
    "0000314808": "VAL", #Valaris LLC - not present?
    "0000354908": "TDY", #teledyne
    "0000356028": "CA" ,#CA inc - not present?
    "0000721683": "TSS", #total systems services
    "0000750556": "STI", #sunstrust banks
    "0000754737": "SCG", #scana corp - not present?
    "0000773910": "APC", #andarko petroleum corp
    "0000801898": "JOY", #joy global inc
    "0000875159": "XL", #xl group - not present?
    "0001015780": "ETFC", #e trade financial corp
    "0001039101": "LLL", #3L technologies inc
    "0001047122": "RTN", #raytheon co
    "0001067983": "BRK-N", #berkshire hathaway - not present
    "0001087423": "RHT", #red hat inc
    "0001108827": "QEP", #qep resources inc
    "0001110783": "MON", #monsanto co
    "0001122304": "AET", #aetna inc
    "0001137411": "COL", #rockwell collins inc
    "0001279363": "WCG", #wellcare health plans inc - not present?
    "0001358071": "CXO", #concho resources inc
    "0001361658": "TNL", #travel and leisure co - not present?
    "0001465128": "HOT", #starwood property trust - STWD not present?
    "0001496048": "BPYU", #brookfield property reit inc - not present?
    "0001578845": "AGN", #allergan plc
    "0001598014": "INFO", #IHS markit ltd
    "0001629995": "CPGX", # columbia pipeline group
    "0001646383": "CSRA"
}

add_cik_ticker_mapping2 = {
    "0001364742": "BLK", #black rock
    "0000945764": "DEN", #denbury - not present?
    "0000009892": "BCR", #cr bard - not present?
    "0000011199": "AMCR", #amcor
    "0000034408": "FAY.DE", #Family dollar sotr - or FDO?! - not present?
    "0000072207": "NBL", #noble energy - not present?
    "0000075829": "PLL", #pall corp
    "0000086144": "SWY", #safeway inc - not present?
    "0000090185": "SIAL", #sigma aldrich - not present?
    "0000203077": "STJ", #st jude medical - not present?
    "0000350563": "TE_OLD", #teco energy  - still up to date? - not present?
    "0000704051": "LM", #legg mason - still up to date? - not present?
    "0000768251": "ALR.DE",#altera corp - still up to date? - not present?
    "0000790070": "EMC_old", #emc corp - still up to date? - not present?
    "0000800459": "HAR", #harman international industires - still up to date? - not present?
    "0000812233": "OI", #owen illinois group
    "0000849213": "PCL", #plum creek timber co - not present?
    "0000850693": "AGN", #allergan plc - not present?
    "0000863157": "PETM", #petsmart - not present?
    "0000912750": "NFX", #newfield exploration - still up to date? - not present?
    "0000916863": "TEG", #integrys energy group - still up to date? - not present?
    "0000921847": "HCBK", #hudson city bancorp - still up to date? - not present?
    "0001004155": "AGL", #agl resources - or GAS(-N)? - still up to date? - not present?
    "0001053112": "CVC", #cablevision systems corp - still up to date? - not present?
    "0001135971": "PCPO", #pepco holdings - still up to date? - not present?
    "0001274057": "HSP", #hospira inc - still up to date? - not present?
    "0001339947": "VIAC", #viacom - still up to date? - not present? M&A
    "0001373835": "SE", #spectra energy - still up to date? - not present?
    "0001385187":"COV", #covidien - still up to date? - not present?
    "0001424847": "LO", #lorillad - still up to date? - not present?
    "0001452575": "MJN", #mead johnson nutrition - still up to date? - not present?
    "0001457543": "CFN", #carefusion corp - still up to date? - not present?
    "0001465112": "DTV", #directv - still up to date? - not present?
    "0001545158": "KHC", #kraft foods group
    "0001620546": "BXLT", #baxalta inc - still up to date? - not present?
    "0001678531": "EVHC", #envision healthcare - still up to date? - not present?
}
