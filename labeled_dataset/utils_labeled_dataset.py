import pickle
from tqdm import tqdm
import pandas as pd
import numpy as np




"""
PAPER 2 
Improvements:
- widened substitutions for avoiding problems with keyword approach


New Paper 1 approach
- widened substitutions
- removal of energy and covid topic --> not in every year and cross-dependent to other topics (e.g. covid disrupted supply chain); energy topic seemed unreliable --> company names, etc.
"""

#--------------------------lemmatization, keep relevant parts of speech
#spacy
import nltk
from nltk.tokenize import sent_tokenize
import re

#lemmatization of text --> business vocabulary
substitutions =[
    #double whitespaces, urls, newline characters, etc.
    (r'\b\w*(?:TextBox|TextLine|LTCurve|LTRect|LTFigure|LTImage|LTLine)\w*\b', ''), #NEW: remove textboxes, etc.
    (r'\b\w*(?:cid.*?){2,}\w*\b', ''), #NEW: remove the "cid"-noise
    (r'\n\s*(\S+(?:\s+\S+){0,2})\s*(?=[\n:])', ''), #NEW: remove newline characters followed by max three words --> headlines and bulletpoints
    (r'/n\s*(\S+(?:\s+\S+){0,2})\s*(?=[/n:])', ''), #NEW: remove newline characters followed by max three words --> headlines and bulletpoints
    (r'\b\w{31,}\b', ''), #NEW: remove words with more than 30 characters
    (r'\n', ''),
    (r'/n', ''), #remove newline characters
    #modification 19.08.25
    (r'\be\.?u\.?\b', 'eu'), #swap e.u/e.u. and u.s/u.s. to eu/us
    (r'\bu\.?s\.?\b', 'us'),
    ### Modification 23.03.25
    #(r'\b[^aeiouAEIOU\s]+\b', ''), #words that do not contain any vowels --> especially abbreviations
    (r'\b[^aeiouAEIOU\s]{3,}\b', ''), ## NEW PAPER 2: Remove non-vowel words longer than 2 letters --> update for paper to compared to words without vowels at all --> keep e.g. the word "by"
    ###
    #(r'\s+', " "), #remove double whitespaces
    (r'http\S+', ''), # remove urls starting with http
    (r'www\.\S+', ''), # remove urls starting with www.
    (r'\S*\.de\S*|\S*\.com\S*|\S*\.us\S*', ''),  # NEW: Match any word containing ".de" or ".com" or ".us" --> e.g.  microsoft.com/investors is now captured
    #(r'\b\w*[\/\\]\w*\b', ''), #NEW: Match any word containing a "/" or a "\" --> hyperlink
    #(r'\b(?:\S+\.de|\S+\.com)\b', ''), #remove urls that end with ".de" or ".com"
    (r'\b\w*\d\w*\b', ''), # NEW: Remove words containing "_<number> or number_text or text_number_text", #-------------->> NEW!
    (r'\b\d+\b', ''), #remove remaining numbers
    (r'\S*@\S*\s*', ''), #remove each word that consists an "@" --> mail adresses
    (r'\b\w*\(at\)\w*\b', ''), #NEW: remove each word that consists an (at) --> variation of e-mail address
    (r'\b\w*\(dot\)\w*\b', ''), #NEW: remove each word that consists an (dot) --> variation of e-mail address or url
    #addition 23.03.25
    (r'\b[Uu]\.?[Ss]\.?\b', 'us'), #NEW PAPER 2: switch u.s. to us to avoid the removal of punctuation within it --> if not, only "u" remains
    ###
    (r'[^\w\s.]', ''),  # remove punctuation except periods
    (r'\b\w{31,}\b', ''), #NEW: remove words with more than 30 characters --> Again as the removal of punctuation may have change word-lengths
    (r'\s+', " "), #remove double whitespaces ----------------->moved it down
    #topic words - single analysis of those words
    (r'supply chain', 'supplychain'),
    (r'supply-chain', 'supplychain'),
    (r'cash flow', 'cashflow'),
    (r'human resources', 'hr'), #as removal of no vowel words takes place before this, it is not affected in the final text and correctly "human resources" is replaced with "hr"
    (r'human resource', 'hr'),
    (r'process management', 'processmanagement'),
    (r'full-time-equivalent', "fte"),
    (r'full time equivalent', 'fte'),
    (r'return on assets', 'roa'),
    (r'return on equity', 'roe'),
    (r'return on interest', 'roi'),
    (r'chief executive officer', "ceo"),
    (r'chief financial officer', "cfo"),
    (r'chief information officer', "cio"),
    (r'chief operations officer', "coo"),
    (r'chief intelligence officer', "cio"),
    (r'chief sales officer', "cso"),
    #NEW: additional things,
    (r'cyber security', 'cybersecurity'),
    (r'table of contents', ''),
    (r'table of content', ''),
    (r'risk management', 'riskmanagement'),
    #further additional things 12.02.25
    #v3 21.03.25
    (r'european union', 'europeanunion'), #"union"
    (r'chief legal officer', 'clo'), #legal
    (r'california consumer privacy act', 'ccpa'), # consumer
    (r'nextgeneration management', 'nextgenerationmanagement'),
    (r'programming environment', 'programmingenvironment'), #environment
    (r'development environment', 'developmentenvironment'),
    (r'mileage credit', 'mileagecredit'), #credit
    (r'credit card', 'creditcard'),
    (r'pre tax', 'pretax'), #tax
    (r'after tax', 'aftertax'),
    # (24.03.25)
    (r'recoupment demand', 'recoupmentdemand'), #demand
    (r'demand for recoupment', 'recoupmentdemand'),
    (r'submerchant account', 'submerchantaccount'), #account
    (r'mid account', 'midaccount'),
    (r'diversity equity', 'diversityequity'), #equity
    (r'train velocity', 'trainvelocity'), #train
    (r'train speed', 'trainspeed'),
    (r'passenger train', 'passengertrain'),
    (r'work train', 'worktrain'),
    (r'desktop workstation', 'desktopworkstation'),
    (r'union carbide', 'unioncarbide'), #union
    (r'option pric', 'optionpric'), #pric
    (r'cloud environment', 'cloudenvironment'), #environment
    #25.03.25
    (r'consumer privacy law', 'consumerprivacylaw'),
    #26.03.25
    (r'dog training', 'dogtraining'), #training
    (r'economic environment', 'economicenvironment'), #environment
    (r'political environment', 'politicalenvironment'),
    (r'security environment', 'securityenvironment'),
    (r'compute networking segment', 'computenetworkingsegment'), #network/work
    (r'computing mellanox networking', 'computingmellanoxnetworking'),
    (r'working capital', 'workingcapital'), #--> needed to remove capital from liquidity/solvency
    (r'stock market', 'stockmarket'), #market --> not sales-related
    (r'client retention', 'clientretention'), #retention
    (r'wallet retention', 'walletretention'),
    (r'risk retention', 'riskretention'), 
    #27.03.
    (r'energy drink', 'energydrink'), #energy
    (r'monster energy', 'monsterenergy'),
    (r'going concern', 'goingconcern'),
    #28.03.
    (r'western union', 'westernunion'), #union
    (r'capital market', 'capitalmarket'),
    (r'omni plastic', 'omniplastic'), #plastic
    (r'human capital', 'humancapital'),
    # 31.03.
    (r'canada revenue agency', 'canadarevenueagency'), #revenue
    # 01.04.
    (r'liquid asset buffer', 'liquidassetbuffer'), #asset
    (r'dog tracking and training device', 'dogtrackingandtrainingdevice'), #training
    (r'intellectual property', 'intellectualproperty'),
    (r'twitter account', 'twitteraccount'), #account
    (r'instagram account', 'instagramaccount'),
    (r'periscope account', 'periscopeaccount'),
    (r'facebook account', 'facebookaccount'),
    #02.04.
    (r'interest rate environment', 'interestrateenvironment'), #environment
    (r'puerto rico department of consumer affair', "puertoricodepartmentofconsumeraffair"), #consumer
    (r'equity market', 'equitymarket'),
    (r'equities market', 'equitiesmarket'),
    (r'revenue bond', 'revenuebond'), #revenue
    # 03.04.
    (r'investigative demand', 'investigativedemand'), #demand
    (r'missile competition', 'missilecompetition'), #competition
    (r'computing environment', 'computingenvironment'), #environment
    (r'container environment', 'containerenvironment'),
    (r'consumer goods', 'consumergoods'), #consumer
    (r'safety performance', 'safetyperformance'), #performance
    (r'social media account', 'socialmediaaccount'), #account
    (r'socialmedia account', "socialmediaaccount"),
    (r'account takeover', 'accounttakeover'),
    (r'esg performance', 'esgperformance'), #performance
    (r'sandbox environment', 'sandboxenvironment'), #environment
    (r'local environment', 'localenvironment'),
    (r'safety management', 'safetymanagement'), #management
    #04.04.
    (r'project manag', 'projectmanag'),
    (r'sterile environment', 'sterileenvironment'), #environment
    (r'train control', 'traincontrol'),
    (r'water risk', 'waterrisk'), #water
    (r'water footprint', 'waterfootprint'),
    (r'authority for consumers and markets', 'authorityforconsumersandmarkets'), #market/consumer
    # 05.04.
    (r'control product', 'controlproduct'),
    (r'email account', 'emailaccount'), #account
    (r'microsoft office account', 'microsoftofficeaccount'),
    (r'president packaging specialty plastics', 'presidentpackagingspecialtyplastics'), #plastic
    (r'talent develop', 'talentdevelop'), #talentdevelopment
    (r'talent manag', 'talentmanag'),
    (r'feedback cultur', 'feedbackcultur'), #feedbackculture
    #06.04.
    (r'waste services company', 'wasteservicescompany'), #waste
    (r'waste management inc', 'wastemanagementinc'),
    (r'waste management holdings inc', 'wastemanagementholdingsinc'),
    (r'median income', 'medianincome'), #income
    #07.04.
    (r'master of business administration', 'mba'), #business
    (r'equity index product', 'equityindexproduct'), #equity
    (r'deposit repricing', 'depositrepricing'), #price
    (r'investor relation', 'investorrelation'), #invest
    (r'transmission incentiv', 'transmissionincentiv'), #incentiv
    (r'consumer data protection act', 'cdpa'), #consumer
    (r'self insured retention', 'selfinsuredretention'), #retention
    #08.04.
    (r'internal revenue code', 'internalrevenuecode'), #revenue
    (r'household income', 'householdincome'), #income
    (r'average income', 'averageincome'),
    (r'competition law', 'competitionlaw'), #competition
    (r'bachelor of science degree in business administration', "bsc in ba"), #business
    (r'school of business', 'schoolofbusiness'), 
    (r'design account', 'designaccount'), #account
    (r'national account contract', 'nationalaccountcontract'), #account
    (r'stock performance', 'stockperformance'), #performance
    (r'stock price', "stockprice"), 
    (r'stockprice performance', 'stockpriceperformance'),
    (r'equity research', 'equityresearch'), #equity
    (r'privately funded charity', 'privatelyfundedcharity'), #fund
    (r'internal revenue service', 'internalrevenueservice'), #revenue
    #09.04.
    (r'middle income', 'middleincome'), #income
    (r'high income', 'highincome'),
    (r'low income', 'lowincome'),
    (r'fare audit', 'fareaudit'), #audit
    (r'debit card', 'debitcard'), #ebit
    (r'debit network', 'debitnetwork'),
    (r'guest environment', 'guestenvironment'), #environment
    (r'schneider electric', 'schneiderelectric'), #electric
    (r'government bond', 'governmentbond'), #government
    (r'brokerage account', 'brokerageaccount'), #account
    (r'internal control', 'internalcontrol'), #control --> added keywords for accounting
    (r'control framework', 'controlframework'), 
    (r'control risk', 'controlrisk'),
    (r'control structur', 'controlstructur'),
    #10.04.
    (r'account compromise', 'accountcompromise'), #account
    (r'account feed', 'accountfeed'),
    (r'vision loss', 'visionloss'), #loss
    (r'regulated discovery service', 'regulateddiscoveryservice'), # regulate
    (r'regulated safety assessment', 'regulatedsafetyassessment'),
    (r'customer complaint', 'customercomplaint'), #complaint
    (r'irish revenue', 'irishrevenue'), #revenue
    (r'ehs performance', 'ehsperformance'), #performance
    (r'quality control', 'qualitycontrol'), #keyword for operations
    (r'sports leagues team', 'sportsleaguesteam'), #team
    (r'sports team', 'sportsteam'),
    (r'aircraft carrier refueling', 'aircraftcarrierrefueling'), #fuel
    (r'business day', 'businessday'), #business
    (r'competition authorit', 'competitionauthorit'), #competition
    # 11.04.
    (r'index account option', 'indexaccountoption') ,#account
    (r'equity index', 'equityindex'), #equity
    (r'equity fund', 'equityfund'),
    (r'pricing service', 'pricingservice'), #price
    (r' financial result', 'financialresult'),
    (r'operational result', 'operationalresult'),
    (r'operating result', 'operatingresult'),
    (r'result of operation', 'operatingresult'), #--> do it like this to avoid keyword blocking
    (r'yearend result', 'yearendresult'),
    (r'competitive environment', 'competitiveenvironment'), #environment
    (r'key account', 'keyaccount'), #account
    (r'customer account', 'customeraccount'),
    (r'remote work', 'remotework'), #work
    (r'work remote', 'workremote'),
    (r'working hours', 'workinghours'),
    (r'work hours', 'workhours'),
    (r'work life', 'worklife'),
    (r'opening balance', 'openingbalance'), #balance
    (r'closing balance', 'closingbalance'),
    (r'bank balance', 'bankbalance'),
    (r'cash balance', 'cashbalance'),
    (r'balance sheet', 'balancesheet'),
    # 12.04.
    (r'onboarding account', 'onboardingaccount'), #account
    (r'equity securit', 'equitysecurit'), #equity/security
    (r'securities act', 'securitiesact'), #securities
    (r'securities exchange', 'securitiesexchange'),
    (r'moderate income', 'moderateincome'), #income
    (r'lending environment', 'lendingenvironment'), # environment
    #14.04. --> v6
    (r'work environment', 'workenvironment'), # environment
    (r'operating environment', 'operatingenvironment'),
    (r'software environment', 'softwareenvironment'),
    (r'hershey income accelerator', 'hersheyincomeaccelerator'), #income
    (r'farmer income', 'farmerincome'),
    (r'realty income corporation', 'realtyincomecorporation'),
    (r'checking account', 'checkingaccount'), #account
    (r'deposit account', 'depositaccount'),
    (r'workplace adjustment', 'workplaceadjustment'), #adjustmente
    (r'loss control service', 'losscontrolservice'), #loss
    (r'specialist loss control', 'specialistlosscontrol'),
    (r'transfer pricing', 'transferpricing'), #pricing
    (r'air system audit', 'airsystemaudit'), #audit
    (r'document filing solution', 'documentfilingsolution'), #filing
    (r'client incentiv', 'clientincentiv'), #incentiv
    (r'threadneedle fund', 'threadneedlefund'), #fund
    (r'exchangetraded fund', 'etf'),
    (r'opened mutual fund', 'openedmutualfund'),
    (r'career develop', 'careerdevelop'),
    (r'equity portfolio', 'equityportfolio'), #equity
    (r'domestic equit', 'domesticequit'), 
    (r'foreign equit', 'foreignequit'),
    (r'inventory loss', 'inventoryloss'), #loss
    (r'index mutual fund', 'indexmutualfund'), #fund
    #15.04.
    (r'regulated water', 'regulatedwater'), #regulat
    (r'tribal government', 'tribalgovernment'), #government
    # 16.04
    # 17.04.
    (r'legal entit', 'legalentit'), #legal
    (r'investor website', 'investorwebsite'), #invest
    (r'cash reward card', 'cashrewardcard'), #cash
    (r'cash back benefit', 'cashbackbenefit'),
    (r'global consumer services group', 'gcsg'), #consumer
    (r'australian competition and consumer commission', 'accc'),
    (r'union pacific', 'unionpacific'), #union
    (r'nutrient loss', 'nutrientloss'), #loss
    (r'fixed income clearing corporation', 'fixedincomeclearingcorporation'), #income
    #19.04.
    (r'nongovernmental organization', 'ngo'), #government
    (r'climate leadership', 'climateleadership'), #leadership
    #23.05. - v7 - expansion dataset
    (r'presuit demand', 'presuitdemand'), #demand
    (r'plead demand', 'pleaddemand'),
    (r'final average earnings formula', "finalaverageearningsformula"), #earnings
    (r'career average earnings formula', "careeraverageearningsformula"),
    (r'casualty treaty retention', 'casualtytreatyretention'), #retention
    (r'property treaty retention', 'propertytreatyretention'),
    (r'promoted account', 'promotedaccount'), #account
    (r'federal express', 'fedex'), #federal
    # 26.05.
    (r'consumer protection', 'consumerprotection'), #consumer protection
    (r'demand registration right', 'demandregistrationright'), #demand
    (r'administrative appeals process', 'administrativeappealsprocess'), #process
    (r'petroleum consultant', 'petroleumconsultant'), #consultant
    (r'demand repayment', 'demandrepayment'), #demand
    (r'repayment of demand', 'demandrepayment'),
    (r'customer environment', 'customerenvironmnent'), #environment
    (r'office of utility consumer counselor', 'oucc'), #consumer
    (r'casualty treaty retention', 'casualtytreatyretention'), #retention
    (r'union security', 'unionsecurity'), #union
    (r'discount rate environment', 'discountrateenvironmnet'), #environmnet
    (r'fiscal incentive', 'fiscalincentive'),#incentive
    (r'demand letter', 'demandletter'), #demand
    (r'sec enforcement staff', 'secenforcementstaff'), #staff
    (r'consumer class action', 'consumerclassaction'), #consumer
    (r'consumer financial protection bureau', 'cfp bureau'), #consumer
    #27.05.
    (r'cloud it environment', 'clouditenvironment'), #environment
    (r'noninterestbearing demand', 'noninterestbearingdemand'), #demand
    (r'ferc staff', 'fercstaff'), #staff
    (r'equityindex price', 'equityindexprice'),
    (r'allied waste', 'alliedwaste'), #waste
    (r'logo account', 'logoaccount'), #account
    (r'securities class action', 'securitiesclassaction'), #securities
    #22.08.
    (r'nan ya plastic', 'nanyaplastic'), #plastic
    (r'monetary union', 'monetaryunion'), #union
    (r'financial inclusion', 'financialinclusion'), #inclusion
    #29.08.
    (r'competition bureau', 'competitionbureau'), #competition
    (r'competition tribunal', 'competitiontribunal'),
    (r'derivative gainsloss', 'derivativegainsloss'), #loss
    #contractions (https://www.sjsu.edu/writingcenter/docs/handouts/Contractions.pdf)
    #left column
    (r"aren't", "are not"),
    (r"can't", "can not"),
    (r"couldn't", "could not"),
    (r"didn't", "did not"),
    (r"doens't", "does not"),
    (r"don't", "do not"),
    (r"hadn't", "had not"),
    (r"hasn't", "has not"),
    (r"haven't", "have not"),
    (r"he'd", "he would not"),
    (r"he'll", "he will"),
    (r"he's", "he is"),
    (r"i'd", "i would"),
    (r"i'll", "i will"),
    (r"i'm", "i am"),
    (r"i've", "i have"),
    (r"isn't", "is not"),
    (r"let's", "let us"),
    (r"mightn't", "might not"),
    (r"mustn't", "must not"),
    (r"shan't", "shall not"),
    (r"she'd", "she would"),
    (r"she'll", "she will"),
    (r"she's", "she is"),
    (r"shouldn't", "should not"),
    (r"that's", "that is"),
    #right column
    (r"there's", "there is"),
    (r"they'd", "they would"),
    (r"they'll", "they will"),
    (r"they're", "they are"),
    (r"they've", "they have"),
    (r"we'd", "we would"),
    (r"we're", "we are"),
    (r"we've", "we have"),
    (r"weren't", "were not"),
    (r"what'll", "what will"),
    (r"what're", "what are"),
    (r"what's", "what is"),
    (r"what've", "what have"),
    (r"where's", "where is"),
    (r"who'd", "whou would"),
    (r"who'll", "who will"),
    (r"who're", "who are"),
    (r"who's", "who is"),
    (r"who've", "who have"),
    (r"won't", "will not"),
    (r"wouldn't", "would not"),
    (r"you'd", "you would"),
    (r"you'll", "you will"),
    (r"you're", "you are"),
    (r"you've", "you have")  
]

def replace_subs(text, substitutions_list):
    """
    function to replace substitutions in text (words + re patterns) --> before tokenizing, etc.

    text: document
    substitutions list: list of tuples containing substitutions and replacements
    """
    text = text.lower() #convert to lower case

    text_subbed = text
    for pattern, replacement in substitutions_list:
        text_subbed = re.sub(pattern, replacement, text_subbed)

    # Ensure single whitespaces
    text_subbed = re.sub(r'\s+', ' ', text_subbed).strip()

    return text_subbed



def clean_sentence(sentence):
    # Remove punctuation using regex (only keeps relevant context)
    sentence = re.sub(r'[^\w\s]', '', sentence)
    # Remove extra whitespace at the beginning and end
    sentence = sentence.strip()
    return sentence




#-------------------------clean the sentences --> punctuation, whitespaces

def subs_sentence_lengths_filter(texts, metadata, substitutions, min_words = 5, max_words = 50):
    """
    Combined function to replace substitutions and filter sentences by sentence length for both metadata and text data
    """
    #Replace substitutions
    texts_subs = [replace_subs(text, substitutions) for text in tqdm(texts, desc = "Replace substitutions")]
    print(f"Substitutions replaced.\nNumber of documens: {len(texts_subs)}")

    #filter sentences based on length
    texts_sent_length = [(i, sent) for (i, sent) in tqdm(enumerate(texts_subs), desc = "Filtering sentences by word count") if min_words < len(sent.split()) < max_words]
    print(f"Number of relevant sentences after filtering by number of words (min: {min_words}; max: {max_words}): {len(texts_sent_length)}")
    print(f"Filtered sentences: {len(texts_subs) - len(texts_sent_length)}")

    sentences = [sent for i, sent in texts_sent_length] #sentences to keep after filtering for sentence length
    print(f"Number of sentences to keep: {len(sentences)}")
    
    indices = [i for i, sent in texts_sent_length] #indices to keep after filtering for sentence length --> corresponding to sentences
    print(f"Number of indices to keep: {len(indices)}")

    #clean whitespaces and punctuation
    sentences_final = [clean_sentence(sent) for sent in tqdm(sentences, desc = "Cleaning punctuation and whitespaces")]
    print(f"Number of documents: {len(sentences_final)}; Sentences cleaned (punctuation, whitespaces after sentence-length-based filter")

    #------------------Filter metadata
    indices_set = set(indices)  # Convert list to set for faster lookup

    metadata_final = [meta for i, meta in tqdm(enumerate(metadata), desc="Metadata filtering") if i in indices_set]

    print(f"Remaining metadata entries: {len(metadata_final)}")
    return sentences_final, metadata_final
    



#--------------------labeled dataset creation
"""
PAPER 2

COMPARED TO PAPER 1:
- coronavirus added to 13
- moved liabilit from cost/expenses to financing/debt --> balance-sheet related rather than P&L cost
- what about "product"?
- removed "capital" from liquidity/solvency --> more financing --> added "workingcapital" and added the substitution for "working capital" to "workingcapital" in subs --> do not add capital to financing/debt as it is a substring of workingcapital --> removed the bigrams regarding working capital
- removed "interest" from liquidity/solvency --> could be more related to financing/debt
- added solvency to liquidity and solvency topic as well as solvent and bankrupt and goingconcern
- added humancapital to HR
- changed incentive to incentiv --> capture incentivization
- added intellectualproperty to topic 7 - litigation, legal, intellectual property

FURTHER Paper 1 improvements (19.08.25)
- remove covid and energy topic
- removed liabilit from financing/debt
- removed "fund" and "share" --> too general
- added SG aspects to ESG
- removed market --> too general: e.g. stock market, etc. + always crossed out with marketing
- removed performance --> too general: ehs performance, esg performance, financial performance
- removed price --> too general: strike price, stock price, price of material, etc.
- removed coverage and bankrupt (bankruptcy code, bankruptcy court, etc.)
"""


topic_names = ["sales", "cost and expense", "profit and loss", "operations", "liquidity and solvency", "investment", "financing and debt", "litigation and intellectual property", "hr", "regulation and tax", "accounting", "environmental, social, governance (ESG)"]

keywords = [
    ["sale", "revenue", "consumer", "demand", "competition", "pricing"], 
    ["cost", "expense", "goodwill", "impairment", "depreciat"], #depreciate to cover up for depreciation and depreciate
    ["profit", "margin", "income", "earning", "loss", "ebit", "operatingresult", "financialresult", "yearendresult"], #ebitda also captured with ebit when exact_match = False
    ["operation", "production", "business", "produce", "supply", "supplier", "process", "manufacture", "manufacturing", "logistic", "transport", "marketing", "advertise", "advertising", "projectmanag", "inventorymanag", "safetymanag", "qualitycontrol"], #marketing/market overlap + supply covers supplychain
    ["liquidity", "solvency", "solvent", "goingconcern", "cash", "workingcapital", "bankbalance"], #cashflow already within cash
    ["expenditure", "m&a", "invest", "asset", "disposal", "divest"], #divestiture within divest
    ["financing", "finance", "debt", "equity", "dividend", "repurchase", "securities", "borrow", "credit", "stockperformance"], #
    ["litigation", "lawsuit", "legal", "dispute", "complaint", "arbitration", "patent", "intellectualproperty"],
    ["employee", "retention", "hiring", "hire", "union", "consultant", "staff", "recruit", "labor", "incentiv", "training", "salary", "wage", "job", "humancapital", "talentmanag", "successionmanag", "hrmanag", "nextgenerationmanag", "talentdevelop", "feedbackcultur", "workremote", "remotework", "worklife", "worklifebalance", "leadership", "careerdevelop", "employment", "employing", "ehsperformance", "genderaffirm"], #insurance removed
    ["regulation", "tax", "government", "legislation", "federal", "regulator", "regulate"], #regulator includes regulatory
    ["account", "audit", "adjustment", "filing", "internalcontrol", "controlframework", "controlstruct", "controlrisk"], #account includes accounting, accounts payable/receivable, etc --> remove "auditor" as it is already in "audit"
    ["plastic", "recycl", "waste", "carbon", "emission", "renewable", "environment", "sustain", "ecologic", "waterrisk", "waterfootprint", "diversity", "inclusion", "ethics", "ethical", "esgperformance"], #environment, "recycle" changed to "recycl" to capture "recycling", etc.
]


label_to_keywords = {i: words for i, words in enumerate(keywords)}


#---------keyword blacklist
"""
COMPARED TO PAPER 1 - VERSION:
- added reliability and reliabilities to the keyword blacklist --> confusion with "liabilit" --> documentation overleaf 18.03.25
- other words (pretax, aftertax) --> see documentation in Overleaf from 21.03.25
- several word replacements due to the LLM check for sense-making of labels for labeled dataset --> see documentation in Overleaf from 21.03.25 + 23.03. + 24.03. + 25.03. + 26.03. + 27.03.
- refinement regarding substring approach
"""


"""
Removed some blackstringed keywords for water, contract and management
19.08.25
removed blacklisted words regarding to energy + share
added investor
"""
keyword_blacklist_substring = [
    "wholesal", "recoupmentdemand", "optionpric", "consumerprivacylaw", "currencypric", "stockmarket", "stockpric", "capitalmarket", "canadarevenueagency", "puertoricodepartmentofconsumeraffair", "revenuebond", "equitiesmarket", "equitymarket", "investigativedemand", "missilecompetition", "consumergoods", "depositrepricing", "internalrevenuecode", "competitionlaw",  "internalrevenueservice",  "irishrevenue", "salem", "competitionauthorit", "pricingservice", "transferpricing", "exchangemarket", "housingmarket", "wholefoodsmarket", "presuit demand", "pleaddemand", "pricingquote", "consumerprotection", "demandregistrationright", "demandrepayment", "demandletter", "consumerclassaction", "noninterestbearingdemand", "equityindexprice", "noncompetition", "salesforce", "ondemand", "competitionbureau", "competitiontribunal",   #sales (sale, demand, consumer)
    "costar", "costa", "benefitcost", "benefitexpense", "costigan", "trucost", "costum", "costco", #costs and expenses (cost, sale)
    "marginal", "learning", "elearning", "lowincome", "nonprofit", "glossar", "medianincome", "householdincome", "averageincome", "gloss", "highincome", "middleincome", "debitcard", "debitnetwork", "visionloss", "debit", "moderateincome", "hersheyincomeaccelerator", "farmerincome", "realtyincomecorporation", "losscontrolservice", "specialistlosscontrol", "inventoryloss", "floss", "nutrientloss", "lossdegradation", "fixedincomeclearingcorporation", "careeraverageearningsformula", "finalaverageearningsformula", "interestearning", "profitec", "fixedincome", "derivativegainsloss",  #profit and loss (margin, earning, income, profit)
    "remanufactur", "reprocess", "schoolofbusiness", "businessday", "administrativeappealsprocess", "cooperation", #operations
    "cashion", "solventborne", "cashrewardcard", "cashback", #liquidity
    "investigat", "liquidassetbuffer", "investorrelation", "investor", "pharmasset", "multiassetclass", "crossasset", #investment (invest) 
    "cybersecurit", "creditcard", "mileagecredit", "capitalcit", "capitalgrill", "capitalburger", "diversityequity", "accredit", "equityindexproduct", "equityresearch", "equityindex", "equityfund", "equitysecurit", "securitiesact", "securitiesexchange", "equityportfolio", "domesticequit", "foreignequit",  "securitiesclassaction", "credited", "webequity", #financing and debt (security/securities, credit, capital, equity)
    "customercomplaint", "legalentit",    #litigation
    "westernunion", "riskretention", "clientretention", "walletretention","dogtraining", "unioncarbide", "europeanunion", "flagstaff", "steam", "laborator", "sewage", "wilshire", "hampshire", "trainspeed", "trainvelocity", "worktrain", "passengertrain", "collaborat", "hillshire", "transmissionincentiv", "selfinsuredretention", "dogtrackingandtrainingdevice", "sportsleaguesteam", "sportsteam", "clientincentiv", "shire", "unionpacific", "climateleadership", "treatyretention", "petroleumconsultant", "unionsecurity", "fiscalincentive", "secenforcementstaff", "fercstaff", "wager", "volkswagen", "monetaryunion", "laborsystem",  #HR and employment (work, insurance, staff, team, labor, wage, hire, union)
    "taxonom", "taxi", "nabpaclitaxel", "aftertax", "pretax", "governmentbond", "pentax", "regulateddiscoveryservice", "regulatedsafetyassessment", "tribalgovernment", "regulatedwater", "marketaxess", "docetaxel", "taxus", "taxol", "paclitaxel",  #regulation and tax (tax, coal)
    "additionsadjustment", "efiling", "submerchantaccount", "midaccount", "facebookaccount", "twitteraccount", "instagramaccount", "periscopeaccount", "accounttakeover", "socialmediaaccount", "emailaccount", "microsoftofficeaccount", "berkshire", "designaccount", "nationalaccountcontract", "accountguard", "fareaudit", "brokerageaccount", "accountcompromise", "accountfeed", "indexaccountoption", "keyaccount", "customeraccount", "onboardingaccount", "documenfilingsolution", "workplaceadjustment", "checkingaccount", "depositaccount", "airsystemaudit", "documentfilingsolution", "promotedaccount", "logoaccount", "profiling", #accounting (adjustment, filing)
 "cloudenvironment", "cloudenvironments", "hydrocarbon", "thermoplastic", "omniplastic", "economicenvironment", "interestrateenvironment", "computingenvironment", "containerenvironment", "localenvironment", "sandboxenvironment", "sterileenvironment", "presidentpackagingspecialtyplastics", "wasteservicescompany", "wastemanagementinc", "wastemanagementholdingsinc", "guestenvironment", "controlenvironment", "lendingenvironment", "competitiveenvironment", "operatingenvironment", "workenvironment", "softwareenvironment", "carbonated", "remission", "gynecolog", "customerenvironment", "discountrateenvironment", "clouditenvironment", "alliedwaste", "bicarbonate", "sesquicarbonate", "carbonate", "nanyaplastic", "financialinclusion", "lifesustaining" #ecologic sustainability (wind)
    #covid pandemic
]

label_to_blacklist = {
    0: ["wholesal", "recoupmentdemand", "optionpric", "consumerprivacylaw", "currencypric", "stockmarket", "stockpric", "capitalmarket", "canadarevenueagency", "puertoricodepartmentofconsumeraffair", "revenuebond", "equitiesmarket", "equitymarket", "investigativedemand", "missilecompetition", "consumergoods", "depositrepricing", "internalrevenuecode", "competitionlaw",  "internalrevenueservice",  "irishrevenue", "salem", "competitionauthorit", "pricingservice", "transferpricing", "exchangemarket", "housingmarket", "wholefoodsmarket", "presuit demand", "pleaddemand", "pricingquote", "consumerprotection", "demandregistrationright", "demandrepayment", "demandletter", "consumerclassaction", "noninterestbearingdemand", "equityindexprice", "noncompetition", "salesforce", "ondemand", "competitionbureau", "competitiontribunal"],
    1: ["costar", "costa", "benefitcost", "benefitexpense", "costigan", "trucost", "costum", "costco"],
    2: ["marginal", "learning", "elearning", "lowincome", "nonprofit", "glossar", "medianincome", "householdincome", "averageincome", "gloss", "highincome", "middleincome", "debitcard", "debitnetwork", "visionloss", "debit", "moderateincome", "hersheyincomeaccelerator", "farmerincome", "realtyincomecorporation", "losscontrolservice", "specialistlosscontrol", "inventoryloss", "floss", "nutrientloss", "lossdegradation", "fixedincomeclearingcorporation", "careeraverageearningsformula", "finalaverageearningsformula", "interestearning", "profitec", "fixedincome", "derivativegainsloss"],
    3: ["remanufactur", "reprocess", "schoolofbusiness", "businessday", "administrativeappealsprocess", "cooperation"],
    4: ["cashion", "solventborne", "cashrewardcard", "cashback"],
    5: ["investigat", "liquidassetbuffer", "investorrelation", "investor", "pharmasset", "multiassetclass", "crossasset"],
    6: ["cybersecurit", "creditcard", "mileagecredit", "capitalcit", "capitalgrill", "capitalburger", "diversityequity", "accredit", "equityindexproduct", "equityresearch", "equityindex", "equityfund", "equitysecurit", "securitiesact", "securitiesexchange", "equityportfolio", "domesticequit", "foreignequit",  "securitiesclassaction", "credited", "webequity"],
    7: ["customercomplaint", "legalentit"],
    8: ["westernunion", "riskretention", "clientretention", "walletretention","dogtraining", "unioncarbide", "europeanunion", "flagstaff", "steam", "laborator", "sewage", "wilshire", "hampshire", "trainspeed", "trainvelocity", "worktrain", "passengertrain", "collaborat", "hillshire", "transmissionincentiv", "selfinsuredretention", "dogtrackingandtrainingdevice", "sportsleaguesteam", "sportsteam", "clientincentiv", "shire", "unionpacific", "climateleadership", "treatyretention", "petroleumconsultant", "unionsecurity", "fiscalincentive", "secenforcementstaff", "fercstaff", "wager", "volkswagen", "monetaryunion", "laborsystem"],
    9: ["taxonom", "taxi", "nabpaclitaxel", "aftertax", "pretax", "governmentbond", "pentax", "regulateddiscoveryservice", "regulatedsafetyassessment", "tribalgovernment", "regulatedwater", "marketaxess", "docetaxel", "taxus", "taxol", "paclitaxel"],
    10: ["additionsadjustment", "efiling", "submerchantaccount", "midaccount", "facebookaccount", "twitteraccount", "instagramaccount", "periscopeaccount", "accounttakeover", "socialmediaaccount", "emailaccount", "microsoftofficeaccount", "berkshire", "designaccount", "nationalaccountcontract", "accountguard", "fareaudit", "brokerageaccount", "accountcompromise", "accountfeed", "indexaccountoption", "keyaccount", "customeraccount", "onboardingaccount", "documenfilingsolution", "workplaceadjustment", "checkingaccount", "depositaccount", "airsystemaudit", "documentfilingsolution", "promotedaccount", "logoaccount", "profiling"],
    11: ["cloudenvironment", "cloudenvironments", "hydrocarbon", "thermoplastic", "omniplastic", "economicenvironment", "interestrateenvironment", "computingenvironment", "containerenvironment", "localenvironment", "sandboxenvironment", "sterileenvironment", "presidentpackagingspecialtyplastics", "wasteservicescompany", "wastemanagementinc", "wastemanagementholdingsinc", "guestenvironment", "controlenvironment", "lendingenvironment", "competitiveenvironment", "operatingenvironment", "workenvironment", "softwareenvironment", "carbonated", "remission", "gynecolog", "customerenvironment", "discountrateenvironment", "clouditenvironment", "alliedwaste", "bicarbonate", "sesquicarbonate", "carbonate", "nanyaplastic", "financialinclusion", "lifesustaining"]
}



# Exclusion words per topic
exclusion_dict = {
    0: ["stock", "fair value", "financial instrument", "derivative", "public offering", "net interest", "interest rate", "interestrate", "monte carlo simulation", "black scholes", "blackscholes", "hedging", "swaption", "senior note", "foreign exchange", " libor ", "multiples method", "unpaid interest", "accrued interest", "debenture", "futurescontract", "loan", "ownership interest", "partnership interest", "joint venture interest", "pension benefit", "municipal bond", "hedge", "capital deployment", "trial management", "commercial paper", "deposit rate", "gender", "ethnic", "benefit pension plan", "congressional committee", "canadarevenueagency", "valuation multiple", "personal information", "senior note", "floor trading", "preferred equit", " bonds ", " bond ", "measurement of inventory", "holder of the note", "issueprice of the note", "trading portfolio", "governmentbond", "infection", "confidentiality obligation", "privacy right", "consumer financial product", " collar ", " collars ", "delaware corporation law", "equitysecurit", "principal amount", "plaintiff", "fixed maturit", "senior subordinated note", "trasury bill", "optioncontract", "implied volatility", "implied market volatility", "var calculation", "treasury", "counterswap", "removal action plan", "statute", "level valuation"],  # Sales 
    1: ["equity market", "gender", "ethnic", "trading portfolio", "governmentbond", "lipper average", "lipper quartile"] , # cost and expenses
    2: ["ethnically diverse", "motor neurons", "brain stem", "muscular", "medicine", "vaccine", "cellular", "natural habitat", "endangered species", "reforestation", "biodiversity", "safety initiative", "safety council", "safety protocol", "women of color", "people of color", "municipal bond", "beneficial owner", "security owner", "drilling program", "human right", "mortality", "mutation", "diversityequity", "hiv treatment", "biodivers", "gender", "ethnic", "research result", "clinical trial management", "welness and benefit program", "monsoon", "tornado", "hurricane", "human life", "destruction", "cocoaproducing", "tradingportfolio", "governmentbond", "lipper average", "lipper quartile", "noncompliance notice", "trading community"],  # profit and loss
    3: ["gender", "ethnic", "senior note", "payment of interest", "trading portfolio", "governmentbond", "lipper average", "lipper quartile"], #operations
    4: ["body function", "fertility", "fertile", "gender", "ethnic", "stockholder", "trading portfolio", "governmentbond", "benefit pension", "qualified plan", "earn benefit", "lipper average", "lipper quartile", "pension formula", "charitable giving", "west africa", "room nights booked", "mobile app booking", "polymer", "viscosity"], #liquidity/solvency
    5: ["impaired", "impairment", "trading portfolio", "governmentbond", "liaba ilities", "lipper average", "lipper quartile", "pension plan"], #investment
    6: ["pay equity ratio", "inclusion", "women of color", "people of color", "ethnicity", "ethnical", "gender", "miniwheats", "honey smacks", "fishing map", "backlog", "miller lite", "trading portfolio", "governmentbond", "lipper average", "lipper quartile", "natural gas distribution", "derivative petition", "nominal defendant", "plaintiff"], #financing/debt
    7: ["parental leave"], #litigation+
    8: ["glucose", "insulin", "medicine", "patient", "disease", "interest rate", "muscular", "cellular", "vaccine", "blood", "hemostatic", "sterile", "black communit", "cocoa", "fercapproved", "adobe", "photoshop", "libor", "neural network", "transmissionincentive", "housing program", "trading portfolio", "governmentbond", "biodivers", " bond ", " bonds ", "company bond", "endoflife lifecycle", "rto membership", "megabit technology"], #HR
    9: ["glucose", "insulin", "medicine", "patient", "disease", "interest rate", "muscular", "cellular", "vaccine", "blood", "hemostatic", "sterile", "black communit", "biodivers"], #Regulation and tax
    10: ["glucose", "insulin", "medicine", "patient", "disease", "interest rate", "muscular", "cellular", "vaccine", "blood", "hemostatic", "women of color", "people of color", "gender", "ethnicity", "ethnical", "black communit", "concierge", "equipment access management", "wan northstar", "temperature point", "john deere", "wan paragon", "loan portfolio", "natural disaster", "gas transmiss", "gas distribut", "structured variable annuity", "macao gaming", "cable management", "inflight entertainment", "centralized climate system", "drug research", "fiber procure", "postpaid customers per account", "subscriptionbased service", "honey platform"], #accounting
    11: ["interest rate", "interestrate", "payment history", "injury", "injuries", "creditcard", "sterile", "client service", "client activity", "radar installation", "missile warning", "sustains relationship", "cloudbased", "nan ya plastic", "hurricane katrina", "inventory not owned"], #ecologic sustainability
}


