#---------------Scope of financial topics
scope_topics = {
    # **Income Statement (Profit & Loss Statement) Topics**
    "sales": "Part of the Income Statement. Covers revenue generation, market trends, consumer demand, pricing strategies, competitive positioning, and contractual agreements related to sales transactions.  **IMPORTANT**: Aspects related to stock and share prices, trading of shares and stocks, stock-based compensation, financial instruments, derivative contracts, and hedging strategies are excluded, as these fall outside the scope of direct sales activities.",
    "costs and expenses": "Part of the Income Statement. Focuses on operating costs related to day-to-day operations, such as cost of goods sold, depreciation, and impairments. **This includes goodwill impairments**, which are recognized as a non-cash expense that reduces the carrying value of goodwill.",
    "profit and loss": "Part of the Income Statement. Examines financial performance through profitability metrics, earnings, margins, net income, and key indicators such as EBIT and EBITDA.",
    #operations
    "operations": "Encompasses core business activities, including production, supply chain management, logistics, and marketing. **Marketing includes strategic positioning, branding, and advertising (such as promotional campaigns and media outreach).** This is just a selection of marketing tasks within a broader operational context.",
    #balance sheet and Cash Flow Statement
    "liquidity and solvency": "Part of the Balance Sheet and Cash Flow Statement. Evaluates a company’s ability to meet financial obligations and generate cash, covering cash flow management, liquidity ratios, and working capital. **Excludes equity structure and ownership interests.**",
    "investment": "Part of the Balance Sheet and Cash Flow Statement. Covers capital allocation, mergers and acquisitions, capital expenditures,  asset purchases, divestitures, and strategic investments.",
    "financing and debt": "Part of the Balance Sheet. Covers capital structure, long- and short-term debt, financing activities, debt management, corporate borrowing activities, issuance and repurchase of shares and equity, as well as **dividends and distribution policies** and funding sources.",
#other topics
    "litigation and intellectual property": "Covers two distinct but related areas: Litigation and Intellectual Property. A sentence belongs to this topic if it discusses at least one of these aspects. Litigation includes legal disputes, lawsuits, government investigations, enforcement actions and arbitration. Intellectual Property covers patents, trademarks, copyrights, trade secrets, and legal protections related to intellectual property.",
    "human resources and employment": "Covers workforce management, hiring, compensation, benefits, labor relations, employee incentives, and workplace policies.",
    "regulation and tax": "Encompasses two distinct but related areas: regulatory compliance and corporate taxation. A sentence belongs to this topic if it discusses at least one of these aspects. Regulatory compliance includes corporate laws, industry standards, sector-specific requirements, and government-imposed policies. Taxation covers corporate taxation, fiscal policies and compliance with local, state, and international tax laws.",
    "accounting": "Covers financial reporting, audits, accounting adjustments, internal controls, and regulatory financial disclosures.",
    "environmental, social, governance (ESG)": "Encompasses environmental impact (e.g., emissions, energy use, sustainability), social factors (e.g., diversity, inclusion, community engagement, human rights), and governance issues (e.g., ethics, compliance, board oversight, transparency)."
}


system_prompt = (
    "You are an expert in financial text classification and NLP.\n"
    "Your task is to analyze a given sentence from a 10-K report, specifically from the perspective of the company that issued the report. Analyze the given text and classify it into **one financial topic** from the topic taxonomy. If no topic fits, return 'None'.\n"
    "Focus on the semantic meaning and context of the sentence, not just keyword matching.\n\n"

    "#### Topic Taxonomy\n"
    "Use the topic taxonomy and the scope of each topic to guide your decision.\n"
    + "\n".join([f"- **{topic}**: {scope}" for topic, scope in scope_topics.items()])

    + "\n\n**IMPORTANT RULES**:\n"
    "##### Multi-Area Topic Assignment Rule:\n"
    "- If a topic includes multiple distinct subtopics (e.g., 'regulation / tax' or 'litigation / legal / intellectual property'), a sentence is correctly assigned as long as it relates to **at least one** of the subtopics. **It does not need to cover all aspects.**\n"
    "- Examples:\n"
      "- A sentence about patents relates to 'intellectual property' and is therefore correctly assigned to 'litigation / legal / intellectual property', even if there is no mention of 'litigation' or 'legal'.\n"
      "- A sentence about corporate tax belongs to 'regulation / tax' even if it does not discuss regulations explicitly.\n"
    
    "\n##### General Guidelines:\n"
    "- Ensure that the topic assignment reflects the perspective **of the company itself, not third parties** mentioned in the sentence.\n"
    "- You must assign exactly one topic from the list below. If a sentence could fit multiple topics, choose the one most directly and substantively discussed. If no financial topic matches, return 'None' as the assigned topic. Do **NOT** invent, modify, or generalize topic names.\n"
    "- Keep the explanation **general** and **concise** (max 2 sentences).\n"
    "- Additionally, estimate the **confidence** in the sentence being correctly assigned to the topic, and provide a percentage probability (0-100%).\n"
    
    "\n##### Response Format (Follow exactly):\n"
    "1. Topic: [One topic from the taxonomy OR 'None']\n"
    "2. Explanation: [One or two sentences explaining the reasoning]\n"
    "3. Probability: [Percentage 0–100]\n"
)


system_prompt_hf = (
    "You are an expert in financial text classification and NLP."
    "Your task is to analyze a given sentence from a 10-K report, specifically from the perspective of the company that issued the report. Analyze the given text and classify it into **one financial topic** from the topic taxonomy. If no topic fits, return 'None'.\n"
    "Focus on the semantic meaning and context of the sentence, not just keyword matching.\n\n"

    "#### Topic Taxonomy\n"
    "Use the topic taxonomy and the scope of each topic to guide your decision. Each topic is described as follows:\n\n"
    + "\n".join([f"- **{topic}**: {scope}\n" for topic, scope in scope_topics.items()])

    + "\n\n**IMPORTANT RULES**:\n"
    "##### Multi-Area Topic Assignment Rule:\n"
    "- If a topic includes multiple distinct subtopics (e.g., 'regulation / tax' or 'litigation / legal / intellectual property'), a sentence is correctly assigned as long as it relates to **at least one** of the subtopics. **It does not need to cover all aspects.**\n"
    "- Examples:\n"
      "- A sentence about patents relates to 'intellectual property' and is therefore correctly assigned to 'litigation / legal / intellectual property', even if there is no mention of 'litigation' or 'legal'.\n"
      "- A sentence about corporate tax belongs to 'regulation / tax' even if it does not discuss regulations explicitly.\n"
    
    "\n##### General Guidelines:\n"
    "- Ensure that the topic assignment reflects the perspective **of the company itself, not third parties** mentioned in the sentence.\n"
    "- You must assign exactly one topic from the list below. If a sentence could fit multiple topics, choose the one most directly and substantively discussed. If no financial topic matches, return 'None' as the assigned topic. Do **NOT** invent, modify, or generalize topic names,\n"
    "- Keep the explanation **general** and **concise** (max 2 sentences).\n"
    "- Additionally, estimate the **confidence** in the sentence being correctly assigned to the topic, and provide a percentage probability (0-100%).\n"
    
    "\n##### Response Format (Follow exactly):\n"
    "You will respond strictly in the following JSON format:\n"
    "{\n"
    "  \"topic\": \"<one topic from taxonomy or \\\"None\\\">\",\n"
    "  \"explanation\": \"<one or two sentences explaining the reasoning>\",\n"
    "  \"probability\": <integer between 0 and 100>\n"
    "}"
)


system_prompt_hf_llm = (
    "You are an expert in financial text classification and NLP."
    "Your task is to analyze a given sentence from a 10-K report, specifically from the perspective of the company that issued the report. Analyze the given text and classify it into **one financial topic** from the topic taxonomy. If no topic fits, return 'None'.\n"
    "Focus on the semantic meaning and context of the sentence, not just keyword matching.\n\n"

    "#### Topic Taxonomy\n"
    "Use the topic taxonomy and the scope of each topic to guide your decision. Each topic is described as follows:\n\n"
    + "\n".join([f"- **{topic}**: {scope}\n" for topic, scope in scope_topics.items()])

    + "\n\n**IMPORTANT RULES**:\n"
    "##### Multi-Area Topic Assignment Rule:\n"
    "- If a topic includes multiple distinct subtopics (e.g., 'regulation / tax' or 'litigation / legal / intellectual property'), a sentence is correctly assigned as long as it relates to **at least one** of the subtopics. **It does not need to cover all aspects.**\n"
    "- Examples:\n"
      "- A sentence about patents relates to 'intellectual property' and is therefore correctly assigned to 'litigation / legal / intellectual property', even if there is no mention of 'litigation' or 'legal'.\n"
      "- A sentence about corporate tax belongs to 'regulation / tax' even if it does not discuss regulations explicitly.\n"
    
    "\n##### General Guidelines:\n"
    "- Ensure that the topic assignment reflects the perspective **of the company itself, not third parties** mentioned in the sentence.\n"
    "- You must assign exactly one topic from the list below. If a sentence could fit multiple topics, choose the one most directly and substantively discussed. If no financial topic matches, return 'None' as the assigned topic. Do **NOT** invent, modify, or generalize topic names,\n"
)



import pandas as pd
import numpy as np

def transform_rag_output(indices, rag_answers, topic_names, prob_threshold=85):
    """
    Function to extract RAG answers and transform the topic assignments.
        
    Parameters:
    - indices_to_check: list of indices in the dataset to check via RAG
    - rag_answers: list of LLMResponse objects with .topic, .explanation, .probability attributes
    - prob_threshold: float, minimum confidence for accepting a RAG topic
        
    Returns:
    - indices_rag: np.array, updated topic indices (with -1 for noise)
    - noise_idx: list of indices in `indices_to_check` identified as noise
    """

    # Extract topic, explanation, probability from RAG output
    extracted_topics_rag = rag_answers["topic"]
    extracted_probs_rag = rag_answers["probability"]
    indices_to_check = rag_answers["sentence_index"]
    
    # Print stats
    in_taxonomy = [idx for idx, topic in zip(indices_to_check, extracted_topics_rag) if topic in topic_names]
    out_of_taxonomy = [idx for idx, topic in zip(indices_to_check, extracted_topics_rag) if topic not in topic_names]
    low_prob = [idx for idx, prob in zip(indices_to_check, extracted_probs_rag) if prob < prob_threshold]
    
    print(f"Number of sentences with valid topics: {len(in_taxonomy)}")
    print(f"Share of valid topics: {len(in_taxonomy) / len(extracted_topics_rag) * 100:.2f}%")
    print(f"Number of noise points (not in taxonomy): {len(out_of_taxonomy)}")
    print(f"Share of noise points (not in taxonomy): {len(out_of_taxonomy) / len(extracted_topics_rag) * 100:.2f}%")
    print(f"Number of noise points (below prob_threshold): {len(low_prob)}")
    print(f"Share of noise points (below prob_threshold): {len(low_prob) / len(extracted_probs_rag) * 100:.2f}%")
    
    # Create topic-to-index mapping
    topic_dict = {topic: i for i, topic in enumerate(topic_names)}
    
    # Assign topic indices or -1 for noise
    rag_int_labels = [
        topic_dict.get(topic, -1) if (prob >= prob_threshold and topic in topic_dict) else -1
        for topic, prob in zip(extracted_topics_rag, extracted_probs_rag)
    ]
    
    # Track noise indices for debugging or filtering
    noise_idx = [idx for idx, label in zip(indices_to_check, rag_int_labels) if label == -1]
    print(f"Number of noise idx (taxonomy + probability): {len(noise_idx)}")
    
    # Get initial topic assignment (closest neighbor)
    indices_rag = indices[:, 0].copy()
    
    # Apply RAG label updates only to selected indices
    indices_rag[indices_to_check] = rag_int_labels
    
    print(f"\n\nNumber of indices (total): {len(indices_rag)}")
    print(f"Number of noise datapoints (total): {np.sum(np.array(indices_rag) == -1)}")
    print(f"Share of noise datapoints (total): {np.sum(np.array(indices_rag) == -1) / len(indices_rag) * 100:.2f}%")
    print(f"Total coverage: {100 - (np.sum(np.array(indices_rag) == -1) / len(indices_rag)) * 100:.1f}%")
    
    return indices_rag, noise_idx