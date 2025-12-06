import pandas as pd
import numpy as np


class TokenizerCleaner:
    def __init__(self,
                 texts
                 substitutions
                ):

    def replace_subs(self, text):
        """
        Replace words to make the text more comparable and avoid unintended keyword matches (words + re pattern) - before tokenizing
        """
        text = text.lower() #convert to lower case

        #initialize subbed text
        text_subbed = text
        for pattern, replacement in substitutions_list:
            text_subbed = re.sub(pattern, replacement, text_subbed)
    
        # Ensure single whitespaces
        text_subbed = re.sub(r'\s+', ' ', text_subbed).strip()
    
        return text_subbed