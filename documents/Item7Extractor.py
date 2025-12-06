import pandas as pd
import pickle
import numpy as np
import os
from tqdm import tqdm
import re



class ItemExtractor:
    """
    Extract text of Item 7 from 10-K report
    """
    def __init__(self, 
                 matches_pattern:str,
                 file_loc:str = "paper2/Data/Text/10-K",
                 output_dir:str = "paper2/Data/text/10-K/item7/item7_text.pkl"
                ):
        
        self.matches_pattern = matches_pattern
        self.file_loc = file_loc
        self.output_dir = output_dir

    def get_file_locs(self):
        """
        Generate the file_locs to iterate over
        """
        file_locs = os.listdir(self.file_loc) #get the filenames
        file_locs = [loc for loc in file_locs if loc.endswith(".pkl")] #focus only on pkl files
        file_locs = [f"{self.file_loc}/{file}" for file in file_locs] #concatenate with folder structure
        return file_locs

    def load_text_meta(self, file_loc):
        """
        Load text and metadata from file loc
        """
        with open(file_loc, "rb") as file:
            result_dict = pickle.load(file)
            
        docs = result_dict["docs"]
        metadata = result_dict["metadata"]
        return docs, metadata

    def generate_matches_df(self, text):
        """
        Get the location of patterns occurring in the text
        """
        try:
            matches = re.compile(self.matches_pattern, re.IGNORECASE) #compile regex pettern
            match_list = [(match.group(), match.start()) for match in matches.finditer(text)] #find patterns in the text
            
            if not match_list:
                print("No matches found.")
                return pd.DataFrame(columns=["SearchTerm", "Start", "Selection"])
            
            matches_array = pd.DataFrame(match_list, columns=["SearchTerm", "Start"]) #store in dataframe
            matches_array["Selection"] = ""  #use this for tagging later
            return matches_array
        except Exception as e:
            print(f"Error: {e}")
            return None
    
    def find_real_item7_bounds(self, matches_df, text):
        """
        Identify the real bounds of item 7
        We got the location of parts, items and the text of items
        Item 7 is in Part II
        --> the real Item 7 should be in the last appeareance of part 2 where Item 7 and Item 8 still follow --> prevent that references in part 3 to part 2 and items 7 and 8 are used
        """
        # Get all relevant tag positions
        part_ii_locs = matches_df[matches_df["SearchTerm"].str.contains("part ii", case=False)]["Start"].tolist()
        part_iii_locs = matches_df[matches_df["SearchTerm"].str.contains("part iii", case=False)]["Start"].tolist()
        item7_locs = matches_df[matches_df["SearchTerm"].str.contains(r'items?\s+7', case=False, regex=True)]["Start"].tolist() #handle cases where Item 7 and 7A are combined --> Items 7 and 7A
        item8_locs = matches_df[matches_df["SearchTerm"].str.contains("item 8", case=False)]["Start"].tolist()
    
        # Sort all positions just to be safe
        part_ii_locs.sort()
        part_iii_locs.sort()
        item7_locs.sort()
        item8_locs.sort()
    
        for part_ii in reversed(part_ii_locs): # reversed to get the latest appearances --> last time part ii is mentioned while item 7 and item 8 have to appear between the next part iii
            # Look for first Item 7 after this PART II
            item7_candidates = [loc for loc in item7_locs if loc > part_ii]
            if not item7_candidates:
                continue
            item7 = item7_candidates[0] #take first element --> first time Item 7 is referenced in part ii
    
            # Look for PART III after this Item 7 --> next part iii following this item 7
            part_iii_candidates = [loc for loc in part_iii_locs if loc > item7]
            if not part_iii_candidates:
                continue
            part_iii = part_iii_candidates[0]
    
            # Find the last Item 8 between Item 7 and PART III
            item8_candidates = [loc for loc in item8_locs if item7 < loc < part_iii]
            if not item8_candidates:
                continue
            item8 = item8_candidates[-1]  # Use last valid Item 8 --> last time Item 8 is referenced in part ii
    
            item7_text = text[item7:item8]
            return item7_text
    
        # No valid pattern found
        return None
    
    
    def split_mda(self,
                  text
                  ):
        """
        wrapper method to find and isolate Item 7 (MD&A) + Item 7A (Market Risk) in stage one parsed 10-K reports
        extract text between item 7 and item 8 --> item7 search text and item8 search text
        """
        matches_df = self.generate_matches_df(text)
        item7_text = self.find_real_item7_bounds(matches_df, text)
        
        return item7_text

    def save_results(self, texts, metadata):
        """
        Save results to output directory
        """
        result_dict = {
            "item7_texts": texts,
            "item7_metadata": metadata
        }

        with open(self.output_dir, "wb") as file:
            pickle.dump(result_dict, file)

        print(f"Texts and metadata saved to {self.output_dir}")

    def run(self):
        """
        Run the splitting process of Item 7
        """
        #initialize lists to store results
        item7_texts = []
        item7_metadata = []

        #get file locs of the folders
        file_locs = self.get_file_locs()

        #iterate over the folders
        for file_loc in tqdm(file_locs, desc = "Folder progress"):
            docs, metadata = self.load_text_meta(file_loc) #load documents and metadata for that folder

            #split each document --> extract Item 7
            split_text = [self.split_mda(doc) for doc in tqdm(docs, desc = "Progress within Folder")]
            print(f"Number of None: {len([x for x in split_text if x is None])}")
            print(f"None docs: {[i for i, doc in enumerate(split_text) if doc is None]}")
        
            # Collect text + metadata only if text is not None
            for (text, meta) in zip(split_text, metadata):
                item7_texts.append(text)
                item7_metadata.append(meta)

        self.save_results(item7_texts, item7_metadata)