import pandas as pd
import pickle
from tqdm import tqdm
import os
import re
import zipfile


class DocLoader:
    """
    Load the documents from zip folders
    Extract text and metadata (full report)
    """
    def __init__(self, 
                 zip_path:str = "10-K", #directory where the zip folders are stored
                 keyword:str ="_10-K_", #define the document we want to consider - e.g. _10-Q_, _10-K-A_, _"10-K_", "_10-Q-A_"
                 output_dir = "paper2/Data/10-K/", #directory where to save the docs and metadata
                 rel_cik_list:list = None, #list of relevant ciks for which we want to extract the reports
                 ):

        self.zip_path = zip_path
        self.rel_cik_list = rel_cik_list
        self.keyword = keyword
        self.output_dir = output_dir

        self.seen_accession_numbers = set()
        self.duplicate_accession_count = 0

        if rel_cik_list is not None:
            print(f"Apply CIK-based filtering: {len(rel_cik_list)} CIK's.")
        else:
            print("No CIK-based filtering")


    def get_zip_paths(self):
        """
        Get the zip paths from the zip directory --> iterate over them later to extract the relevant documents
        """
        zip_paths = os.listdir(self.zip_path) #list all files, folders, etc. in the directory
        zip_paths = [file for file in zip_paths if file.endswith(".zip")] #filter for files which end with ".zip"

        zip_paths = [f"{self.zip_path}/{path}" for path in zip_paths]

        return zip_paths


    def extract_metadata(self, text, regex_pattern):
        """
        Extract metadata from the 10-X documents
        """
        pattern = re.compile(regex_pattern)
    
        #Search for the pattern in the text
        match = re.search(pattern, text)
        
        if match:
            # Extract the matched value and store it in a variable
            result = match.group(1)
            return result
        else:
            # Handle the case when the pattern is not found
            return None
    
    
    def get_metadata(self, text):
        """
        Combined method to extract all relevant metadata from a document.
        """
        period_of_report = self.extract_metadata(text, r'CONFORMED PERIOD OF REPORT:[ \t\n\r\f\v]*(\d+)')
        try:
            year = period_of_report[:4]
        except:
            year = None #assign None if year cannot be extracted

        accession_number = self.extract_metadata(text, r'ACCESSION NUMBER:[ \t\n\r\f\v]*(\S+)')
        filing_date = self.extract_metadata(text, r'FILED AS OF DATE:[ \t\n\r\f\v]*(\d+)')
        change_date = self.extract_metadata(text, r'DATE AS OF CHANGE:[ \t\n\r\f\v]*(\d+)')
        cik = self.extract_metadata(text, r'CENTRAL INDEX KEY:[ \t\n\r\f\v]*(\d+)')
        name = self.extract_metadata(text, r'COMPANY CONFORMED NAME:[ \t\n\r\f\v]*([^\n]+)')
        former_name = self.extract_metadata(text, r'FORMER CONFORMED NAME:[ \t\n\r\f\v]*([^\n]+)')
        date_name_change = self.extract_metadata(text, r'DATE OF NAME CHANGE:[ \t\n\r\f\v]*([^\n]+)')
        submission_type = self.extract_metadata(text, r'CONFORMED SUBMISSION TYPE:[ \t\n\r\f\v]*([^\n]+)')

    
        metadata = {
            "period_of_report": period_of_report,
            "year_of_report": year,
            "accession_number": accession_number,
            "cik": cik,
            "company_name": name,
            "filing_date": filing_date,
            "change_date": change_date,
            "submission_type": submission_type,
            "former_company_name": former_name,
            "date_name_change": date_name_change
        }
    
        return metadata

    def save_results(self, docs, metadata, zip_filename):
        base_name = os.path.splitext(os.path.basename(zip_filename))[0] #
        save_path = os.path.join(self.output_dir, f"{base_name}.pkl")

        with open(save_path, "wb") as file:
            pickle.dump({"docs": docs, "metadata": metadata}, file)

        print(f"Saved to {save_path}")
        
    def load_and_save_data_from_zip(self, zip_path):
        """
        Load text documents from a zip file
        - has to contain special keyword in document name (e.g., "_10-K_")
        - cik has to be in rel_cik_list (if provided)
        - avoids duplicates by accession number
        """
        docs = []
        metadata_list = []
    
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            for filename in tqdm(zip_ref.namelist(), desc="Progress"):
                # check if the keyword is in the filename (if it is a 10-K report)
                if self.keyword not in filename:
                    continue
    
                with zip_ref.open(filename) as file:
                    doc_content = file.read().decode("utf-8")
    
                # --- duplicate check ---
                accession_number = self.extract_metadata(
                    doc_content, r'ACCESSION NUMBER:[ \t\n\r\f\v]*(\S+)'
                )
                if accession_number in self.seen_accession_numbers:
                    self.duplicate_accession_count += 1
                    continue
    
                # --- CIK filtering (if rel_cik_list is provided) ---
                if self.rel_cik_list is not None:
                    cik = self.extract_metadata(
                        doc_content, r'CENTRAL INDEX KEY:[ \t\n\r\f\v]*(\d+)'
                    )
                    if cik not in self.rel_cik_list:
                        continue
    
                # --- passed all checks → keep doc ---
                self.seen_accession_numbers.add(accession_number)
                docs.append(doc_content)
                metadata = self.get_metadata(doc_content)
                metadata_list.append(metadata)
    
        print(f"Number of texts: {len(docs)}")
        print(f"Number of metadata: {len(metadata_list)}")
        self.save_results(docs, metadata_list, zip_path)
    
        return docs, metadata_list

    def run_extraction(self):
        """
        Run the full extraction.
        Generate zip paths
        iterate over each path and load/store the data
        """
        zip_paths = self.get_zip_paths() #get zip paths

        for zip_path in zip_paths:
            docs, metadata = self.load_and_save_data_from_zip(zip_path) #iterate over all zip paths, extract documents and metadata + save them in directory

        print(f"Number of (removed) duplicates by accession number: {self.duplicate_accession_count}")