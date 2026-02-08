import pandas as pd
from tqdm import tqdm


class FilingDateExtractor:
    def __init__(
        self, 
        result_dict, 
        result_cat = "financials",
        table_name = "income_statement",
        date_name = "calendarYear", #name of the date which we want for fiscal year mapping
        filing_date_name = "fillingDate", #name of filing date we want to map with year of  --> Typo in API
    ):
        self.result_dict = result_dict
        self.result_cat = result_cat
        self.table_name = table_name
        self.date_name = date_name
        self.filing_date_name = filing_date_name
        self.clean_df = None
        self.invalid_df = None

    def extract_data(self, remove_duplicates = True):
        """
        Extract entries from the JSON input and transform them to rows followed by transformation into a dataframe
        date: boolean: if True, the date is in %Y-%m-%d format. Switch to False if the year is given directly
        """
        rows = []
    
        #iterate over each ticker
        for ticker, data in tqdm(self.result_dict.items(), desc="Progress"):
            rel_cat = data.get(self.result_cat, []) #extract relevant result category
            rel_data = rel_cat.get(self.table_name, []) #extract the relevant input table
            
    
            for entry in rel_data: #iterate over each entry in the relevant input table
                year = int(entry.get(self.date_name))
                filing_date = entry.get(self.filing_date_name, None)[:10]
                period_of_report = entry.get("date")
                
                row = {
                    "ticker": ticker,
                    "period_of_report": period_of_report,
                    "filing_date": filing_date,
                    "year": year,
                }

                rows.append(row)
           
        #create df from the rows
        df = pd.DataFrame(rows)
        
        #drop duplicate rows of ticker, year --> before sorting to ensure that newest values are kept  
        if remove_duplicates:
            df = df.drop_duplicates(subset = ["ticker", "year"])
    
        #sort values for calculating YoY percentage changes
        df = df.sort_values(["ticker", "year"]).reset_index(drop=True)
        return df

    def sanity_check_filings(self, filing_df, max_delay_days=180):
        """
        Sanity check for filing-period of report mapping --> found some cases where end of period of report > filing date
        """
        df = filing_df.copy()

        #ensure date format
        df['period_of_report'] = pd.to_datetime(df['period_of_report'], errors='coerce')
        df['filing_date'] = pd.to_datetime(df['filing_date'], errors='coerce')
        df = df.dropna(subset=['period_of_report', 'filing_date'])
    
        #calculate delays between filing date and end of period of report
        df['days_delay'] = (df['filing_date'] - df['period_of_report']).dt.days
    
        #basic logical filters --> days of delay > 0; max_delay time between filing and end of period of report
        valid_mask = (
            (df['days_delay'] > 0) & #filing should be after end of period of report, not exactly at the same date
            (df['days_delay'] <= max_delay_days) #filings should be within a reasonable date range after period ends
        )
    
        #fiscal year needs to be close to the year of the end of the period of report
        valid_mask &= (df['year'] - df['period_of_report'].dt.year).abs() <= 1
    
        #drop future dates
        today = pd.Timestamp.today()
        valid_mask &= (df['filing_date'] <= today) & (df['period_of_report'] <= today)
    
        cleaned_df = df[valid_mask].copy().reset_index(drop = True)
        invalid_df = df[~valid_mask].copy().reset_index(drop = True)
    
        print(f"Valid filings: {len(cleaned_df)} / {len(df)} ({100*len(cleaned_df)/len(df):.1f}%)")
        print(f"Removed {len(invalid_df)} invalid rows (e.g., negative delay, year mismatch, or future dates)")
    
        return cleaned_df, invalid_df



    def run(self, remove_duplicates = True, max_delay_dates = 180):
        """
        Wrapper method to extract sanity-checked filing dates
        """
        #1. extract all information
        filing_df = self.extract_data(remove_duplicates)

        #2. sanity check the filing information
        cleaned_df, invalid_df = self.sanity_check_filings(filing_df, max_delay_dates)
        self.clean_df = cleaned_df
        self.invalid_df = invalid_df

        return cleaned_df

    def plot_dfs(self):
        """
        Plot the days of delay between end of period of report and filing dates
        """
        fig, axes = plt.subplots(1, 2, figsize = (16, 4))
        
        axes[0].hist(self.clean_df["days_delay"], edgecolor = "white")
        axes[0].set_title("Cleaned")
        axes[0].grid(alpha = 0.2)
        
        axes[1].hist(self.invalid_df["days_delay"], edgecolor = "white", bins = 50)
        axes[1].set_title("Invalid")
        axes[1].grid(alpha = 0.2)
        
        plt.suptitle("Histogram of days of delay between end of period and filing date")
        plt.tight_layout()
        plt.show()