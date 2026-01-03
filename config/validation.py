import pandas as pd
import logging

logger = logging.getLogger(__name__)

class DataValidator:
    """Validate policy documents before processing"""
    
    @staticmethod
    def validate_raw_data(df):
        """Validate merged_dataset.csv"""
        errors = []
        
        # Check required columns
        required_cols = {"url", "country", "document_type", "guideline_text"}
        missing_cols = required_cols - set(df.columns)
        if missing_cols:
            errors.append(f"Missing columns: {missing_cols}")
        
        # Check for null values
        null_counts = df[required_cols].isnull().sum()
        for col, count in null_counts.items():
            if count > 0:
                errors.append(f"Column '{col}' has {count} null values")
                logger.warning(f"Found {count} null values in {col}")
        
        # Check for empty strings
        for col in required_cols:
            empty_count = (df[col].astype(str).str.strip() == "").sum()
            if empty_count > 0:
                errors.append(f"Column '{col}' has {empty_count} empty values")
                logger.warning(f"Found {empty_count} empty values in {col}")
        
        # Check for duplicates
        dup_count = df.duplicated(subset=["url"]).sum()
        if dup_count > 0:
            errors.append(f"Found {dup_count} duplicate URLs")
            logger.warning(f"Found {dup_count} duplicate URLs")
        
        # Check text length (should be >50 chars for policy)
        short_text = (df["guideline_text"].astype(str).str.len() < 50).sum()
        if short_text > 0:
            logger.warning(f"Found {short_text} documents with <50 characters")
        
        # Check valid countries
        valid_countries = {"Germany", "UK", "USA", "Canada", "Australia"}
        invalid_countries = set(df["country"].unique()) - valid_countries
        if invalid_countries:
            errors.append(f"Invalid countries: {invalid_countries}")
            logger.warning(f"Found invalid countries: {invalid_countries}")
        
        if errors:
            logger.error(f"Validation failed: {'; '.join(errors)}")
            raise ValueError(f"Data validation failed:\n" + "\n".join(errors))
        
        logger.info(f"✅ Data validation passed: {len(df)} documents")
        return True
    
    @staticmethod
    def validate_clean_data(df):
        """Validate final_clean_dataset.csv before topic modeling"""
        errors = []
        
        required_cols = {"url", "clean_text", "tokens", "word_count"}
        missing_cols = required_cols - set(df.columns)
        if missing_cols:
            errors.append(f"Missing columns: {missing_cols}")
        
        # Check that clean_text is not empty
        empty_clean = (df["clean_text"].astype(str).str.strip() == "").sum()
        if empty_clean > 0:
            errors.append(f"Found {empty_clean} empty clean_text values")
        
        # Check that tokens is a valid list
        for idx, tokens in df["tokens"].items():
            try:
                if isinstance(tokens, str):
                    eval(tokens)  # Should be valid Python list
            except:
                errors.append(f"Row {idx}: Invalid token format")
        
        # Check word count > 0
        zero_count = (df["word_count"] == 0).sum()
        if zero_count > 0:
            logger.warning(f"Found {zero_count} documents with 0 words")
        
        if errors:
            logger.error(f"Validation failed: {'; '.join(errors)}")
            raise ValueError(f"Data validation failed:\n" + "\n".join(errors))
        
        logger.info(f"✅ Clean data validation passed: {len(df)} documents")
        return True
