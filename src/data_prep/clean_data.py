import pandas as pd
import os


def load_data(file_path):
    """
    Attempts to load the CSV file using multiple encodings
    (utf-16, windows-1250, utf-8) to safely prevent decoding errors.
    """
    try:
        return pd.read_csv(file_path, sep="\t", encoding="utf-16")
    except Exception as e1:
        try:
            return pd.read_csv(file_path, sep="\t", encoding="windows-1250")
        except Exception as e2:
            try:
                return pd.read_csv(file_path, sep="\t", encoding="utf-8")
            except Exception as e3:
                print("Error loading data: " + str(e3))
                return pd.DataFrame()


def rename_columns(df):
    """
    Translates Czech column names into standardized English names
    required for the machine learning model.
    """
    try:
        df = df.rename(columns={
            "Znacka": "brand",
            "Model": "model",
            "Cena_Kc": "price",
            "Rok": "year",
            "Najezd_KM": "mileage_km",
            "Palivo": "fuel",
            "Motor": "engine_type",
            "Vykon_kW": "power_kw",
            "Pohon": "drivetrain",
            "Prevodovka": "transmission",
            "Cesta_k_obrazku": "image_path"
        })
        return df
    except Exception as e:
        print("Error renaming columns: " + str(e))
        return df


def drop_missing_values(df):
    """
    Removes any rows that contain missing or null values from the dataset.
    """
    try:
        return df.dropna()
    except Exception as e:
        print("Error dropping missing values: " + str(e))
        return df


def convert_numeric_types(df):
    """
    Converts specific columns to strict numeric data types,
    coercing any parsing errors into NaN values.
    """
    try:
        df["price"] = pd.to_numeric(df["price"], errors="coerce")
        df["year"] = pd.to_numeric(df["year"], errors="coerce")
        df["mileage_km"] = pd.to_numeric(df["mileage_km"], errors="coerce")
        df["power_kw"] = pd.to_numeric(df["power_kw"], errors="coerce")
        return df
    except Exception as e:
        print("Error converting numeric types: " + str(e))
        return df


def normalize_text_columns(df):
    """
    Cleans text columns by removing leading/trailing whitespaces
    and converting all characters to lowercase to prevent duplicates.
    """
    try:
        text_cols = ["brand", "model", "fuel", "engine_type", "drivetrain", "transmission"]
        for col in text_cols:
            df[col] = df[col].astype(str).str.strip().str.lower()
        return df
    except Exception as e:
        print("Error normalizing text columns: " + str(e))
        return df


def clean_image_paths(df):
    """
    Normalizes image file paths by replacing Windows-style backslashes
    with standard forward slashes for cross-platform compatibility.
    """
    try:
        df["image_path"] = df["image_path"].str.replace("obrazky_aut\\\\", "cars_photos/", regex=False)
        df["image_path"] = df["image_path"].str.replace("obrazky_aut\\", "cars_photos/", regex=False)
        return df
    except Exception as e:
        print("Error cleaning image paths: " + str(e))
        return df


def save_data(df, output_path):
    """
    Creates the necessary destination directories if they do not exist
    and saves the cleaned dataframe to the disk as a CSV file.
    """
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path, index=False)
    except Exception as e:
        print("Error saving data: " + str(e))


def main():
    """
    Executes the complete data preparation pipeline: loading, renaming,
    cleaning, formatting, and saving the final processed dataset.
    """
    try:
        input_path = "../../data/raw/cars_data_not_clean.csv"
        output_path = "../../data/processed/cleaned_cars_data.csv"

        df = load_data(input_path)

        try:
            if df.empty:
                print("DataFrame is empty. Stopping.")
                return
        except Exception as e:
            pass

        df = rename_columns(df)
        df = drop_missing_values(df)

        df = convert_numeric_types(df)
        df = drop_missing_values(df)

        df = normalize_text_columns(df)
        df = clean_image_paths(df)

        save_data(df, output_path)
        print("Data cleaning completed successfully.")
    except Exception as e:
        print("Error in main execution: " + str(e))


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("Fatal error: " + str(e))