import pandas as pd

files = {
    "Flights": "data/raw/flights.csv",
    "Hotels": "data/raw/hotels.csv",
    "Users": "data/raw/users.csv"
}

for name, path in files.items():
    print("\n" + "=" * 60)
    print(f"{name} Dataset")
    print("=" * 60)

    df = pd.read_csv(path)

    print("Shape:", df.shape)
    print("\nColumns:")
    print(df.columns.tolist())

    print("\nFirst 5 rows:")
    print(df.head())

    print("\nMissing values:")
    print(df.isnull().sum())

    print("\nData types:")
    print(df.dtypes)