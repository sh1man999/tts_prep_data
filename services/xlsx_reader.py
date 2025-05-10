from typing import Generator

import pandas as pd


def read_xlsx(file_path, column_name: str)-> Generator[str, None, None]:
    df = pd.read_excel(file_path)
    for index, row in df.iterrows():
        yield row[column_name]