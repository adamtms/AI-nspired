from modules import  Contrast, Color
from common import generate_csv

generate_csv(
    "test.csv",
    [
        Contrast(),
        Color(),
    ],
    data_path="data/clean",
    verbose=True,
    min_max_similarities=True,
)