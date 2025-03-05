from modules import DinoV2, Color, Contrast, ResNet
from common import generate_csv

generate_csv("csv/final.csv", [DinoV2(), Color(), Contrast(), ResNet()], verbose=True, min_max_similarities=True)
