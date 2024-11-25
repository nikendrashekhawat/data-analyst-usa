import tomllib

with open("../configs/cfg.toml", "rb") as cfg_file:
    configs = tomllib.load(cfg_file)

KAGGLE_USERNAME = configs["kaggle"]["USERNAME"]
KAGGLE_APIKEY = configs["kaggle"]["KEY"]
