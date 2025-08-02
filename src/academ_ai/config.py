import yaml

with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)
    
config["DB_PATH"] = config.get("DB_PATH", "./db.db")