import pandas as pd

ports = pd.read_csv("data/raw/portwatch_ports_raw.csv")
choke = pd.read_csv("data/raw/portwatch_chokepoints_raw.csv")

print("=== PORTS ===")
print(ports.dtypes)
print(ports.head(3).to_string())

print("\n=== CHOKEPOINTS ===")
print(choke.dtypes)
print(choke.head(3).to_string())
