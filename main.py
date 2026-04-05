from src.synthetic_trade.data_generation import (
    load_generation_config,
    generate_advisors,
    generate_clients,
    generate_products,
    generate_trades,
)
from src.synthetic_trade.suitability_rules import apply_suitability_rules

import json
import numpy as np

if __name__ == "__main__":
    with open("config/config.json") as f:
        raw_cfg = json.load(f)

    cfg = load_generation_config(raw_cfg)
    rng = np.random.default_rng(cfg.seed)
    advisors = generate_advisors(cfg, rng)
    clients = generate_clients(advisors, cfg, rng)
    products = generate_products(cfg, rng)
    trades_suitability = apply_suitability_rules(
        generate_trades(advisors, clients, products, cfg, rng),
        clients,
        products,
        raw_cfg["rules"],
    )

    advisors.to_csv("output/advisors.csv", index=False)
    clients.to_csv("output/clients.csv", index=False)
    products.to_csv("output/products.csv", index=False)
    trades_suitability.to_csv("output/trade_suitability.csv", index=False)
