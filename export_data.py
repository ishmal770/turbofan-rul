import argparse, datetime as dt, json, os
import joblib
import numpy as np
import torch
from omegaconf import OmegaConf
from scipy.stats import spearmanr
from utils.metric_dataloader import MetricDataPreprocessor
from utils.tshae_utils import load_tshae_model




TSHAE_DIR = os.path.join("best_models", "FD003", "tshae")


sensors_names = [ ("s1", "T2", "Fan inlet temperature", "Fan"), 
                 ("s2", "T24", "LPC Outlet temperature", "LPC"),
                 ("s3", "T30", "HPC outlet temperature", "HPC"),
                 ("s4", "T50", "LPT outlet temperature", "LPT"),
                 ("s5", "P2", "Fan inlet pressure", "Fan"),
                 ("s6", "P15", "Bypass-duct pressure", "Bypass"),
                 ("s7", "P30", "HPC outlet pressure", "HPC"),
                  ("s8", "Nf", "Physical fan speed", "Fan"),
                  ("s9", "Nc", "Physical core speed", "HPC"),
                  ("s10", "epr", "Engine pressure ratio", "Bypass"),
                  ("s11", "Ps30", "HPC outlet static pressure", "HPC"),
                  ("s12", "phi", "Fuel flw/ Ps30", "Combustor"),
                   ("s13", "NRf", "Corrected fan speed", "Fan"),
                    ("s14", "NRc", "Corrected core speed", "HPC"),
                ("s15", "BPR", "Bypass ratio", "Fan"),
                ("s16", "farB", "Burner fuel-air ratio", "Combustor"),
                 ("s17", "htBleed", "Bleed enthalpy", "HPC"),
                ("s18", "Nf_dmd", "Demanded fan speed", "Fan"),
                ("s19", "PCNfR_dmd", "Demanded corrected fan speed", "Fan"),
                ("s20", "W31", "HPT coolant bleed", "HPT"),
                ("s21", "W32", "LPT coolant bleed", "LPT"),
]

def round_values(a, d):
    return np.round(np.asarray(a, dtype=float), d).tolist()

def augment(z):
    return np.column_stack([z,z[:,0] ** 2, z[:,1] ** 2, z[:,0] *z[:,1]])

def gaussianprocess(gpr, Z):
    try:
        mu, sd=gpr.predict(Z, return_std=True)
    except TypeError:
        mu, sd = gpr.predict(Z)
    return np.asarray(mu).ravel(), np.asarray(sd).ravel()

def run_engine(model, gpr, ds, eid):
    x, rul, = ds.get_run(int(eid))
    with torch.no_grad():
        z, _, _, x_hat = model(x)
    z= z.numpy()
    mu, sd = gaussianprocess(gpr, augment(z))
    return {
        "true_rul": rul.numpy().ravel(), "prediction_rul": mu, "std": sd, "z":z,
        "sensors": x[:, -1, :].numpy(),
        "err": ((x_hat-x)**2).mean(dim=1).numpy(),
    
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpr", default=os.path.join("outputs", "2026-07-23", "20-16-00", "gpr_best_model.pkl"))
    ap.add_argument("--out", default=os.path.join("web", "data"))
    args = ap.parse_args()

    cfg = OmegaConf.load(os.path.join(TSHAE_DIR, ".hydra", "config.yaml"))
    for k in ("train", "test", "val"):
        cfg.data_preprocessor[f"{k}_dl_num_workers"] = 0
    _, test, val = MetricDataPreprocessor(**cfg.data_preprocessor).get_dataloaders()
    model = load_tshae_model(os.path.join(TSHAE_DIR, "best_rve_model.pt")).eval()
    gpr = joblib.load(args.gpr)

    runs = {}
    for split, ds in (("val", val.dataset), ("test", test.dataset)):
        for eid in ds.ids:
            runs[f"{split}-{int(eid)}"] = (split, int(eid), run_engine(model, gpr, ds, eid))
        print(f"{split}: done")

    vals = [v for v in runs.values() if v[0] == "val"]
    healthy = np.concatenate([v[2]["err"][v[2]["true_rul"] >= 100] for v in vals])
    err_ref = np.percentile(healthy, 90, axis=0) + 1e-9
    S = np.concatenate([v[2]["sensors"] for v in vals])
    R = np.concatenate([v[2]["true_rul"] for v in vals])
    P = np.concatenate([v[2]["prediction_rul"] for v in vals])

    meta = []
    for j, (sid, sym, name, comp) in enumerate(sensors_names):
        rho = spearmanr(S[:, j], R).correlation
        meta.append({"id": sid, "symbol": sym, "name": name, "component": comp,
                     "corr_rul": None if np.isnan(rho) else round(float(rho), 3),
                     "err_ref": float(err_ref[j])})

    os.makedirs(os.path.join(args.out, "engines"), exist_ok=True)
    fleet = []
    for key, (split, eid, d) in runs.items():
        anom = d["err"] / err_ref
        with open(os.path.join(args.out, "engines", f"{key}.json"), "w") as f:
            json.dump({"key": key, "split": split, "engine": eid,
                       "cycle": list(range(len(d["true_rul"]))),
                       "true_rul": round_values(d["true_rul"], 1), "prediction_rul": round_values(d["prediction_rul"], 1),
                       "std": round_values(d["std"], 1), "z1": round_values(d["z"][:, 0], 3), "z2": round_values(d["z"][:, 1], 3),
                       "sensors": round_values(d["sensors"].T, 3), "anom": round_values(anom.T, 2)},
                      f, separators=(",", ":"))
            fleet.append({"key": key, "split": split, "engine": eid, "cycles": len(d["true_rul"]),
                    "rul_true_last": round(float(d["true_rul"][-1]), 1),
                    "rul_pred_last": round(float(d["prediction_rul"][-1]), 1),
                    "std_last": round(float(d["std"][-1]), 1),
                    "anom_last": round(float(anom[-10:].mean()), 2)})
    rmse = float(np.sqrt(np.mean((P - R) ** 2)))
    json.dump({"sensors": meta, "units": "standardised (z-score)"},
              open(os.path.join(args.out, "sensors_meta.json"), "w"), indent=1)
    json.dump({"fleet": fleet}, open(os.path.join(args.out, "fleet.json"), "w"), separators=(",", ":"))
    json.dump({"dataset": "NASA C-MAPSS FD003", "val_rmse": rmse,
               "val_mae": float(np.mean(np.abs(P - R))),
               "n_val": len(vals), "n_test": len(runs) - len(vals),
               "generated": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")},
              open(os.path.join(args.out, "model_summary.json"), "w"), indent=1)

    size = sum(os.path.getsize(os.path.join(dp, f)) for dp, _, fs in os.walk(args.out) for f in fs)
    print(f"engines: {len(runs)} | total {size / 1e6:.1f} MB | val RMSE {rmse:.2f}")
    print("sensors most correlated with RUL:",
          sorted([(m["id"], m["corr_rul"]) for m in meta if m["corr_rul"] is not None],
                 key=lambda t: -abs(t[1]))[:5])


if __name__ == "__main__":
    main()
    

    