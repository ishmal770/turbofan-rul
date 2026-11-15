import torch
from omegaconf import OmegaConf
from utils.metric_dataloader import MetricDataPreprocessor
from utils.tshae_utils import load_tshae_model


cfg = OmegaConf.load(r"best_models\FD003\tshae\.hydra\config.yaml")
for k in ("train", "test", "val"):
    cfg.data_preprocessor[f"{k}_dl_num_workers"] = 0

def main():
    train, test,val = MetricDataPreprocessor(**cfg.data_preprocessor).get_dataloaders()
    model = load_tshae_model(r"best_models\FD003\tshae\best_rve_model.pt").eval()
    eid = int(val.dataset.ids[0])
    x, rul = val.dataset.get_run(eid)
    print("engine", eid, "| x", tuple(x.shape), "| rul", tuple(rul.shape))
    print("x min/max", float(x.min()), float(x.max()))

    with torch.no_grad():
        z, mean, log_var, x_hat = model(x)

    print("z", tuple(z.shape), "| x_hat", tuple(x_hat.shape))
    print("val engines", len(val.dataset.ids), "| test engines", len(test.dataset.ids))


if __name__ == "__main__":
    main()