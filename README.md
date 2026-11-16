TurboFan Jet Engine RUL Prediction

Live demo: https://ishmal770.github.io/turbofan-rul/

Predicting how many cycles a jet engine has left before it fails, with a confidence interval, using an autoencoder and a Gaussian Process Regressor on NASA C-MAPSS (FD003).

Overview

Turbofan jet engines power most commercial aircraft, and an engine failure can have catastrophic consequences. For a long time the industry relied on preventive maintenance, where engines are inspected on a fixed schedule. Sometimes a problem still develops between inspections and is caught too late. This pushed the industry toward predictive maintenance, which uses machine learning on the engine's sensor readings to predict its Remaining Useful Life (RUL): how many operating cycles the engine has left until it fails.

This project combines an autoencoder (TSHAE) with a Gaussian Process Regressor (GPR) to predict RUL. On the validation engines it reaches an RMSE of about 19.7 cycles, and I plan to keep improving it over further iterations. It uses the NASA C-MAPSS dataset, subset FD003.

What makes this architecture different?
It can handle the noisy data. C-MAPSS has 21 noisy sensor channels. The autoencoder compresses each window of sensor readings into a 2-D latent space, so the regressor only sees the important structure.
Gaussian Process gives a confidence interval. Gaussian Process returns a prediction and an uncertainty. In a safety setting, knowing how sure the model is matters as much as the prediction itself.



How it is built
Autoencoder (TSHAE). The Time-Series Hybrid Autoencoder takes a window of 32 cycles x 21 sensors and compresses it into a 2-D latent vector. I did not write this part. It comes from the thesis code by Mikhail Zaitsev: https://github.com/insdout/Time-Series-Hybrid-Autoencoder (MIT license).
Gaussian Process Regressor. My contribution is a GPR trained on the latent vector, plus its squares and product, which outputs the RUL mean and standard deviation.
Export to JSON. export_data.py runs the trained models over 19 validation and 100 test engines and saves the predictions, uncertainty, latent points and per-sensor anomaly scores as JSON files.
Website. The pages in web/ read those JSON files and draw the charts with Plotly. The model is not run in the browser, so the site stays free and fast to host on GitHub Pages.


Most of the hours went into training. The autoencoder and the GPR together took over 1.5 hours per training run, mostly because a Gaussian Process becomes slow as the amount of data grows. The rest went into debugging and iterating: changing settings and features to lower the RMSE and improve accuracy, and fixing the many small problems that come with connecting two models.



Dependencies

- Windows 10 or later (also works on macOS and Linux)
- Python 3.10 or newer
- Packages: `torch`, `numpy`, `pandas`, `scikit-learn`, `omegaconf`
- The NASA C-MAPSS FD003 data (included in the `data` folder)

Installing

```bash
git clone https://github.com/ishmal770/turbofan-rul.git
cd turbofan-rul
pip install torch numpy pandas scikit-learn omegaconf
```

The saved model files (`.pkl`) are not stored in the repo because of their size, so retrain them or ask me for them.

Executing the program

To rebuild the JSON results from the trained models:

```bash
python export_data.py
```

To view the website on your computer:

```bash
cd web
python -m http.server 8000
```

Then open http://localhost:8000 in your browser.



