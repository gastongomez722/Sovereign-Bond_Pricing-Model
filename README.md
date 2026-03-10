# Sovereign Bond Pricing Model

This project implements several short-rate interest rate models to price sovereign bonds and analyze model-implied term structures. The goal is find out the reason for the gap between market observed prices and projected fair market values, as market observed values imply unrealistically low implied future short term interest rates. 

Project Report Can be Read Here: https://drive.google.com/file/d/1gR_FQc4BcFJohK53C24rQRPZ58wZFfm9/view?usp=sharing

The project includes implementations of:

- Cox–Ingersoll–Ross (CIR)
- Hull–White
- Black–Karasinski
- Simple diffusion benchmark model

These models are calibrated to observed sovereign bond prices to compare model-implied yields against market data.

---

## Project Motivation

Pricing sovereign bonds requires modeling the evolution of interest rates over time. Short-rate models provide a tractable way to simulate interest rate dynamics and derive bond prices analytically or numerically.

This project explores how different short-rate models perform when calibrated to real sovereign bond data, using the BONTAM bond as a case study.

The objective is to:

- Implement common short-rate models used in fixed income
- Calibrate model parameters to market prices
- Compare model-implied bond prices with observed prices

---

## Repository Structure

Sovereign-Bond-Pricing-Model  
│  
├── data/                  # Sovereign bond price data  
├── notebooks/             # Analysis and experiments  
│   └── bontam_analysis.ipynb  
│  
├── src/                   # Model implementations  
│   ├── cir_model.py  
│   ├── hull_white.py  
│   ├── bk_model.py  
│   ├── calibration.py  
└── README.md  

---

## Models Implemented

### Cox–Ingersoll–Ross (CIR)

A mean-reverting square-root diffusion process commonly used for modeling short-term interest rates.

The CIR process is defined as:

dr_t = κ(θ − r_t)dt + σ√(r_t)dW_t

Key properties:

- Mean-reverting  
- Ensures positive interest rates  
- Widely used in fixed income modeling  

---

### Hull–White

An extension of the Vasicek model allowing time-dependent parameters. It is frequently used for fitting the entire yield curve.

Properties:

- Mean-reverting  
- Flexible term structure fitting  
- Common in interest rate derivatives pricing  

---

### Black–Karasinski

A lognormal short-rate model ensuring positive interest rates through exponential transformation.

Properties:

- Lognormal dynamics  
- Useful for modeling rate distributions with skew  

---

## Calibration

Model parameters are estimated by minimizing the difference between model-implied bond prices and observed market prices.

The calibration process involves:

1. Loading market bond price data  
2. Computing model-implied prices  
3. Minimizing pricing error   

This returns a market price of uncertainty, lying around 1.0-1.3 range, depending on the trading session. This is robust to model specification, time till expiry, and even model parameters

---

## Future Improvements

Potential extensions to this project include:

- regime switching process (especially relevant in Argentina's macroeconomy)
- API calls for macro-data
- Tests for robustness of model fit

---

## Author

Gaston Gomez  

GitHub: https://github.com/gastongomez722