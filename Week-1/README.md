# Week-1 — Motorcycle & Car Data: Scraping + EDA

## Table of Contents
- [Project Overview](#project-overview)
- [Project Structure](#project-structure)
- [Setup](#setup)
- [How to run](#how-to-run)

## Project Overview
This repository contains notebooks for collecting motorcycle listings (web scraping) and performing exploratory data analysis (EDA) on those listings and a `car.csv` dataset. It includes scraping scripts, notebooks with cleaning and visualization, generated EDA reports, and CatBoost training metadata.

## Project Structure
- Data files (raw/processed):
  - `car.csv`
  - `EDA_DF.csv`
  - `eda_motor.csv`
  - `hamrobazar_motorcycles_multitab.csv`
  - `motocycle_243.csv`
  - `motorbike_initial.csv`
- Notebooks:
  - `notebooks/Web Scraping.ipynb`
  - `Week-1 Task.ipynb`
- Results / reports:
  - `output.html`
  - `output_for_motocycle_data.html`
- Model / training metadata:
  - `catboost_info/`
- This file:
  - `README.md`
- Extra:
  - `requirements.txt`
  - `.gitignore`

## Setup
1. Create a virtual environment:
   ```powershell 
   python -m venv .venv
2. Activate (PowerShell):
   ```powershell
   .venv\Scripts\Activate.ps1 

3. Or (Command Prompt):
   ```cmd
   .venv\Scripts\activate.bat
4. Install dependencies:
   ```bash
   pip install -r requirements.txt

## How to run
- Notebooks:
  - Open `notebooks/Web Scraping.ipynb` and `Week-1 Task.ipynb` in Jupyter or VS Code and run cells.
- Scraping:
  - `notebooks/Web Scraping.ipynb` uses Selenium - you must install a matching ChromeDriver (or other driver) and ensure the driver is on your `PATH`.
  - Scraping may be blocked by site rate limits - use polite delays and respect site terms.
- EDA / Reports:
  - `Week-1 Task.ipynb` contains cleaning and visualizations; the profile report is exported to `output_for_motocycle_data.html`.