# Market Dashboard
Daily cross-asset data (GitHub Actions) + Streamlit dashboard.
- `update_data.py` – downloads prices (Yahoo Finance) and rates (FRED) into `data/`
- `.github/workflows/daily_update.yml` – runs it every day at 07:00 Malaysia time
- `app.py` – the dashboard
