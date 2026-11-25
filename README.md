# University Investment Challenge Portfolio Optimizer (Python web app)

The whole dashboard lives in **`app.py`**: the page, the maths from
`Portfolio_Optimizer_M1.xlsx` and the yfinance download. The **Pull latest data** button
at the top of the page makes the server download fresh prices from Yahoo Finance and
recalculate everything.

| File | What it is |
|---|---|
| `app.py` | The web app: settings (`CONFIG`), calculations (`build`), download, web server, page. |
| `requirements.txt` | Python libraries the app needs. |
| `render.yaml` | Tells Render how to start the app. |
| `.python-version` | Python version for Render (3.12). |

## 1. Try it on your computer (optional, 5 minutes)

1. Install Python 3.12 from python.org if you don't have it.
2. Open a terminal in this folder and run:
   ```bash
   pip install -r requirements.txt
   python app.py
   ```
3. Open **http://localhost:5000**. The first load takes a few seconds while it downloads
   10 years of prices. Press **Ctrl + C** in the terminal to stop it.

## 2. Put it online for your CV (free, about 15 minutes)

**a. Put the files on GitHub**
1. Make an account at github.com.
2. Click **+** (top right) → **New repository**. Name: `portfolio-optimizer`, Public, **Create repository**.
3. On the new repo page click **uploading an existing file**, drag in `app.py`,
   `requirements.txt`, `render.yaml` and `README.md`, then **Commit changes**.
   (`.python-version` and `.gitignore` are optional. Files starting with a dot are hidden on Mac:
   press Cmd + Shift + . in Finder to see them.)

**b. Run it on Render**
1. Go to render.com and sign up **with your GitHub account**.
2. Click **New** → **Web Service**, and pick your `portfolio-optimizer` repository.
3. Check the settings (Render usually fills them from `render.yaml`):
   - Runtime: **Python 3**
   - Build command: `pip install -r requirements.txt`
   - Start command: `gunicorn app:app --workers 1 --threads 4 --timeout 120`
   - Instance type: **Free**
4. Click **Create Web Service** (or **Deploy**). After a few minutes you get a link like
   `https://portfolio-optimizer.onrender.com`. That is the link for your CV.
   Direct link to one stock: add `#eog`, `#meta`, … at the end.

**Good to know about the free plan**
- The app goes to sleep after about 15 minutes without visitors. The next visitor waits
  roughly a minute while it wakes up and downloads prices, so open the link yourself
  before an interview.
- Yahoo Finance sometimes limits downloads from cloud servers. If the button reports that
  the download failed, the page keeps showing the last numbers it had; try again later.

## 3. Changing things

Edit `app.py` on GitHub (open the file → pencil icon → **Commit changes**). Render redeploys
automatically within a few minutes.

- **New dividend:** add a line to `"dividends"` in `CONFIG`, e.g.
  `{"date": "2026-12-15", "ticker": "MCD", "amount": 4.10},`
- **Trades:** `"holdings"` has the shares, fill price, commission and amount paid for each stock.
- **Name in the footer:** `"owner"`.

## Method

Matches the Excel model: daily log returns of the adjusted close; expected annual return
e^(mean×252) − 1; annual σ √(sample variance × 252); VARCOVAR = XᵀX/(N−1) on returns minus
their expected daily return; covariance = pandas `cov(ddof=0)`; portfolio σ and Sharpe as in
Calculations N4, N5 and N8; weights = invested ÷ capital as in the Holdings tab.
Not investment advice.
