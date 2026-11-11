-- Point-in-time storage. Every externally sourced row records when it became known.

CREATE TABLE IF NOT EXISTS prices (
    ticker      VARCHAR NOT NULL,
    date        DATE    NOT NULL,
    open        DOUBLE,
    high        DOUBLE,
    low         DOUBLE,
    close       DOUBLE,
    adj_close   DOUBLE,
    volume      BIGINT,
    shares_out  DOUBLE,
    PRIMARY KEY (ticker, date)
);

-- Quarterly fundamentals, long format so new line items need no migration.
CREATE TABLE IF NOT EXISTS fundamentals (
    ticker         VARCHAR NOT NULL,
    statement      VARCHAR NOT NULL,      -- income | balance | cashflow
    fiscal_year    INTEGER NOT NULL,
    fiscal_period  VARCHAR NOT NULL,      -- Q1..Q4
    report_date    DATE,                  -- fiscal period end
    publish_date   DATE    NOT NULL,      -- when it became public: USE THIS FOR JOINS
    restated_date  DATE,                  -- SimFin values may reflect a later restatement
    item           VARCHAR NOT NULL,
    value          DOUBLE,
    PRIMARY KEY (ticker, statement, fiscal_year, fiscal_period, item)
);

CREATE TABLE IF NOT EXISTS posts (
    post_id       VARCHAR PRIMARY KEY,    -- X post id (or hash of url)
    url           VARCHAR,
    author        VARCHAR NOT NULL,
    published_at  TIMESTAMP NOT NULL,
    ingested_at   TIMESTAMP NOT NULL,
    text          VARCHAR,
    tickers       VARCHAR[],
    stance        VARCHAR,                -- bull | bear | neutral
    confidence    DOUBLE,
    raw           JSON
);

CREATE TABLE IF NOT EXISTS news (
    news_id       VARCHAR PRIMARY KEY,    -- hash of url
    url           VARCHAR,
    source        VARCHAR,
    headline      VARCHAR,
    summary       VARCHAR,
    published_at  TIMESTAMP NOT NULL,
    ingested_at   TIMESTAMP NOT NULL,
    tickers       VARCHAR[],
    stance        VARCHAR,
    confidence    DOUBLE,
    raw           JSON
);

-- Market-wide daily series from yfinance: ^VIX (market IV), SPY (benchmark), ^IRX (13w T-bill %).
CREATE TABLE IF NOT EXISTS market_series (
    symbol     VARCHAR NOT NULL,
    date       DATE    NOT NULL,
    open       DOUBLE,
    high       DOUBLE,
    low        DOUBLE,
    close      DOUBLE,
    adj_close  DOUBLE,
    volume     BIGINT,
    PRIMARY KEY (symbol, date)
);

-- Per-stock implied-volatility summary, one row per ticker per recording day.
-- yfinance has no option history, so this table only grows from the day recording starts.
CREATE TABLE IF NOT EXISTS iv_snapshots (
    ticker        VARCHAR NOT NULL,
    date          DATE    NOT NULL,        -- trading day the snapshot describes
    asof_ts       TIMESTAMP NOT NULL,      -- when it was recorded
    spot          DOUBLE,
    iv30_atm      DOUBLE,
    iv60_atm      DOUBLE,
    skew30        DOUBLE,                  -- IV(put, K=0.9S) - IV(call, K=1.1S) at ~30d
    term_slope    DOUBLE,                  -- iv60_atm - iv30_atm
    pc_vol_ratio  DOUBLE,                  -- put/call volume, expiries <= ~60d
    pc_oi_ratio   DOUBLE,                  -- put/call open interest, expiries <= ~60d
    n_contracts   INTEGER,
    PRIMARY KEY (ticker, date)
);

CREATE TABLE IF NOT EXISTS signals (
    ticker      VARCHAR NOT NULL,
    date        DATE    NOT NULL,
    model       VARCHAR NOT NULL,         -- baseline | llm | ensemble
    p_down      DOUBLE,
    p_flat      DOUBLE,
    p_up        DOUBLE,
    created_at  TIMESTAMP NOT NULL,
    PRIMARY KEY (ticker, date, model)
);
