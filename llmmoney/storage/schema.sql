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
