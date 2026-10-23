-- Daily Keepa observations for the shared refresh job.
-- Run once on an existing database after 001_prices_keepa_columns.sql.
ALTER TABLE prices
    ADD COLUMN observed_on DATE NULL AFTER count_used,
    ADD COLUMN source ENUM('keepa_history', 'keepa_daily') NOT NULL
        DEFAULT 'keepa_history' AFTER observed_on,
    ADD UNIQUE KEY unique_daily_price (product_id, observed_on, source);
