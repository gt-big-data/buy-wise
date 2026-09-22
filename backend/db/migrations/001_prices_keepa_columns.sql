-- Adds the secondary Keepa series that keepa_fetch.py now pulls alongside price.
-- Run against an existing database; schema.sql already includes these for fresh setups.
--
--   docker exec -i buywise-mysql mysql -uroot -proot buywise < db/migrations/001_prices_keepa_columns.sql

ALTER TABLE prices
  ADD COLUMN used_price DECIMAL(10,2) NULL AFTER deal_flag,
  ADD COLUMN list_price DECIMAL(10,2) NULL AFTER used_price,
  ADD COLUMN sales_rank INT NULL AFTER list_price,
  ADD COLUMN count_new INT NULL AFTER sales_rank,
  ADD COLUMN count_used INT NULL AFTER count_new;
