-- Track record (#85): store enough with each prediction to score it once its horizon passes.
-- Run against an existing database; schema.sql already includes these for fresh setups.
--
--   docker exec -i buywise-mysql mysql -uroot -proot buywise < db/migrations/003_predictions_track_record.sql
--
-- All new columns are nullable so older rows, and code that still calls insert_prediction(),
-- keep working. Rows written by db.connection.log_prediction() fill every one of them.

ALTER TABLE predictions
  -- what we knew and said when we predicted
  ADD COLUMN price_at_prediction DECIMAL(10,2) NULL AFTER confidence_score,
  ADD COLUMN p_drop FLOAT NULL AFTER price_at_prediction,
  ADD COLUMN horizon_days SMALLINT NULL AFTER p_drop,
  ADD COLUMN drop_threshold FLOAT NULL AFTER horizon_days,
  ADD COLUMN model_version VARCHAR(64) NULL AFTER drop_threshold,
  ADD COLUMN predicted_on DATE NULL AFTER model_version,
  -- filled in by the scoring job once predicted_on + horizon_days has passed
  ADD COLUMN resolved_at TIMESTAMP NULL AFTER predicted_on,
  ADD COLUMN actual_min_price DECIMAL(10,2) NULL AFTER resolved_at,
  ADD COLUMN actual_end_price DECIMAL(10,2) NULL AFTER actual_min_price,
  ADD COLUMN dropped BOOLEAN NULL AFTER actual_end_price,
  ADD COLUMN was_correct BOOLEAN NULL AFTER dropped,
  ADD INDEX idx_unresolved (resolved_at, predicted_on),
  ADD INDEX idx_model_version (model_version);
