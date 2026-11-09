-- The extension now sends a random ID per install instead of the hardcoded 1,
-- so watchlist.user_id becomes a string to match user_activity.user_id.
-- Run against an existing database; schema.sql already includes this for fresh setups.
--
--   docker exec -i buywise-mysql mysql -uroot -proot buywise < db/migrations/002_watchlist_user_id_string.sql

ALTER TABLE watchlist MODIFY user_id VARCHAR(100) NOT NULL;
