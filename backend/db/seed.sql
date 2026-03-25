-- ==============================================================
-- BuyWise Seed Data
-- Fake products, prices, and predictions for local development
-- ==============================================================
USE buywise;

INSERT INTO products (asin, title, brand, category) VALUES
    ('B08N5WRWNW', 'Echo Dot (4th Gen) Smart Speaker', 'Amazon', 'Smart Home'),
    ('B07PXGQC1Q', 'AirPods with Charging Case', 'Apple', 'Electronics'),
    ('B09G9HD6PD', 'Kindle Paperwhite (11th Gen)', 'Amazon', 'Electronics'),
    ('B08L5TNJHG', 'Instant Pot Duo 7-in-1', 'Instant Pot', 'Kitchen'),
    ('B07FZ8S74R', 'Logitech MX Master 3 Mouse', 'Logitech', 'Computers')
ON DUPLICATE KEY UPDATE title = VALUES(title);

INSERT INTO prices (product_id, price, timestamp, availability, deal_flag)
SELECT product_id, 29.99,  NOW() - INTERVAL 30 DAY, TRUE,  FALSE FROM products WHERE asin = 'B08N5WRWNW' UNION ALL
SELECT product_id, 27.49,  NOW() - INTERVAL 14 DAY, TRUE,  TRUE  FROM products WHERE asin = 'B08N5WRWNW' UNION ALL
SELECT product_id, 31.99,  NOW(),                   TRUE,  FALSE FROM products WHERE asin = 'B08N5WRWNW' UNION ALL
SELECT product_id, 159.99, NOW() - INTERVAL 30 DAY, TRUE,  FALSE FROM products WHERE asin = 'B07PXGQC1Q' UNION ALL
SELECT product_id, 149.99, NOW() - INTERVAL 14 DAY, TRUE,  TRUE  FROM products WHERE asin = 'B07PXGQC1Q' UNION ALL
SELECT product_id, 159.99, NOW(),                   TRUE,  FALSE FROM products WHERE asin = 'B07PXGQC1Q' UNION ALL
SELECT product_id, 139.99, NOW() - INTERVAL 30 DAY, TRUE,  FALSE FROM products WHERE asin = 'B09G9HD6PD' UNION ALL
SELECT product_id, 134.99, NOW() - INTERVAL 14 DAY, TRUE,  FALSE FROM products WHERE asin = 'B09G9HD6PD' UNION ALL
SELECT product_id, 129.99, NOW(),                   TRUE,  TRUE  FROM products WHERE asin = 'B09G9HD6PD' UNION ALL
SELECT product_id, 99.99,  NOW() - INTERVAL 30 DAY, TRUE,  FALSE FROM products WHERE asin = 'B08L5TNJHG' UNION ALL
SELECT product_id, 89.99,  NOW() - INTERVAL 14 DAY, TRUE,  TRUE  FROM products WHERE asin = 'B08L5TNJHG' UNION ALL
SELECT product_id, 94.99,  NOW(),                   TRUE,  FALSE FROM products WHERE asin = 'B08L5TNJHG' UNION ALL
SELECT product_id, 99.99,  NOW() - INTERVAL 30 DAY, TRUE,  FALSE FROM products WHERE asin = 'B07FZ8S74R' UNION ALL
SELECT product_id, 99.99,  NOW() - INTERVAL 14 DAY, TRUE,  FALSE FROM products WHERE asin = 'B07FZ8S74R' UNION ALL
SELECT product_id, 94.99,  NOW(),                   TRUE,  FALSE FROM products WHERE asin = 'B07FZ8S74R';

INSERT INTO predictions (product_id, recommendation, pred_7d, pred_14d, pred_30d, confidence_score)
SELECT product_id, 'WAIT', 30.49, 28.99, 27.00, 0.82 FROM products WHERE asin = 'B08N5WRWNW' UNION ALL
SELECT product_id, 'BUY',  155.00, 152.00, 148.00, 0.74 FROM products WHERE asin = 'B07PXGQC1Q' UNION ALL
SELECT product_id, 'BUY',  127.00, 124.00, 119.99, 0.91 FROM products WHERE asin = 'B09G9HD6PD' UNION ALL
SELECT product_id, 'WAIT', 92.00,  88.00,  84.99,  0.68 FROM products WHERE asin = 'B08L5TNJHG' UNION ALL
SELECT product_id, 'BUY',  93.00,  91.00,  89.99,  0.77 FROM products WHERE asin = 'B07FZ8S74R';
