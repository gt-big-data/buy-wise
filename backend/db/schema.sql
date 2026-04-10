-- ==============================================================
-- BuyWise Database Schema
-- Initializes basic database with Product, Price, and Prediction
-- ===============================================================
CREATE database IF NOT EXISTS buywise;
USE buywise;

DROP TABLE IF EXISTS watchlist;
DROP TABLE IF EXISTS predictions;
DROP TABLE IF EXISTS prices;
DROP TABLE IF EXISTS products;

CREATE TABLE products (
	product_id INT auto_increment primary key,
    asin VARCHAR(10) NOT NULL UNIQUE,
    title VARCHAR(255),
    brand VARCHAR(100),
    category VARCHAR(100),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    INDEX idx_asn (asin)
);

CREATE TABLE prices (
	price_id BIGINT AUTO_INCREMENT primary key,
    product_id INT NOT NULL,
    price DECIMAL(10,2) NOT NULL,
    timestamp DATETIME NOT NULL,
    availability BOOLEAN DEFAULT TRUE,
    deal_flag BOOLEAN DEFAULT FALSE,
    
    foreign key (product_id) references products(product_id)
		ON DELETE CASCADE,
        
	INDEX idx_product_time (product_id, timestamp),
    INDEX idx_timestamp (timestamp)
);

CREATE TABLE predictions (
	prediction_id BIGINT auto_increment primary key,
    product_id INT NOT NULL,
    recommendation ENUM('BUY', 'WAIT'),
    pred_7d DECIMAL(10,2),
    pred_14d DECIMAL(10,2),
    pred_30d DECIMAL(10,2),
    confidence_score FLOAT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    foreign key (product_id) references products(product_id)
		ON DELETE CASCADE,
        
	INDEX idx_product_created (product_id, created_at)
);

CREATE TABLE watchlist (
    watchlist_id BIGINT AUTO_INCREMENT PRIMARY KEY,
    user_id INT NOT NULL,
    product_id INT NOT NULL,
    recommendation_at_add ENUM('BUY', 'WAIT') NOT NULL,
    target_price DECIMAL(10,2),
    added_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    FOREIGN KEY (product_id) REFERENCES products(product_id)
        ON DELETE CASCADE,

-- add the following fk when the user relation is added
    -- FOREIGN KEY (user_id) REFERENCES users(user_id)
    -- ON DELETE CASCADE

    UNIQUE KEY unique_watchlist (user_id, product_id),
    INDEX idx_user (user_id),
    INDEX idx_user_product (user_id, product_id)
);

    
