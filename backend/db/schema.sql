-- ==============================================================
-- BuyWise Database Schema
-- Initializes basic database with Product, Price, and Prediction
-- ===============================================================
CREATE database IF NOT EXISTS buywise;
USE buywise;

DROP TABLE IF EXISTS predictions;
DROP TABLE IF EXISTS user_activity;
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

CREATE TABLE user_activity (
    activity_id BIGINT AUTO_INCREMENT PRIMARY KEY,
    asin VARCHAR(10) NOT NULL,
    recommendation_shown ENUM('BUY', 'WAIT') NOT NULL,
    action VARCHAR(50) NOT NULL,
    user_id VARCHAR(100),
    timestamp DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    INDEX idx_activity_time (timestamp),
    INDEX idx_activity_user_time (user_id, timestamp),
    INDEX idx_activity_asin_time (asin, timestamp)
);

    
