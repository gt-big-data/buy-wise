import os
from datetime import datetime
from typing import Optional

from dotenv import load_dotenv
import mysql.connector
from mysql.connector import pooling

load_dotenv()

dbconfig = {
    "host": os.getenv("DB_HOST", "localhost"),
    "user": os.getenv("DB_USER"),
    "password": os.getenv("DB_PASSWORD"),
    "database": os.getenv("DB_NAME", "buywise")
}


connection_pool = pooling.MySQLConnectionPool(
    pool_name="mypool",
    pool_size=5,
    **dbconfig
)

def get_connection():
    return connection_pool.get_connection()

def get_product(asin):
    conn = None
    cursor = None
    try:
        conn = get_connection()
        cursor = conn.cursor(dictionary=True)
        query = "SELECT * FROM products WHERE asin = %s"
        cursor.execute(query, (asin,))
        return cursor.fetchone()
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()

def insert_product(asin, title, brand, category):
    conn = None
    cursor = None

    
    try:
        conn = get_connection()
        cursor = conn.cursor(dictionary=True)
        query = """
        INSERT INTO products (asin, title, brand, category)
        VALUES (%s, %s, %s, %s)
        ON DUPLICATE KEY UPDATE
                title = VALUES(title),
                brand = VALUES(brand),
                category = VALUES(category)

        """

        cursor.execute(query, (asin, title, brand, category))
        conn.commit()
    except Exception as e:
        if conn:
            conn.rollback()

    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()




def insert_price(
    product_id,
    price,
    availability=True,
    deal_flag=False,
    *,
    recorded_at: Optional[datetime] = None,
):
    """Insert a price row. Pass ``recorded_at`` for historical samples (e.g. Keepa); else UTC now."""
    conn = None
    cursor = None
    ts = recorded_at if recorded_at is not None else datetime.utcnow()
    try:
        conn = get_connection()
        cursor = conn.cursor()
        query = """
            INSERT INTO prices (product_id, price, timestamp, availability, deal_flag)
            VALUES (%s, %s, %s, %s, %s)
        """
        cursor.execute(query, (product_id, price, ts, availability, deal_flag))
        conn.commit()
    except Exception as e:
        if conn:
            conn.rollback()
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()

def get_price_history(product_id, limit=100):
    conn = None
    cursor = None
    try:
        conn = get_connection()
        cursor = conn.cursor(dictionary=True)
        query = """
            SELECT *
            FROM prices
            WHERE product_id = %s
            ORDER BY timestamp DESC
            LIMIT %s
        """
        cursor.execute(query, (product_id, limit))
        return cursor.fetchall()
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()

def insert_prediction(product_id, pred_7d, pred_14d, pred_30d, recommendation, confidence):
    conn = None
    cursor = None

    try:
        conn = get_connection()
        cursor = conn.cursor()

        query = """
            INSERT INTO predictions (
                product_id,
                pred_7d,
                pred_14d,
                pred_30d,
                recommendation,
                confidence_score
            )
            VALUES (%s, %s, %s, %s, %s, %s)
        """

        values = (product_id, pred_7d, pred_14d, pred_30d, recommendation, confidence)

        cursor.execute(query, values)
        conn.commit()
    except Exception as e:
        if conn:
            conn.rollback()
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()

def get_latest_prediction(product_id):
    conn = None
    cursor = None
    try:
        conn = get_connection()
        cursor = conn.cursor(dictionary=True)
        query = """
            SELECT *
            FROM predictions
            WHERE product_id = %s
            ORDER BY created_at DESC
            LIMIT 1
        """
        cursor.execute(query, (product_id,))
        return cursor.fetchone()
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()

def insert_user_activity(
    asin,
    recommendation_shown,
    action,
    timestamp,
    user_id=None,
):
    conn = None
    cursor = None
    try:
        conn = get_connection()
        cursor = conn.cursor()
        query = """
            INSERT INTO user_activity (
                asin,
                recommendation_shown,
                action,
                user_id,
                timestamp
            )
            VALUES (%s, %s, %s, %s, %s)
        """
        cursor.execute(
            query,
            (asin, recommendation_shown, action, user_id, timestamp),
        )
        conn.commit()
        return cursor.lastrowid
    except Exception:
        if conn:
            conn.rollback()
        raise
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()

def get_recent_user_activity(limit=20, user_id=None):
    conn = None
    cursor = None
    try:
        conn = get_connection()
        cursor = conn.cursor(dictionary=True)
        if user_id:
            query = """
                SELECT *
                FROM user_activity
                WHERE user_id = %s
                ORDER BY timestamp DESC, activity_id DESC
                LIMIT %s
            """
            cursor.execute(query, (user_id, limit))
        else:
            query = """
                SELECT *
                FROM user_activity
                ORDER BY timestamp DESC, activity_id DESC
                LIMIT %s
            """
            cursor.execute(query, (limit,))
        return cursor.fetchall()
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()
