from connection import insert_product, insert_price, insert_prediction, get_product, get_latest_prediction

# 1. Insert product
insert_product("B001", "Test Product", "TestBrand", "Electronics")

# 2. Fetch it
product = get_product("B001")
print("Product:", product)

# 3. Insert price
insert_price(product["product_id"], 199.99)

# 4. Insert prediction
insert_prediction(product["product_id"], 180.00, 170.00, 160.00, "WAIT", 0.85)

pred = get_latest_prediction(product["product_id"])
print("test: ", pred)

print("Done testing!")