import pytest
from asin import extract_asin


# --- Valid product URLs ---

def test_standard_dp_url():
    assert extract_asin("https://www.amazon.com/dp/B08N5WRWNW") == "B08N5WRWNW"

def test_dp_url_with_product_title_slug():
    assert extract_asin("https://www.amazon.com/Apple-AirPods-Charging-Case-Latest/dp/B07PXGQC1Q/") == "B07PXGQC1Q"

def test_gp_product_url():
    assert extract_asin("https://www.amazon.com/gp/product/B09G9HD6PD") == "B09G9HD6PD"

def test_exec_obidos_url():
    assert extract_asin("https://www.amazon.com/exec/obidos/ASIN/B000000000") == "B000000000"

def test_o_asin_url():
    assert extract_asin("https://www.amazon.com/o/ASIN/B000000001") == "B000000001"

def test_non_us_domain_uk():
    assert extract_asin("https://www.amazon.co.uk/dp/B08N5WRWNW") == "B08N5WRWNW"

def test_non_us_domain_de():
    assert extract_asin("https://www.amazon.de/dp/B08N5WRWNW") == "B08N5WRWNW"

def test_http_scheme():
    assert extract_asin("http://www.amazon.com/dp/B08N5WRWNW") == "B08N5WRWNW"

def test_url_with_query_params():
    assert extract_asin("https://www.amazon.com/dp/B08N5WRWNW?ref=sr_1_1&keywords=test") == "B08N5WRWNW"

def test_asin_returned_uppercase():
    # lowercase in URL should still return uppercase ASIN
    assert extract_asin("https://www.amazon.com/dp/b08n5wrwnw") == "B08N5WRWNW"


# --- Invalid / non-product URLs ---

def test_non_amazon_domain():
    assert extract_asin("https://www.google.com/dp/B08N5WRWNW") is None

def test_amazon_search_page():
    assert extract_asin("https://www.amazon.com/s?k=airpods") is None

def test_amazon_homepage():
    assert extract_asin("https://www.amazon.com/") is None

def test_ftp_scheme():
    assert extract_asin("ftp://www.amazon.com/dp/B08N5WRWNW") is None

def test_non_string_input():
    assert extract_asin(None) is None

def test_integer_input():
    assert extract_asin(12345) is None

def test_empty_string():
    assert extract_asin("") is None

def test_asin_too_short():
    assert extract_asin("https://www.amazon.com/dp/B08N5WRW") is None

def test_asin_too_long():
    assert extract_asin("https://www.amazon.com/dp/B08N5WRWNW1") is None

def test_fake_amazon_domain():
    # e.g. amazon.com.fake.com — should not match
    assert extract_asin("https://www.amazon.com.fake.com/dp/B08N5WRWNW") is None
