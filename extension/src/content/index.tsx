import { isAmazonProductPage, extractASIN } from "./amazon";

console.log("BuyWise content script loaded");

if (isAmazonProductPage()) {
  const asin = extractASIN();
  console.log("BuyWise detected Amazon product page", { asin });
}
