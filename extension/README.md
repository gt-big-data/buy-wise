# BuyWise Extension

A Chrome extension that shows price recommendations, confidence scores, and price history charts on Amazon product pages.

---

## Setup

**Prerequisites:** Node.js and npm installed.

1. Install dependencies:

```bash
cd extension
npm install
```

2. Build:

```bash
npm run build
```

The compiled extension is output to `extension/dist/`.

---

## Loading in Chrome

1. Open Chrome and navigate to `chrome://extensions`
2. Enable **Developer mode** (toggle in the top-right corner)
3. Click **Load unpacked**
4. Select the `extension/dist/` folder

The BuyWise extension icon will appear in your Chrome toolbar.

---

## Using the Extension

Navigate to any Amazon product page (e.g. `amazon.com/dp/XXXXXXXXXX`). The BuyWise panel will automatically appear in the top-right corner of the page with:

- A **BUY** or **WAIT** recommendation with confidence score
- A projected savings estimate
- A price history chart with time range and axis toggles
- An expandable explanation of the recommendation

Clicking the extension icon in the toolbar also reopens the panel if it was closed.

---

## Development

To rebuild automatically on file changes:

```bash
npm run watch
```

After each rebuild, go to `chrome://extensions` and click the refresh icon on the BuyWise card to reload the extension. Then refresh the Amazon tab.

### Project structure

```
extension/
├── dist/               # Compiled output — load this folder in Chrome
├── src/
│   ├── components/     # All UI components (RecommendationBanner, PriceChart, etc.)
│   ├── content/        # Content script — injected into Amazon pages
│   ├── popup/          # Extension toolbar popup
│   └── background/     # Service worker
├── public/
│   └── popup.html
└── webpack.config.js
```

### Making changes

- **UI components** live in `src/components/` — edit these and rebuild to see changes on Amazon pages
- **Do not edit `dist/` directly** — it gets overwritten on every build
- The content script (`src/content/index.tsx`) controls when and how the panel mounts

---

## Permissions

| Permission | Reason |
|---|---|
| `activeTab` | Read the current tab's URL to detect Amazon product pages and extract the ASIN |
| `storage` | Save user actions (buy/wait) locally |
| `host_permissions: amazon.com` | Inject the content script on Amazon pages |
