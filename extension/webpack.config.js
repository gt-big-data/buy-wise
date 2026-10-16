const path = require("path");
const webpack = require("webpack");
const CopyPlugin = require("copy-webpack-plugin");

const BACKEND_URL = (process.env.BACKEND_URL || "http://localhost:8000").replace(/\/+$/, "");

module.exports = {
  mode: "production",
  entry: {
    popup: "./src/popup/index.tsx",
    content: "./src/content/index.tsx",
    background: "./src/background/index.ts"
  },
  output: {
    path: path.resolve(__dirname, "dist"),
    filename: "[name].js"
  },
  resolve: {
    extensions: [".ts", ".tsx", ".js"]
  },
  plugins: [
    new webpack.DefinePlugin({
      __BACKEND_URL__: JSON.stringify(BACKEND_URL)
    }),
    new CopyPlugin({
      patterns: [
        {
          from: "public",
          to: ".",
          // The manifest has to allow whatever backend the build points at
          transform(content, absoluteFrom) {
            if (!absoluteFrom.endsWith("manifest.json")) return content;
            return content
              .toString()
              .replace("http://localhost:8000/*", `${new URL(BACKEND_URL).origin}/*`);
          }
        }
      ]
    })
  ],
  module: {
    rules: [
      {
        test: /\.tsx?$/,
        use: "ts-loader",
        exclude: /node_modules/
      },
      {
        test: /\.css$/,
        use: ["style-loader", "css-loader"]
      }
    ]
  }
};
