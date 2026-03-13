import React from "react";
import {
  Area,
  AreaChart,
  CartesianGrid,
  Line,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis
} from "recharts";
import { PricePoint } from "../types";

type PriceChartProps = {
  title: string;
  points: PricePoint[];
  predictedBestPrice: number;
};

const formatMoney = (value: number): string => `$${Math.round(value)}`;

const CustomTooltip = ({
  active,
  payload,
  label
}: {
  active?: boolean;
  payload?: Array<{ value?: number; dataKey?: string }>;
  label?: string;
}) => {
  if (!active || !payload || payload.length === 0) {
    return null;
  }

  return (
    <div className="buywise-tooltip">
      <div className="buywise-tooltip-label">{label}</div>
      {payload.map((item) => {
        if (typeof item.value !== "number") {
          return null;
        }

        const lineLabel =
          item.dataKey === "actual" ? "Observed price" : "Predicted price";

        return (
          <div key={`${item.dataKey}-${item.value}`} className="buywise-tooltip-row">
            {lineLabel}: {formatMoney(item.value)}
          </div>
        );
      })}
    </div>
  );
};

const PriceChart: React.FC<PriceChartProps> = ({
  title,
  points,
  predictedBestPrice
}) => {
  return (
    <div className="buywise-chart-card">
      <div className="buywise-section-title">{title}</div>

      <div className="buywise-chart-wrap">
        <ResponsiveContainer width="100%" height={320}>
          <AreaChart
            data={points}
            margin={{ top: 16, right: 16, bottom: 16, left: 0 }}
          >
            <defs>
              <linearGradient id="buywiseAreaFill" x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%" stopColor="#5ea55f" stopOpacity={0.38} />
                <stop offset="100%" stopColor="#5ea55f" stopOpacity={0.08} />
              </linearGradient>
            </defs>

            <CartesianGrid stroke="#d7ddd5" vertical={false} />
            <XAxis
              dataKey="label"
              tick={{ fontSize: 13, fill: "#696f69" }}
              axisLine={{ stroke: "#8d938c" }}
              tickLine={false}
            />
            <YAxis
              tickFormatter={formatMoney}
              tick={{ fontSize: 13, fill: "#696f69" }}
              axisLine={{ stroke: "#8d938c" }}
              tickLine={false}
              domain={["dataMin - 10", "dataMax + 10"]}
            />
            <Tooltip content={<CustomTooltip />} />

            <Area
              type="monotone"
              dataKey="actual"
              stroke="#2f7d32"
              fill="url(#buywiseAreaFill)"
              strokeWidth={4}
              connectNulls
            />
            <Line
              type="monotone"
              dataKey="actual"
              stroke="#2f7d32"
              strokeWidth={4}
              dot={{ r: 4, fill: "#ffffff", stroke: "#2f7d32", strokeWidth: 2 }}
              activeDot={{ r: 6 }}
              connectNulls
            />
            <Line
              type="monotone"
              dataKey="predicted"
              stroke="#2f7d32"
              strokeWidth={4}
              strokeDasharray="6 6"
              dot={{ r: 4, fill: "#ffffff", stroke: "#2f7d32", strokeWidth: 2 }}
              activeDot={{ r: 6 }}
              connectNulls
              label={({ x, y, value }: { x?: number; y?: number; value?: number }) => {
                if (
                  value !== predictedBestPrice ||
                  typeof x !== "number" ||
                  typeof y !== "number"
                ) {
                  return <g />;
                }

                return (
                  <g>
                    <rect
                      x={x - 26}
                      y={y - 46}
                      width="52"
                      height="32"
                      rx="6"
                      fill="#1f5e22"
                    />
                    <text
                      x={x}
                      y={y - 24}
                      textAnchor="middle"
                      fill="#ffffff"
                      fontSize="14"
                      fontWeight="700"
                    >
                      {`$${Math.round(predictedBestPrice)}`}
                    </text>
                  </g>
                );
              }}
            />
          </AreaChart>
        </ResponsiveContainer>
      </div>

      <div className="buywise-chart-footer">
        <span>0w</span>
        <div className="buywise-chart-timeline">
          <div className="buywise-chart-timeline-line" />
          <div className="buywise-chart-timeline-dot" />
          <span className="buywise-chart-timeline-label">Prediction Time</span>
        </div>
        <span>4w</span>
      </div>
    </div>
  );
};

export default PriceChart;
