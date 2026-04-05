import React, { useMemo, useState } from "react";
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

type RangeOption = "2W" | "1M" | "ALL";
type AxisMode = "price" | "change";

type ChartDatum = {
  label: string;
  actual?: number;
  predicted?: number;
  actualDelta?: number;
  predictedDelta?: number;
};

const RANGE_OPTIONS: RangeOption[] = ["2W", "1M", "ALL"];

const formatMoney = (value: number): string => `$${Math.round(value)}`;
const formatDelta = (value: number): string => `${value > 0 ? "+" : ""}${value.toFixed(1)}%`;

const firstForecastOnlyIndex = (points: PricePoint[]): number => {
  const idx = points.findIndex(
    (p) => typeof p.predicted === "number" && typeof p.actual !== "number"
  );
  return idx === -1 ? points.length : idx;
};

const clampRange = (points: PricePoint[], range: RangeOption): PricePoint[] => {
  if (range === "ALL" || points.length === 0) {
    return points;
  }

  const split = firstForecastOnlyIndex(points);
  const history = points.slice(0, split);
  const forecastTail = points.slice(split);

  if (range === "1M") {
    return points;
  }

  if (range === "2W") {
    const window = 15;
    const histShown = history.slice(Math.max(0, history.length - window));
    return [...histShown, ...forecastTail];
  }

  return points;
};

const toChartData = (points: PricePoint[], mode: AxisMode): ChartDatum[] => {
  const baseline = points.find((point) => typeof point.actual === "number" || typeof point.predicted === "number");
  const baselineValue = baseline?.actual ?? baseline?.predicted ?? 1;

  return points.map((point) => {
    const actualDelta =
      typeof point.actual === "number" ? ((point.actual - baselineValue) / baselineValue) * 100 : undefined;
    const predictedDelta =
      typeof point.predicted === "number"
        ? ((point.predicted - baselineValue) / baselineValue) * 100
        : undefined;

    if (mode === "change") {
      return {
        label: point.label,
        actual: undefined,
        predicted: undefined,
        actualDelta,
        predictedDelta
      };
    }

    return {
      label: point.label,
      actual: point.actual,
      predicted: point.predicted,
      actualDelta,
      predictedDelta
    };
  });
};

const CustomTooltip = ({
  active,
  payload,
  label,
  mode
}: {
  active?: boolean;
  payload?: Array<{ value?: number; dataKey?: string }>;
  label?: string;
  mode: AxisMode;
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

        const lineLabel = item.dataKey?.includes("actual") ? "Observed" : "Predicted";

        return (
          <div key={`${item.dataKey}-${item.value}`} className="buywise-tooltip-row">
            {lineLabel}: {mode === "price" ? formatMoney(item.value) : formatDelta(item.value)}
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
  const [range, setRange] = useState<RangeOption>("1M");
  const [axisMode, setAxisMode] = useState<AxisMode>("price");

  const visiblePoints = useMemo(() => clampRange(points, range), [points, range]);
  const chartData = useMemo(() => toChartData(visiblePoints, axisMode), [visiblePoints, axisMode]);

  const predictedLineKey = axisMode === "price" ? "predicted" : "predictedDelta";
  const actualLineKey = axisMode === "price" ? "actual" : "actualDelta";
  const bestPredictionValue =
    axisMode === "price"
      ? predictedBestPrice
      : (() => {
          const baseline = visiblePoints.find((point) => typeof point.actual === "number" || typeof point.predicted === "number");
          const baselineValue = baseline?.actual ?? baseline?.predicted ?? 1;
          return ((predictedBestPrice - baselineValue) / baselineValue) * 100;
        })();

  const yFormatter = axisMode === "price" ? formatMoney : formatDelta;

  return (
    <div className="buywise-chart-card buywise-animate-in-up">
      <div className="buywise-chart-header">
        <div>
          <div className="buywise-section-title">{title}</div>
          <div className="buywise-section-subtitle">Compare recent movement with the current forecast.</div>
        </div>

        <div className="buywise-chart-controls">
          <div className="buywise-segmented-control" role="tablist" aria-label="Time range">
            {RANGE_OPTIONS.map((option) => (
              <button
                key={option}
                type="button"
                className={`buywise-segmented-control__button ${range === option ? "is-active" : ""}`}
                onClick={() => setRange(option)}
              >
                {option}
              </button>
            ))}
          </div>

          <div className="buywise-segmented-control" role="tablist" aria-label="Chart units">
            <button
              type="button"
              className={`buywise-segmented-control__button ${axisMode === "price" ? "is-active" : ""}`}
              onClick={() => setAxisMode("price")}
            >
              Price
            </button>
            <button
              type="button"
              className={`buywise-segmented-control__button ${axisMode === "change" ? "is-active" : ""}`}
              onClick={() => setAxisMode("change")}
            >
              % change
            </button>
          </div>
        </div>
      </div>

      <div className="buywise-chart-wrap">
        <ResponsiveContainer width="100%" height={250}>
          <AreaChart
            data={chartData}
            margin={{ top: 16, right: 16, bottom: 16, left: 0 }}
          >
            <defs>
              <linearGradient id="buywiseAreaFill" x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%" stopColor="#5ea55f" stopOpacity={0.34} />
                <stop offset="100%" stopColor="#5ea55f" stopOpacity={0.05} />
              </linearGradient>
            </defs>

            <CartesianGrid stroke="#d7ddd5" vertical={false} />
            <XAxis
              dataKey="label"
              tick={{ fontSize: 12, fill: "#667164" }}
              axisLine={{ stroke: "#c7d0c4" }}
              tickLine={false}
            />
            <YAxis
              tickFormatter={yFormatter}
              tick={{ fontSize: 12, fill: "#667164" }}
              axisLine={{ stroke: "#c7d0c4" }}
              tickLine={false}
              width={54}
              domain={axisMode === "price" ? ["dataMin - 8", "dataMax + 8"] : ["auto", "auto"]}
            />
            <Tooltip content={<CustomTooltip mode={axisMode} />} />

            <Area
              type="monotone"
              dataKey={actualLineKey}
              stroke="#2f7d32"
              fill="url(#buywiseAreaFill)"
              strokeWidth={3}
              connectNulls
              animationDuration={800}
            />
            <Line
              type="monotone"
              dataKey={actualLineKey}
              stroke="#2f7d32"
              strokeWidth={3}
              dot={{ r: 3, fill: "#ffffff", stroke: "#2f7d32", strokeWidth: 2 }}
              activeDot={{ r: 5 }}
              connectNulls
              animationDuration={900}
            />
            <Line
              type="monotone"
              dataKey={predictedLineKey}
              stroke="#1d6b25"
              strokeWidth={3}
              strokeDasharray="6 6"
              dot={{ r: 3, fill: "#ffffff", stroke: "#1d6b25", strokeWidth: 2 }}
              activeDot={{ r: 5 }}
              connectNulls
              animationDuration={1050}
              label={({ x, y, value }: { x?: number; y?: number; value?: number }) => {
                if (
                  typeof value !== "number" ||
                  Math.abs(value - bestPredictionValue) > 0.001 ||
                  typeof x !== "number" ||
                  typeof y !== "number"
                ) {
                  return <g />;
                }

                const pillLabel = axisMode === "price" ? formatMoney(predictedBestPrice) : formatDelta(bestPredictionValue);

                return (
                  <g>
                    <rect
                      x={x - 28}
                      y={y - 42}
                      width="56"
                      height="28"
                      rx="8"
                      fill="#1f5e22"
                    />
                    <text
                      x={x}
                      y={y - 24}
                      textAnchor="middle"
                      fill="#ffffff"
                      fontSize="12"
                      fontWeight="700"
                    >
                      {pillLabel}
                    </text>
                  </g>
                );
              }}
            />
          </AreaChart>
        </ResponsiveContainer>
      </div>

      <div className="buywise-chart-footer">
        <span>{range === "2W" ? "2w view" : range === "1M" ? "1m view" : "full history"}</span>
        <div className="buywise-chart-timeline">
          <div className="buywise-chart-timeline-line" />
          <div className="buywise-chart-timeline-dot" />
          <span className="buywise-chart-timeline-label">Forecast start</span>
        </div>
        <span>{axisMode === "price" ? "absolute price" : "relative to first point"}</span>
      </div>
    </div>
  );
};

export default PriceChart;
