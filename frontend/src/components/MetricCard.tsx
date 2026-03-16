interface MetricCardProps {
  label: string;
  value: string;
  color?: "default" | "profit" | "loss" | "neutral";
}

export default function MetricCard({
  label,
  value,
  color = "default",
}: MetricCardProps) {
  const colorClasses = {
    default: "text-white",
    profit: "text-profit",
    loss: "text-loss",
    neutral: "text-yellow-400",
  };

  return (
    <div className="bg-surface rounded-lg p-4 border border-border">
      <p className="text-xs text-slate-500 uppercase tracking-wider mb-1">
        {label}
      </p>
      <p className={`text-xl font-bold ${colorClasses[color]}`}>{value}</p>
    </div>
  );
}
