import { useState, useRef } from "react";

const API_URL = "http://localhost:8000/predict";

type Status = "idle" | "loading" | "success" | "error";

interface FormState {
  recency: string;
  frequency: string;
  monetary: string;
  segment: "UK" | "Global";
  horizon: "30" | "60" | "90";
}

const INITIAL_FORM: FormState = {
  recency: "",
  frequency: "",
  monetary: "",
  segment: "UK",
  horizon: "30",
};

interface PredictionResult {
  predicted_spend: number;
  horizon_days: number;
  label: string;
}

function Spinner() {
  return (
    <svg
      className="animate-spin h-5 w-5 text-slate-900"
      xmlns="http://www.w3.org/2000/svg"
      fill="none"
      viewBox="0 0 24 24"
    >
      <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
      <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v8z" />
    </svg>
  );
}

function MetricBadge({ label, value, color }: { label: string; value: string; color: string }) {
  return (
    <div
      className={`flex flex-col items-center px-4 py-2 rounded-xl border ${color} bg-opacity-10 backdrop-blur-sm`}
    >
      <span className="text-xs font-semibold uppercase tracking-widest opacity-70 mb-0.5">{label}</span>
      <span className="text-sm font-bold">{value}</span>
    </div>
  );
}

function InputField({
  label,
  name,
  value,
  onChange,
  placeholder,
  prefix,
}: {
  label: string;
  name: string;
  value: string;
  onChange: (e: React.ChangeEvent<HTMLInputElement>) => void;
  placeholder: string;
  prefix?: string;
}) {
  return (
    <div>
      <label className="block text-xs font-bold uppercase tracking-widest text-slate-400 mb-2">
        {label}
      </label>
      <div className="relative">
        {prefix && (
          <span className="absolute left-4 top-1/2 -translate-y-1/2 text-slate-400 font-bold select-none">
            {prefix}
          </span>
        )}
        <input
          type="number"
          name={name}
          value={value}
          onChange={onChange}
          placeholder={placeholder}
          min={0}
          className={`w-full bg-slate-900/50 border border-slate-700 rounded-xl py-3 text-white
            focus:border-emerald-500 focus:ring-1 focus:ring-emerald-500 outline-none transition-all
            ${prefix ? "pl-8 pr-4" : "px-4"}`}
        />
      </div>
    </div>
  );
}

export default function App() {
  const [form, setForm] = useState<FormState>(INITIAL_FORM);
  const [status, setStatus] = useState<Status>("idle");
  const [result, setResult] = useState<PredictionResult | null>(null);
  const [error, setError] = useState<string>("");
  const resultRef = useRef<HTMLDivElement>(null);

  const handleChange = (
    e: React.ChangeEvent<HTMLInputElement | HTMLSelectElement>
  ) => {
    const { name, value } = e.target;
    setForm((prev) => ({ ...prev, [name]: value }));
    // Reset result when inputs change
    if (status === "success") {
      setStatus("idle");
      setResult(null);
    }
  };

  const handlePredict = async () => {
    const recency = parseInt(form.recency);
    const frequency = parseInt(form.frequency);
    const monetary = parseFloat(form.monetary);

    if (isNaN(recency) || isNaN(frequency) || isNaN(monetary)) {
      setError("Please fill in all numeric fields before predicting.");
      setStatus("error");
      return;
    }
    if (recency < 0 || frequency < 1 || monetary < 0) {
      setError("Recency ≥ 0, Frequency ≥ 1, Monetary ≥ 0 are required.");
      setStatus("error");
      return;
    }

    setStatus("loading");
    setError("");
    setResult(null);

    try {
      const res = await fetch(API_URL, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          recency,
          frequency,
          monetary,
          is_uk: form.segment === "UK" ? 1 : 0,
          horizon: parseInt(form.horizon),
        }),
      });

      if (!res.ok) {
        const detail = await res.json().catch(() => ({}));
        throw new Error(detail?.detail || `Server error ${res.status}`);
      }

      const data: PredictionResult = await res.json();
      setResult(data);
      setStatus("success");

      setTimeout(() => {
        resultRef.current?.scrollIntoView({ behavior: "smooth", block: "center" });
      }, 100);
    } catch (err: unknown) {
      const msg =
        err instanceof TypeError && err.message.includes("fetch")
          ? "Cannot reach the API. Make sure FastAPI is running on localhost:8000."
          : err instanceof Error
          ? err.message
          : "Unexpected error occurred.";
      setError(msg);
      setStatus("error");
    }
  };

  const handleReset = () => {
    setForm(INITIAL_FORM);
    setStatus("idle");
    setResult(null);
    setError("");
  };

  const horizonColors: Record<string, string> = {
    "30": "text-emerald-400",
    "60": "text-sky-400",
    "90": "text-violet-400",
  };
  const horizonGlows: Record<string, string> = {
    "30": "rgba(52,211,153,0.5)",
    "60": "rgba(56,189,248,0.5)",
    "90": "rgba(167,139,250,0.5)",
  };

  return (
    <div
      className="min-h-screen flex flex-col items-center justify-center px-4 py-12 text-slate-200"
      style={{
        background: "radial-gradient(circle at top right, #1e1e38 0%, #0b0f19 100%)",
        fontFamily: "'Inter', sans-serif",
      }}
    >
      {/* ── Header ── */}
      <div className="text-center mb-10 max-w-2xl">
        <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full mb-6 border border-emerald-500/30 bg-emerald-500/10 text-emerald-400 text-xs font-bold tracking-widest uppercase">
          Live AI Inference
        </div>
        <h1 className="text-4xl md:text-5xl font-extrabold text-white mb-4 tracking-tight">
          Customer Lifetime Value Engine
        </h1>
        <p className="text-slate-400 text-lg">
          Predict short-term future spend using Random Forest regression based on historical RFM segments.
        </p>
      </div>

      {/* ── Card ── */}
      <div className="bg-white/5 border border-white/10 backdrop-blur-xl rounded-3xl p-8 w-full max-w-xl shadow-2xl">
        <div className="flex gap-3 justify-center mb-8 flex-wrap">
          <MetricBadge label="Algorithm" value="Random Forest" color="border-indigo-500/50 text-indigo-300" />
          <MetricBadge label="Data" value="Online Retail II" color="border-sky-500/50 text-sky-300" />
          <MetricBadge label="Features" value="RFM + Geo" color="border-emerald-500/50 text-emerald-300" />
        </div>

        {/* ── Inputs ── */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-5 mb-5">
          <InputField
            label="Recency (Days Since Last Order)"
            name="recency"
            value={form.recency}
            onChange={handleChange}
            placeholder="e.g. 14"
          />
          <InputField
            label="Frequency (Total Past Orders)"
            name="frequency"
            value={form.frequency}
            onChange={handleChange}
            placeholder="e.g. 5"
          />
          <div className="md:col-span-2">
            <InputField
              label="Monetary (Total Past Spend)"
              name="monetary"
              value={form.monetary}
              onChange={handleChange}
              placeholder="e.g. 350.00"
              prefix="$"
            />
          </div>
        </div>

        {/* ── Selects ── */}
        <div className="grid grid-cols-2 gap-4 mb-8">
          <div>
            <label className="block text-xs font-bold uppercase tracking-widest text-slate-400 mb-2">
              Customer Segment
            </label>
            <select
              name="segment"
              value={form.segment}
              onChange={handleChange}
              className="w-full bg-slate-900/50 border border-slate-700 rounded-xl px-4 py-3 text-white outline-none focus:border-indigo-500 cursor-pointer"
            >
              <option value="UK">🇬🇧 UK Market</option>
              <option value="Global">🌍 Global Market</option>
            </select>
          </div>
          <div>
            <label className="block text-xs font-bold uppercase tracking-widest text-slate-400 mb-2">
              Prediction Horizon
            </label>
            <select
              name="horizon"
              value={form.horizon}
              onChange={handleChange}
              className="w-full bg-slate-900/50 border border-slate-700 rounded-xl px-4 py-3 text-white outline-none focus:border-indigo-500 cursor-pointer"
            >
              <option value="30">Next 30 Days</option>
              <option value="60">Next 60 Days</option>
              <option value="90">Next 90 Days</option>
            </select>
          </div>
        </div>

        {/* ── Error ── */}
        {status === "error" && (
          <div className="mb-6 p-4 bg-red-500/10 border border-red-500/20 rounded-xl text-red-400 text-sm">
            <span className="font-bold">Error: </span>
            {error}
          </div>
        )}

        {/* ── Predict button ── */}
        <button
          onClick={handlePredict}
          disabled={status === "loading"}
          className="w-full bg-emerald-400 hover:bg-emerald-300 disabled:opacity-50 disabled:cursor-not-allowed
            text-slate-900 font-bold text-lg py-4 rounded-xl transition-all flex justify-center items-center
            shadow-[0_0_20px_rgba(16,185,129,0.3)] hover:shadow-[0_0_30px_rgba(16,185,129,0.5)]"
        >
          {status === "loading" ? (
            <>
              <Spinner />
              <span className="ml-2">Predicting…</span>
            </>
          ) : (
            `Predict ${form.horizon}-Day Spend →`
          )}
        </button>

        {/* ── Result ── */}
        {status === "success" && result !== null && (
          <div
            ref={resultRef}
            className="mt-8 p-6 rounded-2xl text-center"
            style={{
              background: "rgba(255,255,255,0.04)",
              border: "1px solid rgba(255,255,255,0.1)",
              animation: "fadeUp 0.45s ease-out forwards",
            }}
          >
            <style>{`
              @keyframes fadeUp {
                from { opacity: 0; transform: translateY(14px); }
                to   { opacity: 1; transform: translateY(0); }
              }
            `}</style>

            {/* Horizon pill */}
            <span
              className="inline-block px-3 py-1 rounded-full text-xs font-bold uppercase tracking-widest mb-3"
              style={{
                background: `${horizonGlows[form.horizon]}22`,
                border: `1px solid ${horizonGlows[form.horizon]}55`,
                color: `${horizonGlows[form.horizon]}`,
              }}
            >
              {result.label}
            </span>

            {/* Dollar amount */}
            <p
              className={`text-5xl font-black tracking-tight ${horizonColors[form.horizon]}`}
              style={{
                textShadow: `0 0 20px ${horizonGlows[form.horizon]}`,
              }}
            >
              $
              {result.predicted_spend.toLocaleString("en-US", {
                minimumFractionDigits: 2,
                maximumFractionDigits: 2,
              })}
            </p>

            {/* Summary */}
            <p className="text-xs text-slate-500 mt-3 leading-relaxed">
              R={form.recency}d · F={form.frequency} orders · M=${parseFloat(form.monetary).toFixed(2)} ·{" "}
              {form.segment === "UK" ? "🇬🇧 UK" : "🌍 Global"} · {result.horizon_days}-day window
            </p>

            <button
              onClick={handleReset}
              className="mt-4 text-xs text-slate-500 hover:text-slate-300 underline underline-offset-2 transition-colors"
            >
              ← Reset
            </button>
          </div>
        )}
      </div>

      <p className="text-slate-600 text-xs mt-8">
        Trained on UCI Online Retail II · Random Forest Regressor
      </p>
    </div>
  );
}