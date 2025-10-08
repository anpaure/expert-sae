"use client";

import React, { useMemo, useRef, useState } from "react";
import { Card, CardContent } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Slider } from "@/components/ui/slider";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer } from "recharts";
import { Upload, Search, Download, Filter, Gauge, Layers, List, Sparkles } from "lucide-react";

// ---- Types expected from the exporter JSON ----
type ExpertHit = { p: number; prompt: string; generated: string; token?: number };

type ExpertEntry = {
  id: number;
  stats: { count: number; meanTopP: number; meanEntropy: number };
  top: ExpertHit[];
};

type DashboardData = {
  meta: { layer: number; n_experts: number; keep: number; samples: number };
  experts: ExpertEntry[];
};

// ---- Types for SAE feature viewer ----
type TokenWithScore = {
  text: string;
  score: number;
};

type FeatureActivation = {
  prompt: string;
  generated: string;
  tokens: TokenWithScore[];
  max_score: number;
  layer: number;
};

type FeatureData = {
  meta: { feature_id: number; d_sae: number; samples: number; k_keep: number; checkpoint: string };
  activations: FeatureActivation[];
};

// ---- Small helpers ----
function clip(s: string, n = 200) {
  if (!s) return "";
  return s.length > n ? s.slice(0, n) + "…" : s;
}

function toCSV(rows: Array<Record<string, any>>) {
  if (!rows.length) return "";
  const headers = Object.keys(rows[0]);
  const esc = (v: any) => {
    const s = String(v ?? "");
    if (s.includes(",") || s.includes("\n") || s.includes('"')) {
      return '"' + s.replace(/"/g, '""') + '"';
    }
    return s;
  };
  const out = [headers.join(",")].concat(rows.map(r => headers.map(h => esc(r[h])).join(",")));
  return out.join("\n");
}

export default function ExpertRouterDashboard() {
  // Router data state
  const [data, setData] = useState<DashboardData | null>(null);
  const [expertId, setExpertId] = useState<number | null>(null);
  const [q, setQ] = useState("");
  const [minProb, setMinProb] = useState(0);
  const [sortBy, setSortBy] = useState<"count" | "meanTopP">("count");
  const fileRef = useRef<HTMLInputElement | null>(null);

  // Feature viewer state
  const [featureData, setFeatureData] = useState<FeatureData | null>(null);
  const [allFeatures, setAllFeatures] = useState<FeatureData[]>([]);
  const [selectedFeatureIdx, setSelectedFeatureIdx] = useState<number>(0);
  const [minScore, setMinScore] = useState(0);
  const [viewMode, setViewMode] = useState<"router" | "features">("router");
  const featureFileRef = useRef<HTMLInputElement | null>(null);

  const onPickFile = () => fileRef.current?.click();
  const onFile = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const f = e.target.files?.[0];
    if (!f) return;
    const text = await f.text();
    try {
      const obj = JSON.parse(text) as DashboardData;
      setData(obj);
      setExpertId(0);
      setViewMode("router");
    } catch (err) {
      alert("Invalid JSON file");
    }
  };

  const onPickFeatureFile = () => featureFileRef.current?.click();
  const onFeatureFile = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files;
    if (!files || files.length === 0) return;

    const loadedFeatures: FeatureData[] = [];

    for (let i = 0; i < files.length; i++) {
      const f = files[i];
      try {
        const text = await f.text();
        const obj = JSON.parse(text) as FeatureData;
        loadedFeatures.push(obj);
      } catch (err) {
        console.error(`Failed to load ${f.name}:`, err);
      }
    }

    if (loadedFeatures.length === 0) {
      alert("No valid feature JSON files loaded");
      return;
    }

    // Sort by feature_id
    loadedFeatures.sort((a, b) => a.meta.feature_id - b.meta.feature_id);

    setAllFeatures(loadedFeatures);
    setSelectedFeatureIdx(0);
    setFeatureData(loadedFeatures[0]);
    setViewMode("features");
  };

  const expertsSorted = useMemo(() => {
    if (!data) return [] as ExpertEntry[];
    return [...data.experts].sort((a, b) => {
      if (sortBy === "count") return (b.stats?.count ?? 0) - (a.stats?.count ?? 0);
      return (b.stats?.meanTopP ?? 0) - (a.stats?.meanTopP ?? 0);
    });
  }, [data, sortBy]);

  const current = useMemo(() => {
    if (!data || expertId == null) return null;
    return data.experts.find(e => e.id === expertId) ?? null;
  }, [data, expertId]);

  const filteredHits = useMemo(() => {
    if (!current) return [] as ExpertHit[];
    const ql = q.trim().toLowerCase();
    return current.top.filter(h => h.p >= (minProb / 100) && (
      !ql || h.prompt.toLowerCase().includes(ql) || h.generated.toLowerCase().includes(ql)
    ));
  }, [current, q, minProb]);

  const histData = useMemo(() => {
    if (!current) return [] as { bin: string; count: number }[];
    const bins = new Array(10).fill(0);
    for (const h of current.top) {
      const idx = Math.max(0, Math.min(9, Math.floor(h.p * 10)));
      bins[idx]++;
    }
    return bins.map((c, i) => ({ bin: `${(i/10).toFixed(1)}–${((i+1)/10).toFixed(1)}`, count: c }));
  }, [current]);

  const exportCSV = () => {
    if (!current) return;
    const rows = filteredHits.map((h, idx) => ({
      rank: idx + 1,
      expert: current.id,
      p: h.p.toFixed(6),
      token: h.token ?? "",
      prompt: h.prompt,
      generated: h.generated,
    }));
    const csv = toCSV(rows);
    const blob = new Blob([csv], { type: "text/csv" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `expert_${current.id}_contexts.csv`;
    a.click();
    URL.revokeObjectURL(url);
  };

  // Feature viewer helpers
  const filteredActivations = useMemo(() => {
    if (!featureData) return [] as FeatureActivation[];
    return featureData.activations.filter(a => a.max_score >= minScore);
  }, [featureData, minScore]);

  const getColorForScore = (score: number, maxScore: number) => {
    if (score <= 0) return "transparent";
    const intensity = Math.min(1, score / (maxScore * 0.8)); // cap at 0.8 for readability
    const r = Math.round(255 * intensity);
    const g = Math.round(255 * (1 - intensity * 0.5));
    const b = Math.round(255 * (1 - intensity));
    return `rgb(${r}, ${g}, ${b})`;
  };

  // Render actual tokens with their scores
  const renderColoredTokens = (tokens: TokenWithScore[], maxScore: number) => {
    if (!tokens || tokens.length === 0) return null;

    return (
      <span>
        {tokens.map((token, idx) => {
          const bgColor = getColorForScore(token.score, maxScore);
          return (
            <span
              key={idx}
              style={{ backgroundColor: bgColor }}
              className={token.score > 0 ? "px-0.5 rounded" : ""}
              title={token.score > 0 ? `Score: ${token.score.toFixed(4)}` : undefined}
            >
              {token.text}
            </span>
          );
        })}
      </span>
    );
  };

  return (
    <div className="min-h-screen bg-neutral-50 text-neutral-900">
      <header className="sticky top-0 z-10 bg-white border-b border-neutral-200">
        <div className="max-w-7xl mx-auto px-4 py-3 flex items-center gap-3">
          <Sparkles className="w-5 h-5"/>
          <h1 className="text-lg font-semibold">Expert Router Dashboard</h1>
          <div className="flex-1" />
          <div className="flex gap-2">
            <Button
              variant={viewMode === "router" ? "default" : "outline"}
              onClick={() => setViewMode("router")}
              className="gap-2"
            >
              <Layers className="w-4 h-4"/>Router
            </Button>
            <Button
              variant={viewMode === "features" ? "default" : "outline"}
              onClick={() => setViewMode("features")}
              className="gap-2"
            >
              <Sparkles className="w-4 h-4"/>Features
            </Button>
          </div>
          {viewMode === "router" && (
            <Button variant="secondary" onClick={onPickFile} className="gap-2"><Upload className="w-4 h-4"/>Load Router</Button>
          )}
          {viewMode === "features" && (
            <Button variant="secondary" onClick={onPickFeatureFile} className="gap-2"><Upload className="w-4 h-4"/>Load Feature</Button>
          )}
          <input ref={fileRef} type="file" accept="application/json" className="hidden" onChange={onFile}/>
          <input ref={featureFileRef} type="file" accept="application/json" multiple className="hidden" onChange={onFeatureFile}/>
        </div>
      </header>

      <main className="max-w-7xl mx-auto px-4 py-4 grid grid-cols-12 gap-4">
        {viewMode === "router" && (
          <>
            {/* Sidebar */}
            <aside className="col-span-12 lg:col-span-4">
              <Card className="shadow-sm">
                <CardContent className="p-4 space-y-4">
                  <div className="flex items-center gap-2 text-neutral-700">
                    <Layers className="w-4 h-4"/>
                    <div className="text-sm">{data ? (<>
                      <span className="font-medium">Layer {data.meta.layer}</span>
                      <span className="mx-2">•</span>
                      <span>{data.meta.n_experts} experts</span>
                      <span className="mx-2">•</span>
                      <span>{data.meta.samples} tokens</span>
                    </>) : (<span>Load a JSON export to begin</span>)}</div>
                  </div>

                  <div className="flex items-center gap-2">
                    <Filter className="w-4 h-4"/>
                    <Select value={sortBy} onValueChange={v => setSortBy(v as any)}>
                      <SelectTrigger className="h-9 w-48">
                        <SelectValue placeholder="Sort" />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="count">Sort by count</SelectItem>
                        <SelectItem value="meanTopP">Sort by mean top‑p</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>

                  <div className="space-y-2 max-h-[55vh] overflow-auto pr-1">
                    {expertsSorted.map((e) => (
                      <button key={e.id}
                              onClick={() => setExpertId(e.id)}
                              className={`w-full text-left rounded-xl p-3 border transition ${expertId===e.id?"border-neutral-900 bg-neutral-100":"border-neutral-200 hover:bg-neutral-50"}`}>
                        <div className="flex items-center justify-between">
                          <div className="font-medium">Expert {e.id}</div>
                          <div className="text-xs text-neutral-500">n={e.stats?.count ?? 0}</div>
                        </div>
                        <div className="text-xs text-neutral-600">mean top‑p ≈ {(e.stats?.meanTopP ?? 0).toFixed(3)} • H ≈ {(e.stats?.meanEntropy ?? 0).toFixed(2)}</div>
                      </button>
                    ))}
                  </div>
                </CardContent>
              </Card>
            </aside>

        {/* Main panel */}
        <section className="col-span-12 lg:col-span-8 space-y-4">
          <Card className="shadow-sm">
            <CardContent className="p-4 space-y-3">
              <div className="flex items-center gap-3">
                <List className="w-4 h-4"/>
                <div className="font-semibold">{current ? `Expert ${current.id}` : "No expert selected"}</div>
                <div className="flex-1"/>
                <Button variant="outline" className="gap-2" onClick={exportCSV} disabled={!current}><Download className="w-4 h-4"/>Export CSV</Button>
              </div>

              {current && (
                <div className="grid grid-cols-12 gap-3">
                  <div className="col-span-12 sm:col-span-4">
                    <Card className="border-neutral-200">
                      <CardContent className="p-3">
                        <div className="text-xs text-neutral-500"># contexts</div>
                        <div className="text-2xl font-semibold">{current.stats?.count ?? current.top.length}</div>
                      </CardContent>
                    </Card>
                  </div>
                  <div className="col-span-6 sm:col-span-4">
                    <Card className="border-neutral-200">
                      <CardContent className="p-3">
                        <div className="text-xs text-neutral-500">mean top‑p</div>
                        <div className="text-2xl font-semibold">{(current.stats?.meanTopP ?? 0).toFixed(3)}</div>
                      </CardContent>
                    </Card>
                  </div>
                  <div className="col-span-6 sm:col-span-4">
                    <Card className="border-neutral-200">
                      <CardContent className="p-3">
                        <div className="text-xs text-neutral-500">mean entropy</div>
                        <div className="text-2xl font-semibold">{(current.stats?.meanEntropy ?? 0).toFixed(2)}</div>
                      </CardContent>
                    </Card>
                  </div>
                </div>
              )}

              {current && (
                <div className="h-48">
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart data={histData}>
                      <XAxis dataKey="bin" tick={{ fontSize: 11 }} interval={0} angle={-20} textAnchor="end" />
                      <YAxis width={28} tick={{ fontSize: 11 }} />
                      <Tooltip />
                      <Bar dataKey="count" />
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              )}
            </CardContent>
          </Card>

          {/* Filters */}
          <Card className="shadow-sm">
            <CardContent className="p-4 space-y-3">
              <div className="flex flex-wrap items-center gap-3">
                <div className="flex items-center gap-2 w-full sm:w-auto">
                  <Search className="w-4 h-4"/>
                  <Input placeholder="Search prompt/generated…" value={q} onChange={e=>setQ(e.target.value)} className="h-9"/>
                </div>
                <div className="flex items-center gap-3">
                  <Gauge className="w-4 h-4"/>
                  <div className="text-xs text-neutral-600 w-24">min p ≥ {minProb/100}</div>
                  <Slider value={[minProb]} onValueChange={(v)=>setMinProb(v[0])} min={0} max={100} step={5} className="w-56"/>
                </div>
              </div>
            </CardContent>
          </Card>

          {/* Context table */}
          <Card className="shadow-sm">
            <CardContent className="p-0">
              <div className="overflow-auto">
                <table className="w-full text-sm">
                  <thead className="sticky top-0 bg-white border-b">
                    <tr className="text-left">
                      <th className="p-3 w-16">#</th>
                      <th className="p-3 w-20">p</th>
                      <th className="p-3">Prompt</th>
                      <th className="p-3">Generated</th>
                    </tr>
                  </thead>
                  <tbody>
                    {current && filteredHits.map((h, i) => (
                      <tr key={i} className="border-b hover:bg-neutral-50">
                        <td className="p-3 text-neutral-500">{i+1}</td>
                        <td className="p-3 font-mono">{h.p.toFixed(4)}</td>
                        <td className="p-3 align-top">{clip(h.prompt, 300)}</td>
                        <td className="p-3 align-top">{clip(h.generated, 300)}</td>
                      </tr>
                    ))}
                    {(!current || filteredHits.length===0) && (
                      <tr><td className="p-6 text-neutral-500" colSpan={4}>No rows — load a file, choose an expert, or relax filters.</td></tr>
                    )}
                  </tbody>
                </table>
              </div>
            </CardContent>
          </Card>
        </section>
          </>
        )}

        {viewMode === "features" && (
          <section className="col-span-12 space-y-4">
            <Card className="shadow-sm">
              <CardContent className="p-4 space-y-3">
                <div className="flex items-center gap-3">
                  <Sparkles className="w-4 h-4"/>
                  <div className="font-semibold">
                    {featureData ? `Feature ${featureData.meta.feature_id} / ${featureData.meta.d_sae}` : "No feature data loaded"}
                  </div>
                  <div className="flex-1"/>
                  {allFeatures.length > 1 && (
                    <Select
                      value={selectedFeatureIdx.toString()}
                      onValueChange={(v) => {
                        const idx = parseInt(v);
                        setSelectedFeatureIdx(idx);
                        setFeatureData(allFeatures[idx]);
                        setMinScore(0); // Reset filter when switching features
                      }}
                    >
                      <SelectTrigger className="h-9 w-48">
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        {allFeatures.map((f, idx) => (
                          <SelectItem key={idx} value={idx.toString()}>
                            Feature {f.meta.feature_id} ({f.activations.length} samples)
                          </SelectItem>
                        ))}
                      </SelectContent>
                    </Select>
                  )}
                </div>

                {featureData && (
                  <div className="grid grid-cols-12 gap-3">
                    <div className="col-span-6 sm:col-span-3">
                      <Card className="border-neutral-200">
                        <CardContent className="p-3">
                          <div className="text-xs text-neutral-500">Feature ID</div>
                          <div className="text-2xl font-semibold">{featureData.meta.feature_id}</div>
                        </CardContent>
                      </Card>
                    </div>
                    <div className="col-span-6 sm:col-span-3">
                      <Card className="border-neutral-200">
                        <CardContent className="p-3">
                          <div className="text-xs text-neutral-500">d_sae</div>
                          <div className="text-2xl font-semibold">{featureData.meta.d_sae}</div>
                        </CardContent>
                      </Card>
                    </div>
                    <div className="col-span-6 sm:col-span-3">
                      <Card className="border-neutral-200">
                        <CardContent className="p-3">
                          <div className="text-xs text-neutral-500">samples</div>
                          <div className="text-2xl font-semibold">{featureData.meta.samples}</div>
                        </CardContent>
                      </Card>
                    </div>
                    <div className="col-span-6 sm:col-span-3">
                      <Card className="border-neutral-200">
                        <CardContent className="p-3">
                          <div className="text-xs text-neutral-500">with activations</div>
                          <div className="text-2xl font-semibold">{featureData.activations.length}</div>
                        </CardContent>
                      </Card>
                    </div>
                  </div>
                )}

                {featureData && featureData.activations.length > 0 && (
                  <div className="flex items-center gap-3">
                    <Gauge className="w-4 h-4"/>
                    <div className="text-xs text-neutral-600 w-32">
                      min score ≥ {minScore.toFixed(2)}
                      <span className="text-neutral-400 ml-1">(max: {featureData.activations[0].max_score.toFixed(2)})</span>
                    </div>
                    <Slider
                      value={[minScore]}
                      onValueChange={(v) => setMinScore(v[0])}
                      min={0}
                      max={featureData.activations[0].max_score}
                      step={featureData.activations[0].max_score / 100}
                      className="flex-1"
                    />
                  </div>
                )}
              </CardContent>
            </Card>

            {/* Feature activation contexts */}
            <Card className="shadow-sm">
              <CardContent className="p-4 space-y-4">
                <div className="text-sm font-medium">Color-coded token activations (red = high, blue = low)</div>
                <div className="space-y-4 max-h-[70vh] overflow-auto">
                  {featureData && filteredActivations.map((a, i) => {
                    const globalMaxScore = featureData.activations[0]?.max_score ?? 1;
                    return (
                      <div key={i} className="border rounded-lg p-4 space-y-2 bg-white">
                        <div className="flex items-center justify-between text-xs text-neutral-600">
                          <span>Sample #{i + 1}</span>
                          <span className="font-mono font-semibold">max score: {a.max_score.toFixed(4)}</span>
                        </div>
                        <div className="space-y-1">
                          <div className="text-xs font-medium text-neutral-700">Prompt:</div>
                          <div className="text-sm bg-neutral-50 p-2 rounded leading-relaxed">
                            {a.prompt}
                          </div>
                        </div>
                        {a.tokens && a.tokens.length > 0 && (
                          <div className="space-y-1">
                            <div className="text-xs font-medium text-neutral-700">Generated (color-coded by token activation):</div>
                            <div className="text-sm bg-neutral-50 p-2 rounded leading-relaxed font-mono">
                              {renderColoredTokens(a.tokens, globalMaxScore)}
                            </div>
                          </div>
                        )}
                        <div className="text-xs text-neutral-500">
                          Layer {a.layer} • {a.tokens?.filter(t => t.score > 0).length || 0} active tokens
                        </div>
                      </div>
                    );
                  })}
                  {(!featureData || filteredActivations.length === 0) && (
                    <div className="p-6 text-neutral-500 text-center">
                      No activations — load a feature file or lower the min score filter.
                    </div>
                  )}
                </div>
              </CardContent>
            </Card>
          </section>
        )}
      </main>
    </div>
  );
}
