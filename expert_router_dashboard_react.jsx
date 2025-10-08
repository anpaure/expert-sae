import React, { useMemo, useRef, useState } from "react";
import { Card, CardContent } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Slider } from "@/components/ui/slider";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer } from "recharts";
import { Upload, Search, Download, Filter, Gauge, Layers, List, Sparkles } from "lucide-react";

// ---- Types expected from the exporter JSON ----
// {
//   meta: { layer: number, n_experts: number, keep: number, samples: number },
//   experts: Array<{
//     id: number,
//     stats: { count: number, meanTopP: number, meanEntropy: number },
//     top: Array<{ p: number, prompt: string, generated: string, token?: number }>
//   }>
// }

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
  const [data, setData] = useState<DashboardData | null>(null);
  const [expertId, setExpertId] = useState<number | null>(null);
  const [q, setQ] = useState("");
  const [minProb, setMinProb] = useState(0);
  const [sortBy, setSortBy] = useState<"count" | "meanTopP">("count");
  const fileRef = useRef<HTMLInputElement | null>(null);

  const onPickFile = () => fileRef.current?.click();
  const onFile = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const f = e.target.files?.[0];
    if (!f) return;
    const text = await f.text();
    try {
      const obj = JSON.parse(text) as DashboardData;
      setData(obj);
      setExpertId(0);
    } catch (err) {
      alert("Invalid JSON file");
    }
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
    return current.top.filter(h => h.p >= minProb && (
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

  return (
    <div className="min-h-screen bg-neutral-50 text-neutral-900">
      <header className="sticky top-0 z-10 bg-white border-b border-neutral-200">
        <div className="max-w-7xl mx-auto px-4 py-3 flex items-center gap-3">
          <Sparkles className="w-5 h-5"/>
          <h1 className="text-lg font-semibold">Expert Router Dashboard</h1>
          <div className="flex-1" />
          <Button variant="secondary" onClick={onPickFile} className="gap-2"><Upload className="w-4 h-4"/>Load JSON</Button>
          <input ref={fileRef} type="file" accept="application/json" className="hidden" onChange={onFile}/>
        </div>
      </header>

      <main className="max-w-7xl mx-auto px-4 py-4 grid grid-cols-12 gap-4">
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
      </main>
    </div>
  );
}
