import { useState, useEffect, useRef, useCallback } from "react";

// ── API config ──────────────────────────────────────────
const API = "http://localhost:8000";

// ── Helpers ─────────────────────────────────────────────
const fmt = (d) => new Date(d).toLocaleTimeString();
const pct = (n) => `${(n * 100).toFixed(1)}%`;

const EMERGENCY_META = {
  fire:    { color: "#FF4500", icon: "🔥", label: "FIRE",    bg: "rgba(255,69,0,0.12)"    },
  medical: { color: "#FF6B35", icon: "🚑", label: "MEDICAL", bg: "rgba(255,107,53,0.12)"  },
  crime:   { color: "#E63946", icon: "🚔", label: "CRIME",   bg: "rgba(230,57,70,0.12)"   },
  normal:  { color: "#4CAF7D", icon: "✅", label: "NORMAL",  bg: "rgba(76,175,125,0.12)"  },
};

// ── CSS injected once ────────────────────────────────────
const GLOBAL_CSS = `
  @import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Rajdhani:wght@400;500;600;700&display=swap');

  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

  :root {
    --bg0: #0a0b0d;
    --bg1: #0f1114;
    --bg2: #161820;
    --bg3: #1e2130;
    --red:  #FF4500;
    --red2: #FF6B35;
    --amber:#FFB347;
    --green:#4CAF7D;
    --text: #E8EAF0;
    --muted:#6B7280;
    --border: rgba(255,69,0,0.18);
    --glow: 0 0 18px rgba(255,69,0,0.25);
    --font-mono: 'Share Tech Mono', monospace;
    --font-ui: 'Rajdhani', sans-serif;
  }

  body { background: var(--bg0); color: var(--text); font-family: var(--font-ui); }
  ::-webkit-scrollbar { width: 4px; }
  ::-webkit-scrollbar-track { background: var(--bg1); }
  ::-webkit-scrollbar-thumb { background: var(--red); border-radius: 2px; }

  @keyframes pulse-ring {
    0%   { box-shadow: 0 0 0 0 rgba(255,69,0,0.5); }
    70%  { box-shadow: 0 0 0 10px rgba(255,69,0,0); }
    100% { box-shadow: 0 0 0 0 rgba(255,69,0,0); }
  }
  @keyframes scan-line {
    0%   { top: 0; }
    100% { top: 100%; }
  }
  @keyframes blink { 0%,100% { opacity:1; } 50% { opacity:0; } }
  @keyframes slide-in {
    from { opacity:0; transform: translateY(10px); }
    to   { opacity:1; transform: translateY(0); }
  }
  @keyframes fade-in {
    from { opacity:0; }
    to   { opacity:1; }
  }
  @keyframes shimmer {
    0%   { background-position: -200% center; }
    100% { background-position:  200% center; }
  }
  @keyframes alert-flash {
    0%,100% { background: rgba(255,69,0,0.08); }
    50%     { background: rgba(255,69,0,0.22); }
  }
`;

// ── Sub-components ───────────────────────────────────────

function StatusDot({ online }) {
  return (
    <span style={{
      display: "inline-block", width: 8, height: 8, borderRadius: "50%",
      background: online ? "#4CAF7D" : "#E63946",
      animation: online ? "pulse-ring 2s infinite" : "blink 1s infinite",
      marginRight: 6, flexShrink: 0,
    }} />
  );
}

function StatCard({ label, value, color, icon, sub }) {
  return (
    <div style={{
      background: "linear-gradient(135deg, #161820 0%, #1a1d28 100%)",
      border: `1px solid ${color}33`,
      borderRadius: 8, padding: "14px 18px",
      position: "relative", overflow: "hidden",
      flex: 1, minWidth: 120,
    }}>
      <div style={{
        position: "absolute", top: 0, left: 0, right: 0, height: 2,
        background: `linear-gradient(90deg, transparent, ${color}, transparent)`,
      }} />
      <div style={{ fontSize: 22, marginBottom: 4 }}>{icon}</div>
      <div style={{
        fontFamily: "var(--font-mono)", fontSize: 28, fontWeight: 700,
        color, lineHeight: 1, marginBottom: 4,
      }}>{value}</div>
      <div style={{ fontSize: 11, color: "var(--muted)", fontWeight: 600, letterSpacing: "0.08em", textTransform: "uppercase" }}>{label}</div>
      {sub && <div style={{ fontSize: 10, color: "var(--muted)", marginTop: 2 }}>{sub}</div>}
    </div>
  );
}

function ConfidenceBar({ scores }) {
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 5, marginTop: 8 }}>
      {Object.entries(scores).sort((a,b) => b[1]-a[1]).map(([cls, score]) => {
        const meta = EMERGENCY_META[cls] || { color: "#888", label: cls.toUpperCase() };
        return (
          <div key={cls} style={{ display: "flex", alignItems: "center", gap: 8 }}>
            <div style={{ width: 56, fontSize: 10, color: meta.color, fontFamily: "var(--font-mono)", fontWeight: 700 }}>
              {meta.label}
            </div>
            <div style={{ flex: 1, height: 6, background: "var(--bg0)", borderRadius: 3, overflow: "hidden" }}>
              <div style={{
                height: "100%", width: `${score * 100}%`,
                background: `linear-gradient(90deg, ${meta.color}88, ${meta.color})`,
                borderRadius: 3, transition: "width 0.6s ease",
              }} />
            </div>
            <div style={{ width: 40, fontSize: 10, color: "var(--muted)", fontFamily: "var(--font-mono)", textAlign: "right" }}>
              {pct(score)}
            </div>
          </div>
        );
      })}
    </div>
  );
}

function ResultCard({ result }) {
  const meta = EMERGENCY_META[result.emergency_type] || EMERGENCY_META.normal;
  const isEmergency = result.status === "emergency";

  return (
    <div style={{
      border: `1px solid ${meta.color}55`,
      borderRadius: 10, padding: 16,
      background: meta.bg,
      animation: isEmergency ? "alert-flash 1.5s ease 3" : "fade-in 0.4s ease",
      position: "relative", overflow: "hidden",
    }}>
      {isEmergency && (
        <div style={{
          position: "absolute", top: 0, left: 0, right: 0, bottom: 0,
          background: "linear-gradient(135deg, transparent 60%, rgba(255,69,0,0.04))",
          pointerEvents: "none",
        }} />
      )}

      {/* Header */}
      <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 12 }}>
        <span style={{ fontSize: 28 }}>{meta.icon}</span>
        <div>
          <div style={{ fontFamily: "var(--font-mono)", fontSize: 18, color: meta.color, fontWeight: 700 }}>
            {meta.label} {isEmergency ? "DETECTED" : result.status === "uncertain" ? "— UNCERTAIN" : "— CLEAR"}
          </div>
          <div style={{ fontSize: 11, color: "var(--muted)" }}>
            Confidence: <span style={{ color: meta.color }}>{pct(result.confidence)}</span>
            &nbsp;·&nbsp;{result.inference_ms}ms&nbsp;·&nbsp;{fmt(result.timestamp)}
          </div>
        </div>
        {result.flagged && (
          <div style={{
            marginLeft: "auto", padding: "3px 8px", borderRadius: 4,
            background: "rgba(255,179,71,0.15)", border: "1px solid #FFB34788",
            fontSize: 10, color: "#FFB347", fontWeight: 700, letterSpacing: "0.05em",
          }}>⚠ FLAGGED</div>
        )}
      </div>

      {/* Confidence bars */}
      <ConfidenceBar scores={result.all_scores} />

      {/* Dispatch info */}
      {result.dispatch.services.length > 0 && (
        <div style={{
          marginTop: 12, padding: "10px 12px",
          background: "rgba(0,0,0,0.3)", borderRadius: 6,
          borderLeft: `3px solid ${meta.color}`,
        }}>
          <div style={{ fontSize: 11, color: meta.color, fontWeight: 700, marginBottom: 6, letterSpacing: "0.08em" }}>
            DISPATCH ORDER
          </div>
          <div style={{ display: "flex", gap: 8, flexWrap: "wrap" }}>
            {result.dispatch.services.map(s => (
              <span key={s} style={{
                padding: "2px 8px", borderRadius: 4,
                background: `${meta.color}22`, border: `1px solid ${meta.color}44`,
                fontSize: 11, color: meta.color, fontFamily: "var(--font-mono)",
              }}>{s.replace("_", " ").toUpperCase()}</span>
            ))}
          </div>
          <div style={{ marginTop: 6, fontSize: 11, color: "var(--muted)" }}>
            {result.dispatch.message}
          </div>
        </div>
      )}
    </div>
  );
}

function ChatMessage({ msg }) {
  const isUser = msg.role === "user";
  return (
    <div style={{
      display: "flex", flexDirection: isUser ? "row-reverse" : "row",
      gap: 10, animation: "slide-in 0.25s ease",
      alignItems: "flex-start",
    }}>
      {/* Avatar */}
      <div style={{
        width: 32, height: 32, borderRadius: "50%", flexShrink: 0,
        background: isUser ? "var(--bg3)" : "linear-gradient(135deg, var(--red), var(--red2))",
        display: "flex", alignItems: "center", justifyContent: "center",
        fontSize: 14, border: `1px solid ${isUser ? "var(--border)" : "transparent"}`,
      }}>
        {isUser ? "👤" : "🚨"}
      </div>

      <div style={{ maxWidth: "75%", display: "flex", flexDirection: "column", gap: 6 }}>
        {/* Image preview */}
        {msg.imageUrl && (
          <img src={msg.imageUrl} alt="upload" style={{
            maxWidth: "100%", maxHeight: 180, borderRadius: 8,
            border: "1px solid var(--border)", objectFit: "cover",
          }} />
        )}

        {/* Text bubble */}
        {msg.text && (
          <div style={{
            padding: "10px 14px", borderRadius: isUser ? "12px 4px 12px 12px" : "4px 12px 12px 12px",
            background: isUser ? "var(--bg3)" : "var(--bg2)",
            border: `1px solid ${isUser ? "var(--border)" : "rgba(255,255,255,0.06)"}`,
            fontSize: 14, lineHeight: 1.6, color: "var(--text)",
            fontFamily: msg.mono ? "var(--font-mono)" : "var(--font-ui)",
          }}>{msg.text}</div>
        )}

        {/* Result card */}
        {msg.result && <ResultCard result={msg.result} />}

        <div style={{ fontSize: 10, color: "var(--muted)", textAlign: isUser ? "right" : "left" }}>
          {fmt(msg.ts)}
        </div>
      </div>
    </div>
  );
}

function LogRow({ entry, idx }) {
  const meta = EMERGENCY_META[entry.emergency_type] || EMERGENCY_META.normal;
  return (
    <div style={{
      display: "grid", gridTemplateColumns: "80px 1fr 70px 70px",
      gap: 8, padding: "6px 10px", borderRadius: 4,
      background: idx % 2 === 0 ? "rgba(255, 255, 255, 0.02)" : "transparent",
      fontSize: 11, alignItems: "center",
      animation: idx === 0 ? "slide-in 0.3s ease" : "none",
    }}>
      <span style={{ color: "var(--muted)", fontFamily: "var(--font-mono)" }}>
        {fmt(entry.timestamp)}
      </span>
      <span style={{ color: meta.color, fontFamily: "var(--font-mono)", fontWeight: 700 }}>
        {meta.icon} {entry.emergency_type.toUpperCase()}
      </span>
      <span style={{
        color: entry.confidence > 0.8 ? "var(--green)" : entry.confidence > 0.6 ? "var(--amber)" : "var(--red)",
        fontFamily: "var(--font-mono)",
      }}>
        {pct(entry.confidence)}
      </span>
      <span style={{
        padding: "1px 6px", borderRadius: 3, fontSize: 10, textAlign: "center",
        background: entry.flagged ? "rgba(255,179,71,0.12)" : "rgba(76,175,125,0.1)",
        color: entry.flagged ? "#FFB347" : "#4CAF7D",
        border: `1px solid ${entry.flagged ? "#FFB34740" : "#4CAF7D40"}`,
      }}>
        {entry.flagged ? "FLAGGED" : "OK"}
      </span>
    </div>
  );
}

// ── Main App ─────────────────────────────────────────────
export default function EmergencyAI() {
  const [messages, setMessages]     = useState([{
    role: "assistant", ts: new Date(),
    text: "EmergencyAI online. Upload an image or describe a situation to begin analysis.",
  }]);
  const [input, setInput]           = useState("");
  const [loading, setLoading]       = useState(false);
  const [apiOnline, setApiOnline]   = useState(false);
  const [stats, setStats]           = useState({ total:0, fire:0, medical:0, crime:0, normal:0, flagged:0 });
  const [logs, setLogs]             = useState([]);
  const [dragOver, setDragOver]     = useState(false);
  const [activeTab, setActiveTab]   = useState("chat");

  const chatEndRef  = useRef(null);
  const fileRef     = useRef(null);
  const pollRef     = useRef(null);

  // inject global CSS once
  useEffect(() => {
    const style = document.createElement("style");
    style.textContent = GLOBAL_CSS;
    document.head.appendChild(style);
    return () => document.head.removeChild(style);
  }, []);

  // scroll chat to bottom
  useEffect(() => { chatEndRef.current?.scrollIntoView({ behavior: "smooth" }); }, [messages]);

  // health check + polling
  const fetchStatus = useCallback(async () => {
    try {
      const r = await fetch(`${API}/health`);
      setApiOnline(r.ok);
    } catch { setApiOnline(false); }
  }, []);

  const fetchStats = useCallback(async () => {
    try {
      const r = await fetch(`${API}/logs/stats`);
      if (r.ok) setStats(await r.json());
    } catch {}
  }, []);

  const fetchLogs = useCallback(async () => {
    try {
      const r = await fetch(`${API}/logs?limit=30`);
      if (r.ok) {
        const d = await r.json();
        setLogs(d.logs || []);
      }
    } catch {}
  }, []);

  useEffect(() => {
    fetchStatus(); fetchStats(); fetchLogs();
    pollRef.current = setInterval(() => {
      fetchStatus(); fetchStats(); fetchLogs();
    }, 4000);
    return () => clearInterval(pollRef.current);
  }, [fetchStatus, fetchStats, fetchLogs]);

  // send text message
  const sendText = async () => {
    const text = input.trim();
    if (!text || loading) return;
    setInput("");

    const userMsg = { role: "user", ts: new Date(), text };
    setMessages(m => [...m, userMsg]);
    setLoading(true);

    // Simple keyword triage response (chatbot without LLM)
    await new Promise(r => setTimeout(r, 600));
    const lower = text.toLowerCase();
    let reply = "I'm designed to analyze images for emergencies. Please upload an image for AI-powered analysis.";
    if (lower.includes("fire") || lower.includes("smoke"))
      reply = "🔥 You've described a potential fire situation. Please upload an image so I can analyse it and determine dispatch requirements.";
    else if (lower.includes("accident") || lower.includes("injur") || lower.includes("blood"))
      reply = "🚑 Medical emergency suspected. Upload a photo of the scene so I can assess severity and dispatch the appropriate services.";
    else if (lower.includes("crime") || lower.includes("robbery") || lower.includes("fight"))
      reply = "🚔 Criminal activity described. Upload an image to verify and I'll alert the appropriate units.";
    else if (lower.includes("hello") || lower.includes("hi"))
      reply = "EmergencyAI ready. I can analyse images for fire, medical, and crime emergencies. Upload an image to begin.";

    setMessages(m => [...m, { role: "assistant", ts: new Date(), text: reply }]);
    setLoading(false);
  };

  // send image
  const sendImage = async (file) => {
    if (!file || loading) return;
    if (!file.type.startsWith("image/")) {
      setMessages(m => [...m, { role: "assistant", ts: new Date(), text: "⚠ Please upload an image file (JPG, PNG, WEBP)." }]);
      return;
    }

    const imageUrl = URL.createObjectURL(file);
    const userMsg  = { role: "user", ts: new Date(), text: `Analysing: ${file.name}`, imageUrl };
    setMessages(m => [...m, userMsg]);
    setLoading(true);

    if (!apiOnline) {
      await new Promise(r => setTimeout(r, 800));
      setMessages(m => [...m, {
        role: "assistant", ts: new Date(),
        text: "⚠ Backend offline. Start the Python server:\n\nuvicorn api:app --reload --port 8000",
        mono: true,
      }]);
      setLoading(false);
      return;
    }

    try {
      const form = new FormData();
      form.append("file", file);
      const res  = await fetch(`${API}/analyze-image`, { method: "POST", body: form });
      if (!res.ok) throw new Error(`Server error ${res.status}`);
      const result = await res.json();

      setMessages(m => [...m, { role: "assistant", ts: new Date(), result }]);
      fetchStats(); fetchLogs();
    } catch (err) {
      setMessages(m => [...m, {
        role: "assistant", ts: new Date(),
        text: `Analysis failed: ${err.message}`,
      }]);
    }
    setLoading(false);
  };

  // drag + drop
  const onDrop = (e) => {
    e.preventDefault(); setDragOver(false);
    const file = e.dataTransfer.files[0];
    if (file) sendImage(file);
  };

  const TABS = ["chat", "logs"];

  return (
    <div style={{
      minHeight: "100vh", background: "var(--bg0)",
      fontFamily: "var(--font-ui)", display: "flex", flexDirection: "column",
      position: "relative", overflow: "hidden",
    }}>

      {/* Background grid texture */}
      <div style={{
        position: "fixed", inset: 0, pointerEvents: "none", zIndex: 0,
        backgroundImage: `
          linear-gradient(rgba(255,69,0,0.03) 1px, transparent 1px),
          linear-gradient(90deg, rgba(255,69,0,0.03) 1px, transparent 1px)
        `,
        backgroundSize: "40px 40px",
      }} />

      {/* ── HEADER ── */}
      <header style={{
        position: "relative", zIndex: 10,
        display: "flex", alignItems: "center", justifyContent: "space-between",
        padding: "0 24px", height: 58,
        background: "linear-gradient(180deg, #0f1114 0%, rgba(15,17,20,0.95) 100%)",
        borderBottom: "1px solid var(--border)",
        backdropFilter: "blur(10px)",
      }}>
        {/* Logo */}
        <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
          <div style={{
            width: 36, height: 36, borderRadius: 8,
            background: "linear-gradient(135deg, var(--red), var(--red2))",
            display: "flex", alignItems: "center", justifyContent: "center",
            fontSize: 18, boxShadow: "var(--glow)",
            animation: "pulse-ring 3s infinite",
          }}>🚨</div>
          <div>
            <div style={{
              fontFamily: "var(--font-mono)", fontSize: 18, fontWeight: 700,
              color: "var(--text)", letterSpacing: "0.06em",
            }}>
              EMERGENCY<span style={{ color: "var(--red)" }}>AI</span>
            </div>
            <div style={{ fontSize: 9, color: "var(--muted)", letterSpacing: "0.15em", textTransform: "uppercase" }}>
              Emergency Detection System
            </div>
          </div>
        </div>

        {/* Nav tabs */}
        <div style={{ display: "flex", gap: 4 }}>
          {TABS.map(tab => (
            <button key={tab} onClick={() => setActiveTab(tab)} style={{
              padding: "5px 16px", borderRadius: 6, border: "none", cursor: "pointer",
              fontFamily: "var(--font-ui)", fontSize: 12, fontWeight: 700,
              letterSpacing: "0.08em", textTransform: "uppercase",
              background: activeTab === tab ? "rgba(255,69,0,0.15)" : "transparent",
              color: activeTab === tab ? "var(--red)" : "var(--muted)",
              borderBottom: activeTab === tab ? "2px solid var(--red)" : "2px solid transparent",
              transition: "all 0.2s",
            }}>{tab === "chat" ? "⚡ Analysis" : "📋 Incident Log"}</button>
          ))}
        </div>

        {/* Status */}
        <div style={{ display: "flex", alignItems: "center", gap: 16 }}>
          <div style={{ display: "flex", alignItems: "center", fontSize: 12 }}>
            <StatusDot online={apiOnline} />
            <span style={{ color: apiOnline ? "#4CAF7D" : "#E63946", fontFamily: "var(--font-mono)", fontWeight: 700 }}>
              {apiOnline ? "SYSTEM ONLINE" : "BACKEND OFFLINE"}
            </span>
          </div>
          <div style={{
            fontFamily: "var(--font-mono)", fontSize: 11, color: "var(--muted)",
            padding: "3px 8px", background: "var(--bg2)", borderRadius: 4,
            border: "1px solid var(--border)",
          }}>
            {new Date().toLocaleTimeString()}
          </div>
        </div>
      </header>

      {/* ── MAIN ── */}
      <main style={{ flex: 1, display: "flex", flexDirection: "column", position: "relative", zIndex: 1, overflow: "hidden" }}>

        {/* ── STATS BAR ── */}
        <div style={{
          padding: "12px 24px",
          borderBottom: "1px solid rgba(255,255,255,0.04)",
          background: "rgba(15,17,20,0.8)",
        }}>
          <div style={{ display: "flex", gap: 10 }}>
            <StatCard label="Total Incidents" value={stats.total}   color="#E8EAF0" icon="📊" />
            <StatCard label="Fire"            value={stats.fire}    color="#FF4500" icon="🔥" />
            <StatCard label="Medical"         value={stats.medical} color="#FF6B35" icon="🚑" />
            <StatCard label="Crime"           value={stats.crime}   color="#E63946" icon="🚔" />
            <StatCard label="Normal"          value={stats.normal}  color="#4CAF7D" icon="✅" />
            <StatCard label="Flagged"         value={stats.flagged} color="#FFB347" icon="⚠" sub="Low confidence" />
          </div>
        </div>

        {/* ── TAB CONTENT ── */}
        <div style={{ flex: 1, overflow: "hidden", display: "flex", flexDirection: "column" }}>

          {/* ANALYSIS TAB */}
          {activeTab === "chat" && (
            <div style={{ flex: 1, display: "flex", flexDirection: "column", overflow: "hidden" }}>

              {/* Chat messages */}
              <div style={{
                flex: 1, overflowY: "auto", padding: "20px 24px",
                display: "flex", flexDirection: "column", gap: 16,
              }}>
                {messages.map((msg, i) => <ChatMessage key={i} msg={msg} />)}

                {loading && (
                  <div style={{ display: "flex", gap: 10, alignItems: "center", animation: "fade-in 0.3s ease" }}>
                    <div style={{
                      width: 32, height: 32, borderRadius: "50%",
                      background: "linear-gradient(135deg, var(--red), var(--red2))",
                      display: "flex", alignItems: "center", justifyContent: "center", fontSize: 14,
                    }}>🚨</div>
                    <div style={{
                      padding: "10px 16px", background: "var(--bg2)", borderRadius: "4px 12px 12px 12px",
                      border: "1px solid rgba(255,255,255,0.06)",
                    }}>
                      <div style={{ display: "flex", gap: 5, alignItems: "center" }}>
                        {[0,1,2].map(i => (
                          <div key={i} style={{
                            width: 6, height: 6, borderRadius: "50%", background: "var(--red)",
                            animation: `blink 1.2s ${i*0.2}s infinite`,
                          }} />
                        ))}
                        <span style={{ fontSize: 12, color: "var(--muted)", marginLeft: 6 }}>
                          Analysing...
                        </span>
                      </div>
                    </div>
                  </div>
                )}
                <div ref={chatEndRef} />
              </div>

              {/* Drop zone + input */}
              <div style={{
                padding: "12px 24px 20px",
                background: "rgba(10,11,13,0.9)",
                borderTop: "1px solid var(--border)",
              }}>
                {/* Drop zone */}
                <div
                  onDragOver={e => { e.preventDefault(); setDragOver(true); }}
                  onDragLeave={() => setDragOver(false)}
                  onDrop={onDrop}
                  onClick={() => fileRef.current?.click()}
                  style={{
                    border: `1.5px dashed ${dragOver ? "var(--red)" : "var(--border)"}`,
                    borderRadius: 8, padding: "10px 16px", marginBottom: 10,
                    display: "flex", alignItems: "center", justifyContent: "center", gap: 10,
                    cursor: "pointer", transition: "all 0.2s",
                    background: dragOver ? "rgba(255,69,0,0.06)" : "transparent",
                  }}
                >
                  <span style={{ fontSize: 18 }}>📎</span>
                  <span style={{ fontSize: 13, color: "var(--muted)" }}>
                    Drop image here or <span style={{ color: "var(--red)" }}>click to upload</span>
                    <span style={{ fontSize: 11, marginLeft: 8 }}>(JPG, PNG, WEBP · max 10MB)</span>
                  </span>
                  <input
                    ref={fileRef} type="file" accept="image/*" style={{ display: "none" }}
                    onChange={e => sendImage(e.target.files[0])}
                  />
                </div>

                {/* Text input */}
                <div style={{ display: "flex", gap: 10 }}>
                  <input
                    value={input}
                    onChange={e => setInput(e.target.value)}
                    onKeyDown={e => e.key === "Enter" && sendText()}
                    placeholder="Describe the situation or ask a question…"
                    style={{
                      flex: 1, padding: "10px 16px", borderRadius: 8,
                      background: "var(--bg2)", border: "1px solid var(--border)",
                      color: "var(--text)", fontSize: 14, fontFamily: "var(--font-ui)",
                      outline: "none", transition: "border-color 0.2s",
                    }}
                    onFocus={e => e.target.style.borderColor = "var(--red)"}
                    onBlur={e  => e.target.style.borderColor = "var(--border)"}
                  />
                  <button
                    onClick={sendText}
                    disabled={!input.trim() || loading}
                    style={{
                      padding: "10px 20px", borderRadius: 8, border: "none", cursor: "pointer",
                      background: input.trim() && !loading
                        ? "linear-gradient(135deg, var(--red), var(--red2))"
                        : "var(--bg3)",
                      color: input.trim() && !loading ? "white" : "var(--muted)",
                      fontFamily: "var(--font-ui)", fontWeight: 700, fontSize: 13,
                      letterSpacing: "0.06em", transition: "all 0.2s",
                      boxShadow: input.trim() && !loading ? "var(--glow)" : "none",
                    }}
                  >SEND</button>
                </div>
              </div>
            </div>
          )}

          {/* LOGS TAB */}
          {activeTab === "logs" && (
            <div style={{ flex: 1, overflowY: "auto", padding: 24 }}>
              <div style={{
                background: "var(--bg1)", borderRadius: 10,
                border: "1px solid var(--border)", overflow: "hidden",
              }}>
                {/* Table header */}
                <div style={{
                  display: "grid", gridTemplateColumns: "80px 1fr 70px 70px",
                  gap: 8, padding: "8px 10px",
                  background: "var(--bg2)", borderBottom: "1px solid var(--border)",
                  fontSize: 10, color: "var(--muted)", fontWeight: 700,
                  letterSpacing: "0.1em", textTransform: "uppercase",
                  fontFamily: "var(--font-mono)",
                }}>
                  <span>TIME</span>
                  <span>TYPE</span>
                  <span>CONF.</span>
                  <span>STATUS</span>
                </div>

                {logs.length === 0 ? (
                  <div style={{ padding: 40, textAlign: "center", color: "var(--muted)", fontSize: 13 }}>
                    No incidents logged yet. Upload an image to begin analysis.
                  </div>
                ) : (
                  logs.map((entry, i) => <LogRow key={i} entry={entry} idx={i} />)
                )}
              </div>
            </div>
          )}
        </div>
      </main>

      {/* ── FOOTER ── */}
      <footer style={{
        padding: "6px 24px", borderTop: "1px solid rgba(255,255,255,0.04)",
        background: "var(--bg0)", display: "flex", justifyContent: "space-between",
        alignItems: "center", fontSize: 10, color: "var(--muted)",
        fontFamily: "var(--font-mono)", position: "relative", zIndex: 10,
      }}>
        <span>EMERGENCYAI v1.0 · LOCAL MODEL · NO DATA LEAVES THIS MACHINE</span>
        <span style={{ display: "flex", alignItems: "center", gap: 6 }}>
          <StatusDot online={apiOnline} />
          API: localhost:8000
        </span>
      </footer>
    </div>
  );
}