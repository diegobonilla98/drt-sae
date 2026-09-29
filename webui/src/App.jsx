import { useEffect, useMemo, useState } from "react";
import NeuronGraph3D from "./NeuronGraph3D";

const defaultPrompt = "The Eiffel Tower is located in the city of";
const defaultLabels = "cat, dog, car, person, tree, building, food, cow";
const palette = ["#ff7a62", "#48d3cf", "#bb85ff", "#f6b94c", "#7295dc", "#e682a8", "#7bd88f", "#ecf06c"];

function formatNumber(value, digits = 2) {
  if (value === null || value === undefined) return "-";
  return Number(value).toLocaleString(undefined, { maximumFractionDigits: digits });
}

function shortId(id) {
  return `E-${String(id).slice(-5)}`;
}

function activationWidth(expert, experts) {
  const highest = Math.max(...experts.map((item) => item.activation), 1);
  return `${Math.max(8, (expert.activation / highest) * 100)}%`;
}

function formatLocation(expert, modality) {
  if (modality === "text" && expert.token_index !== undefined) {
    const label = expert.token_text ? `"${expert.token_text.trim()}"` : `index ${expert.token_index}`;
    return `Token ${expert.token_index}: ${label}`;
  }
  if (modality === "image" && expert.patch_index !== undefined) {
    const row = expert.patch_row ?? Math.floor(expert.patch_index / 14);
    const col = expert.patch_col ?? expert.patch_index % 14;
    return `Patch ${expert.patch_index} (row ${row}, col ${col})`;
  }
  return null;
}

function TokenLocalization({ tokens, experts, selectedId, predictionPosition, palette }) {
  const expertByToken = useMemo(() => {
    const map = new Map();
    experts.forEach((expert, index) => {
      if (expert.token_index === undefined) return;
      const existing = map.get(expert.token_index) ?? [];
      existing.push({ expert, color: palette[index % palette.length] });
      map.set(expert.token_index, existing);
    });
    return map;
  }, [experts, palette]);
  if (!tokens.length) {
    return <div className="localization-empty">Run inference to map expert activations onto tokens.</div>;
  }
  return (
    <div className="token-localization">
      {tokens.map((token) => {
        const hits = expertByToken.get(token.index) ?? [];
        const selectedHit = hits.find((item) => item.expert.expert_id === selectedId);
        const accent = selectedHit?.color ?? hits[0]?.color;
        const isPrediction = token.index === predictionPosition;
        return (
          <span
            key={token.index}
            className={["token-chip", hits.length ? "has-expert" : "", selectedHit ? "selected" : "", isPrediction ? "prediction-site" : ""].filter(Boolean).join(" ")}
            style={accent ? { "--accent": accent } : undefined}
            title={hits.map((item) => item.expert.label).join(", ") || (isPrediction ? "Next-token prediction site" : undefined)}
          >
            {token.text}
          </span>
        );
      })}
    </div>
  );
}

function PatchLocalization({ imageData, patchGrid, experts, selectedId, palette }) {
  const cells = useMemo(() => Array.from({ length: patchGrid * patchGrid }, (_, index) => index), [patchGrid]);
  const expertByPatch = useMemo(() => {
    const map = new Map();
    experts.forEach((expert, index) => {
      if (expert.patch_index === undefined) return;
      map.set(expert.patch_index, { expert, color: palette[index % palette.length] });
    });
    return map;
  }, [experts, palette]);
  if (imageData) {
    return (
      <div className="patch-localization image-mode">
        <div className="patch-localization-frame">
          <img src={imageData} alt="Localization overlay" />
          {experts.filter((expert) => expert.patch_box).map((expert, index) => (
            <div
              key={expert.expert_id}
              className={`patch-box ${expert.expert_id === selectedId ? "selected" : ""}`}
              style={{
                left: `${expert.patch_box.left}%`,
                top: `${expert.patch_box.top}%`,
                width: `${expert.patch_box.width}%`,
                height: `${expert.patch_box.height}%`,
                borderColor: palette[index % palette.length],
              }}
            />
          ))}
        </div>
      </div>
    );
  }
  if (!experts.some((expert) => expert.patch_index !== undefined)) {
    return <div className="localization-empty">Run inference to map expert activations onto image patches.</div>;
  }
  return (
    <div className="patch-localization grid-mode" style={{ gridTemplateColumns: `repeat(${patchGrid}, 1fr)` }}>
      {cells.map((patchIndex) => {
        const hit = expertByPatch.get(patchIndex);
        return (
          <div
            key={patchIndex}
            className={["patch-cell", hit ? "has-expert" : "", hit?.expert.expert_id === selectedId ? "selected" : ""].filter(Boolean).join(" ")}
            style={hit ? { "--accent": hit.color } : undefined}
            title={hit ? hit.expert.label : `Patch ${patchIndex}`}
          >
            {hit ? shortId(hit.expert.expert_id) : ""}
          </div>
        );
      })}
    </div>
  );
}

function LocalizationReport({ modality, result, experts, selectedId, imageData, palette }) {
  const selected = experts.find((expert) => expert.expert_id === selectedId);
  const location = selected ? formatLocation(selected, modality) : null;
  return (
    <section className="localization-panel">
      <div className="section-heading">
        <span>Feature localization</span>
        <span>{location || (modality === "text" ? "Per-token map" : "Per-patch map")}</span>
      </div>
      {modality === "text" ? (
        <TokenLocalization tokens={result?.tokens ?? []} experts={experts} selectedId={selectedId} predictionPosition={result?.prediction_position} palette={palette} />
      ) : (
        <PatchLocalization imageData={imageData} patchGrid={result?.patch_grid ?? 14} experts={experts} selectedId={selectedId} palette={palette} />
      )}
      {selected && location && <div className="localization-selected">{shortId(selected.expert_id)} strongest at {location}</div>}
    </section>
  );
}

function Icon({ name }) {
  const paths = {
    text: "M5 5h14M12 5v14M8 19h8",
    image: "M4 5h16v14H4zM4 15l5-5 3 3 2-2 6 6M16 8h.01",
    spark: "M12 2l1.9 6.1L20 10l-6.1 1.9L12 18l-1.9-6.1L4 10l6.1-1.9z",
    play: "M8 5v14l11-7z",
    upload: "M12 16V4m0 0L7 9m5-5 5 5M5 20h14",
    cube: "m12 3 8 4.5v9L12 21l-8-4.5v-9zM12 12l8-4.5M12 12 4 7.5M12 12v9",
    close: "M6 6l12 12M18 6 6 18"
  };
  return <svg viewBox="0 0 24 24" aria-hidden="true"><path d={paths[name]} /></svg>;
}

function ExpertCard({ expert, experts, index, selected, onSelect, modality }) {
  const accent = palette[index % palette.length];
  const location = formatLocation(expert, modality);
  return (
    <button className={`expert-card ${selected ? "selected" : ""}`} style={{ "--accent": accent }} onClick={() => onSelect(expert.expert_id)}>
      <span className="expert-card-index">{String(index + 1).padStart(2, "0")}</span>
      <span className="expert-card-main">
        <strong>{expert.label}</strong>
        <span>{shortId(expert.expert_id)}</span>
        {location && <span className="expert-location">{location}</span>}
      </span>
      <span className="expert-card-metrics">
        <span className="metric-label">Activation</span>
        <span className="activation-bar"><i style={{ width: activationWidth(expert, experts) }} /></span>
        <strong>{formatNumber(expert.activation)}</strong>
        <span className={expert.causal_effect >= 0 ? "effect positive" : "effect negative"}>{expert.causal_effect >= 0 ? "+" : ""}{formatNumber(expert.causal_effect, 3)}</span>
      </span>
    </button>
  );
}

function DetailPanel({ expert, modality }) {
  if (!expert) return <section className="detail-panel empty">Run an inference to inspect an expert.</section>;
  const location = formatLocation(expert, modality);
  return (
    <section className="detail-panel">
      <div className="section-heading">
        <span>Selected expert</span>
        <span className="expert-tag">{shortId(expert.expert_id)}</span>
      </div>
      <h2>{expert.label}</h2>
      <p>{expert.description}</p>
      {location && (
        <div className="location-detail">
          <span>Strongest site</span>
          <strong>{location}</strong>
        </div>
      )}
      <div className="detail-stats">
        <div><span>Current activation</span><strong>{formatNumber(expert.activation)}</strong></div>
        <div><span>Causal effect</span><strong className={expert.causal_effect >= 0 ? "positive" : "negative"}>{expert.causal_effect >= 0 ? "+" : ""}{formatNumber(expert.causal_effect, 3)}</strong></div>
        <div><span>Training max</span><strong>{formatNumber(expert.max_training_activation)}</strong></div>
        <div><span>Active examples</span><strong>{formatNumber(expert.active_count, 0)}</strong></div>
      </div>
      <div className="evidence">
        <span>Representative evidence</span>
        <div>{expert.evidence?.length ? expert.evidence.map((item, index) => <em key={`${item}-${index}`}>{item}</em>) : <em>No saved examples</em>}</div>
      </div>
      {modality === "image" && expert.patch_index !== undefined && <div className="patch-note">Patch grid cell {expert.patch_row},{expert.patch_col}</div>}
    </section>
  );
}

function GenerationPanel({ result }) {
  if (!result?.generated_text) return null;
  return (
    <section className="generation-panel">
      <div className="section-heading"><span>Generated continuation</span><span>{result.generated_text.length} chars</span></div>
      {result.intervention && (
        <div className="prediction-compare">
          <span>Before</span><strong>{result.intervention.baseline_generated_text || result.intervention.baseline_prediction}</strong>
          <span>After</span><strong>{result.generated_text}</strong>
        </div>
      )}
      <p>{result.generated_text}</p>
    </section>
  );
}

function InterventionPanel({ experts, expert, multiplier, onMultiplierChange, onApply, onSelect, isRunning, intervention, result }) {
  const [query, setQuery] = useState("");
  const matches = useMemo(() => {
    const normalizedQuery = query.trim().toLowerCase();
    if (!normalizedQuery) return [];
    return experts
      .map((item) => {
        const label = item.label.toLowerCase();
        const evidence = (item.evidence ?? []).join(" ").toLowerCase();
        const description = item.description.toLowerCase();
        const score = label.startsWith(normalizedQuery) ? 0 : label.includes(normalizedQuery) ? 1 : evidence.includes(normalizedQuery) ? 2 : description.includes(normalizedQuery) ? 3 : -1;
        return { item, score };
      })
      .filter(({ score }) => score >= 0)
      .sort((left, right) => left.score - right.score)
      .slice(0, 5)
      .map(({ item }) => item);
  }, [experts, query]);
  const label = expert ? `${shortId(expert.expert_id)} - ${expert.label}` : "Select an active expert";
  return (
    <section className="intervention-panel">
      <div className="section-heading"><span>Concept amplification</span><span>{intervention ? "Applied" : "Causal test"}</span></div>
      <p>Increase an active expert's decoded contribution, then rerun inference to compare the output.</p>
      <label className="concept-finder" htmlFor="concept-finder">
        <span>Find active concept</span>
        <input id="concept-finder" value={query} onChange={(event) => setQuery(event.target.value)} placeholder="person, drug, medication..." />
      </label>
      {matches.length > 0 && <div className="concept-matches">{matches.map((item) => <button key={item.expert_id} className={item.expert_id === expert?.expert_id ? "selected" : ""} onClick={() => onSelect(item.expert_id)}>{shortId(item.expert_id)} {item.label}</button>)}</div>}
      <strong className="intervention-expert">{label}</strong>
      <label className="range-label" htmlFor="boost-multiplier"><span>Amplification</span><output>{multiplier.toFixed(1)}x</output></label>
      <input id="boost-multiplier" type="range" min="1.1" max="5" step="0.1" value={multiplier} onChange={(event) => onMultiplierChange(Number(event.target.value))} />
      {intervention && <div className="intervention-result"><span>{intervention.scope}</span><div><em>Before</em><strong>{intervention.baseline_prediction}</strong></div><div><em>After</em><strong>{result?.prediction}</strong></div><div><em>Outcome</em><strong>{intervention.prediction_changed ? "Prediction changed" : "Prediction held steady"}</strong></div></div>}
      <button className="boost-button" onClick={onApply} disabled={!expert || isRunning}>Boost & rerun</button>
    </section>
  );
}

function App() {
  const [modality, setModality] = useState("text");
  const [prompt, setPrompt] = useState(defaultPrompt);
  const [labels, setLabels] = useState(defaultLabels);
  const [imageData, setImageData] = useState("");
  const [imageName, setImageName] = useState("");
  const [result, setResult] = useState(null);
  const [selectedId, setSelectedId] = useState(null);
  const [isRunning, setIsRunning] = useState(false);
  const [notice, setNotice] = useState("");
  const [maxNewTokens, setMaxNewTokens] = useState(4);
  const [boostMultiplier, setBoostMultiplier] = useState(1.5);

  useEffect(() => {
    const controller = new AbortController();
    fetch(`/api/demo/${modality}`, { signal: controller.signal })
      .then((response) => response.ok ? response.json() : Promise.reject(new Error("Saved demo is unavailable.")))
      .then((data) => {
        setResult(data);
        setSelectedId((currentId) => data.experts.some((expert) => expert.expert_id === currentId) ? currentId : data.experts[0]?.expert_id ?? null);
        setNotice("Loaded saved explanation");
      })
      .catch((error) => {
        if (error.name !== "AbortError") setNotice(error.message);
      });
    return () => controller.abort();
  }, [modality]);

  const selected = result?.experts.find((expert) => expert.expert_id === selectedId) ?? result?.experts[0];
  const handleImage = (event) => {
    const file = event.target.files?.[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = () => setImageData(String(reader.result));
    reader.readAsDataURL(file);
    setImageName(file.name);
  };
  const runInference = async (intervention = null) => {
    setIsRunning(true);
    setNotice("");
    const endpoint = modality === "text" ? "/api/infer/text" : "/api/infer/image";
    const body = modality === "text"
      ? { prompt, max_experts: 12, max_new_tokens: maxNewTokens, intervention }
      : { image_data: imageData, candidate_labels: labels.split(",").map((item) => item.trim()).filter(Boolean), max_experts: 12, intervention };
    if (modality === "image" && !imageData) {
      setNotice("Choose an image before running visual inference.");
      setIsRunning(false);
      return;
    }
    try {
      const response = await fetch(endpoint, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
      const data = await response.json();
      if (!response.ok) throw new Error(data.detail || "Inference failed.");
      setResult(data);
      setSelectedId((currentId) => data.experts.some((expert) => expert.expert_id === currentId) ? currentId : data.experts[0]?.expert_id ?? null);
      setNotice(intervention ? "Amplified inference complete" : "Inference complete");
    } catch (error) {
      setNotice(error.message);
    } finally {
      setIsRunning(false);
    }
  };
  const experts = result?.experts ?? [];
  const applyBoost = () => {
    if (!selected) return;
    runInference({ expert_id: selected.expert_id, multiplier: boostMultiplier });
  };
  return (
    <main className="app-shell">
      <header className="topbar">
        <div className="brand"><Icon name="cube" /><span>Neuron Atlas</span></div>
        <div className="model-context"><span>Gemma-2-2B / CLIP ViT-B/16</span><i /> <span>Layer {result?.layer_id ?? (modality === "text" ? 13 : 9)}</span></div>
        <button className="run-button" onClick={runInference} disabled={isRunning}><Icon name="play" />{isRunning ? "Analyzing" : "Run inference"}</button>
      </header>
      <div className="workspace">
        <aside className="input-rail">
          <div className="section-heading"><span>Inference input</span><span className="status-dot" title={notice}>{notice || "Ready"}</span></div>
          <div className="modality-switch">
            <button className={modality === "text" ? "active" : ""} onClick={() => setModality("text")}><Icon name="text" />Text</button>
            <button className={modality === "image" ? "active" : ""} onClick={() => setModality("image")}><Icon name="image" />Image</button>
          </div>
          {modality === "text" ? <>
            <label className="field-label" htmlFor="prompt">Prompt <span>{prompt.length} / 4096</span></label>
            <textarea id="prompt" value={prompt} onChange={(event) => setPrompt(event.target.value)} />
            <div className="token-preview"><span>Input tokens</span><div>{prompt.split(/\s+/).filter(Boolean).slice(-8).map((token, index) => <em key={`${token}-${index}`}>{token}</em>)}</div></div>
            <label className="range-label" htmlFor="max-new-tokens"><span>Continuation length</span><output>{maxNewTokens} tokens</output></label>
            <input id="max-new-tokens" type="range" min="1" max="32" step="1" value={maxNewTokens} onChange={(event) => setMaxNewTokens(Number(event.target.value))} />
          </> : <>
            <label className="drop-zone" htmlFor="image-input">
              {imageData ? <img src={imageData} alt="Selected inference input" /> : <><Icon name="upload" /><strong>Drop an image here</strong><span>or select a file</span></>}
              <input id="image-input" type="file" accept="image/png,image/jpeg,image/webp" onChange={handleImage} />
            </label>
            <div className="file-name">{imageName || "PNG, JPG, or WEBP"}</div>
            <label className="field-label" htmlFor="labels">Candidate labels</label>
            <textarea id="labels" className="labels-input" value={labels} onChange={(event) => setLabels(event.target.value)} />
          </>}
          <div className="rail-footer"><span>Routed Top-K SAE</span><strong>{modality === "text" ? "32" : "16"} routes</strong></div>
        </aside>
        <section className="analysis-space">
          <div className="analysis-header">
            <div><span className="eyeless-label">Neuron constellation</span><strong>{modality === "text" ? "Next-token circuit" : "Visual concept circuit"}</strong></div>
          </div>
          <NeuronGraph3D
            experts={experts}
            selectedId={selected?.expert_id}
            onSelect={setSelectedId}
            palette={palette}
            expertCountLabel={`${experts.length || 0} experts mapped`}
          />
          <LocalizationReport modality={modality} result={result} experts={experts} selectedId={selected?.expert_id} imageData={imageData} palette={palette} />
          <section className="contribution-panel">
            <div className="section-heading"><span>Causal contribution</span><span>Ablation delta</span></div>
            <div className="contribution-list">
              {experts.map((expert, index) => <button key={expert.expert_id} className={selected?.expert_id === expert.expert_id ? "selected" : ""} onClick={() => setSelectedId(expert.expert_id)}>
                <span>{String(index + 1).padStart(2, "0")}</span><i style={{ backgroundColor: palette[index % palette.length] }} /><strong>{expert.label}</strong><em className={expert.causal_effect >= 0 ? "positive" : "negative"}>{expert.causal_effect >= 0 ? "+" : ""}{formatNumber(expert.causal_effect, 3)}</em>
              </button>)}
            </div>
          </section>
        </section>
        <aside className="inspector-rail">
          <section className="prediction-block">
            <span>Prediction</span>
            <h1>{result?.prediction || "-"}</h1>
            <div className="confidence-row"><span>Confidence</span><strong>{formatNumber((result?.confidence ?? 0) * 100, 1)}%</strong></div>
            <div className="confidence-bar"><i style={{ width: `${Math.min(100, (result?.confidence ?? 0) * 100)}%` }} /></div>
            <div className="prediction-stats"><div><span>Experts active</span><strong>{result?.active_expert_count ?? 0}</strong></div><div><span>Positive effect</span><strong>+{formatNumber(result?.summary?.positive_causal_effect, 3)}</strong></div></div>
          </section>
          <section className="experts-panel">
            <div className="section-heading"><span>Used experts</span><span>{experts.length} shown</span></div>
            <div className="expert-stack">{experts.map((expert, index) => <ExpertCard key={expert.expert_id} expert={expert} experts={experts} index={index} selected={selected?.expert_id === expert.expert_id} onSelect={setSelectedId} modality={modality} />)}</div>
          </section>
          {modality === "text" && <GenerationPanel result={result} />}
          <DetailPanel expert={selected} modality={modality} />
          <InterventionPanel experts={experts} expert={selected} multiplier={boostMultiplier} onMultiplierChange={setBoostMultiplier} onApply={applyBoost} onSelect={setSelectedId} isRunning={isRunning} intervention={result?.intervention} result={result} />
        </aside>
      </div>
    </main>
  );
}

export default App;
