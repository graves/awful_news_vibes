// src/viz_export.rs
use anyhow::{Context, Result};
use regex::Regex;
use serde::Serialize;
use serde_json::json;
use std::{
    collections::{BTreeSet, HashMap, HashSet},
    fs,
    path::Path,
};
use url::Url;

use crate::models::{ClusterMember, StoryCluster};
use crate::out_models::MetaInsights;

/* -------------------------------------------------------------------------- */
/* Entry point                                                                */
/* -------------------------------------------------------------------------- */

/// Public entry point: write all D3-ready visualization JSONs into `out/<date>/`.
pub fn write_all_viz(
    out_dir_for_date: &Path, // e.g., out/2025-10-28
    date: &str,              // "YYYY-MM-DD" (anchor 'today')
    clusters: &[StoryCluster],
    insights_json: &str,
    editions: &[String], // ordered editions used this run
) -> Result<()> {
    fs::create_dir_all(out_dir_for_date).with_context(|| format!("create {:?}", out_dir_for_date))?;

    // 1) Lifecycles
    let lifecycles = build_lifecycles(clusters, editions);
    write_json(out_dir_for_date.join("viz.lifecycles.json"), &lifecycles)?;

    // 2) Momentum (derived from lifecycles)
    let momentum = build_momentum(&lifecycles);
    write_json(out_dir_for_date.join("viz.momentum.json"), &momentum)?;

    // 3) Narrative Divergence (heuristic from digest text)
    let divergence = build_divergence(clusters);
    write_json(out_dir_for_date.join("viz.divergence.json"), &divergence)?;

    // 4) Emotion Weather (aggregate from digests per edition presence)
    let emotion = build_emotion(clusters, editions);
    write_json(out_dir_for_date.join("viz.emotion.json"), &emotion)?;

    // 5) Meta-Topic Compass (heuristic quadrant + weight, normalized vectors, raw scores)
    let compass = build_compass(clusters);
    write_json(out_dir_for_date.join("viz.compass.json"), &compass)?;

    // 6) Silences (surface from insights.json if present)
    let silences = build_silences(insights_json);
    write_json(out_dir_for_date.join("viz.silences.json"), &silences)?;

    // 7) Word Clouds (cleaned tokens; per-outlet + per-cluster)
    let clouds = build_clouds(clusters);
    write_json(out_dir_for_date.join("viz.clouds.json"), &clouds)?;

    // 8) Narrative Fingerprints (radial 0..1 features)
    let fingerprints = build_fingerprints(clusters);
    write_json(out_dir_for_date.join("viz.fingerprints.json"), &fingerprints)?;

    // 9) Cartogram stub / Entities graph / Influence stub
    write_json(out_dir_for_date.join("viz.cartogram.json"), &json!({ "themes": [], "positions": {} }))?;

    let entities = build_entities(clusters);
    write_json(out_dir_for_date.join("viz.entities.json"), &entities)?;

    write_json(
        out_dir_for_date.join("viz.influence.json"),
        &json!({ "outlets": [], "edges": [], "method": "none" }),
    )?;

    // 10) Per-day index
    let counts = json!({
        "clusters": clusters.len(),
        "articles": estimate_article_count(clusters),
        "outlets": estimate_outlet_count(clusters),
    });
    let idx = json!({
        "date": date,
        "version": 1,
        "counts": counts,
        "files": [
            "viz.lifecycles.json",
            "viz.momentum.json",
            "viz.divergence.json",
            "viz.emotion.json",
            "viz.compass.json",
            "viz.silences.json",
            "viz.clouds.json",
            "viz.fingerprints.json",
            "viz.cartogram.json",
            "viz.entities.json",
            "viz.influence.json"
        ]
    });
    write_json(out_dir_for_date.join("viz.index.json"), &idx)?;

    Ok(())
}

fn write_json<P: AsRef<Path>, T: ?Sized + Serialize>(path: P, value: &T) -> Result<()> {
    fs::write(path, serde_json::to_vec_pretty(value)?)
        .map(|_| ())
        .map_err(|e| e.into())
}

/* -------------------------------------------------------------------------- */
/* 1) Lifecycles                                                              */
/* -------------------------------------------------------------------------- */

#[derive(Serialize)]
struct VLife {
    id: String,
    title: String,
    presence: Vec<u8>, // 0/1 per edition
    volume: Vec<u32>,  // #articles per edition
    class: String,     // flashstorm | wildfire | slow_freeze | resurrection | steady
}

#[derive(Serialize)]
struct VLifeBundle {
    editions: Vec<String>,
    clusters: Vec<VLife>,
}

fn build_lifecycles(clusters: &[StoryCluster], editions: &[String]) -> VLifeBundle {
    let mut out = Vec::with_capacity(clusters.len());
    let date_to_idxs = build_date_to_idxs(editions);

    for c in clusters {
        // Count members per edition_id (robust mapping via date & URL as fallback)
        let mut vol = vec![0u32; editions.len()];
        for m in &c.members {
            let ed_idxs = edition_indices_for_member(m, editions, &date_to_idxs);
            for idx in ed_idxs {
                vol[idx] += 1;
            }
        }
        let presence: Vec<u8> = vol.iter().map(|&v| if v > 0 { 1 } else { 0 }).collect();
        let class = classify_lifecycle(&vol, &presence);

        out.push(VLife {
            id: c.cluster_id.clone(),
            title: c.canonical_title.clone(),
            presence,
            volume: vol,
            class,
        });
    }

    VLifeBundle {
        editions: editions.to_vec(),
        clusters: out,
    }
}

fn classify_lifecycle(vol: &[u32], pres: &[u8]) -> String {
    let total: u32 = vol.iter().sum();
    let peak = *vol.iter().max().unwrap_or(&0);
    let duration = pres.iter().filter(|&&x| x == 1).count();
    
    // Must meet minimum importance threshold first
    if total < 4 && peak < 3 && duration < 3 {
        return "steady".into(); // default for low-importance stories
    }
    
    // Flashstorm: brief intense spike
    // Peak ≥3 in single edition, duration ≤2, most volume concentrated
    if peak >= 3 && duration <= 2 {
        let peak_idx = vol.iter().position(|&v| v == peak).unwrap_or(0);
        let peak_concentration = vol[peak_idx] as f32 / total as f32;
        if peak_concentration >= 0.6 {
            return "flashstorm".into();
        }
    }
    
    // Resurrection: gap in coverage
    if has_resurrection(pres) && duration >= 3 {
        return "resurrection".into();
    }
    
    // Wildfire: growing story with sustained attention
    // Peak ≥2, appears in ≥3 editions, has actual growth pattern
    if peak >= 2 && duration >= 3 && has_growth_pattern(vol) {
        return "wildfire".into();
    }
    
    // Slow freeze: declining story
    // Starts with ≥2, declines over ≥3 editions
    if duration >= 3 && has_decline_pattern(vol) {
        let first_nonzero = vol.iter().find(|&&x| x > 0).copied().unwrap_or(0);
        if first_nonzero >= 2 {
            return "slow_freeze".into();
        }
    }
    
    // Steady: consistent coverage
    "steady".into()
}

fn has_growth_pattern(vol: &[u32]) -> bool {
    // True growth: later editions have MORE articles than early ones
    // Not just "no decreases" but actual increases
    
    let nonzero: Vec<u32> = vol.iter().copied().filter(|&x| x > 0).collect();
    if nonzero.len() < 2 {
        return false;
    }
    
    // Check if last half has higher average than first half
    let mid = nonzero.len() / 2;
    let early_avg: f32 = nonzero[..mid].iter().sum::<u32>() as f32 / mid as f32;
    let late_avg: f32 = nonzero[mid..].iter().sum::<u32>() as f32 / (nonzero.len() - mid) as f32;
    
    late_avg > early_avg * 1.2 // Must be 20% higher
}

fn has_decline_pattern(vol: &[u32]) -> bool {
    // True decline: earlier editions have MORE articles than later ones
    
    let nonzero: Vec<u32> = vol.iter().copied().filter(|&x| x > 0).collect();
    if nonzero.len() < 2 {
        return false;
    }
    
    // Check if first half has higher average than last half
    let mid = nonzero.len() / 2;
    let early_avg: f32 = nonzero[..mid].iter().sum::<u32>() as f32 / mid as f32;
    let late_avg: f32 = nonzero[mid..].iter().sum::<u32>() as f32 / (nonzero.len() - mid) as f32;
    
    early_avg > late_avg * 1.2 // Must be 20% higher
}

fn has_resurrection(pres: &[u8]) -> bool {
    let mut seen_one = false;
    let mut seen_zero_after_one = false;
    for &b in pres {
        if b == 1 {
            if seen_zero_after_one {
                return true;
            }
            seen_one = true;
        } else if b == 0 && seen_one {
            seen_zero_after_one = true;
        }
    }
    false
}

/* -------------------------------------------------------------------------- */
/* 2) Momentum                                                                */
/* -------------------------------------------------------------------------- */

#[derive(Serialize)]
struct VMomentum {
    id: String,
    title: String,
    velocity: Vec<f32>, // normalized deltas per step
    gravity: f32,       // share of editions with presence
    prediction: String, // rising | flat | falling
}

#[derive(Serialize)]
struct VMomentumBundle {
    editions: Vec<String>,
    trails: Vec<VMomentum>,
}

fn build_momentum(life: &VLifeBundle) -> VMomentumBundle {
    let mut trails = Vec::with_capacity(life.clusters.len());
    for c in &life.clusters {
        let vels = diffs_norm(&c.volume);
        let gravity =
            c.presence.iter().filter(|&&x| x == 1).count() as f32 / (c.presence.len() as f32);
        let last = *vels.last().unwrap_or(&0.0);
        let prediction = if last > 0.05 {
            "rising"
        } else if last < -0.05 {
            "falling"
        } else {
            "flat"
        }
        .to_string();

        trails.push(VMomentum {
            id: c.id.clone(),
            title: c.title.clone(),
            velocity: vels,
            gravity,
            prediction,
        });
    }
    VMomentumBundle {
        editions: life.editions.clone(),
        trails,
    }
}

fn diffs_norm(v: &[u32]) -> Vec<f32> {
    if v.is_empty() {
        return vec![];
    }
    let maxv = *v.iter().max().unwrap_or(&1) as f32;
    let mut out = vec![0.0]; // first step has no previous
    for w in v.windows(2) {
        let d = (w[1] as f32 - w[0] as f32) / maxv.max(1.0);
        out.push(d);
    }
    out
}

/* -------------------------------------------------------------------------- */
/* 3) Narrative Divergence (heuristic from digest text)                       */
/* -------------------------------------------------------------------------- */

#[derive(Serialize)]
struct VDivPoint {
    outlet: String,
    x: f32,
    y: f32,
    /// Signed polarity: speculation (+) vs certainty (−)
    certainty: f32,
    /// Convenience magnitude in 0..1 (0 speculative … 1 certain)
    certainty01: f32,
}
#[derive(Serialize)]
struct VDivCluster {
    id: String,
    title: String,
    points: Vec<VDivPoint>,
}
#[derive(Serialize)]
struct VDivBundle {
    axes: Vec<String>,
    clusters: Vec<VDivCluster>,
}

fn build_divergence(clusters: &[StoryCluster]) -> VDivBundle {
    let mut out = Vec::new();
    for c in clusters {
        let outlets = unique_outlets(&c.members);
        let mut points = Vec::new();
        // NOTE: we still use cluster-level digest for all outlets (member-level text may not be available here).
        for o in outlets {
            let (x, y, c_pol) = stance_from_text(&c.digest_abridged);
            let c_mag = ((c_pol + 1.0) / 2.0).clamp(0.0, 1.0);
            points.push(VDivPoint {
                outlet: o,
                x,
                y,
                certainty: c_pol,   // + = speculation, − = certainty (see doc above)
                certainty01: c_mag, // 0..1 magnitude where 1 ~ certain
            });
        }
        out.push(VDivCluster {
            id: c.cluster_id.clone(),
            title: c.canonical_title.clone(),
            points,
        });
    }
    VDivBundle {
        axes: vec![
            "blame→cause".into(),
            "risk→optimism".into(),
            "speculation→certainty".into(),
        ],
        clusters: out,
    }
}

/* -------------------------------------------------------------------------- */
/* 4) Emotion Weather                                                         */
/* -------------------------------------------------------------------------- */

#[derive(Serialize)]
struct VEmotionSeries {
    edition: String,
    scores: [f32; 4], // [anxiety, optimism, panic, ambiguity]
}
#[derive(Serialize)]
struct VEmotion {
    grid: [String; 4],
    series: Vec<VEmotionSeries>,
    top_examples: HashMap<String, Vec<String>>,
}

fn build_emotion(clusters: &[StoryCluster], editions: &[String]) -> VEmotion {
    // accumulate per edition
    let mut per_edition_scores: Vec<[f32; 4]> = vec![[0.0; 4]; editions.len()];
    let mut tops: HashMap<String, Vec<(String, f32)>> = HashMap::new(); // theme -> (cluster_id, score)
    let date_to_idxs = build_date_to_idxs(editions);

    for c in clusters {
        let (anx, opt, pan, amb) = emotion_from_text(&c.digest_abridged);

        // Use precomputed mapping for efficiency
        let mut ed_idxs = edition_indices_for_cluster_with_idx(&c.members, editions, &date_to_idxs);

        // Fallback mapping via date/URL if none matched directly
        if ed_idxs.is_empty() {
            let mut set = BTreeSet::new();
            for m in &c.members {
                for idx in edition_indices_for_member(m, editions, &date_to_idxs) {
                    set.insert(idx);
                }
            }
            ed_idxs = set.into_iter().collect::<Vec<_>>();
        }

        for &idx in &ed_idxs {
            per_edition_scores[idx][0] += anx;
            per_edition_scores[idx][1] += opt;
            per_edition_scores[idx][2] += pan;
            per_edition_scores[idx][3] += amb;
        }
        push_top(&mut tops, "anxiety", c.cluster_id.clone(), anx);
        push_top(&mut tops, "optimism", c.cluster_id.clone(), opt);
        push_top(&mut tops, "panic", c.cluster_id.clone(), pan);
        push_top(&mut tops, "ambiguity", c.cluster_id.clone(), amb);
    }

    // normalize columns
    let mut series = Vec::new();
    let maxes = column_maxes(&per_edition_scores);
    for (i, ed) in editions.iter().enumerate() {
        let mut s = per_edition_scores[i];
        for j in 0..4 {
            if maxes[j] > 0.0 {
                s[j] = (s[j] / maxes[j]).clamp(0.0, 1.0);
            }
        }
        series.push(VEmotionSeries {
            edition: ed.clone(),
            scores: s,
        });
    }

    let mut top_examples: HashMap<String, Vec<String>> = HashMap::new();
    for (k, mut v) in tops {
        v.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        top_examples.insert(k, v.into_iter().take(5).map(|(id, _)| id).collect());
    }

    VEmotion {
        grid: ["anxiety".into(), "optimism".into(), "panic".into(), "ambiguity".into()],
        series,
        top_examples,
    }
}

/* -------------------------------------------------------------------------- */
/* 5) Meta-Topic Compass                                                      */
/* -------------------------------------------------------------------------- */

#[derive(Serialize)]
struct VCompassCluster {
    id: String,
    title: String,
    quad: String,
    weight: f32,
    /// Raw bucket scores used to choose the quadrant (already scaled).
    scores: HashMap<String, f32>,
    /// Normalized 2D vector used for centroid [x, y]
    vector: [f32; 2],
}
#[derive(Serialize)]
struct VCompass {
    quadrants: [String; 4],
    clusters: Vec<VCompassCluster>,
    centroid: Option<[f32; 2]>,
}

fn build_compass(clusters: &[StoryCluster]) -> VCompass {
    let mut out = Vec::new();
    let mut sum: [f32; 2] = [0.0, 0.0];
    let mut n = 0.0;

    for c in clusters {
        let (quad, vx, vy, scores) = compass_from_text(&c.digest_abridged);

        // Weight = attention mass (size × span), lightly compressed (sqrt) and scaled.
        let members = c.members.len() as f32;
        let ed_span = unique_edition_count(&c.members) as f32;
        let weight = ((members.sqrt() * ed_span.sqrt()) / 10.0).clamp(0.05, 1.0);

        sum[0] += vx * weight;
        sum[1] += vy * weight;
        n += weight;

        out.push(VCompassCluster {
            id: c.cluster_id.clone(),
            title: c.canonical_title.clone(),
            quad,
            weight,
            scores,
            vector: [vx, vy],
        });
    }

    let centroid = if n > 0.0 { Some([sum[0] / n, sum[1] / n]) } else { None };

    VCompass {
        quadrants: ["structural".into(), "conflict".into(), "human".into(), "future".into()],
        clusters: out,
        centroid,
    }
}

/* -------------------------------------------------------------------------- */
/* 6) Silences                                                                */
/* -------------------------------------------------------------------------- */

#[derive(Serialize)]
struct VSilences {
    expectations: Vec<serde_json::Value>,
    by_outlet: Vec<serde_json::Value>,
}

fn build_silences(insights_json: &str) -> VSilences {
    // Parse MetaInsights if possible; otherwise return empty shells
    let parsed: Result<MetaInsights, _> = serde_json::from_str(insights_json);
    if let Ok(m) = parsed {
        // Map fields defensively; ignore missing ones.
        let expectations: Vec<serde_json::Value> = m
            .silences
            .iter()
            .map(|s| json!({ "theme": s.theme, "expectedButMissing": s.expectedButMissing }))
            .collect();
        VSilences { expectations, by_outlet: vec![] }
    } else {
        VSilences { expectations: vec![], by_outlet: vec![] }
    }
}

/* -------------------------------------------------------------------------- */
/* 7) Word Clouds (CLEANED)                                                   */
/* -------------------------------------------------------------------------- */

#[derive(Serialize)]
struct VCloudByOutlet {
    outlet: String,
    tokens: Vec<(String, u32)>,
}
#[derive(Serialize)]
struct VClouds {
    by_outlet: Vec<VCloudByOutlet>,
    by_cluster: HashMap<String, Vec<(String, u32)>>,
}

fn build_clouds(clusters: &[StoryCluster]) -> VClouds {
    // Aggregate cluster-level tokens (cleaned, outlet-agnostic)
    let mut cluster_tokens: HashMap<String, Vec<(String, u32)>> = HashMap::new();

    // For outlets, we recompute per outlet with outlet-specific stopwords so brand/etc. is removed.
    let mut outlet_tokens: HashMap<String, HashMap<String, u32>> = HashMap::new();

    for c in clusters {
        // by-cluster tokens (use a generic stop set; no outlet brand removal here)
        let toks_cluster = top_tokens_clean(&c.digest_abridged, None, 30);
        cluster_tokens.insert(c.cluster_id.clone(), toks_cluster.clone());

        // per-outlet aggregation with outlet-aware cleaning
        let outlets = unique_outlets(&c.members);
        for o in outlets {
            let entry = outlet_tokens.entry(o.clone()).or_default();
            let toks_outlet = top_tokens_clean(&c.digest_abridged, Some(&o), 30);
            for (t, w) in toks_outlet {
                *entry.entry(t).or_insert(0) += w;
            }
        }
    }

    let mut by_outlet = Vec::new();
    for (o, m) in outlet_tokens {
        let mut v: Vec<(String, u32)> = m.into_iter().collect();
        v.sort_by_key(|(_, w)| std::cmp::Reverse(*w));
        // Enforce diversity: avoid repeated stems dominating (very light heuristic)
        dedupe_by_stem(&mut v);
        v.truncate(60);
        by_outlet.push(VCloudByOutlet { outlet: o, tokens: v });
    }

    VClouds { by_outlet, by_cluster: cluster_tokens }
}

/* -------------------------------------------------------------------------- */
/* 8) Fingerprints                                                            */
/* -------------------------------------------------------------------------- */

#[derive(Serialize)]
struct VFingerprint {
    id: String,
    title: String,
    radial: HashMap<String, f32>,
}
#[derive(Serialize)]
struct VFingerprints {
    clusters: Vec<VFingerprint>,
}

fn build_fingerprints(clusters: &[StoryCluster]) -> VFingerprints {
    let mut out = Vec::new();
    for c in clusters {
        let (risk, optimism, blame, cause, certainty) = fingerprint_from_text(&c.digest_abridged);
        let mut radial = HashMap::new();
        radial.insert("risk".into(), risk);
        radial.insert("optimism".into(), optimism);
        radial.insert("blame".into(), blame);
        radial.insert("cause".into(), cause);
        radial.insert("certainty".into(), certainty);
        out.push(VFingerprint {
            id: c.cluster_id.clone(),
            title: c.canonical_title.clone(),
            radial,
        });
    }
    VFingerprints { clusters: out }
}

/* -------------------------------------------------------------------------- */
/* 9) Entities graph (co-mention)                                             */
/* -------------------------------------------------------------------------- */

#[derive(Serialize)]
struct VEntityNode {
    id: String,    // entity string
    label: String, // display label
    count: u32,    // in how many clusters it appears
}

#[derive(Serialize)]
struct VEntityEdge {
    source: String,
    target: String,
    weight: u32, // co-mentions across clusters
}

#[derive(Serialize)]
struct VEntities {
    nodes: Vec<VEntityNode>,
    edges: Vec<VEntityEdge>,
}

fn build_entities(clusters: &[StoryCluster]) -> VEntities {
    use std::cmp::Ordering;

    let mut entity_counts: HashMap<String, u32> = HashMap::new();
    let mut co_counts: HashMap<(String, String), u32> = HashMap::new();

    for c in clusters {
        // Unique entities within a cluster to avoid double-counting duplicate strings
        let mut ents: Vec<String> = c.entities.iter().map(|s| s.trim().to_string()).collect();
        ents.retain(|e| !e.is_empty());
        ents.sort();
        ents.dedup();

        for e in &ents {
            *entity_counts.entry(e.clone()).or_insert(0) += 1;
        }

        for i in 0..ents.len() {
            for j in (i + 1)..ents.len() {
                let a = ents[i].clone();
                let b = ents[j].clone();
                let key = if a.cmp(&b) == Ordering::Less { (a, b) } else { (b, a) };
                *co_counts.entry(key).or_insert(0) += 1;
            }
        }
    }

    let mut nodes: Vec<VEntityNode> = entity_counts
        .into_iter()
        .map(|(id, count)| VEntityNode {
            label: id.clone(),
            id,
            count,
        })
        .collect();
    nodes.sort_by_key(|n| std::cmp::Reverse(n.count));

    // Optional pruning knobs:
    let min_edge_weight = 1u32;
    let mut edges: Vec<VEntityEdge> = co_counts
        .into_iter()
        .filter(|(_, w)| *w >= min_edge_weight)
        .map(|((a, b), w)| VEntityEdge {
            source: a,
            target: b,
            weight: w,
        })
        .collect();
    edges.sort_by_key(|e| std::cmp::Reverse(e.weight));

    VEntities { nodes, edges }
}

/* -------------------------------------------------------------------------- */
/* Helpers                                                                    */
/* -------------------------------------------------------------------------- */

fn estimate_article_count(clusters: &[StoryCluster]) -> usize {
    clusters.iter().map(|c| c.members.len()).sum()
}

fn estimate_outlet_count(clusters: &[StoryCluster]) -> usize {
    let mut s = BTreeSet::new();
    for c in clusters {
        for m in &c.members {
            s.insert(outlet_from_source(&m.source));
        }
    }
    s.len()
}

fn unique_outlets(members: &[ClusterMember]) -> Vec<String> {
    let mut s = BTreeSet::new();
    for m in members {
        s.insert(outlet_from_source(&m.source));
    }
    s.into_iter().collect()
}

fn unique_edition_count(members: &[ClusterMember]) -> usize {
    let mut s = BTreeSet::new();
    for m in members {
        s.insert(m.edition_id.clone());
    }
    s.len()
}

/// Build a map `YYYY-MM-DD` → indices of all editions occurring on that date.
fn build_date_to_idxs(editions: &[String]) -> HashMap<String, Vec<usize>> {
    let mut date_to_idxs: HashMap<String, Vec<usize>> = HashMap::new();
    for (i, ed) in editions.iter().enumerate() {
        if ed.len() >= 10 {
            let date = ed[..10].to_string();
            date_to_idxs.entry(date).or_default().push(i);
        }
    }
    date_to_idxs
}

/// Robust per-member mapping to edition indices:
/// 1) Exact edition_id match
/// 2) If edition_id is a date (YYYY-MM-DD), map to all editions on that date
/// 3) Else, try to extract a date from the source URL and map to all editions on that date
fn edition_indices_for_member(
    m: &ClusterMember,
    editions: &[String],
    date_to_idxs: &HashMap<String, Vec<usize>>,
) -> Vec<usize> {
    // 1) Exact edition-id match
    if let Some(i) = editions.iter().position(|e| e == &m.edition_id) {
        return vec![i];
    }
    // 2) edition-id looks like date
    if looks_like_date(&m.edition_id) {
        let date = m.edition_id[..10].to_string();
        if let Some(v) = date_to_idxs.get(&date) {
            return v.clone();
        }
    }
    // 3) Extract date from source URL/text
    if let Some(date) = extract_date_from_str(&m.source) {
        if let Some(v) = date_to_idxs.get(&date) {
            return v.clone();
        }
    }
    vec![]
}

/// Union of indices across all members (using robust mapping) — version that reuses a precomputed map.
fn edition_indices_for_cluster_with_idx(
    members: &[ClusterMember],
    editions: &[String],
    date_to_idxs: &HashMap<String, Vec<usize>>,
) -> Vec<usize> {
    let mut s = BTreeSet::new();
    for m in members {
        for idx in edition_indices_for_member(m, editions, date_to_idxs) {
            s.insert(idx);
        }
    }
    s.into_iter().collect()
}

/// Backward-compatible version that constructs its own date index (kept for other call sites).
fn edition_indices_for_cluster(members: &[ClusterMember], editions: &[String]) -> Vec<usize> {
    let date_to_idxs = build_date_to_idxs(editions);
    edition_indices_for_cluster_with_idx(members, editions, &date_to_idxs)
}

fn column_maxes(v: &[[f32; 4]]) -> [f32; 4] {
    let mut maxes = [0.0; 4];
    for row in v {
        for j in 0..4 {
            if row[j] > maxes[j] {
                maxes[j] = row[j];
            }
        }
    }
    maxes
}

fn push_top(map: &mut HashMap<String, Vec<(String, f32)>>, key: &str, id: String, score: f32) {
    map.entry(key.to_string()).or_default().push((id, score));
}

/* ------------------ lightweight keyword heuristics over text --------------- */

fn stance_from_text(txt: &str) -> (f32, f32, f32) {
    // (x, y, certainty_polarity) with boundary-aware counting + smoothing
    let t = txt.to_lowercase();

    // blame vs cause
    let blame = count_tokens(&t, &["blame", "fault", "accuse", "responsible"]);
    let cause = count_tokens(&t, &["cause", "drivers"])
        + count_phrases(&t, &["due to", "because of"]);

    // risk vs optimism
    let risk = count_tokens(&t, &["risk", "threat", "concern", "uncertain", "volatile", "danger"]);
    let prog =
        count_tokens(&t, &["progress", "improved", "optimistic", "advance", "breakthrough", "recovery"]);

    // certainty vs speculation
    let certain = count_tokens(&t, &["confirmed", "official", "definitive", "certified"]);
    let spec = count_tokens(&t, &["speculation", "rumors", "unconfirmed", "alleged"]);

    let x = norm_smooth(blame as f32, cause as f32); // blame vs cause (-1..1)
    let y = norm_smooth(risk as f32, prog as f32);   // risk vs optimism (-1..1)
    let c = norm_smooth(certain as f32, spec as f32); // certainty vs speculation (-1..1)

    (x, y, c)
}

fn emotion_from_text(txt: &str) -> (f32, f32, f32, f32) {
    let t = txt.to_lowercase();
    let anxiety = count_tokens(&t, &["fear", "worry", "anxiety", "turmoil", "instability"]) as f32;
    let optimism = count_tokens(&t, &["optimism", "hope", "progress", "recovery", "breakthrough"]) as f32;
    let panic = count_tokens(&t, &["panic", "crisis", "emergency", "collapse"]) as f32;
    let ambiguity = count_tokens(&t, &["unclear", "uncertain", "ambiguous", "conflicting"]) as f32;
    (anxiety, optimism, panic, ambiguity)
}

// Returns (quad, vx, vy, scores)
//  - vx = (structural - human) / sum
//  - vy = (conflict  - future) / sum
// where sum = structural + conflict + human + future (if sum==0 → vector [0,0]).
fn compass_from_text(txt: &str) -> (String, f32, f32, HashMap<String, f32>) {
    let t = txt.to_lowercase();

    // Balanced keyword sets.
    // Structural: institutions, policy, budgets, courts, sanctions, trade, climate (structural angle).
    let structural = count_tokens(
        &t,
        &[
            "policy",
            "policies",
            "economy",
            "budget",
            "budgets",
            "institution",
            "institutions",
            "regulation",
            "regulations",
            "law",
            "laws",
            "tax",
            "taxes",
            "welfare",
            "infrastructure",
            "agency",
            "agencies",
            "court",
            "courts",
            "legislation",
            "congress",
            "parliament",
            "administration",
            "sanction",
            "sanctions",
            "tariff",
            "tariffs",
            "trade",
            "climate",
            "emissions",
            "carbon",
        ],
    ) as f32;

    // Conflict: war/violence/legal fights/protests/escalations.
    let conflict = count_tokens(
        &t,
        &[
            "war",
            "conflict",
            "clashes",
            "attack",
            "attacks",
            "strike",
            "strikes",
            "missile",
            "bomb",
            "bombing",
            "battle",
            "battles",
            "riot",
            "riots",
            "protest",
            "protests",
            "arrest",
            "arrests",
            "indict",
            "indicted",
            "charges",
            "trial",
            "ceasefire",
            "tensions",
            "crackdown",
            "shooting",
            "shootings",
        ],
    ) as f32;

    // Human: social/people-centered content.
    let human = count_tokens(
        &t,
        &[
            "health",
            "culture",
            "identity",
            "education",
            "family",
            "victims",
            "workers",
            "students",
            "women",
            "children",
            "community",
            "refugees",
            "migrants",
            "rights",
            "deaths",
            "injured",
            "mental",
            "care",
        ],
    ) as f32;

    // Future: genuinely forward-looking; removed "climate" to reduce skew.
    let future = count_tokens(
        &t,
        &[
            "technology",
            "ai",
            "innovation",
            "research",
            "forecast",
            "forecasts",
            "projection",
            "projections",
            "plan",
            "plans",
            "roadmap",
            "prototype",
            "pilot",
            "next",
            "upcoming",
            "future",
            "launch",
            "launches",
        ],
    ) as f32;

    let sum = structural + conflict + human + future;
    let (vx, vy) = if sum > 0.0 {
        ((structural - human) / sum, (conflict - future) / sum)
    } else {
        (0.0, 0.0)
    };

    // Choose quadrant by max score (stable tie-breaker via order below).
    let mut quad = "structural";
    let mut maxv = structural;
    if conflict > maxv {
        quad = "conflict";
        maxv = conflict;
    }
    if human > maxv {
        quad = "human";
        maxv = human;
    }
    if future > maxv {
        quad = "future";
    }

    let mut scores = HashMap::new();
    // Expose *normalized* bucket shares so they sum to 1.0 when sum>0.
    if sum > 0.0 {
        scores.insert("structural".into(), structural / sum);
        scores.insert("conflict".into(), conflict / sum);
        scores.insert("human".into(), human / sum);
        scores.insert("future".into(), future / sum);
    } else {
        scores.insert("structural".into(), 0.0);
        scores.insert("conflict".into(), 0.0);
        scores.insert("human".into(), 0.0);
        scores.insert("future".into(), 0.0);
    }

    (quad.to_string(), vx, vy, scores)
}

fn fingerprint_from_text(txt: &str) -> (f32, f32, f32, f32, f32) {
    let (x, y, c) = stance_from_text(txt);
    // stance axes: x=(cause−blame)/sum, y=(optimism−risk)/sum, c=(spec−certainty)/sum
    // Convert so each feature reflects its *named* quantity.
    let risk = ((-y + 1.0) / 2.0).clamp(0.0, 1.0);       // high when risk>optimism
    let optimism = 1.0 - risk;

    let blame = ((-x + 1.0) / 2.0).clamp(0.0, 1.0);      // high when blame>cause
    let cause = 1.0 - blame;

    let certainty = ((-c + 1.0) / 2.0).clamp(0.0, 1.0);  // high when certainty>speculation
    (risk, optimism, blame, cause, certainty)
}

/* --------------------------- CLEAN TOKENIZATION ---------------------------- */

fn top_tokens_clean(text: &str, outlet_host: Option<&str>, limit: usize) -> Vec<(String, u32)> {
    // Pre-clean
    let mut s = strip_urls(text);
    s = normalize_apostrophes(&s);
    s = normalize_acronyms(&s);

    // Tokenize (letters and hyphens)
    let token_re: Regex = regex_cached(r"[A-Za-z][A-Za-z\-]+");
    let mut bag: HashMap<String, u32> = HashMap::new();

    let global_stop = global_stopwords();
    let domain_stop = domain_stopwords();
    let months = months_set();

    // Build per-outlet stop set from host parts (e.g., "www.bbc.com" => {"bbc","com"})
    let outlet_stop = outlet_host.map(|h| outlet_host_parts(h)).unwrap_or_else(HashSet::new);

    for cap in token_re.find_iter(&s) {
        let mut t = cap.as_str().to_lowercase();

        // Whitelist short acronyms (do not length-filter them)
        let keep_short = matches!(t.as_str(), "us" | "uk" | "eu" | "ai" | "un" | "nato");

        if t.len() < 2 && !keep_short {
            continue;
        }

        // Filter: global/domain/outlet stop words
        if global_stop.contains(&t) {
            continue;
        }
        if domain_stop.contains(&t) {
            continue;
        }
        if outlet_stop.contains(&t) {
            continue;
        }

        // Drop months & year-like tokens
        if months.contains(&t) || is_year_like(&t) {
            continue;
        }

        // Crude lemmatization (protect acronyms)
        if !keep_short {
            if t.ends_with("ies") && t.len() > 3 {
                t = t.trim_end_matches("ies").to_string() + "y";
            } else if t.ends_with('s') && t.len() > 3 {
                t.pop();
            }
        }

        *bag.entry(t).or_insert(0) += 1;
    }

    // Convert to vec, sort by freq, keep top N
    let mut v: Vec<(String, u32)> = bag.into_iter().collect();

    // Deweight extremely common “generic news” terms that slipped through
    for (w, c) in &mut v {
        if GENERIC_NEWS_WORDS.contains(&w.as_str()) {
            *c = (*c as f32 * 0.5) as u32;
        }
    }

    v.sort_by_key(|(_, c)| std::cmp::Reverse(*c));
    // Encourage diversity: avoid stems repeated too much
    dedupe_by_stem(&mut v);
    v.retain(|(_, c)| *c > 0);
    v.truncate(limit);
    v
}

fn strip_urls(s: &str) -> String {
    let re: Regex = regex_cached(r"https?://\S+");
    re.replace_all(s, " ").to_string()
}

fn normalize_apostrophes(s: &str) -> String {
    // Remove possessives ’s/'s and unify quotes
    let apos_re: Regex = regex_cached(r"[’']s\b");
    let mut out = apos_re.replace_all(s, "").to_string();
    out = out.replace(['“', '”', '‘', '’'], "\"");
    out
}

fn normalize_acronyms(s: &str) -> String {
    // U.S. -> US, E.U. -> EU, etc.
    s.replace("U.S.", "US").replace("U.K.", "UK").replace("E.U.", "EU")
}

fn is_year_like(w: &str) -> bool {
    w.len() == 4 && w.chars().all(|c| c.is_ascii_digit())
}

fn outlet_from_source(src: &str) -> String {
    if let Ok(u) = Url::parse(src) {
        if let Some(h) = u.host_str() {
            return h.trim_start_matches("www.").to_string();
        }
    }
    // Fallback for non-URL or unexpected strings
    src.trim()
        .trim_start_matches("www.")
        .split('/')
        .next()
        .unwrap_or(src)
        .to_lowercase()
        .to_string()
}

fn outlet_host_parts(host: &str) -> HashSet<String> {
    // Trim leading www.* prefixes on the whole host, then split.
    let h = {
        let re = regex_cached(r"^www\d*\.");
        re.replace(host, "").to_string()
    };
    h.split('.')
        .filter(|p| !p.is_empty())
        .map(|p| p.to_string())
        .collect()
}

fn dedupe_by_stem(items: &mut Vec<(String, u32)>) {
    let mut seen: HashSet<String> = HashSet::new();
    items.retain(|(w, _)| {
        let stem = light_stem(w);
        if seen.contains(&stem) {
            false
        } else {
            seen.insert(stem);
            true
        }
    });
}

fn light_stem(w: &str) -> String {
    if w.len() > 4 && w.ends_with('s') {
        return w[..w.len() - 1].to_string();
    }
    if w.ends_with("ing") && w.len() > 5 {
        return w[..w.len() - 3].to_string();
    }
    if w.ends_with("ed") && w.len() > 4 {
        return w[..w.len() - 2].to_string();
    }
    w.to_string()
}

/* ------------------------------ Stop words -------------------------------- */

fn global_stopwords() -> &'static HashSet<String> {
    use once_cell::sync::Lazy;
    static SET: Lazy<HashSet<String>> = Lazy::new(|| {
        let words = [
            // english
            "the","a","an","and","or","but","if","then","of","to","in","on","for","with","as","by",
            "is","are","was","were","be","been","being","that","this","it","its","at","from","into",
            "over","under","about","after","before","between","during","without","within","than",
            "not","no","yes","more","most","less","least","very","much","many","some","any","such",
            // pipeline/meta smells
            "source","article","articles","news","updated","update","analysis","live","breaking",
            "tag","tags","frame","frames","entity","entities","generated","ongoing","potential",
            "changed","what","title","titles",
        ];
        words.iter().map(|s| s.to_string()).collect()
    });
    &SET
}

fn domain_stopwords() -> &'static HashSet<String> {
    use once_cell::sync::Lazy;
    static SET: Lazy<HashSet<String>> = Lazy::new(|| {
        let words = [
            "http", "https", "com", "org", "net", "www", "amp", "m", "co", "io", "newsroom", "press", "blog", "text",
            "apnews",
        ];
        words.iter().map(|s| s.to_string()).collect()
    });
    &SET
}

fn months_set() -> &'static HashSet<String> {
    use once_cell::sync::Lazy;
    static SET: Lazy<HashSet<String>> = Lazy::new(|| {
        let words = [
            "january","february","march","april","may","june","july","august","september","october","november","december",
            "mon","tue","wed","thu","fri","sat","sun",
        ];
        words.iter().map(|s| s.to_string()).collect()
    });
    &SET
}

static GENERIC_NEWS_WORDS: &[&str] = &[
    "public", "state", "national", "government", "official", "leaders", "support", "policy", "movement", "issues",
    "concern",
];

/* ------------------------------- Utilities -------------------------------- */

/// Count tokens with word boundaries (e.g., "cause" won't match "because").
fn count_tokens(text: &str, tokens: &[&str]) -> usize {
    if tokens.is_empty() {
        return 0;
    }
    let pattern = tokens
        .iter()
        .map(|t| regex::escape(t))
        .collect::<Vec<_>>()
        .join("|");
    let re = regex_cached(&format!(r"\b(?:{})\b", pattern));
    re.find_iter(text).count()
}

/// Count multi-word phrases with flexible whitespace (e.g., "due\s+to") and boundaries.
fn count_phrases(text: &str, phrases: &[&str]) -> usize {
    phrases
        .iter()
        .map(|p| {
            let p = regex::escape(p).replace(r"\ ", r"\s+");
            let re = regex_cached(&format!(r"\b{}\b", p));
            re.find_iter(text).count()
        })
        .sum()
}

/// Laplace-smoothed polarity; reduces saturation at ±1.
fn norm_smooth(left: f32, right: f32) -> f32 {
    let alpha = 1.0; // Laplace smoothing
    let num = (right + alpha) - (left + alpha);
    let den = (right + left) + 2.0 * alpha;
    let raw = if den > 0.0 { num / den } else { 0.0 };
    // gentle compression to avoid hard walls
    (raw * 0.85).clamp(-0.95, 0.95)
}

/// Heuristic: "YYYY-MM-DD"
fn looks_like_date(s: &str) -> bool {
    s.len() >= 10
        && s.as_bytes().get(4) == Some(&b'-')
        && s.as_bytes().get(7) == Some(&b'-')
        && s[..4].chars().all(|c| c.is_ascii_digit())
        && s[5..7].chars().all(|c| c.is_ascii_digit())
        && s[8..10].chars().all(|c| c.is_ascii_digit())
}

/// Extract "YYYY-MM-DD" from strings like ".../YYYY/MM/DD/..." or "...YYYY-MM-DD..."
fn extract_date_from_str(s: &str) -> Option<String> {
    let re: Regex = regex_cached(r"(?P<y>20\d{2})[/-](?P<m>\d{2})[/-](?P<d>\d{2})");
    if let Some(c) = re.captures(s) {
        let y = c.name("y")?.as_str();
        let m = c.name("m")?.as_str();
        let d = c.name("d")?.as_str();
        return Some(format!("{y}-{m}-{d}"));
    }
    None
}

/// Thread-safe regex cache (clone-on-read).
fn regex_cached(pat: &str) -> Regex {
    use once_cell::sync::Lazy;
    use std::sync::RwLock;

    static REGEX_CACHE: Lazy<RwLock<HashMap<&'static str, Regex>>> =
        Lazy::new(|| RwLock::new(HashMap::new()));

    if let Some(r) = REGEX_CACHE.read().unwrap().get(pat) {
        return r.clone();
    }
    let compiled = Regex::new(pat).unwrap();
    let mut w = REGEX_CACHE.write().unwrap();
    // Leak the pattern string to get a &'static str key (finite set of patterns)
    let key: &'static str = Box::leak(pat.to_string().into_boxed_str());
    w.insert(key, compiled.clone());
    compiled
}