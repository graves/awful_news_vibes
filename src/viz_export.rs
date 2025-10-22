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

use crate::models::{Article, ClusterMember, StoryCluster};
use crate::out_models::MetaInsights;

/* -------------------------------------------------------------------------- */
/* Entry point                                                                */
/* -------------------------------------------------------------------------- */

/// Public entry point: write all D3-ready visualization JSONs into `out/<date>/`.
pub fn write_all_viz(
    out_dir_for_date: &Path, // e.g., out/2025-10-28
    date: &str,              // "YYYY-MM-DD" (anchor 'today')
    clusters: &[StoryCluster],
    articles: &[Article],    // All articles for word cloud generation from original content
    insights_json: &str,
    editions: &[String], // ordered editions used this run
) -> Result<()> {
    write_all_viz_internal(out_dir_for_date, date, clusters, articles, insights_json, editions)?;
    
    // Update the root-level index.json
    if let Some(root_dir) = out_dir_for_date.parent() {
        update_root_index(root_dir, date)?;
    }
    
    Ok(())
}

fn write_all_viz_internal(
    out_dir_for_date: &Path,
    date: &str,
    clusters: &[StoryCluster],
    articles: &[Article],
    insights_json: &str,
    editions: &[String],
) -> Result<()> {
    fs::create_dir_all(out_dir_for_date).with_context(|| format!("create {:?}", out_dir_for_date))?;

    // 1) Lifecycles
    let lifecycles = build_lifecycles(clusters, editions);
    write_json(out_dir_for_date.join("viz.lifecycles.json"), &lifecycles)?;

    // 2) Momentum (derived from lifecycles)
    let momentum = build_momentum(&lifecycles);
    write_json(out_dir_for_date.join("viz.momentum.json"), &momentum)?;

    // 3) Narrative Divergence (from original article content per outlet)
    let divergence = build_divergence(clusters, articles);
    write_json(out_dir_for_date.join("viz.divergence.json"), &divergence)?;

    // 4) Emotion Weather (from original article content per edition)
    let emotion = build_emotion(clusters, articles, editions);
    write_json(out_dir_for_date.join("viz.emotion.json"), &emotion)?;

    // 5) Meta-Topic Compass (from original article content)
    let compass = build_compass(clusters, articles);
    write_json(out_dir_for_date.join("viz.compass.json"), &compass)?;

    // 6) Silences (surface from insights.json if present)
    let silences = build_silences(insights_json);
    write_json(out_dir_for_date.join("viz.silences.json"), &silences)?;

    // 7) Word Clouds (cleaned tokens; per-outlet + per-cluster) - NOW FROM ORIGINAL CONTENT
    let clouds = build_clouds(clusters, articles);
    write_json(out_dir_for_date.join("viz.clouds.json"), &clouds)?;

    // 8) Narrative Fingerprints (from original article content)
    let fingerprints = build_fingerprints(clusters, articles);
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
    
    // Minimum importance threshold
    if total < 4 && peak < 3 && duration < 3 {
        return "steady".into();
    }
    
    // Flashstorm: brief intense spike (≥60% of volume in peak edition)
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
    if peak >= 2 && duration >= 3 && has_growth_pattern(vol) {
        return "wildfire".into();
    }
    
    // Slow freeze: declining story
    if duration >= 3 && has_decline_pattern(vol) {
        let first_nonzero = vol.iter().find(|&&x| x > 0).copied().unwrap_or(0);
        if first_nonzero >= 2 {
            return "slow_freeze".into();
        }
    }
    
    "steady".into()
}

fn has_growth_pattern(vol: &[u32]) -> bool {
    let nonzero: Vec<u32> = vol.iter().copied().filter(|&x| x > 0).collect();
    if nonzero.len() < 2 {
        return false;
    }
    
    // Later editions must have 20% more average than early editions
    let mid = nonzero.len() / 2;
    let early_avg: f32 = nonzero[..mid].iter().sum::<u32>() as f32 / mid as f32;
    let late_avg: f32 = nonzero[mid..].iter().sum::<u32>() as f32 / (nonzero.len() - mid) as f32;
    
    late_avg > early_avg * 1.2
}

fn has_decline_pattern(vol: &[u32]) -> bool {
    let nonzero: Vec<u32> = vol.iter().copied().filter(|&x| x > 0).collect();
    if nonzero.len() < 2 {
        return false;
    }
    
    // Earlier editions must have 20% more average than later editions
    let mid = nonzero.len() / 2;
    let early_avg: f32 = nonzero[..mid].iter().sum::<u32>() as f32 / mid as f32;
    let late_avg: f32 = nonzero[mid..].iter().sum::<u32>() as f32 / (nonzero.len() - mid) as f32;
    
    early_avg > late_avg * 1.2
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
    /// Signed polarity: -1 (certain/definitive) to +1 (speculative/unconfirmed)
    certainty: f32,
    /// Normalized to 0..1 where 0=certain, 1=speculative (inverse for compatibility)
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

fn build_divergence(clusters: &[StoryCluster], articles: &[Article]) -> VDivBundle {
    // Build article lookup
    let article_map: HashMap<String, &Article> = articles.iter()
        .map(|a| (a.id.clone(), a))
        .collect();
    
    let mut out = Vec::new();
    for c in clusters {
        // Collect article content per outlet
        let mut outlet_content: HashMap<String, String> = HashMap::new();
        
        for member in &c.members {
            if let Some(article) = article_map.get(&member.article_id) {
                if !article.content.is_empty() {
                    let outlet = outlet_from_source(&member.source);
                    outlet_content.entry(outlet.clone()).or_default().push_str(&article.content);
                    outlet_content.get_mut(&outlet).unwrap().push(' ');
                }
            }
        }
        
        let mut points = Vec::new();
        for (outlet, content) in outlet_content {
            let (x, y, c_pol) = stance_from_text(&content);
            let c_mag = ((c_pol + 1.0) / 2.0).clamp(0.0, 1.0);
            points.push(VDivPoint {
                outlet,
                x,
                y,
                certainty: c_pol,
                certainty01: c_mag,
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
            "certain→speculative".into(),
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

fn build_emotion(clusters: &[StoryCluster], articles: &[Article], editions: &[String]) -> VEmotion {
    // Build article lookup
    let article_map: HashMap<String, &Article> = articles.iter()
        .map(|a| (a.id.clone(), a))
        .collect();
    
    // accumulate per edition
    let mut per_edition_scores: Vec<[f32; 4]> = vec![[0.0; 4]; editions.len()];
    let mut tops: HashMap<String, Vec<(String, f32)>> = HashMap::new();
    let date_to_idxs = build_date_to_idxs(editions);

    for c in clusters {
        // Collect article content for this cluster
        let mut cluster_content = String::new();
        for member in &c.members {
            if let Some(article) = article_map.get(&member.article_id) {
                if !article.content.is_empty() {
                    cluster_content.push_str(&article.content);
                    cluster_content.push(' ');
                }
            }
        }
        
        let (anx, opt, pan, amb) = emotion_from_text(&cluster_content);

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

fn build_compass(clusters: &[StoryCluster], articles: &[Article]) -> VCompass {
    // Build article lookup
    let article_map: HashMap<String, &Article> = articles.iter()
        .map(|a| (a.id.clone(), a))
        .collect();
    
    let mut out = Vec::new();
    let mut sum: [f32; 2] = [0.0, 0.0];
    let mut n = 0.0;

    for c in clusters {
        // Collect article content for this cluster
        let mut cluster_content = String::new();
        for member in &c.members {
            if let Some(article) = article_map.get(&member.article_id) {
                if !article.content.is_empty() {
                    cluster_content.push_str(&article.content);
                    cluster_content.push(' ');
                }
            }
        }
        
        // Enhanced: pass entities to boost category scores
        let entity_names: Vec<String> = c.entities.iter().cloned().collect();
        let (quad, vx, vy, scores) = compass_from_text_enhanced(&cluster_content, &entity_names);

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
/* 7) Word Clouds (CLEANED + FIXED multi-counting)                           */
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

fn build_clouds(clusters: &[StoryCluster], articles: &[Article]) -> VClouds {
    // Build article lookup by ID
    let article_map: HashMap<String, &Article> = articles.iter()
        .map(|a| (a.id.clone(), a))
        .collect();

    // Aggregate cluster-level tokens from ORIGINAL CONTENT
    let mut cluster_tokens: HashMap<String, Vec<(String, u32)>> = HashMap::new();

    // For outlets: collect original article content per outlet
    let mut outlet_article_content: HashMap<String, Vec<String>> = HashMap::new();

    for c in clusters {
        // Collect all article content for this cluster
        let mut cluster_content = String::new();
        
        for member in &c.members {
            if let Some(article) = article_map.get(&member.article_id) {
                if !article.content.is_empty() {
                    cluster_content.push_str(&article.content);
                    cluster_content.push(' ');
                    
                    // Also track per-outlet
                    let outlet = outlet_from_source(&member.source);
                    outlet_article_content.entry(outlet).or_default().push(article.content.clone());
                }
            }
        }
        
        // by-cluster tokens from ORIGINAL CONTENT
        let toks_cluster = top_tokens_clean(&cluster_content, None, 30);
        cluster_tokens.insert(c.cluster_id.clone(), toks_cluster);
    }

    // Now tokenize per-outlet with all their article content
    let mut by_outlet = Vec::new();
    for (outlet, contents) in outlet_article_content {
        let combined_content = contents.join(" ");
        let mut toks = top_tokens_clean(&combined_content, Some(&outlet), 60);
        
        // Enforce diversity: avoid repeated stems dominating
        dedupe_by_stem(&mut toks);
        toks.truncate(60);
        
        by_outlet.push(VCloudByOutlet { outlet, tokens: toks });
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

fn build_fingerprints(clusters: &[StoryCluster], articles: &[Article]) -> VFingerprints {
    // Build article lookup
    let article_map: HashMap<String, &Article> = articles.iter()
        .map(|a| (a.id.clone(), a))
        .collect();
    
    let mut out = Vec::new();
    for c in clusters {
        // Collect article content for this cluster
        let mut cluster_content = String::new();
        for member in &c.members {
            if let Some(article) = article_map.get(&member.article_id) {
                if !article.content.is_empty() {
                    cluster_content.push_str(&article.content);
                    cluster_content.push(' ');
                }
            }
        }
        
        let (risk, optimism, blame, cause, certainty) = fingerprint_from_text(&cluster_content);
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

/* ------------------ Improved keyword heuristics over text ------------------ */

fn stance_from_text(txt: &str) -> (f32, f32, f32) {
    let t = txt.to_lowercase();

    // Blame: Attribution of fault/wrongdoing (added: guilty, liable, scapegoat, faulted)
    let blame = count_tokens(&t, &[
        "blame", "blamed", "blames", "blaming",
        "fault", "faulted",
        "accuse", "accused", "accusation",
        "guilty",
        "liable",
        "scapegoat", "scapegoating",
    ]) + count_phrases(&t, &["at fault", "to blame for"]);
    
    // Cause: Causal explanation/analysis (added: reason, underlying, trigger, contribute, root cause)
    let cause = count_tokens(&t, &[
        "cause", "causes", "caused", "causing",
        "driver", "drivers", "driving",
        "factor", "factors",
        "reason", "reasons",
        "trigger", "triggered", "triggers",
        "contribute", "contributed", "contributes",
        "underlying",
        "lead", "led", "leading",
    ]) + count_phrases(&t, &["due to", "because of", "result of", "stems from", "root cause", "leads to"]);

    // Risk: Threat/danger/uncertainty (added: hazard, vulnerable, peril, exposed)
    let risk = count_tokens(&t, &[
        "risk", "risks", "risky",
        "threat", "threats", "threaten", "threatening",
        "concern", "concerns", "concerned", "concerning",
        "uncertain", "uncertainty",
        "volatile", "volatility",
        "danger", "dangers", "dangerous",
        "hazard", "hazards", "hazardous",
        "vulnerable", "vulnerability",
        "peril", "perilous",
        "jeopardy",
        "exposed", "exposure",
    ]) + count_phrases(&t, &["at risk"]);
    
    // Optimism/Progress: Positive outlook/improvement (added: promising, encouraging, gains, success, upbeat)
    let prog = count_tokens(&t, &[
        "progress", "progressing",
        "improved", "improvement", "improving", "improves",
        "optimistic", "optimism",
        "advance", "advances", "advancing", "advanced",
        "breakthrough", "breakthroughs",
        "recovery", "recovering", "recovered",
        "positive", "positively",
        "promising",
        "encouraged", "encouraging",
        "gains", "gain", "gained",
        "success", "successful", "successfully",
        "better",
        "upbeat",
        "turnaround",
    ]);

    // Certainty: Definitive/confirmed statements (added: proven, established, documented, validated)
    let certain = count_tokens(&t, &[
        "confirmed", "confirms", "confirm",
        "official", "officially",
        "definitive", "definitively",
        "certified",
        "verified", "verify",
        "proven", "proof",
        "established",
        "documented",
        "validated", "validation",
        "conclusive",
    ]) + count_phrases(&t, &["evidence shows"]);
    
    // Speculation: Unconfirmed/hedged statements (added: claims, suggest, may/might/could, possibly, believed)
    let spec = count_tokens(&t, &[
        "speculation", "speculate", "speculative",
        "rumors", "rumor", "rumored",
        "unconfirmed",
        "alleged", "allegedly", "allegation",
        "reportedly",
        "claim", "claims", "claimed",
        "suggest", "suggests", "suggested",
        "may", "might", "could",
        "possibly", "possible",
        "potential", "potentially",
        "believed",
        "appears",
    ]) + count_phrases(&t, &["sources say", "unnamed sources", "believed to", "appears to"]);

    let x = norm_smooth(blame as f32, cause as f32);
    let y = norm_smooth(risk as f32, prog as f32);
    let c = norm_smooth(certain as f32, spec as f32);

    (x, y, c)
}

fn emotion_from_text(txt: &str) -> (f32, f32, f32, f32) {
    let t = txt.to_lowercase();
    
    // Anxiety: General fear/worry (added: nervous, uneasy, tense, alarm, dread)
    let anxiety = count_tokens(&t, &[
        "fear", "fears", "feared", "fearful",
        "worry", "worried", "worries", "worrying",
        "anxiety", "anxious",
        "turmoil",
        "instability", "unstable",
        "nervous", "nervously",
        "uneasy", "unease",
        "tense", "tension",
        "alarm", "alarmed", "alarming",
        "dread", "dreaded",
    ]) as f32;
    
    // Optimism: Positive outlook (already comprehensive, added: hopeful, encouraged, uplifting)
    let optimism = count_tokens(&t, &[
        "optimism", "optimistic",
        "hope", "hopeful", "hoping",
        "progress", "progressing",
        "recovery", "recovering",
        "breakthrough", "breakthroughs",
        "positive", "positively",
        "encouraged", "encouraging",
        "promising",
        "uplifting",
    ]) as f32;
    
    // Panic: Acute crisis/emergency (added: chaos, catastrophe, disaster, dire, urgent)
    let panic = count_tokens(&t, &[
        "panic", "panicked", "panicking",
        "crisis",
        "emergency", "emergencies",
        "collapse", "collapsed", "collapsing",
        "chaos", "chaotic",
        "catastrophe", "catastrophic",
        "disaster", "disastrous",
        "urgent", "urgently", "urgency",
        "dire",
        "critical",
    ]) as f32;
    
    // Ambiguity: Confusion/contradiction (added: confusion, disputed, mixed, questioned)
    let ambiguity = count_tokens(&t, &[
        "unclear",
        "uncertain", "uncertainty",
        "ambiguous", "ambiguity",
        "conflicting",
        "contradictory", "contradiction", "contradicts",
        "confusion", "confused", "confusing",
        "disputed", "dispute", "disputes",
        "debated", "debate",
        "questioned", "questioning",
    ]) + count_phrases(&t, &["mixed signals"]) as usize;
    
    (anxiety, optimism, panic, ambiguity as f32)
}

/// Enhanced compass analysis with weighted keywords and entity boosting
fn compass_from_text_enhanced(txt: &str, entities: &[String]) -> (String, f32, f32, HashMap<String, f32>) {
    let t = txt.to_lowercase();
    
    // Define weighted keywords - stronger signals get higher multipliers
    struct WeightedKeywords {
        words: Vec<&'static str>,
        weight: f32,
    }
    
    // STRUCTURAL - weighted by signal strength
    let structural_keywords = vec![
        WeightedKeywords { words: vec!["policy", "policies", "law", "laws", "legislation", "regulation"], weight: 2.5 },
        WeightedKeywords { words: vec!["economy", "economic", "fiscal", "monetary", "budget"], weight: 2.0 },
        WeightedKeywords { words: vec!["government", "administration", "parliament", "congress", "senate"], weight: 1.8 },
        WeightedKeywords { words: vec!["institution", "institutions", "agency", "agencies", "ministry"], weight: 1.5 },
        WeightedKeywords { words: vec!["court", "courts", "legal", "judicial"], weight: 1.5 },
        WeightedKeywords { words: vec!["tax", "taxes", "taxation", "tariff", "tariffs", "trade", "sanction", "sanctions"], weight: 1.3 },
        WeightedKeywords { words: vec!["reform", "reforms", "governance", "infrastructure", "spending", "welfare"], weight: 1.0 },
    ];
    
    // CONFLICT - weighted by intensity
    let conflict_keywords = vec![
        WeightedKeywords { words: vec!["war", "wars", "warfare", "combat"], weight: 3.0 },
        WeightedKeywords { words: vec!["attack", "attacks", "bomb", "bombs", "bombing", "missile", "missiles"], weight: 2.8 },
        WeightedKeywords { words: vec!["violence", "violent", "assault", "assaults", "shooting"], weight: 2.5 },
        WeightedKeywords { words: vec!["battle", "battles", "fighting", "fought", "strike", "strikes"], weight: 2.3 },
        WeightedKeywords { words: vec!["conflict", "conflicts", "clash", "clashes", "hostilities"], weight: 2.0 },
        WeightedKeywords { words: vec!["protest", "protests", "riot", "riots", "crackdown"], weight: 1.8 },
        WeightedKeywords { words: vec!["arrest", "arrests", "charges", "charged", "trial", "indict"], weight: 1.5 },
        WeightedKeywords { words: vec!["tensions", "confrontation", "ceasefire", "aggression"], weight: 1.0 },
    ];
    
    // HUMAN - weighted by impact
    let human_keywords = vec![
        WeightedKeywords { words: vec!["deaths", "died", "dead", "casualties", "killed"], weight: 3.0 },
        WeightedKeywords { words: vec!["suffering", "trauma", "traumatic", "victims"], weight: 2.5 },
        WeightedKeywords { words: vec!["displaced", "refugees", "humanitarian", "survivors"], weight: 2.3 },
        WeightedKeywords { words: vec!["injured", "injuries", "wounded", "harm"], weight: 2.0 },
        WeightedKeywords { words: vec!["health", "healthcare", "medical", "mental"], weight: 1.8 },
        WeightedKeywords { words: vec!["children", "families", "women", "community", "communities"], weight: 1.5 },
        WeightedKeywords { words: vec!["poverty", "hunger", "rights", "education", "workers"], weight: 1.3 },
        WeightedKeywords { words: vec!["culture", "identity", "social", "care", "lives"], weight: 1.0 },
    ];
    
    // FUTURE - weighted by forward-looking nature
    let future_keywords = vec![
        WeightedKeywords { words: vec!["climate", "emissions", "carbon", "environmental"], weight: 2.5 },
        WeightedKeywords { words: vec!["ai", "artificial", "technology", "technologies", "innovation"], weight: 2.3 },
        WeightedKeywords { words: vec!["emerging", "transformation", "transition", "evolving"], weight: 2.0 },
        WeightedKeywords { words: vec!["forecast", "projection", "trend", "trends", "outlook"], weight: 1.8 },
        WeightedKeywords { words: vec!["strategy", "strategic", "vision", "plan", "planning"], weight: 1.5 },
        WeightedKeywords { words: vec!["research", "development", "launch", "prototype"], weight: 1.3 },
        WeightedKeywords { words: vec!["future", "next", "upcoming", "potential"], weight: 0.8 },
    ];
    
    // Count with weights
    let mut structural = 0.0f32;
    for wk in &structural_keywords {
        structural += count_tokens(&t, &wk.words) as f32 * wk.weight;
    }
    
    let mut conflict = 0.0f32;
    for wk in &conflict_keywords {
        conflict += count_tokens(&t, &wk.words) as f32 * wk.weight;
    }
    
    let mut human = 0.0f32;
    for wk in &human_keywords {
        human += count_tokens(&t, &wk.words) as f32 * wk.weight;
    }
    
    let mut future = 0.0f32;
    for wk in &future_keywords {
        future += count_tokens(&t, &wk.words) as f32 * wk.weight;
    }
    
    // Entity-based boosting (entities provide strong categorical signals)
    for entity in entities {
        let e = entity.to_lowercase();
        
        // Government/institutional entities → structural boost
        if e.contains("government") || e.contains("ministry") || e.contains("department") 
            || e.contains("agency") || e.contains("parliament") || e.contains("senate")
            || e.contains("congress") || e.contains("court") || e.contains("federal")
            || e.contains("central bank") || e.contains("reserve") {
            structural += 5.0;
        }
        
        // Military/conflict entities → conflict boost
        if e.contains("military") || e.contains("army") || e.contains("forces")
            || e.contains("defense") || e.contains("pentagon") || e.contains("nato")
            || e.contains("battalion") || e.contains("troops") || e.contains("militia")
            || e.contains("hamas") || e.contains("hezbollah") || e.contains("taliban") {
            conflict += 5.0;
        }
        
        // Humanitarian/social entities → human boost
        if e.contains("red cross") || e.contains("unicef") || e.contains("refugee")
            || e.contains("hospital") || e.contains("victims") || e.contains("civilians")
            || e.contains("community") || e.contains("humanitarian") || e.contains("ngo")
            || e.contains("charity") || e.contains("relief") {
            human += 5.0;
        }
        
        // Tech/climate/innovation entities → future boost
        if e.contains("tech") || e.contains("google") || e.contains("microsoft")
            || e.contains("apple") || e.contains("amazon") || e.contains("meta")
            || e.contains("openai") || e.contains("climate") || e.contains("renewable")
            || e.contains("startup") || e.contains("innovation") || e.contains("research") {
            future += 5.0;
        }
    }
    
    let sum = structural + conflict + human + future;
    let (vx, vy) = if sum > 0.0 {
        // Negative = structural (left), Positive = human (right)
        // Negative = conflict (bottom), Positive = future (top)
        ((human - structural) / sum, (future - conflict) / sum)
    } else {
        (0.0, 0.0)
    };

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

/// Original unweighted compass function (kept for compatibility/fallback)
fn compass_from_text(txt: &str) -> (String, f32, f32, HashMap<String, f32>) {
    let t = txt.to_lowercase();

    // Structural: Institutions/policy/systems (added: reform, governance, fiscal, systemic, ministry)
    let structural = count_tokens(
        &t,
        &[
            "policy", "policies",
            "economy", "economic",
            "budget", "budgets", "budgetary",
            "institution", "institutions", "institutional",
            "regulation", "regulations", "regulatory",
            "law", "laws", "legal",
            "tax", "taxes", "taxation",
            "welfare",
            "infrastructure",
            "agency", "agencies",
            "court", "courts",
            "legislation", "legislative",
            "congress", "congressional",
            "parliament", "parliamentary",
            "administration",
            "sanction", "sanctions",
            "tariff", "tariffs",
            "trade", "trading",
            "reform", "reforms",
            "governance",
            "fiscal",
            "monetary",
            "systemic",
            "ministry", "ministries",
            "senate",
            "spending",
        ],
    ) as f32;

    // Conflict: Violence/confrontation/legal battles (added: violence, combat, fighting, raid, assault, hostilities)
    let conflict = count_tokens(
        &t,
        &[
            "war", "wars", "warfare",
            "conflict", "conflicts",
            "clash", "clashes", "clashing",
            "attack", "attacks", "attacked", "attacking",
            "strike", "strikes", "striking",
            "missile", "missiles",
            "bomb", "bombs", "bombing", "bombed",
            "battle", "battles",
            "riot", "riots", "rioting",
            "protest", "protests", "protesting",
            "arrest", "arrests", "arrested",
            "indict", "indicted", "indictment",
            "charges", "charged",
            "trial", "trials",
            "ceasefire",
            "tensions",
            "crackdown",
            "shooting", "shootings", "shot",
            "violence", "violent", "violently",
            "combat",
            "confrontation", "confront",
            "fighting", "fought",
            "raid", "raids", "raided",
            "assault", "assaults", "assaulted",
            "hostilities",
            "aggression", "aggressive",
        ],
    ) as f32;

    // Human: People-centered/social impact (added: suffering, casualties, displaced, trauma, survivors, poverty)
    let human = count_tokens(
        &t,
        &[
            "health", "healthcare",
            "culture", "cultural",
            "identity",
            "education", "educational",
            "family", "families",
            "victims",
            "workers", "labor",
            "students",
            "women",
            "children",
            "community", "communities",
            "refugees",
            "migrants", "migration",
            "rights",
            "deaths", "died", "dead",
            "injured", "injuries",
            "mental",
            "care", "caring",
            "humanitarian",
            "suffering",
            "casualties",
            "displaced", "displacement",
            "trauma", "traumatic",
            "survivors", "survived",
            "lives",
            "social",
            "poverty",
            "hunger", "hungry",
        ],
    ) as f32;

    // Future: Forward-looking/planning/emerging trends (added: trend, outlook, strategy, emerging, transformation)
    let future = count_tokens(
        &t,
        &[
            "technology", "technologies", "technological",
            "ai", "artificial",
            "innovation", "innovative",
            "research", "researching",
            "forecast", "forecasts", "forecasting",
            "projection", "projections", "projected",
            "plan", "plans", "planning", "planned",
            "roadmap",
            "prototype", "prototypes",
            "pilot",
            "next",
            "upcoming",
            "future",
            "launch", "launches", "launching",
            "climate",
            "emissions",
            "carbon",
            "trend", "trends", "trending",
            "outlook",
            "vision", "visionary",
            "strategy", "strategic",
            "development", "developing",
            "evolving", "evolution",
            "transformation", "transforming",
            "transition", "transitioning",
            "emerging",
            "potential",
        ],
    ) as f32;

    let sum = structural + conflict + human + future;
    let (vx, vy) = if sum > 0.0 {
        ((structural - human) / sum, (conflict - future) / sum)
    } else {
        (0.0, 0.0)
    };

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

        // Improved lemmatization (protect acronyms)
        if !keep_short {
            t = lemmatize_word(&t);
        }

        *bag.entry(t).or_insert(0) += 1;
    }

    // Convert to vec, sort by freq, keep top N
    let mut v: Vec<(String, u32)> = bag.into_iter().collect();

    // Deweight extremely common "generic news" terms that slipped through
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

/// Intelligent lemmatization with pattern recognition and proper noun detection
fn lemmatize_word(word: &str) -> String {
    // Handle common irregular plurals first
    match word {
        "countries" => return "country".to_string(),
        "stories" => return "story".to_string(),
        "cities" => return "city".to_string(),
        "parties" => return "party".to_string(),
        "companies" => return "company".to_string(),
        "policies" => return "policy".to_string(),
        "agencies" => return "agency".to_string(),
        "people" => return "person".to_string(),
        "children" => return "child".to_string(),
        "men" => return "man".to_string(),
        "women" => return "woman".to_string(),
        _ => {}
    }

    let mut w = word.to_string();
    
    // Don't lemmatize short words (likely acronyms or proper nouns)
    if w.len() <= 3 {
        return w;
    }
    
    // Handle -ies → -y plurals (but not words ending in -ries where the 'r' is part of the suffix)
    if w.ends_with("ies") && w.len() > 4 {
        let stem = &w[..w.len()-3];
        // Don't convert if it would create an odd stem (e.g., "series" → "ser" is wrong)
        if stem.len() >= 2 && !matches!(stem, "ser" | "quer" | "specif") {
            return stem.to_string() + "y";
        }
    }
    
    // Check if word ends with 's' - be very selective about removing it
    if w.ends_with('s') && w.len() > 4 {
        // NEVER remove 's' from these endings (they're part of the word, not plural markers)
        let non_plural_endings = [
            "ss", "us", "is", "as",           // class, focus, analysis, canvas
            "ness", "less", "ous", "ious",    // business, unless, famous, religious  
            "ess", "ess",                      // princess, goddess
            "sis", "osis", "asis",            // crisis, diagnosis, basis
        ];
        
        if non_plural_endings.iter().any(|&ending| w.ends_with(ending)) {
            return w;
        }
        
        // Check for proper nouns / entities (capitalized words are likely proper nouns)
        // This is heuristic since we've already lowercased, but we can check patterns
        
        // Names of places/people often end in specific patterns - don't lemmatize
        let proper_noun_patterns = [
            "mas",     // Hamas, Thomas, Christmas
            "ris",     // Paris, Morris
            "uis",     // Louis
            "les",     // Angeles, Naples, Charles
            "nos",     // Buenos Aires components
            "os",      // Lagos, Carlos, Philippines-related
            "as",      // Texas, Vegas, Atlas, Douglas, Nicholas
            "es",      // Jones, Davies, Wales
        ];
        
        // If it matches a proper noun pattern, be cautious
        let matches_proper = proper_noun_patterns.iter().any(|&pattern| w.ends_with(pattern));
        
        if matches_proper {
            // Additional check: if removing 's' would leave a very short or odd stem, keep it
            let without_s = &w[..w.len()-1];
            if without_s.len() < 3 || !looks_like_valid_stem(without_s) {
                return w;
            }
        }
        
        // For other words ending in 's', check if it looks like a plural
        // Common plural patterns: consonant + s (books, cats, dogs)
        let without_s = &w[..w.len()-1];
        
        // If the stem ends with a vowel + consonant, it's likely a plural
        // e.g., "books" → "book", "articles" → "article"
        if is_likely_plural_stem(without_s) {
            return without_s.to_string();
        }
        
        // Otherwise, keep the 's'
        return w;
    }
    
    w
}

/// Check if removing 's' would create a valid-looking English stem
fn looks_like_valid_stem(stem: &str) -> bool {
    if stem.len() < 3 {
        return false;
    }
    
    // Check for valid consonant-vowel patterns
    let chars: Vec<char> = stem.chars().collect();
    let vowels = ['a', 'e', 'i', 'o', 'u', 'y'];
    
    // Must contain at least one vowel
    if !chars.iter().any(|c| vowels.contains(c)) {
        return false;
    }
    
    // Check for problematic endings that suggest it's not a good stem
    let bad_endings = ["ha", "te", "ri", "ma", "lu"];  // hama, testa, pari, etc.
    if bad_endings.iter().any(|&ending| stem.ends_with(ending)) {
        return false;
    }
    
    true
}

/// Check if a word stem (without final 's') looks like it came from a plural
fn is_likely_plural_stem(stem: &str) -> bool {
    if stem.len() < 2 {
        return false;
    }
    
    let chars: Vec<char> = stem.chars().collect();
    let vowels = ['a', 'e', 'i', 'o', 'u'];
    let last = chars[chars.len() - 1];
    
    // Common plural patterns:
    // 1. Ends with a consonant (not 's') → likely plural (book+s, cat+s)
    // 2. Ends with 'e' after consonant → likely plural (article+s, mistake+s)
    
    if !vowels.contains(&last) && last != 's' {
        return true; // consonant ending → likely plural
    }
    
    if last == 'e' && chars.len() >= 2 {
        let second_last = chars[chars.len() - 2];
        if !vowels.contains(&second_last) {
            return true; // consonant + e → likely plural
        }
    }
    
    // Ends with double letters → likely plural (add+s → adds)
    if chars.len() >= 2 && chars[chars.len() - 1] == chars[chars.len() - 2] {
        return true;
    }
    
    false
}

fn strip_urls(s: &str) -> String {
    let re: Regex = regex_cached(r"https?://\S+");
    re.replace_all(s, " ").to_string()
}

fn normalize_apostrophes(s: &str) -> String {
    // Remove possessives 's/'s and unify quotes
    let apos_re: Regex = regex_cached(r"['']s\b");
    let mut out = apos_re.replace_all(s, "").to_string();
    out = out.replace(['"', '"', '\'', '\''], "\"");
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
            "has","have","had","will","would","should","could","may","might","can",
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
            // Major outlet brand names to exclude from word clouds
            "reuters", "ap", "bbc", "cnn", "npr", "nbc", "abc", "cbs", "fox", "msnbc",
            "aljazeera", "jazeera", "al", "guardian", "times", "post", "journal", "telegraph",
            "bloomberg", "economist", "politico", "axios", "vox", "vice", "buzzfeed",
            "huffpost", "slate", "salon", "hill", "beast", "newsweek", "forbes",
            "cnbc", "wsj", "nyt", "wapo", "ft", "afp", "dpa", "tass", "xinhua",
            "independent", "express", "mirror", "mail", "sun", "sky", "itv", "rte",
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

/* -------------------------------------------------------------------------- */
/* Root Index Management                                                      */
/* -------------------------------------------------------------------------- */

/// Updates or creates the root-level index.json file (e.g., viz/index.json)
/// This file lists all available dates for the date picker in the JS visualization.
fn update_root_index(root_dir: &Path, new_date: &str) -> Result<()> {
    let index_path = root_dir.join("index.json");
    
    // Read existing index if it exists
    let mut dates: BTreeSet<String> = BTreeSet::new();
    
    if index_path.exists() {
        let existing_content = fs::read_to_string(&index_path)
            .with_context(|| format!("Failed to read existing index at {:?}", index_path))?;
        
        if let Ok(existing_index) = serde_json::from_str::<serde_json::Value>(&existing_content) {
            if let Some(dates_array) = existing_index.get("dates").and_then(|v| v.as_array()) {
                for date_val in dates_array {
                    if let Some(date_str) = date_val.as_str() {
                        dates.insert(date_str.to_string());
                    }
                }
            }
        }
    }
    
    // Add the new date
    dates.insert(new_date.to_string());
    
    // Convert to sorted Vec
    let dates_vec: Vec<String> = dates.into_iter().collect();
    
    // Latest is the most recent date (last in sorted order)
    let latest = dates_vec.last().cloned();
    
    // Create the index structure
    let index = json!({
        "dates": dates_vec,
        "latest": latest,
        "version": 1
    });
    
    // Write the index
    write_json(&index_path, &index)
        .with_context(|| format!("Failed to write root index to {:?}", index_path))?;
    
    Ok(())
}
