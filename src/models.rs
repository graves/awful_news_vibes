use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Edition {
    pub edition_id: String,   // e.g., "2025-10-18-evening"
    pub published_at: String, // ISO8601
    pub articles: Vec<Article>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Article {
    pub id: String,
    pub title: String,
    pub source: String,
    pub category: String,
    pub tags: Vec<String>,
    pub key_takeaways: Vec<String>,
    pub named_entities: Vec<NamedEntity>,
    pub important_dates: Vec<DatedCtx>,
    pub important_timeframes: Vec<SpanCtx>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NamedEntity {
    pub name: String,
    pub kind: String,
    pub context: String,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatedCtx {
    pub date: String,
    pub context: String,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpanCtx {
    pub span: String,
    pub context: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct StanceVector {
    pub sentiment: f32,        // [-1.0, 1.0]
    pub certainty: f32,        // [0.0, 1.0]
    pub blame_vs_cause: f32,   // [-1.0 blame, +1.0 cause]
    pub risk_vs_optimism: f32, // [-1.0 risk, +1.0 optimism]
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClusterMember {
    pub article_id: String,
    pub source: String,
    pub edition_id: String,
    pub title: String,
    pub key_points: Vec<String>, // compressed/squeezed from key_takeaways
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StoryCluster {
    pub cluster_id: String,
    pub canonical_title: String,
    pub topics: BTreeSet<String>,
    pub entities: BTreeSet<String>,
    pub date_refs: BTreeSet<String>,
    pub timeframes: BTreeSet<String>,
    pub members: Vec<ClusterMember>,
    pub digest_abridged: String, // ≤ ~450–500 tokens
    pub stance_matrix: BTreeMap<String, StanceVector>, // source -> stance
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VizClusterIndex {
    pub cluster_id: String,
    pub title: String,
    pub stance_summary: StanceVector,
    pub birth_edition: String,
    pub last_seen_edition: String,
    pub related_entities: Vec<String>,
    pub related_themes: Vec<String>,
    pub story_velocity: f32, // editions per day normalized
}
