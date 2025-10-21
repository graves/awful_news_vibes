use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClusterDigest {
    pub eventSummary: String,
    pub changesSincePrevious: Vec<String>,
    pub crossSourceFrames: Vec<CrossSourceFrame>,
    pub entities: Vec<EntityRole>,
    pub nextMilestones: Vec<Milestone>,
    pub uncertaintyRisks: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CrossSourceFrame {
    pub source: String,
    pub frame: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EntityRole {
    pub role: String,
    pub name: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Milestone {
    pub dateOrSpan: String,
    pub whyItMatters: String,
}

/* Insights */
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetaInsights {
    pub sentimentSwing: Vec<SentimentSwing>,
    pub narrativeDrift: Vec<NarrativeDrift>,
    pub attentionPeaks: Vec<AttentionPeak>,
    pub silences: Vec<Silence>,
    pub storyMortality: Vec<StoryMortality>,
    pub topMilestones: Vec<TopMilestone>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SentimentSwing {
    pub topic: String,
    pub direction: String, // up|down|flat
    pub evidence: Vec<String>, // cluster ids
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NarrativeDrift {
    pub axis: String,
    pub examples: Vec<NarrativeExample>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NarrativeExample {
    pub clusterId: String,
    pub sources: Vec<String>,
    pub contrast: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AttentionPeak {
    pub theme: String,
    pub clusters: Vec<String>,
    pub whyNow: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Silence {
    pub theme: String,
    pub expectedButMissing: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StoryMortality {
    pub theme: String,
    pub ceasedSince: String,
    pub note: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TopMilestone {
    pub date: String,
    pub what: String,
    pub clusters: Vec<String>,
}

/* Final post */
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DailyMetaPost {
    pub metaClimate: String,
    pub yesterdaysMomentum: Vec<String>,
    pub flashpointsGainingHeat: Vec<Flashpoint>,
    pub crossOutletSplits: Vec<String>,
    pub notablyAbsent: Vec<String>,
    pub nextMilestones: Vec<NextMilestone>,
    pub interpretiveMeta: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Flashpoint {
    pub title: String,
    pub whyNow: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NextMilestone {
    pub date: String,
    pub what: String,
    pub whyItMatters: String,
}