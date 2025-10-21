pub fn user_cluster_compression(cluster_members_json: &str) -> String {
    format!(r#"You'll receive multiple articles about the same event.
Return a digest with this structure:
- One-line event summary.
- What changed since previous editions (bullets).
- Cross-source frames (bullets: source → frame in ~6–12 words).
- Named entities (role: name).
- Next milestones (date or timeframe → why it matters).
- Uncertainty/risks in 1–2 bullets.

ARTICLES JSON:
<{json}>

CONSTRAINTS:
- ≤ 450 tokens total.
- No repetition.
- Use ISO dates where possible."#, json = cluster_members_json)
}

pub fn user_insights(digests_json: &str, index_min: &str) -> String {
    format!(r#"Given a set of cluster digests from yesterday's editions and today's morning edition, extract meta insights:

Output JSON with:
{{{{
  "sentiment_swing": [{{{{\"topic\":\"\", \"direction\":\"up|down|flat\", \"evidence\":[\"cluster_id\", ...]}}}}],
  "narrative_drift": [{{{{\"axis\":\"blame→cause|risk→optimism|policy→politics|emergency→accountability\", \"examples\":[{{{{\"cluster_id\":\"\",\"sources\":[\"A\",\"B\"],\"contrast\":\"\"}}}}]}}}}],
  "attention_peaks": [{{{{\"theme\":\"\",\"clusters\":[\"...\"],\"why_now\":\"\"}}}}],
  "silences": [{{{{\"theme\":\"\",\"expected_but_missing\":\"\"}}}}],
  "story_mortality": [{{{{\"theme\":\"\",\"ceased_since\":\"\",\"note\":\"\"}}}}],
  "top_milestones": [{{{{\"date\":\"\",\"what\":\"\",\"clusters\":[\"...\"]}}}}]
}}}}

CONSTRAINTS:
- Use concise evidence references (cluster_id).
- Avoid summarizing each story; surface cross-cluster patterns only.
- ≤ 2500 tokens.

INPUTS:
CLUSTER_DIGESTS_JSON:
<{digests}>

CLUSTER_INDEX_MIN:
<{index_min}>"#, digests = digests_json, index_min = index_min)
}

pub fn user_final_meta(insights_json: &str, cluster_index_min: &str) -> String {
    format!(r#"Using (a) the insights JSON and (b) the list of cluster digests, produce a readable post with sections:
- Meta Climate (1–2 sentence thesis).
- Yesterday's Momentum (3–6 bullets).
- Flashpoints Gaining Heat (3–6 bullets with why now).
- Cross-Outlet Splits (2–4 bullets showing divergences).
- Notably Absent (1–3 bullets).
- Next Milestones (dated bullets; say why they matter).
- Interpretive Meta (short concluding paragraph).

Tone: neutral, observant, explanatory. Avoid punditry. Keep to ≤ 1800 tokens.

INPUTS:
INSIGHTS JSON:
<{insights}>

CLUSTER INDEX (id + title only):
<{index}>"#, insights = insights_json, index = cluster_index_min)
}
