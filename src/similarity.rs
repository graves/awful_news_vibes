use itertools::Itertools;
use std::collections::BTreeSet;
use unicode_normalization::UnicodeNormalization;

use crate::models::Article;

#[derive(Debug, Clone, Copy)]
pub struct SimilarityWeights {
    pub entities: f32,   // 0.35
    pub dates: f32,      // 0.20
    pub timeframes: f32, // 0.10
    pub tags: f32,       // 0.15
    pub text: f32,       // 0.20
}

fn to_set<T: AsRef<str>>(xs: impl IntoIterator<Item = T>) -> BTreeSet<String> {
    xs.into_iter()
        .map(|s| s.as_ref().nfc().collect::<String>().to_lowercase())
        .collect()
}

fn jaccard(a: &BTreeSet<String>, b: &BTreeSet<String>) -> f32 {
    if a.is_empty() && b.is_empty() {
        return 1.0;
    }
    let inter = a.intersection(b).count() as f32;
    let union = a.union(b).count() as f32;
    if union == 0.0 { 0.0 } else { inter / union }
}

fn stemish_tokens<S: AsRef<str>>(s: S) -> BTreeSet<String> {
    let mut out = BTreeSet::new();
    for t in s.as_ref().split(|c: char| !c.is_alphanumeric()) {
        if t.len() >= 3 {
            out.insert(t.to_lowercase());
        }
    }
    out
}

pub fn article_similarity(a: &Article, b: &Article, w: SimilarityWeights) -> f32 {
    let ents_a = to_set(a.named_entities.iter().map(|e| &e.name));
    let ents_b = to_set(b.named_entities.iter().map(|e| &e.name));
    let dates_a = to_set(a.important_dates.iter().map(|d| &d.date));
    let dates_b = to_set(b.important_dates.iter().map(|d| &d.date));
    let tf_a = to_set(a.important_timeframes.iter().map(|d| &d.span));
    let tf_b = to_set(b.important_timeframes.iter().map(|d| &d.span));
    let tags_a = to_set(&a.tags);
    let tags_b = to_set(&b.tags);

    let text_a = stemish_tokens(format!("{} {}", a.title, a.key_takeaways.join(" ")));
    let text_b = stemish_tokens(format!("{} {}", b.title, b.key_takeaways.join(" ")));

    let s = w.entities * jaccard(&ents_a, &ents_b)
        + w.dates * jaccard(&dates_a, &dates_b)
        + w.timeframes * jaccard(&tf_a, &tf_b)
        + w.tags * jaccard(&tags_a, &tags_b)
        + w.text * jaccard(&text_a, &text_b);

    // heuristics: exact date overlaps boost
    let date_boost = if !dates_a.is_empty() && !dates_b.is_empty() && !dates_a.is_disjoint(&dates_b)
    {
        0.05
    } else {
        0.0
    };
    (s + date_boost).min(1.0)
}
