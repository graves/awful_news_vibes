use anyhow::Result;
use rayon::prelude::*;
use std::collections::{BTreeMap, BTreeSet};
use xxhash_rust::xxh3::xxh3_64;
use tracing::{debug, info};

use crate::models::{Article, ClusterMember, StanceVector, StoryCluster};
use crate::similarity::{SimilarityWeights, article_similarity};

pub struct ClusterParams {
    pub threshold: f32,                // e.g., 0.62
    pub max_members_for_digest: usize, // choose top 2–3 representative members
}

pub fn cluster_articles(
    articles: &[Article],
    weights: SimilarityWeights,
    params: ClusterParams,
) -> Result<Vec<StoryCluster>> {
    debug!(
        "Clustering started - articles={}, threshold={}, max_members={}",
        articles.len(), params.threshold, params.max_members_for_digest
    );
    
    let mut assigned = vec![false; articles.len()];
    let mut clusters = Vec::new();
    let total = articles.len();

    for i in 0..articles.len() {
        if i % 50 == 0 && i > 0 {
            let pct = (i as f32 / total as f32 * 100.0) as u32;
            info!("Clustering progress - processed={}/{} ({}%), clusters={}", i, total, pct, clusters.len());
        }
        if assigned[i] {
            continue;
        }

        let mut members_idx = vec![i];
        assigned[i] = true;

        // grow cluster - SINGLE PASS ONLY for speed, PARALLELIZED similarity checks
        // Collect unassigned indices to avoid checking assigned articles repeatedly
        let unassigned: Vec<usize> = (0..articles.len())
            .filter(|&j| !assigned[j])
            .collect();
        
        // Parallel similarity calculation - this is the bottleneck!
        let seed_article = &articles[i];
        let similar_indices: Vec<usize> = unassigned
            .par_iter()
            .filter(|&&j| {
                article_similarity(&articles[j], seed_article, weights) >= params.threshold
            })
            .copied()
            .collect();
        
        // Mark as assigned and add to cluster
        for j in similar_indices {
            assigned[j] = true;
            members_idx.push(j);
        }

        // canonical title = medoid by avg distance
        let (medoid_idx, _) = members_idx
            .iter()
            .map(|&idx| {
                let sum_dist: f32 = members_idx
                    .iter()
                    .filter(|&&other| other != idx)
                    .map(|&other| {
                        1.0 - article_similarity(&articles[idx], &articles[other], weights)
                    })
                    .sum();
                (idx, sum_dist)
            })
            .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
            .unwrap();

        let canonical_title = articles[medoid_idx].title.clone();

        // collect sets
        let mut topics = BTreeSet::new();
        let mut entities = BTreeSet::new();
        let mut date_refs = BTreeSet::new();
        let mut timeframes = BTreeSet::new();

        for &idx in &members_idx {
            topics.extend(articles[idx].tags.iter().cloned());
            for e in &articles[idx].named_entities {
                entities.insert(e.name.clone());
            }
            for d in &articles[idx].important_dates {
                date_refs.insert(d.date.clone());
            }
            for t in &articles[idx].important_timeframes {
                timeframes.insert(t.span.clone());
            }
        }

        // representative members: pick up to max_members_for_digest by cross-source diversity then recency (last in list)
        let mut reps = members_idx.clone();
        reps.sort_by_key(|&idx| (&articles[idx].source, &articles[idx].title));
        reps.truncate(params.max_members_for_digest);

        let members: Vec<ClusterMember> = reps
            .into_iter()
            .map(|idx| ClusterMember {
                article_id: articles[idx].id.clone(),
                source: articles[idx].source.clone(),
                edition_id: "UNKNOWN".into(), // filled later if desired
                title: articles[idx].title.clone(),
                key_points: squeeze_key_points(&articles[idx].key_takeaways, 3),
            })
            .collect();

        // stable cluster_id hash
        let seed = format!(
            "{}|{}|{}",
            canonical_title,
            entities.iter().cloned().collect::<Vec<_>>().join(","),
            date_refs.iter().cloned().collect::<Vec<_>>().join(",")
        );
        let cluster_id = format!("{:016x}", xxh3_64(seed.as_bytes()));

        clusters.push(StoryCluster {
            cluster_id,
            canonical_title,
            topics,
            entities,
            date_refs,
            timeframes,
            members,
            digest_abridged: String::new(),
            stance_matrix: BTreeMap::new(),
        });
    }

    // Show cluster size distribution
    let sizes: Vec<usize> = clusters.iter().map(|c| c.members.len()).collect();
    if !sizes.is_empty() {
        let max_size = sizes.iter().max().unwrap();
        let min_size = sizes.iter().min().unwrap();
        let avg_size = sizes.iter().sum::<usize>() as f32 / sizes.len() as f32;
        debug!("Cluster size distribution - min={}, max={}, avg={:.1}", min_size, max_size, avg_size);
    }

    Ok(clusters)
}

fn squeeze_key_points(pts: &[String], max_n: usize) -> Vec<String> {
    pts.iter()
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .take(max_n)
        .collect()
}
