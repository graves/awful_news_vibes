use anyhow::{Result, bail};
use reqwest::Client;
use serde_json::json;
use tracing::{debug, info, warn, error};

use crate::budget::{assert_global_budget, cap_digest};
use crate::cluster::{ClusterParams, cluster_articles};
use crate::compress::llm_call;
use crate::fetch::{fetch_edition_opt, normalize_editions};
use crate::models::{Article, StoryCluster};
use crate::out_models::{ClusterDigest, DailyMetaPost, MetaInsights};
use crate::prompts::{user_cluster_compression, user_insights, user_final_meta};
use crate::render::{render_cluster_digest_text, render_final_markdown};
use crate::similarity::SimilarityWeights;
use crate::viz_export::write_all_viz; // ← emit D3-ready files
use awful_aj::{config::AwfulJadeConfig, template::ChatTemplate};

fn cluster_index_min(clusters: &[StoryCluster]) -> String {
    let mini: Vec<_> = clusters
        .iter()
        .map(|c| json!({ "cluster_id": c.cluster_id, "title": c.canonical_title }))
        .collect();
    serde_json::to_string_pretty(&mini).unwrap()
}

pub async fn run_daily(
    _cfg: &AwfulJadeConfig,
    cluster_cfg: &AwfulJadeConfig,
    vibe_cfg: &AwfulJadeConfig,
    tpl_cluster: &ChatTemplate,
    tpl_insights: &ChatTemplate,
    tpl_final: &ChatTemplate,
    ymd_yesterday: &str,
    ymd_today: &str,
    output_dir: &str,
) -> Result<()> {
    let pipeline_start = std::time::Instant::now();
    info!("Pipeline started - date_range={} to {}", ymd_yesterday, ymd_today);

    let client = Client::builder().build()?;

    // 1) fetch editions: yesterday (morning, afternoon, evening), today (morning)
    let fetch_start = std::time::Instant::now();
    debug!("Fetching 4 editions: {}/{{morning,afternoon,evening}}, {}/morning", ymd_yesterday, ymd_today);

    let mut fetched = Vec::new();

    for slot in ["morning", "afternoon", "evening"] {
        match fetch_edition_opt(&client, ymd_yesterday, slot).await? {
            Some(ed) => {
                debug!("Successfully fetched: {}/{}", ymd_yesterday, slot);
                fetched.push(ed);
            }
            None => {
                warn!("Missing edition: {}/{}", ymd_yesterday, slot);
            }
        }
    }
    match fetch_edition_opt(&client, ymd_today, "morning").await? {
        Some(ed) => {
            debug!("Successfully fetched: {}/morning", ymd_today);
            fetched.push(ed);
        }
        None => {
            warn!("Missing edition: {}/morning", ymd_today);
        }
    }

    if fetched.is_empty() {
        error!(
            "No editions available - checked {}/{{morning,afternoon,evening}} and {}/morning",
            ymd_yesterday, ymd_today
        );
        bail!(
            "No editions available for processing (checked {}/{{morning,afternoon,evening}} and {}/morning).",
            ymd_yesterday,
            ymd_today
        );
    }

    let fetch_elapsed = fetch_start.elapsed();
    info!("Edition fetch completed - duration={:.2}s, editions={}", fetch_elapsed.as_secs_f32(), fetched.len());

    let eds = normalize_editions(fetched.clone());

    // 2) pool all articles, *retaining edition_id alongside each article* so we never lose it
    let mut pairs: Vec<(Article, String)> = Vec::new(); // (article, edition_id)
    for ed in &eds {
        debug!("Processing edition: {} ({} articles)", ed.edition_id, ed.articles.len());
        for a in &ed.articles {
            pairs.push((a.clone(), ed.edition_id.clone()));
        }
    }
    debug!("Total article-edition pairs pooled: {}", pairs.len());

    // 2.5) deduplicate by exact title *within the same edition* (do NOT erase cross-edition carryover)
    let before = pairs.len();
    use std::collections::HashSet;
    let mut seen: HashSet<(String, String)> = HashSet::new(); // (title, edition_id)
    pairs.retain(|(a, ed_id)| {
        let key = (a.title.clone(), ed_id.clone());
        if seen.contains(&key) {
            false
        } else {
            seen.insert(key);
            true
        }
    });
    let after = pairs.len();
    let removed = before - after;
    if removed > 0 {
        info!("Deduplication - removed={} duplicates, retained={} unique articles", removed, after);
    } else {
        debug!("Deduplication - no duplicates found, retained={} articles", after);
    }

    // 2.6) Build:
    //  - articles Vec<Article> for clustering
    //  - ed_of map: article_id -> edition_id (to repopulate ClusterMember.edition_id later)
    let mut articles: Vec<Article> = Vec::with_capacity(pairs.len());
    let mut ed_of: std::collections::HashMap<String, String> = std::collections::HashMap::new();
    for (a, ed_id) in pairs.into_iter() {
        ed_of.insert(a.id.clone(), ed_id);
        articles.push(a);
    }
    debug!("Articles prepared for clustering: {}", articles.len());

    // 3) cluster
    let cluster_start = std::time::Instant::now();
    let weights = SimilarityWeights {
        entities: 0.35,
        dates: 0.20,
        timeframes: 0.10,
        tags: 0.15,
        text: 0.20,
    };
    let params = ClusterParams {
        threshold: 0.62,
        max_members_for_digest: 7,
    };
    debug!("Clustering articles - threshold={}, max_members={}", params.threshold, params.max_members_for_digest);
    let mut clusters = cluster_articles(&articles, weights, params)?;
    let cluster_elapsed = cluster_start.elapsed();
    info!("Clustering completed - duration={:.2}s, clusters={}", cluster_elapsed.as_secs_f32(), clusters.len());

    // 3.1) Restore correct edition_id for all cluster members from ed_of map
    let mut fixed = 0usize;
    let mut missing = 0usize;
    for c in clusters.iter_mut() {
        for m in c.members.iter_mut() {
            if let Some(ed) = ed_of.get(&m.article_id) {
                if m.edition_id != *ed {
                    m.edition_id = ed.clone();
                    fixed += 1;
                }
            } else {
                // This should be rare; log and keep as-is (likely "UNKNOWN" from clusterer)
                missing += 1;
            }
        }
    }
    if fixed > 0 {
        debug!("Edition mapping - adjusted={} members, missing={}", fixed, missing);
    } else {
        debug!("Edition mapping - all members already correct");
    }

    // 4) per-cluster compression → strict JSON → render compact text
    let total_clusters = clusters.len();
    let compression_start = std::time::Instant::now();
    info!(
        "Cluster compression starting - clusters={}, batch_size=12, total_llm_calls={}",
        total_clusters, total_clusters + 2
    );
    let mut total_llm_time = 0.0f32;
    let mut completed = 0;

    // Process clusters in batches of 12 (parallel)
    const BATCH_SIZE: usize = 12;
    for batch_start in (0..clusters.len()).step_by(BATCH_SIZE) {
        let batch_end = (batch_start + BATCH_SIZE).min(clusters.len());
        let batch_size = batch_end - batch_start;

        let batch_start_time = std::time::Instant::now();

        // Prepare user prompts for this batch (owned strings)
        let mut user_prompts = Vec::new();
        for i in batch_start..batch_end {
            let members_json = serde_json::to_string(&clusters[i].members)?;
            let user = user_cluster_compression(&members_json);
            user_prompts.push(user);
        }

        // Create futures for all clusters in this batch - use cluster_cfg
        let mut tasks = Vec::new();
        for user in &user_prompts {
            tasks.push(llm_call(cluster_cfg, tpl_cluster, user));
        }

        // Execute all tasks in parallel
        let results = futures::future::join_all(tasks).await;

        let batch_elapsed = batch_start_time.elapsed().as_secs_f32();
        total_llm_time += batch_elapsed;

        // Process results
        for (i, result) in results.into_iter().enumerate() {
            let digest_json = result?;
            let digest: ClusterDigest = serde_json::from_str(&digest_json)?;
            let digest_text = render_cluster_digest_text(&digest);

            clusters[batch_start + i].digest_abridged = digest_text;
            cap_digest(&mut clusters[batch_start + i], 200);
        }

        completed += batch_size;

        // Progress logging with timing
        let pct = (completed as f32 / total_clusters as f32 * 100.0) as u32;
        let avg_batch_time = total_llm_time / ((batch_start / BATCH_SIZE) + 1) as f32;
        let remaining_batches =
            ((total_clusters - completed) as f32 / BATCH_SIZE as f32).ceil() as u32;
        let eta_seconds = avg_batch_time * remaining_batches as f32;
        let eta_minutes = (eta_seconds / 60.0) as u32;
        let eta_secs = (eta_seconds % 60.0) as u32;

        info!(
            "Compression progress: {}/{} ({}%) | Batch of {}: {:.1}s | Avg batch: {:.1}s | ETA: {}m {}s",
            completed,
            total_clusters,
            pct,
            batch_size,
            batch_elapsed,
            avg_batch_time,
            eta_minutes,
            eta_secs
        );
    }

    let compression_elapsed = compression_start.elapsed();
    let avg_per_call = compression_elapsed.as_secs_f32() / total_clusters as f32;
    info!(
        "Cluster compression completed - duration={:.2}s, avg_per_cluster={:.2}s",
        compression_elapsed.as_secs_f32(),
        avg_per_call
    );

    // 5) insights (strict JSON)
    let insights_start = std::time::Instant::now();
    debug!("Generating insights from {} cluster digests", clusters.len());

    // Create super-condensed digests for insights (strip entities and other noise)
    let insights_digests: Vec<_> = clusters
        .iter()
        .map(|c| {
            // Extract only essential lines: event summary + changes + frames (skip entities, milestones details)
            let lines: Vec<&str> = c.digest_abridged.lines().collect();
            let mut condensed = String::new();
            let mut in_entities = false;
            let mut in_milestones = false;

            for line in lines {
                if line.starts_with("Entities:") {
                    in_entities = true;
                    continue;
                } else if line.starts_with("Next milestones:") {
                    in_milestones = true;
                    continue;
                } else if line.starts_with("What changed:")
                    || line.starts_with("Cross-source frames:")
                    || line.starts_with("Uncertainty")
                {
                    in_entities = false;
                    in_milestones = false;
                }

                if !in_entities && !in_milestones {
                    condensed.push_str(line);
                    condensed.push('\n');
                }
            }

            json!({"cluster_id": c.cluster_id, "digest": condensed.trim()})
        })
        .collect();

    let digests_json = serde_json::to_string(&insights_digests)?;
    let idx_min = cluster_index_min(&clusters);
    let user_i = user_insights(&digests_json, &idx_min);
    let insights_json = llm_call(vibe_cfg, tpl_insights, &user_i).await?;
    let _insights: MetaInsights = serde_json::from_str(&insights_json)?; // parsed for validation

    let insights_elapsed = insights_start.elapsed();
    info!("Insights generation completed - duration={:.2}s", insights_elapsed.as_secs_f32());

    // 6) final daily post (strict JSON → Markdown)
    let final_start = std::time::Instant::now();
    debug!("Generating final daily post from insights");
    let user_f = user_final_meta(&insights_json, &idx_min);
    let final_json = llm_call(vibe_cfg, tpl_final, &user_f).await?;
    let final_struct: DailyMetaPost = serde_json::from_str(&final_json)?;
    let final_post_md = render_final_markdown(&final_struct);
    
    let final_elapsed = final_start.elapsed();
    info!("Final post generation completed - duration={:.2}s", final_elapsed.as_secs_f32());

    // 7) budget guard
    assert_global_budget(&clusters, &insights_json, &final_post_md, 150_000)?;

    // 8) persist to date-scoped directory
    let persist_start = std::time::Instant::now();
    let date_dir = std::path::Path::new(output_dir).join(ymd_today);
    std::fs::create_dir_all(&date_dir)?;
    debug!("Output directory: {}", date_dir.display());

    std::fs::write(
        date_dir.join("clusters.full.json"),
        serde_json::to_vec_pretty(&clusters)?,
    )?;
    debug!("Wrote clusters.full.json");

    std::fs::write(
        date_dir.join("insights.full.json"),
        insights_json.as_bytes(),
    )?;
    debug!("Wrote insights.full.json");

    std::fs::write(date_dir.join("meta_post.json"), final_json.as_bytes())?;
    debug!("Wrote meta_post.json");

    std::fs::write(date_dir.join("meta_post.md"), final_post_md.as_bytes())?;
    debug!("Wrote meta_post.md");

    // Editions order for this run (yesterday 3 + today morning)
    let editions_ordered: Vec<String> = vec![
        format!("{}-{}", ymd_yesterday, "morning"),
        format!("{}-{}", ymd_yesterday, "afternoon"),
        format!("{}-{}", ymd_yesterday, "evening"),
        format!("{}-{}", ymd_today, "morning"),
    ];

    // Emit D3-ready visual JSONs
    write_all_viz(
        &date_dir,
        ymd_today,
        &clusters,
        &insights_json,
        &editions_ordered,
    )?;
    debug!("Wrote viz bundle");
    
    let persist_elapsed = persist_start.elapsed();
    info!("Output persisted - duration={:.2}s, directory={}", persist_elapsed.as_secs_f32(), date_dir.display());

    let pipeline_elapsed = pipeline_start.elapsed();
    info!(
        "Pipeline completed successfully - total_duration={:.2}s, clusters={}, editions={}",
        pipeline_elapsed.as_secs_f32(), clusters.len(), fetched.len()
    );
    Ok(())
}
