use crate::models::StoryCluster;
use anyhow::{Result, bail};

pub fn approx_tokens(s: &str) -> usize {
    // heuristic ~4 chars/token
    (s.chars().count() + 3) / 4
}

pub fn cap_digest(cluster: &mut StoryCluster, max_tokens: usize) {
    if approx_tokens(&cluster.digest_abridged) > max_tokens {
        let mut s = cluster.digest_abridged.clone();
        while approx_tokens(&s) > max_tokens {
            s.pop();
        }
        cluster.digest_abridged = s;
    }
}

pub fn assert_global_budget(
    clusters: &[StoryCluster],
    insights_json: &str,
    final_post: &str,
    hard_cap: usize,
) -> Result<()> {
    let mut total = 0usize;
    for c in clusters {
        total += approx_tokens(&c.digest_abridged);
    }
    total += approx_tokens(insights_json);
    total += approx_tokens(final_post);

    if total > hard_cap {
        bail!(
            "Token budget exceeded: {} > {} (tighten per-cluster caps or reduce cluster count)",
            total,
            hard_cap
        );
    }
    Ok(())
}
