// src/render.rs
use crate::out_models::{ClusterDigest, DailyMetaPost};

pub fn render_cluster_digest_text(d: &ClusterDigest) -> String {
    let mut out = String::new();
    out.push_str(&format!("{}\n", d.eventSummary.trim()));

    if !d.changesSincePrevious.is_empty() {
        out.push_str("\nWhat changed:\n");
        for b in d.changesSincePrevious.iter().take(5) {
            out.push_str(&format!("- {}\n", b));
        }
    }

    if !d.crossSourceFrames.is_empty() {
        out.push_str("\nCross-source frames:\n");
        for f in d.crossSourceFrames.iter().take(6) {
            out.push_str(&format!("- {} → {}\n", f.source, f.frame));
        }
    }

    if !d.entities.is_empty() {
        out.push_str("\nEntities:\n");
        for e in d.entities.iter().take(8) {
            out.push_str(&format!("- {}: {}\n", e.role, e.name));
        }
    }

    if !d.nextMilestones.is_empty() {
        out.push_str("\nNext milestones:\n");
        for m in d.nextMilestones.iter().take(5) {
            out.push_str(&format!("- {} → {}\n", m.dateOrSpan, m.whyItMatters));
        }
    }

    if !d.uncertaintyRisks.is_empty() {
        out.push_str("\nUncertainty / risks:\n");
        for u in d.uncertaintyRisks.iter().take(2) {
            out.push_str(&format!("- {}\n", u));
        }
    }

    out
}

pub fn render_final_markdown(p: &DailyMetaPost) -> String {
    let mut md = String::new();
    md.push_str("# Daily Meta Vibe\n\n");

    md.push_str("## Meta Climate\n");
    md.push_str(&format!("{}\n\n", p.metaClimate.trim()));

    if !p.yesterdaysMomentum.is_empty() {
        md.push_str("## Yesterday’s Momentum\n");
        for b in &p.yesterdaysMomentum {
            md.push_str(&format!("- {}\n", b));
        }
        md.push('\n');
    }

    if !p.flashpointsGainingHeat.is_empty() {
        md.push_str("## Flashpoints Gaining Heat\n");
        for f in &p.flashpointsGainingHeat {
            md.push_str(&format!("- **{}** — {}\n", f.title, f.whyNow));
        }
        md.push('\n');
    }

    if !p.crossOutletSplits.is_empty() {
        md.push_str("## Cross-Outlet Splits\n");
        for s in &p.crossOutletSplits {
            md.push_str(&format!("- {}\n", s));
        }
        md.push('\n');
    }

    if !p.notablyAbsent.is_empty() {
        md.push_str("## Notably Absent\n");
        for s in &p.notablyAbsent {
            md.push_str(&format!("- {}\n", s));
        }
        md.push('\n');
    }

    if !p.nextMilestones.is_empty() {
        md.push_str("## Next Milestones\n");
        for m in &p.nextMilestones {
            md.push_str(&format!("- **{}** — {} (why: {})\n", m.date, m.what, m.whyItMatters));
        }
        md.push('\n');
    }

    md.push_str("## Interpretive Meta\n");
    md.push_str(&format!("{}\n", p.interpretiveMeta.trim()));

    md
}