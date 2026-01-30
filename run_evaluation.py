import sys
import logging
from pathlib import Path
from evaluation_metrics import EvaluationPipeline
import json
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Run complete evaluation pipeline (LDA + BERT separately)"""
    
    logger.info("="*70)
    logger.info("RQ2 EVALUATION PIPELINE - LDA vs BERT COMPARISON")
    logger.info("="*70)
    logger.info(f"Execution Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("")
    
    try:
        # Initialize pipeline
        pipeline = EvaluationPipeline('config/config.yaml')
        
        # ========== RUN LDA PIPELINE ==========
        logger.info("\n" + "🔴 "*35)
        logger.info("STARTING LDA EVALUATION (7 METRICS)")
        logger.info("🔴 "*35)
        results_lda = pipeline.run_lda_pipeline()
        
        # ========== RUN BERT PIPELINE ==========
        logger.info("\n" + "🟣 "*35)
        logger.info("STARTING BERT EVALUATION (6 METRICS)")
        logger.info("🟣 "*35)
        results_bert = pipeline.run_bert_pipeline()
        
        # ========== GENERATE COMPARISON REPORT ==========
        generate_comparison_report(results_lda, results_bert)
        
        logger.info("\n" + "="*70)
        logger.info("✅ EVALUATION PIPELINE COMPLETED SUCCESSFULLY")
        logger.info("="*70)
        
        print_summary(results_lda, results_bert)
        
        logger.info("\nOutput Files Generated:")
        logger.info("\n🔴 LDA Results (8 files):")
        logger.info("  • analysis_results/lda_chi_square_results.txt")
        logger.info("  • analysis_results/lda_coherence_scores.csv")
        logger.info("  • analysis_results/lda_perplexity_scores.txt ← LDA ONLY")
        logger.info("  • analysis_results/lda_topic_diversity.csv")
        logger.info("  • analysis_results/lda_silhouette_scores.txt")
        logger.info("  • analysis_results/lda_davies_bouldin_index.txt")
        logger.info("  • analysis_results/lda_cosine_similarity.csv")
        logger.info("  • analysis_results/lda_evaluation_report.json")
        
        logger.info("\n🟣 BERT Results (7 files):")
        logger.info("  • analysis_results/bert_chi_square_results.txt")
        logger.info("  • analysis_results/bert_coherence_scores.csv")
        logger.info("  • analysis_results/bert_topic_diversity.csv")
        logger.info("  • analysis_results/bert_silhouette_scores.txt")
        logger.info("  • analysis_results/bert_davies_bouldin_index.txt")
        logger.info("  • analysis_results/bert_cosine_similarity.csv")
        logger.info("  • analysis_results/bert_evaluation_report.json")
        
        logger.info("\n📊 Comparison Reports:")
        logger.info("  • analysis_results/lda_vs_bert_comparison.json")
        logger.info("="*70 + "\n")
        
    except Exception as e:
        logger.error(f"❌ Pipeline execution failed: {e}", exc_info=True)
        sys.exit(1)


def generate_comparison_report(results_lda, results_bert):
    """Generate side-by-side comparison report"""
    
    comparison = {
        'timestamp': datetime.now().isoformat(),
        'pipeline': 'RQ2 Evaluation - LDA vs BERT',
        'lda_metrics_count': len(results_lda),
        'bert_metrics_count': len(results_bert),
        'lda_results': _serialize(results_lda),
        'bert_results': _serialize(results_bert),
        'comparison_notes': {
            'lda_unique': ['Perplexity'],
            'bert_unique': [],
            'shared_metrics': [
                'Chi-Square',
                'Coherence (c_v)',
                'Topic Diversity',
                'Silhouette Score',
                'Davies-Bouldin Index',
                'Cosine Similarity'
            ]
        }
    }
    
    output_path = Path('analysis_results/lda_vs_bert_comparison.json')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(comparison, f, indent=2)
    
    logger.info(f"\n✓ Comparison report saved to {output_path}")


def print_summary(results_lda, results_bert):
    """Print console summary"""
    
    print("\n" + "="*70)
    print("📊 FINAL EVALUATION SUMMARY")
    print("="*70)
    
    print("\n🔴 LDA RESULTS (7 metrics):")
    for metric, value in results_lda.items():
        print(f"  ✓ {metric}")
        if isinstance(value, dict):
            for k, v in value.items():
                if k != 'per_topic_scores':  # Don't print every per-topic score
                    print(f"    → {k}: {v}")
    
    print("\n🟣 BERT RESULTS (6 metrics):")
    for metric, value in results_bert.items():
        print(f"  ✓ {metric}")
        if isinstance(value, dict):
            for k, v in value.items():
                if k != 'per_topic_scores':  # Don't print every per-topic score
                    print(f"    → {k}: {v}")
    
    print("\n" + "="*70)


def _serialize(results):
    """Convert numpy types to JSON-serializable formats"""
    import numpy as np
    
    report = {}
    for key, value in results.items():
        if isinstance(value, dict):
            report[key] = {}
            for k, v in value.items():
                if isinstance(v, (np.ndarray, list)):
                    report[key][k] = [float(x) for x in v]
                elif isinstance(v, (np.floating, np.integer)):
                    report[key][k] = float(v)
                else:
                    report[key][k] = v
    return report


if __name__ == '__main__':
    main()