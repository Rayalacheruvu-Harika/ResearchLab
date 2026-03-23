import sys
from pathlib import Path
from analysis.evaluation_metrics import EvaluationPipeline
import json
from datetime import datetime
from config import CONFIG

def main():
    """Run complete evaluation pipeline (LDA + BERT separately)"""
    
    print("="*70)
    print("RQ2 EVALUATION PIPELINE - LDA vs BERT COMPARISON")
    print("="*70)
    print(f"Execution Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("")
    
    try:
        # Initialize pipeline
        pipeline = EvaluationPipeline('config/config.yaml')
        
        # ========== RUN LDA PIPELINE ==========
        print("\n" + " "*35)
        print("STARTING LDA EVALUATION (7 METRICS)")
        print(" "*35)
        results_lda = pipeline.run_lda_pipeline()
        
        # ========== RUN BERT PIPELINE ==========
        print("\n" + " "*35)
        print("STARTING BERT EVALUATION (6 METRICS)")
        print(" "*35)
        results_bert = pipeline.run_bert_pipeline()
        
        # ========== GENERATE COMPARISON REPORT ==========
        generate_comparison_report(results_lda, results_bert)
        
        print("\n" + "="*70)
        print(" EVALUATION PIPELINE COMPLETED SUCCESSFULLY")
        print("="*70)
        
        print_summary(results_lda, results_bert)
        
        print("\nOutput Files Generated:")
        print("\n LDA Results (8 files):")
        print("  • analysis_results/lda_chi_square_results.txt")
        print("  • analysis_results/lda_coherence_scores.csv")
        print("  • analysis_results/lda_perplexity_scores.txt ← LDA ONLY")
        print("  • analysis_results/lda_topic_diversity.csv")
        print("  • analysis_results/lda_silhouette_scores.txt")
        print("  • analysis_results/lda_davies_bouldin_index.txt")
        print("  • analysis_results/lda_cosine_similarity.csv")
        print("  • analysis_results/lda_evaluation_report.json")
        
        print("\n BERT Results (7 files):")
        print("  • analysis_results/bert_chi_square_results.txt")
        print("  • analysis_results/bert_coherence_scores.csv")
        print("  • analysis_results/bert_topic_diversity.csv")
        print("  • analysis_results/bert_silhouette_scores.txt")
        print("  • analysis_results/bert_davies_bouldin_index.txt")
        print("  • analysis_results/bert_cosine_similarity.csv")
        print("  • analysis_results/bert_evaluation_report.json")
        
        print("\n Comparison Reports:")
        print("  • analysis_results/lda_vs_bert_comparison.json")
        print("="*70 + "\n")
        
    except Exception as e:
        print(f" Pipeline execution failed: {e}")
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
    
    output_path = Path(CONFIG["evaluation_metrics"]["comparison_report"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(comparison, f, indent=2)
    
    print(f"\n Comparison report saved to {output_path}")


def print_summary(results_lda, results_bert):
    """Print console summary"""
    
    print("\n" + "="*70)
    print(" FINAL EVALUATION SUMMARY")
    print("="*70)
    
    print("\n LDA RESULTS (7 metrics):")
    for metric, value in results_lda.items():
        print(f"   {metric}")
        if isinstance(value, dict):
            for k, v in value.items():
                if k != 'per_topic_scores':  # Don't print every per-topic score
                    print(f"     {k}: {v}")
    
    print("\n BERT RESULTS (6 metrics):")
    for metric, value in results_bert.items():
        print(f"   {metric}")
        if isinstance(value, dict):
            for k, v in value.items():
                if k != 'per_topic_scores':  # Don't print every per-topic score
                    print(f"     {k}: {v}")
    
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