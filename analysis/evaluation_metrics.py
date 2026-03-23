"""
 # METRICS ALLOCATED:

LDA MODEL (7 METRICS):
  Chi-Square Test (Country × Topic)        
  Coherence Score (c_v)                     
  Perplexity                               
  Topic Diversity                          
  Silhouette Score (TF-IDF clustering)      
  Davies-Bouldin Index                      
  Cosine Similarity (within-country)        

BERT MODEL (6 METRICS):
  Chi-Square Test (Country × Topic)        
  Coherence Score (c_v) - BERT topics      
  Topic Diversity                       
  Silhouette Score (BERT embeddings)       
  Davies-Bouldin Index                     
  Cosine Similarity (within-country)       

"""

import pandas as pd
import numpy as np
import yaml
from pathlib import Path
from scipy.spatial.distance import pdist, squareform, cosine
from scipy.stats import chi2_contingency
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import CoherenceModel
from gensim.corpora import Dictionary
from gensim.models import LdaModel
import json
import warnings

warnings.filterwarnings('ignore')



class EvaluationPipeline:
    """Separate evaluation pipelines for LDA and BERT topic models"""

    def __init__(self, config_path='config/config.yaml'):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        self.results_lda = {}
        self.results_bert = {}
        self.data_lda = None
        self.data_bert = None
        self.X_tfidf = None

    # ============ DATA LOADING ============

    def load_lda_data(self):
        """Load clean_data and merge with LDA topics"""
        print(" Loading LDA data...")
        try:
            clean_data = pd.read_csv(self.config['data']['clean'])
            lda_results = pd.read_csv(self.config['data']['lda_results'])
            
            print(f"   clean_data.csv: {clean_data.shape}")
            print(f"   lda_topic_model_results.csv: {lda_results.shape}")
            
            if len(clean_data) == len(lda_results):
                self.data_lda = clean_data.copy()
                self.data_lda['lda_topic'] = lda_results['lda_topic'].values
                print(f"   Merged: added 'lda_topic' column")
                return self.data_lda
            else:
                print(f"   Row count mismatch: {len(clean_data)} vs {len(lda_results)}")
                return None
        except Exception as e:
            print(f" LDA data loading failed: {e}")
            return None

    def load_bert_data(self):
        """Load clean_data and merge with BERT topics"""
        print(" Loading BERT data...")
        try:
            clean_data = pd.read_csv(self.config['data']['clean'])
            bert_results = pd.read_csv(self.config['data'].get('bert_results', 'data/bert_topic_model_results.csv'))
            
            print(f"   clean_data.csv: {clean_data.shape}")
            print(f"   bert_topic_model_results.csv: {bert_results.shape}")
            
            # Normalize BERT topic column name
            topic_col_candidates = ['bert_topic', 'topic', 'Topic', 'topic_id', 'cluster']
            bert_topic_col = None
            for col in topic_col_candidates:
                if col in bert_results.columns:
                    bert_topic_col = col
                    break
            
            if bert_topic_col is None:
                print(f"   No topic column found in BERT results")
                return None
            
            if len(clean_data) == len(bert_results):
                self.data_bert = clean_data.copy()
                self.data_bert['bert_topic'] = bert_results[bert_topic_col].values
                print(f"   Merged: added 'bert_topic' column from '{bert_topic_col}'")
                return self.data_bert
            else:
                print(f"   Row count mismatch: {len(clean_data)} vs {len(bert_results)}")
                return None
        except Exception as e:
            print(f" BERT data loading failed: {e}")
            return None

    def _build_tfidf_matrix(self):
        """Build TF-IDF matrix once for reuse"""
        if self.X_tfidf is not None:
            return
        
        print(" Building TF-IDF matrix from clean_data...")
        try:
            vectorizer = TfidfVectorizer(max_features=100)
            data = self.data_lda if self.data_lda is not None else self.data_bert
            self.X_tfidf = vectorizer.fit_transform(data['clean_text']).toarray()
            print(f"   TF-IDF shape: {self.X_tfidf.shape}")
        except Exception as e:
            print(f"   TF-IDF build failed: {e}")

    # ============ CHI-SQUARE TEST (SHARED) ============

    def compute_chi_square_lda(self):
        """Chi-Square: Country × LDA Topic distribution"""
        print("\n Computing Chi-Square Test (Country × LDA Topic)")
        try:
            data = self.data_lda.copy()
            contingency = pd.crosstab(data['country'], data['lda_topic'])
            chi2, p_value, dof, _ = chi2_contingency(contingency)
            
            result = {
                'chi_square_statistic': float(chi2),
                'p_value': float(p_value),
                'degrees_of_freedom': int(dof),
                'interpretation': 'SIGNIFICANT (p<0.05)' if p_value < 0.05 else 'NOT SIGNIFICANT (p≥0.05)'
            }
            
            output_path = self.config['evaluation_metrics'].get(
                'chi_square_lda_output', 'analysis_results/lda_chi_square_results.txt'
            )
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w',encoding="utf-8") as f:
                f.write(f"Chi-Square Test: Country × LDA Topic Distribution\n")
                f.write(f"{'='*60}\n\n")
                f.write(f"Chi-square statistic: {chi2:.4f}\n")
                f.write(f"p-value: {p_value:.6f}\n")
                f.write(f"Degrees of freedom: {dof}\n")
                f.write(f"Result: {result['interpretation']}\n\n")
                f.write(f"Interpretation:\n")
                f.write(f"  - Tests if topic distributions differ across countries\n")
                f.write(f"  - p < 0.05: Significant country-level differences in LDA topics\n")
                f.write(f"  - p ≥ 0.05: No significant differences\n")
            
            self.results_lda['chi_square'] = result
            print(f"   χ²={chi2:.4f}, p={p_value:.6f} → {result['interpretation']}")
        except Exception as e:
            print(f"   Chi-Square (LDA) failed: {e}")

    def compute_chi_square_bert(self):
        """Chi-Square: Country × BERT Topic distribution"""
        print("\n Computing Chi-Square Test (Country × BERT Topic)")
        try:
            data = self.data_bert.copy()
            contingency = pd.crosstab(data['country'], data['bert_topic'])
            chi2, p_value, dof, _ = chi2_contingency(contingency)
            
            result = {
                'chi_square_statistic': float(chi2),
                'p_value': float(p_value),
                'degrees_of_freedom': int(dof),
                'interpretation': 'SIGNIFICANT (p<0.05)' if p_value < 0.05 else 'NOT SIGNIFICANT (p≥0.05)'
            }
            
            output_path = self.config['evaluation_metrics'].get(
                'chi_square_bert_output', 'analysis_results/bert_chi_square_results.txt'
            )
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w',encoding="utf-8") as f:
                f.write(f"Chi-Square Test: Country × BERT Topic Distribution\n")
                f.write(f"{'='*60}\n\n")
                f.write(f"Chi-square statistic: {chi2:.4f}\n")
                f.write(f"p-value: {p_value:.6f}\n")
                f.write(f"Degrees of freedom: {dof}\n")
                f.write(f"Result: {result['interpretation']}\n\n")
                f.write(f"Interpretation:\n")
                f.write(f"  - Tests if topic distributions differ across countries\n")
                f.write(f"  - p < 0.05: Significant country-level differences in BERT topics\n")
                f.write(f"  - p ≥ 0.05: No significant differences\n")
            
            self.results_bert['chi_square'] = result
            print(f"   χ²={chi2:.4f}, p={p_value:.6f} → {result['interpretation']}")
        except Exception as e:
            print(f"   Chi-Square (BERT) failed: {e}")

    # ============ COHERENCE (LDA) ============

    def compute_coherence_lda(self):
        """Coherence (c_v metric) for LDA topics"""
        print("\n Computing Coherence Score (c_v) for LDA")
        try:
            lda_data = pd.read_csv(self.config['data']['lda_results'])
            
            if 'tokens' not in lda_data.columns:
                print("   'tokens' column not found in LDA results. Skipping.")
                return
            
            texts = lda_data['tokens'].apply(
                lambda x: eval(x) if isinstance(x, str) else x
            ).tolist()
            dictionary = Dictionary(texts)
            
            lda_model_path = self.config['data'].get(
                'lda_model', 'lda_analysis/lda_model.gensim'
            )
            
            if not Path(lda_model_path).exists():
                print(f"   LDA model not found at {lda_model_path}")
                return
            
            lda_model = LdaModel.load(lda_model_path)
            
            coherence_model = CoherenceModel(
                model=lda_model,
                texts=texts,
                dictionary=dictionary,
                coherence='c_v'
            )
            
            overall_coherence = coherence_model.get_coherence()
            per_topic_coherence = coherence_model.get_coherence_per_topic()
            
            output_path = self.config['evaluation_metrics'].get(
                'coherence_lda_output', 'analysis_results/lda_coherence_scores.csv'
            )
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            
            coherence_df = pd.DataFrame({
                'Topic': range(len(per_topic_coherence)),
                'Coherence_Score_cv': per_topic_coherence
            })
            coherence_df.to_csv(output_path, index=False)
            
            self.results_lda['coherence'] = {
                'overall_score': float(overall_coherence),
                'per_topic_scores': per_topic_coherence.tolist() if hasattr(per_topic_coherence, 'tolist') else list(per_topic_coherence),
                'metric': 'c_v (Röder et al. 2015)',
                'interpretation': 'HIGH' if overall_coherence > 0.5 else 'MODERATE' if overall_coherence > 0.3 else 'LOW'
            }
            
            print(f"   Overall Coherence (c_v): {overall_coherence:.4f} ({self.results_lda['coherence']['interpretation']})")
        except Exception as e:
            print(f"   Coherence (LDA) failed: {e}")

    # ============ COHERENCE (BERT) ============

    def compute_coherence_bert(self):
        """Coherence (c_v metric) for BERT topics"""
        print("\n Computing Coherence Score (c_v) for BERT")
        try:
            from bertopic import BERTopic
            
            bert_data = pd.read_csv(self.config['data'].get(
                'bert_results', 'data/bert_topic_model_results.csv'
            ))
            
            if 'tokens' not in bert_data.columns:
                print("   'tokens' column not found in BERT results. Skipping.")
                return
            
            texts = bert_data['tokens'].apply(
                lambda x: eval(x) if isinstance(x, str) else x
            ).tolist()
            dictionary = Dictionary(texts)
            
            # Try to load BERT model
            bert_model_paths = [
                self.config['data'].get('bertopic_model', 'models/bertopic_model/bertopic_model'),
                'bertopic_analysis/bertopic_model',
                'models/bertopic_model'
            ]
            
            loaded_model = None
            for path in bert_model_paths:
                try:
                    loaded_model = BERTopic.load(path)
                    print(f"   Loaded BERTopic model from: {path}")
                    break
                except (FileNotFoundError, Exception):
                    continue
            
            if loaded_model is None:
                print(f"   BERTopic model not found. Skipping BERT coherence.")
                return
            
            topic_dict = loaded_model.get_topics()
            if not topic_dict or len(topic_dict) == 0:
                print("   No topics found in BERT model.")
                return
            
            # Extract top words per topic
            topics_for_coherence = []
            for topic_id in sorted(topic_dict.keys()):
                if topic_id >= 0:  # Skip noise topic
                    top_words = [word for word, _ in topic_dict[topic_id][:10]]
                    topics_for_coherence.append(top_words)
            
            if len(topics_for_coherence) == 0:
                print("   No valid topics extracted from BERT.")
                return
            
            coherence_model = CoherenceModel(
                topics=topics_for_coherence,
                texts=texts,
                dictionary=dictionary,
                coherence='c_v'
            )
            
            overall_coherence = coherence_model.get_coherence()
            per_topic_coherence = coherence_model.get_coherence_per_topic()
            
            output_path = self.config['evaluation_metrics'].get(
                'coherence_bert_output', 'analysis_results/bert_coherence_scores.csv'
            )
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            
            coherence_df = pd.DataFrame({
                'Topic': range(len(per_topic_coherence)),
                'Coherence_Score_cv': per_topic_coherence
            })
            coherence_df.to_csv(output_path, index=False)
            
            self.results_bert['coherence'] = {
                'overall_score': float(overall_coherence),
                'per_topic_scores': per_topic_coherence.tolist() if hasattr(per_topic_coherence, 'tolist') else list(per_topic_coherence),
                'metric': 'c_v (Röder et al. 2015)',
                'interpretation': 'HIGH' if overall_coherence > 0.5 else 'MODERATE' if overall_coherence > 0.3 else 'LOW'
            }
            
            print(f"   Overall Coherence (c_v): {overall_coherence:.4f} ({self.results_bert['coherence']['interpretation']})")
        except ImportError:
            print("   BERTopic not installed. Skipping BERT coherence.")
        except Exception as e:
            print(f"   Coherence (BERT) failed: {e}")

    # ============ PERPLEXITY (LDA ONLY) ============

    def compute_perplexity_lda(self):
        """Perplexity for LDA model - BERT doesn't have this metric"""
        print("\n Computing Perplexity (LDA ONLY - BERT N/A)")
        try:
            lda_data = pd.read_csv(self.config['data']['lda_results'])
            
            if 'tokens' not in lda_data.columns:
                print("   'tokens' column not found. Skipping perplexity.")
                return
            
            texts = lda_data['tokens'].apply(
                lambda x: eval(x) if isinstance(x, str) else x
            ).tolist()
            dictionary = Dictionary(texts)
            corpus = [dictionary.doc2bow(text) for text in texts]
            
            lda_model_path = self.config['data'].get(
                'lda_model', 'lda_analysis/lda_model.gensim'
            )
            
            if not Path(lda_model_path).exists():
                print(f"   LDA model not found at {lda_model_path}")
                return
            
            lda_model = LdaModel.load(lda_model_path)
            perplexity = lda_model.log_perplexity(corpus)
            
            output_path = self.config['evaluation_metrics'].get(
                'perplexity_lda_output', 'analysis_results/lda_perplexity_scores.txt'
            )
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w') as f:
                f.write(f"Perplexity (LDA Model)\n")
                f.write(f"{'='*50}\n\n")
                f.write(f"Perplexity Score: {perplexity:.4f}\n\n")
                f.write(f"Interpretation:\n")
                f.write(f"  - Measures negative log-likelihood on test set\n")
                f.write(f"  - Lower values = better generalization\n")
                f.write(f"  - Typical range: -10 to -5\n")
                f.write(f"  - Your score: {perplexity:.4f}\n")
                f.write(f"\nNote: BERT models don't use perplexity metric\n")
            
            self.results_lda['perplexity'] = {
                'score': float(perplexity),
                'interpretation': 'GOOD' if perplexity > -10 else 'NEEDS_REVIEW'
            }
            
            print(f"   Perplexity: {perplexity:.4f}")
        except Exception as e:
            print(f"   Perplexity (LDA) failed: {e}")

    # ============ TOPIC DIVERSITY (LDA) ============

    def compute_topic_diversity_lda(self):
        """Topic diversity for LDA topics"""
        print("\n Computing Topic Diversity (LDA)")
        try:
            data = self.data_lda.copy()
            diversity_scores = []
            
            for topic_id in sorted(data['lda_topic'].unique()):
                if topic_id >= 0:
                    topic_docs = data[data['lda_topic'] == topic_id]['clean_text'].str.split().tolist()
                    if len(topic_docs) >= 2:
                        vocab = set()
                        for doc in topic_docs[:10]:
                            vocab.update(doc)
                        diversity = len(vocab) / (len(topic_docs) * 10 + 1)
                        diversity_scores.append(diversity)
            
            overall_diversity = np.mean(diversity_scores) if diversity_scores else 0.0
            
            output_path = self.config['evaluation_metrics'].get(
                'topic_diversity_lda_output', 'analysis_results/lda_topic_diversity.csv'
            )
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            
            diversity_df = pd.DataFrame({
                'Topic': range(len(diversity_scores)),
                'Diversity_Score': diversity_scores
            })
            diversity_df.to_csv(output_path, index=False)
            
            self.results_lda['topic_diversity'] = {
                'overall_score': float(overall_diversity),
                'per_topic_scores': diversity_scores,
                'interpretation': 'HIGH' if overall_diversity > 0.5 else 'LOW'
            }
            
            print(f"   Topic Diversity: {overall_diversity:.4f} ({self.results_lda['topic_diversity']['interpretation']})")
        except Exception as e:
            print(f"   Topic Diversity (LDA) failed: {e}")

    # ============ TOPIC DIVERSITY (BERT) ============

    def compute_topic_diversity_bert(self):
        """Topic diversity for BERT topics"""
        print("\n Computing Topic Diversity (BERT)")
        try:
            data = self.data_bert.copy()
            diversity_scores = []
            
            for topic_id in sorted(data['bert_topic'].unique()):
                if topic_id >= 0:  # Skip noise
                    topic_docs = data[data['bert_topic'] == topic_id]['clean_text'].str.split().tolist()
                    if len(topic_docs) >= 2:
                        vocab = set()
                        for doc in topic_docs[:10]:
                            vocab.update(doc)
                        diversity = len(vocab) / (len(topic_docs) * 10 + 1)
                        diversity_scores.append(diversity)
            
            overall_diversity = np.mean(diversity_scores) if diversity_scores else 0.0
            
            output_path = self.config['evaluation_metrics'].get(
                'topic_diversity_bert_output', 'analysis_results/bert_topic_diversity.csv'
            )
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            
            diversity_df = pd.DataFrame({
                'Topic': range(len(diversity_scores)),
                'Diversity_Score': diversity_scores
            })
            diversity_df.to_csv(output_path, index=False)
            
            self.results_bert['topic_diversity'] = {
                'overall_score': float(overall_diversity),
                'per_topic_scores': diversity_scores,
                'interpretation': 'HIGH' if overall_diversity > 0.5 else 'LOW'
            }
            
            print(f"   Topic Diversity: {overall_diversity:.4f} ({self.results_bert['topic_diversity']['interpretation']})")
        except Exception as e:
            print(f"   Topic Diversity (BERT) failed: {e}")

    # ============ SILHOUETTE SCORE (LDA) ============

    def compute_silhouette_lda(self):
        """Silhouette Score for LDA topic clustering"""
        print("\n Computing Silhouette Score (LDA)")
        try:
            self._build_tfidf_matrix()
            
            if self.X_tfidf is None:
                print("   TF-IDF matrix not available.")
                return
            
            data = self.data_lda.copy()
            labels = data['lda_topic'].values
            
            silhouette = silhouette_score(self.X_tfidf, labels)
            
            output_path = self.config['evaluation_metrics'].get(
                'silhouette_lda_output', 'analysis_results/lda_silhouette_scores.txt'
            )
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w') as f:
                f.write(f"Silhouette Score (LDA Topic Clustering on TF-IDF)\n")
                f.write(f"{'='*50}\n\n")
                f.write(f"Score: {silhouette:.4f}\n\n")
                f.write(f"Interpretation:\n")
                f.write(f"  - Range: -1 to +1\n")
                f.write(f"  - > 0.5: Well-separated clusters\n")
                f.write(f"  - 0.3-0.5: Overlapping clusters\n")
                f.write(f"  - < 0.3: Poor cluster quality\n")
                f.write(f"\nResult: {'GOOD' if silhouette > 0.5 else 'ACCEPTABLE' if silhouette > 0.3 else 'POOR'}\n")
            
            self.results_lda['silhouette'] = {
                'score': float(silhouette),
                'interpretation': 'GOOD' if silhouette > 0.5 else 'ACCEPTABLE' if silhouette > 0.3 else 'POOR'
            }
            
            print(f"   Silhouette Score: {silhouette:.4f} ({self.results_lda['silhouette']['interpretation']})")
        except Exception as e:
            print(f"   Silhouette (LDA) failed: {e}")

    # ============ SILHOUETTE SCORE (BERT) ============

    def compute_silhouette_bert(self):
        """Silhouette Score for BERT topic clustering"""
        print("\n Computing Silhouette Score (BERT)")
        try:
            self._build_tfidf_matrix()
            
            if self.X_tfidf is None:
                print("   TF-IDF matrix not available.")
                return
            
            data = self.data_bert.copy()
            labels = data['bert_topic'].values
            
            silhouette = silhouette_score(self.X_tfidf, labels)
            
            output_path = self.config['evaluation_metrics'].get(
                'silhouette_bert_output', 'analysis_results/bert_silhouette_scores.txt'
            )
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w') as f:
                f.write(f"Silhouette Score (BERT Topic Clustering on TF-IDF)\n")
                f.write(f"{'='*50}\n\n")
                f.write(f"Score: {silhouette:.4f}\n\n")
                f.write(f"Interpretation:\n")
                f.write(f"  - Range: -1 to +1\n")
                f.write(f"  - > 0.5: Well-separated clusters\n")
                f.write(f"  - 0.3-0.5: Overlapping clusters\n")
                f.write(f"  - < 0.3: Poor cluster quality\n")
                f.write(f"\nResult: {'GOOD' if silhouette > 0.5 else 'ACCEPTABLE' if silhouette > 0.3 else 'POOR'}\n")
            
            self.results_bert['silhouette'] = {
                'score': float(silhouette),
                'interpretation': 'GOOD' if silhouette > 0.5 else 'ACCEPTABLE' if silhouette > 0.3 else 'POOR'
            }
            
            print(f"   Silhouette Score: {silhouette:.4f} ({self.results_bert['silhouette']['interpretation']})")
        except Exception as e:
            print(f"   Silhouette (BERT) failed: {e}")

    # ============ DAVIES-BOULDIN INDEX (LDA) ============

    def compute_davies_bouldin_lda(self):
        """Davies-Bouldin Index for LDA topic clustering"""
        print("\n Computing Davies-Bouldin Index (LDA)")
        try:
            self._build_tfidf_matrix()
            
            if self.X_tfidf is None:
                print("   TF-IDF matrix not available.")
                return
            
            data = self.data_lda.copy()
            labels = data['lda_topic'].values
            
            dbi = davies_bouldin_score(self.X_tfidf, labels)
            
            output_path = self.config['evaluation_metrics'].get(
                'davies_bouldin_lda_output', 'analysis_results/lda_davies_bouldin_index.txt'
            )
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w') as f:
                f.write(f"Davies-Bouldin Index (LDA)\n")
                f.write(f"{'='*40}\n\n")
                f.write(f"DBI: {dbi:.4f}\n\n")
                f.write(f"Interpretation:\n")
                f.write(f"  - Lower values = more distinct clusters\n")
                f.write(f"  - Optimal range: 0.0-1.0\n")
                f.write(f"  - DBI > 2: Poor separation\n")
                f.write(f"\nResult: {'EXCELLENT' if dbi < 0.5 else 'GOOD' if dbi < 1.0 else 'NEEDS_REVIEW'}\n")
            
            self.results_lda['davies_bouldin'] = {
                'score': float(dbi),
                'interpretation': 'EXCELLENT' if dbi < 0.5 else 'GOOD' if dbi < 1.0 else 'NEEDS_REVIEW'
            }
            
            print(f"   Davies-Bouldin Index: {dbi:.4f} ({self.results_lda['davies_bouldin']['interpretation']})")
        except Exception as e:
            print(f"   Davies-Bouldin (LDA) failed: {e}")

    # ============ DAVIES-BOULDIN INDEX (BERT) ============

    def compute_davies_bouldin_bert(self):
        """Davies-Bouldin Index for BERT topic clustering"""
        print("\n Computing Davies-Bouldin Index (BERT)")
        try:
            self._build_tfidf_matrix()
            
            if self.X_tfidf is None:
                print("   TF-IDF matrix not available.")
                return
            
            data = self.data_bert.copy()
            labels = data['bert_topic'].values
            
            dbi = davies_bouldin_score(self.X_tfidf, labels)
            
            output_path = self.config['evaluation_metrics'].get(
                'davies_bouldin_bert_output', 'analysis_results/bert_davies_bouldin_index.txt'
            )
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w') as f:
                f.write(f"Davies-Bouldin Index (BERT)\n")
                f.write(f"{'='*40}\n\n")
                f.write(f"DBI: {dbi:.4f}\n\n")
                f.write(f"Interpretation:\n")
                f.write(f"  - Lower values = more distinct clusters\n")
                f.write(f"  - Optimal range: 0.0-1.0\n")
                f.write(f"  - DBI > 2: Poor separation\n")
                f.write(f"\nResult: {'EXCELLENT' if dbi < 0.5 else 'GOOD' if dbi < 1.0 else 'NEEDS_REVIEW'}\n")
            
            self.results_bert['davies_bouldin'] = {
                'score': float(dbi),
                'interpretation': 'EXCELLENT' if dbi < 0.5 else 'GOOD' if dbi < 1.0 else 'NEEDS_REVIEW'
            }
            
            print(f"   Davies-Bouldin Index: {dbi:.4f} ({self.results_bert['davies_bouldin']['interpretation']})")
        except Exception as e:
            print(f"   Davies-Bouldin (BERT) failed: {e}")

    # ============ COSINE SIMILARITY (LDA) ============

    def compute_cosine_similarity_lda(self):
        """Within-country cosine similarity for LDA topics"""
        print("\n Computing Cosine Similarity (LDA)")
        try:
            self._build_tfidf_matrix()
            
            if self.X_tfidf is None:
                print("   TF-IDF matrix not available.")
                return
            
            data = self.data_lda.copy()
            country_similarity = {}
            
            for country in data['country'].unique():
                country_indices = data[data['country'] == country].index.tolist()
                if len(country_indices) > 1:
                    country_X = self.X_tfidf[country_indices]
                    distances = pdist(country_X, metric='cosine')
                    similarities = 1 - distances
                    country_similarity[country] = {
                        'mean_similarity': float(np.mean(similarities)),
                        'std_similarity': float(np.std(similarities)),
                        'n_documents': len(country_indices)
                    }
            
            output_path = self.config['evaluation_metrics'].get(
                'cosine_similarity_lda_output', 'analysis_results/lda_cosine_similarity.csv'
            )
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            
            similarity_df = pd.DataFrame(country_similarity).T
            similarity_df.to_csv(output_path)
            
            self.results_lda['cosine_similarity'] = country_similarity
            
            print(f"   Computed for {len(country_similarity)} countries")
        except Exception as e:
            print(f"   Cosine Similarity (LDA) failed: {e}")

    # ============ COSINE SIMILARITY (BERT) ============

    def compute_cosine_similarity_bert(self):
        """Within-country cosine similarity for BERT topics"""
        print("\n Computing Cosine Similarity (BERT)")
        try:
            self._build_tfidf_matrix()
            
            if self.X_tfidf is None:
                print("   TF-IDF matrix not available.")
                return
            
            data = self.data_bert.copy()
            country_similarity = {}
            
            for country in data['country'].unique():
                country_indices = data[data['country'] == country].index.tolist()
                if len(country_indices) > 1:
                    country_X = self.X_tfidf[country_indices]
                    distances = pdist(country_X, metric='cosine')
                    similarities = 1 - distances
                    country_similarity[country] = {
                        'mean_similarity': float(np.mean(similarities)),
                        'std_similarity': float(np.std(similarities)),
                        'n_documents': len(country_indices)
                    }
            
            output_path = self.config['evaluation_metrics'].get(
                'cosine_similarity_bert_output', 'analysis_results/bert_cosine_similarity.csv'
            )
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            
            similarity_df = pd.DataFrame(country_similarity).T
            similarity_df.to_csv(output_path)
            
            self.results_bert['cosine_similarity'] = country_similarity
            
            print(f"   Computed for {len(country_similarity)} countries")
        except Exception as e:
            print(f"   Cosine Similarity (BERT) failed: {e}")

    # ============ SAVE REPORTS ============

    def save_lda_report(self):
        """Save LDA evaluation report"""
        print("\n Saving LDA evaluation report...")
        try:
            output_path = self.config['evaluation_metrics'].get(
                'lda_report_output', 'analysis_results/lda_evaluation_report.json'
            )
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            
            report = self._serialize_results(self.results_lda)
            with open(output_path, 'w') as f:
                json.dump(report, f, indent=2)
            
            print(f"   Saved to {output_path}")
        except Exception as e:
            print(f"   Failed to save LDA report: {e}")

    def save_bert_report(self):
        """Save BERT evaluation report"""
        print("\n Saving BERT evaluation report...")
        try:
            output_path = self.config['evaluation_metrics'].get(
                'bert_report_output', 'analysis_results/bert_evaluation_report.json'
            )
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            
            report = self._serialize_results(self.results_bert)
            with open(output_path, 'w') as f:
                json.dump(report, f, indent=2)
            
            print(f"   Saved to {output_path}")
        except Exception as e:
            print(f"   Failed to save BERT report: {e}")

    @staticmethod
    def _serialize_results(results):
        """Convert numpy types to JSON-serializable formats"""
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

    # ============ MAIN PIPELINE ============

    def run_lda_pipeline(self):
        """Execute all LDA metrics"""
        print("\n" + "=" * 70)
        print("STARTING LDA EVALUATION PIPELINE (7 METRICS)")
        print("=" * 70)
        
        if self.load_lda_data() is None:
            print(" Cannot proceed without LDA data. Exiting.")
            return self.results_lda
        
        self.compute_chi_square_lda()
        self.compute_coherence_lda()
        self.compute_perplexity_lda()
        self.compute_topic_diversity_lda()
        self._build_tfidf_matrix()
        self.compute_silhouette_lda()
        self.compute_davies_bouldin_lda()
        self.compute_cosine_similarity_lda()
        
        self.save_lda_report()
        
        print("\n" + "=" * 70)
        print("LDA EVALUATION COMPLETE ")
        print("=" * 70)
        
        return self.results_lda

    def run_bert_pipeline(self):
        """Execute all BERT metrics"""
        print("\n" + "=" * 70)
        print("STARTING BERT EVALUATION PIPELINE (6 METRICS)")
        print("=" * 70)
        
        if self.load_bert_data() is None:
            print(" Cannot proceed without BERT data. Exiting.")
            return self.results_bert
        
        self.compute_chi_square_bert()
        self.compute_coherence_bert()
        self.compute_topic_diversity_bert()
        self._build_tfidf_matrix()
        self.compute_silhouette_bert()
        self.compute_davies_bouldin_bert()
        self.compute_cosine_similarity_bert()
        
        self.save_bert_report()
        
        print("\n" + "=" * 70)
        print("BERT EVALUATION COMPLETE ")
        print("=" * 70)
        
        return self.results_bert

    def run_all(self):
        """Execute both LDA and BERT pipelines"""
        self.run_lda_pipeline()
        self.run_bert_pipeline()


if __name__ == '__main__':
    pipeline = EvaluationPipeline('config/config.yaml')
    pipeline.run_all()
    
    print("\n" + "=" * 70)
    print(" FINAL EVALUATION SUMMARY")
    print("=" * 70)
    
    print("\n LDA RESULTS (7 metrics):")
    for metric, result in pipeline.results_lda.items():
        print(f"   {metric}: {result}")
    
    print("\n BERT RESULTS (6 metrics):")
    for metric, result in pipeline.results_bert.items():
        print(f"   {metric}: {result}")