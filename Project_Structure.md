

# ResearchLab — Project Structure



**ResearchLab/**
├── .gitattributes
├── .gitignore
├── Project_Structure.md
├── README.md
├── requirements.txt
│
├── **config**/
│   ├── config.yaml
│   └── __init__.py
│
├── **data/**
│   ├── urls.txt
│   ├── raw_genai_data.csv
│   ├── raw_extracted_data.xlsx
│   ├── merged_dataset.csv
│   ├── final_clean_dataset.csv
│   ├── final_clean_dataset_sentiment_stance.csv
│   ├── final_clean_dataset_with_stance.csv
│   ├── manual_corrected_data.xlsx
│   ├── bert_topic_model_results.csv
│   ├── bert_topic_summary.csv
│   ├── bert_topic_labels.csv
│   ├── lda_topic_model_results.csv
│   ├── lda_topic_summary.csv
│   ├── lda_topics.txt
│   ├── lda_visualization.html
│   ├── university_policy_topic_mapping.csv
│   ├── university_topic_matrix.csv
│   └── university_topic_statistics.csv
│
├── **manual/**
│   ├── university_list.csv
│   ├── annotations.csv
│   ├── sentiment_manual.xlsx
│   └── merge_manual_data.py
│
├── **extraction/**
│   └── extract_data.py
│
├── **cleaning/**
│   └── clean_data.py
│
├── **topic_modeling/**
│   ├── train_lda_model.py
│   ├── lda_uni_topic_modeling.py
│   ├── bert_topic_modeling.py
│   └── bert_topic_labeling.py
│
├── **framing_analysis/**
│   ├── framing.py
│   └── framing_interpret.py
│
├── **sentiment_stance/**
│   ├── sentiment_stance_analysis.py
│   ├── roberta_sentiment.py
│   ├── policy_tone_distilbert.py
│   ├── policy_sentiment_analysis.py
│   ├── policy_stance_analysis.py
│   ├── stance_detection.py
│   ├── manual_annotation_evaluation.py
│   ├── manual_tone_frequency.py
│   └── manual_tone_groups.py
│
├── **analysis/**
│   ├── country_distribution.py
│   ├── country_topic_analysis_all.py
│   ├── country_topic_profiles.py
│   ├── uni_multi_topic_profiling.py
│   ├── topic_cooccurence.py
│   ├── word_cloud.py
│   ├── evaluation_metrics.py
│   ├── run_evaluation.py
│   ├── affective_language_analysis.py
│   ├── curriculum_framing_analysis.py
│   ├── country_framing_summary.py
│   └── institutional_roles_analysis.py
│
├── **models/**
│   ├── bertopic_model/
│   │   └── bertopic_model.pkl
│   └── lda/
│       ├── dictionary.pkl
│       ├── lda_model.gensim
│       ├── lda_model.gensim.expElogbeta.npy
│       ├── lda_model.gensim.id2word
│       └── lda_model.gensim.state
│
├── **analysis_results/**
│   ├── bert_chi_square_results.txt
│   ├── bert_coherence_scores.csv
│   ├── bert_cosine_similarity.csv
│   ├── bert_davies_bouldin_index.txt
│   ├── bert_evaluation_report.json
│   ├── bert_silhouette_scores.txt
│   ├── bert_topic_diversity.csv
│   ├── lda_chi_square_results.txt
│   ├── lda_coherence_scores.csv
│   ├── lda_cosine_similarity.csv
│   ├── lda_davies_bouldin_index.txt
│   ├── lda_evaluation_report.json
│   ├── lda_perplexity_scores.txt
│   ├── lda_silhouette_scores.txt
│   ├── lda_topic_diversity.csv
│   ├── lda_vs_bert_comparison.json
│   ├── country_topic_counts.csv
│   ├── country_topic_proportions.csv
│   ├── heatmap_topics_counts.png
│   ├── heatmap_topics_proportions.png
│   ├── overall_policy_wordcloud.png
│   ├── topic_frequency_bar.png
│   │
│   ├── country/
│   │   ├── country_topic_heatmap_data.csv
│   │   ├── country_topic_profile_table.csv
│   │   ├── heatmap_topic_importance.png
│   │   ├── primary_topic_bar_chart.png
│   │   └── topic_cooccurrence_network.png
│   │
│   ├── framing/
│   │   ├── university_framing_profiles.csv
│   │   ├── overall_frame_distribution.png
│   │   └── dominant_frame_by_country.png
│   │
│   ├── multi_topic_profiling/
│   │   ├── country_top_topics.csv
│   │   └── university_multi_topic_profiles.csv
│   │
│   ├── rq1/
│   │   ├── affective_language_university.csv
│   │   ├── institutional_roles_university.csv
│   │   ├── policy_frames.csv
│   │   ├── rq1_country_summary.csv
│   │   └── figures/
│   │       ├── affective_by_country_university.png
│   │       ├── affective_overall_university.png
│   │       ├── affective_stacked_university.png
│   │       ├── institutional_roles_by_country_grouped.png
│   │       ├── institutional_roles_small_multiples_legend.png
│   │       ├── policy_frames.png
│   │       ├── policy_frames_stacked_lollipop.png
│   │       └── rq1_country_summary_heatmap.png
│   │
│   ├── rq3/
│   │   ├── policy_sentiment_university.csv
│   │   └── figures/
│   │       ├── policy_sentiment_by_country_percentage.png
│   │       └── policy_sentiment_overall_percentage.png
│   │
│   ├── sentiment_analysis/
│   │   ├── policy_tone_distilbert.csv
│   │   ├── sentiment_results.csv
│   │   ├── sentiment_summary.csv
│   │   └── figures/
│   │       ├── policy_tone_overall_bar.png
│   │       └── policy_tone_stacked_by_country.png
│   │
│   ├── stance_detection/
│   │   ├── stance_examples_with_quotes.csv
│   │   ├── stance_extremes_summary.csv
│   │   ├── stance_rq3_report.txt
│   │   ├── stance_scores_document_level.csv
│   │   ├── stance_scores_university_level.csv
│   │   └── figures/
│   │       ├── stance_distribution_by_country.png
│   │       ├── stance_extremes_bar_chart.png
│   │       ├── stance_scatter_universities.png
│   │       └── stance_university_examples.png
│   │
│   ├── manual_sentiment/
│   │   ├── country_tone_distribution_university_final.png
│   │   ├── manual_labels_by_country_top10_each.png
│   │   ├── overall_tone_distribution_university_final.png
│   │   ├── top10_manual_labels.csv
│   │   └── top10_manual_labels.png
│   │
│   └── topic_cooccurrence/
│       ├── topic_cooccurrence_heatmap.png
│       ├── topic_cooccurrence_matrix.csv
│       ├── topic_cooccurrence_network.png
│       └── topic_cooccurrence_pairs.csv
