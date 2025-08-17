 bq query --use_legacy_sql=false \
--format=csv \
--replace \
--destination_table=analytics.ds_model_input_export \
'SELECT * FROM analytics.ds__model_input__hourly'

bq extract --destination_format=CSV \
analytics.ds_model_input_export \
gs://model-staging-bucket/model_input_export.csv

gsutil cp gs://model-staging-bucket/model_input_export.csv ~/

pip install polars xgboost 
