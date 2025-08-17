
start_time=$(date +%s)

echo "Script started at $(date)"

#Create table from view
bq query --use_legacy_sql=false \
--format=csv \
--replace \
--destination_table=analytics.ds_model_input_export \
'SELECT *
FROM analytics.ds__model_input__60_min
WHERE ts >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 120 DAY)'

#Export table as csv to statging bucket
bq extract --destination_format=CSV \
analytics.ds_model_input_export \
gs://model-staging-bucket/model_input_export.csv

#Copy csv to local VM disk
gsutil cp gs://model-staging-bucket/model_input_export.csv ~/

#Install python, git, venv
sudo apt-get update
sudo apt-get install -y python3 python3-pip  python3-venv git
sudo update-alternatives --install /usr/bin/python python /usr/bin/python3 1

# Start venv and activate the virtual environment
python3 -m venv venv
source venv/bin/activate

# Upgrade pip inside the venv
pip install --upgrade pip
REPO_DIR="dispnese-prediction"
REPO_URL="https://github.com/danieleduardofajardof/dispnese-prediction.git"

if [ -d "$REPO_DIR" ]; then
    echo "Repository exists, pulling latest changes..."
    cd "$REPO_DIR"
    git pull
    cd ..
else
    echo "Repository does not exist, cloning..."
    git clone "$REPO_URL"
fi

# Install your requirements

pip install -r dispnese-prediction/requirements.txt

# Run prediction
python dispnese-prediction/predict.py

#Copy result to output bucket
gsutil cp future*.csv gs://model-output-preds/

rm -f *.csv

echo "All CSV files deleted."

echo "Script finished at $(date)"
echo "Total time elapsed: $elapsed seconds"
