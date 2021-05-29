# disaster_response_pipeline

### Run programs
```bash
cd data
python process_data.py disaster_messages.csv disaster_categories.csv DisasterResponse.db

cd ../models
python train_classifier.py ../data/DisasterResponse.db classifier.pkl

cd ../app
python run.py

In another terminal,
env|grep WORK

```

Browse https://SPACEID-3001.SPACEDOMAIN with the output from `env|grep WORK`

