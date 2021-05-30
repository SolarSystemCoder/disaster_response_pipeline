# disaster_response_pipeline

### Disaster classification project
This project is for classifying categories based on input text messages. So, in emergency, by classifying text messages to the right category, departments under the categories can support people in need properly

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

### Files

/data/process_data.py --> a script to run ETL job
/data/disaster_categories.csv --> categories input file
/data/disaster_messages.csv --> messages input file
/data/DisasterResponse.db  --> ETL output file for sqlight
/models/train_classifier.py  --> a script to build a NLP model
/models/classifier.pkl  --> a model output
/app/run.py  --> a script to run flask service
/app/templates/*  --> a files to make htmls
