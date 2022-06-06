# Data and scripts for target-coherence metric


```convert_targetcoherence``` converts the data from  ```tcdata_wneg folder``` into data used for training the tc metric. 


To train the tc metric, run
```bash
test python run_dialclassifier.py --model_name_or_path tmp/test-alv2-t1neg_response_target_specific --validation_file ottersnegdata/total_dev_wneg.jsonl ---output_dir tmp/test-alv2-t1neg_response_target_specific  --per_device_eval_batch_size 120 --overwrite_cache   --do_eval  --overwrite_output_dir --load_best_model_at_end --types_to_avoid neg_response_target_specific --metric_for_best_model accuracy
```

Trained model is uploaded to google drive folder

```run_targetcoherence``` can be used for using the metric

```human_ratings_correlation_tc``` file contins human ratings collected for correlation analysis

