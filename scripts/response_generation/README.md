# scripts for response generation training and inference

Train, test files and model files will be uploaded in the google drive folder (link in the main Readme of the main folder)

Sample command for training model on data created in the last step
```bash
python train_pathottersgpt.py --train_file augfp4_out_of_domain_train_out_goldkeywords.jsonl --validation_file augfp4_out_of_domain_valid_out_istest_goldkeywords-key1.jsonl  --model_name_or_path ../../data_prep/daily_dialogue_act/train_pathseqdd/alv2-tcfiltered0.8_duppathremoved_len12_ppl1.3-vorgresp/checkpoint-500/ --output_dir ftddpaths-oodpaths-pkrv1_exp3_thres2.0/ --num_train_epochs 3 --do_train --overwrite_output_dir --eval_steps 50 --save_steps 50 --evaluation_strategy steps --fp16 --load_best_model_at_end --logging_steps 50 --pad_to_max_length --per_device_train_batch_size 10
```

Sample command for predicting response
```bash
python predict_otters_pathgpt.py  --model_string MODEL_PATH --condition_lambda 0.0 --precondition_topk 20 --do_sample --test_file FOLDER/FILEPREFIX*_test_out_direct.jsonl
```

Sample pat for evaluating model outputs
```bash
python eval_pathotters.py --input_file FILEIN >> metrics.txt
```


