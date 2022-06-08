# scripts for path generation for otters and then alignment

```augment_otters_withpaths```  Perform keyword extraction followed by creating paths between source and target concepts

```prepare_pathgpt_forotters``` use this notebook to create final aligned paths. The output files from this notebook are used for response generation training and inference

```Hyperparameters``` - 
Please note there are multiple hyperparameters involved in the final path-response pairs selected for response generation. A few examples:

* apply_ranking_topk can be set to any value between 1 to 7
* In function get_filtered_paths, on the line if ```scores[i]<2.0 x min_score```, the threshold can be varied. On line ```if len(pathwords_norel)<2``` the value 2 can be changed.

These hyperparameters effect the final performance of the model and we select the parameters that lead to better performance through hyperparameter tuning.