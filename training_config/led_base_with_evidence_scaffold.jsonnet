# local transformer_model = "facebook/bart-base";
local transformer_model = "allenai/led-base-16384";
# local transformer_model = "allenai/led-large-16384-arxiv";
# local transformer_model = "allenai/led-large-16384";
#local epochs = 10;
local epochs = 50;
local batch_size = 1;
#local num_gradient_accumulation_steps = 2;
#local num_gradient_accumulation_steps = 4;
local num_gradient_accumulation_steps = 8;
#local num_gradient_accumulation_steps = 16;
#local num_gradient_accumulation_steps = 32;
#local num_gradient_accumulation_steps = 64;

#local train_data_path = "TODO";
#local dev_data_path = "TODO";

# local train_data_path = "/data/qasper-train-v0.1.json";
# local dev_data_path = "/data/qasper-dev-v0.1.json";

# local train_data_path = "/net/nfs2.corp/allennlp/vidhishab/data/qasper-train-dev-v0.1/qasper-train-v0.1.json";
# local dev_data_path = "/net/nfs2.corp/allennlp/vidhishab/data/qasper-train-dev-v0.1/qasper-dev-v0.1.json";

local train_data_path = "/home/vidhisha/qasper-train-dev-v0.1/qasper-train-v0.1.json";
local dev_data_path = "/home/vidhisha/qasper-train-dev-v0.1/qasper-dev-v0.1.json";

local training_data_size = 2672;
local num_gpus = 1;
local use_margin_loss_for_evidence = false;

# local resume_model_dir = "/home/vidhisha/longdoc_pretrain/saved_models/led_base_1sentpass_evscaff_wsectionmarks_wglobalonsectionrougeLsum_top3_inplen16320_att1024_bs2_acc4/arxiv-epoch=04-step=0-val_rougeLsum_epochval/";
# local resume_model_file = "val_rougeLsum_epoch=0.3967.ckpt";

{
    "dataset_reader": {
        "type": "qasper",
        "transformer_model_name": transformer_model,
	"max_document_length": 15360,
	# "max_document_length": 1024,
	#"max_document_length": 16384,
	#"paragraph_separator": "madeupword0000",
	"for_training": true,
	"insert_extra_sep_for_null": false,
	"use_sentence_level_evidence": false,
	"use_margin_loss_for_evidence": use_margin_loss_for_evidence,
	"include_global_attention_on_para_indices": true
    },
    "validation_dataset_reader": {
        "type": "qasper",
        "transformer_model_name": transformer_model,
	"max_document_length": 15360,
	#"max_document_length": 16384,
	#"paragraph_separator": "madeupword0000",
	"for_training": false,
	"insert_extra_sep_for_null": false,
	"use_sentence_level_evidence": false,
	"use_margin_loss_for_evidence": use_margin_loss_for_evidence,
	"include_global_attention_on_para_indices": true
    },
    "train_data_path": train_data_path,
    "validation_data_path": dev_data_path,
    "vocabulary": {
        "type": "empty",
    },
    "model": {
        "type": "qasper_baseline",
        "transformer_model_name": transformer_model,
	"attention_window_size": 1536,
	"gradient_checkpointing": true,
	"use_only_evidence_loss": false,
    "use_evidence_scaffold": true,
    "use_margin_loss_for_evidence": use_margin_loss_for_evidence,
    "use_single_margin_loss": false,
	"attention_dropout": 0.1,
	"per_reference_level_metrics": false,
    #"freeze_non_position_weights": true,
    #"randomly_initialize_model": true,
	#"resume_model_dir": resume_model_dir,
	#"resume_model_file": resume_model_file,
    },
    "data_loader": {
        "batch_size": batch_size,
    },
    "trainer": {
      "optimizer": {
        "type": "adam",
        "lr": 5e-5,
        #"lr": 1e-5,
        #"lr": 5e-4,
        #"lr": 1e-4,
        #"lr": 5e-6,
        #"lr": 1e-6,
      },
      "learning_rate_scheduler": {
        "type": "slanted_triangular",
        "num_epochs": epochs,
        #"cut_frac": 0.1,
        #"cut_frac": 0.2,
        "cut_frac": 0.05,
        "num_steps_per_epoch": std.ceil(training_data_size / (batch_size * num_gradient_accumulation_steps * num_gpus)),
      },
      "callbacks": [
	{"type": "tensorboard",
	# "summary_interval":1,
	"should_log_learning_rate":true,},
      ],
      "grad_clipping": 1.0,
      "num_epochs": epochs,
      "num_gradient_accumulation_steps": num_gradient_accumulation_steps,
      "patience": epochs,
      "validation_metric": "+answer_f1",
      #"validation_metric": "+evidence_f1",
      "enable_default_callbacks": false,
      "use_amp": false,
      "cuda_device": 0,
    },
    #"pytorch_seed": 15371,
    #"pytorch_seed": 1234,
    "pytorch_seed": 6487,
}
