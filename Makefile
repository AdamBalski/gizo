DATA_DIR := data
TARGET_DIR := target
ZIP_PATH := $(DATA_DIR)/blood-cells.zip
KAGGLE_URL := https://www.kaggle.com/api/v1/datasets/download/paultimothymooney/blood-cells
UNZIP_STAMP := $(DATA_DIR)/.dataset_extracted

BBOX_JSON := $(TARGET_DIR)/bounding-boxes.json
FEATURES_JSON := $(TARGET_DIR)/features.json
EVAL_JSON := $(TARGET_DIR)/eval-results.json
SUMMARY_PNG := $(TARGET_DIR)/eval-summary.png
TREE_PNG := $(TARGET_DIR)/decision-tree.png

.PHONY: download clean-data clean create-bounding-boxes extract-features eval summary render-decision-tree

download: $(UNZIP_STAMP)

$(UNZIP_STAMP): $(ZIP_PATH)
	unzip -q -o $(ZIP_PATH) -d $(DATA_DIR)
	touch $(UNZIP_STAMP)

$(ZIP_PATH):
	mkdir -p $(DATA_DIR)
	curl -L -o $(ZIP_PATH) $(KAGGLE_URL)

clean-data:
	rm -rf $(DATA_DIR)

clean:
	rm -rf $(TARGET_DIR)

$(BBOX_JSON):
	python3 bounding_boxes_creation.py

create-bounding-boxes: $(BBOX_JSON)

$(FEATURES_JSON): $(BBOX_JSON)
	python3 feature_extraction.py

extract-features: $(FEATURES_JSON)

$(EVAL_JSON): $(FEATURES_JSON)
	python3 evaluator.py

eval: $(EVAL_JSON)

$(SUMMARY_PNG): $(EVAL_JSON)
	python3 plotter.py

summary: $(SUMMARY_PNG)

$(TREE_PNG): $(FEATURES_JSON)
	python3 tree_plotter.py

render-decision-tree: $(TREE_PNG)
