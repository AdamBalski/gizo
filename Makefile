DATA_DIR := data
TARGET_DIR := target
ZIP_PATH := $(DATA_DIR)/blood-cells.zip
KAGGLE_URL := https://www.kaggle.com/api/v1/datasets/download/paultimothymooney/blood-cells
UNZIP_STAMP := $(DATA_DIR)/.dataset_extracted
VENV_DIR := venv
VENV_BIN := $(VENV_DIR)/bin
VENV_PYTHON := $(VENV_BIN)/python
VENV_PIP := $(VENV_BIN)/pip

BBOX_JSON := $(TARGET_DIR)/bounding-boxes.json
FEATURES_JSON := $(TARGET_DIR)/features.json
EVAL_JSON := $(TARGET_DIR)/eval-results.json
SUMMARY_PNG := $(TARGET_DIR)/eval-summary.png
TREE_PNG := $(TARGET_DIR)/decision-tree.png

.PHONY: download clean-data clean clean-all create-bounding-boxes extract-features eval summary render-decision-tree venv check-venv clean-venv

# Check if venv exists and create if needed
check-venv:
	@if [ ! -d "$(VENV_DIR)" ]; then \
		echo "Creating virtual environment..."; \
		python3 -m venv $(VENV_DIR); \
		echo "Installing dependencies..."; \
		$(VENV_PIP) install -r requirements.txt; \
		echo "Virtual environment ready"; \
	fi

# Ensure venv exists before running any Python commands
$(BBOX_JSON): check-venv
	$(VENV_PYTHON) bounding_boxes_creation.py

$(FEATURES_JSON): check-venv $(BBOX_JSON)
	$(VENV_PYTHON) feature_extraction.py

$(EVAL_JSON): check-venv $(FEATURES_JSON)
	$(VENV_PYTHON) evaluator.py

$(SUMMARY_PNG): check-venv $(EVAL_JSON)
	$(VENV_PYTHON) plotter.py

$(TREE_PNG): check-venv $(FEATURES_JSON)
	$(VENV_PYTHON) tree_plotter.py

venv: check-venv
	@echo "Virtual environment is ready at $(VENV_DIR)"
	@echo "Activate with: source $(VENV_BIN)/activate"

clean-venv:
	rm -rf $(VENV_DIR)
	@echo "Removed virtual environment"

download: $(UNZIP_STAMP)

$(UNZIP_STAMP): $(ZIP_PATH)
	unzip -q -o $(ZIP_PATH) -d $(DATA_DIR)
	touch $(UNZIP_STAMP)

data-trim:
	find data -type d -print0 | while IFS= read -r -d '' dir; do \
	  find "$$dir" -maxdepth 1 -type f -print0 | \
	    xargs -0 stat -f '%SB %N' -t '%Y%m%d%H%M%S' | \
	    sort -r | \
	    tail -n +11 | \
	    cut -d' ' -f2- | \
	    tr '\n' '\0' | \
	    xargs -0 rm -- ; \
	done

$(ZIP_PATH):
	mkdir -p $(DATA_DIR)
	curl -L -o $(ZIP_PATH) $(KAGGLE_URL)

clean-data:
	rm -rf $(DATA_DIR)

clean:
	rm -rf $(TARGET_DIR)

clean-all: clean clean-data clean-venv

create-bounding-boxes: $(BBOX_JSON)

extract-features: $(FEATURES_JSON)

eval: $(EVAL_JSON)

summary: $(SUMMARY_PNG)

render-decision-tree: $(TREE_PNG)
