.PHONY: .check_csv .check_csv2 .check_train_result .check_predict_result prep describe histogram scatter

SHELL			:= /usr/bin/zsh

PY				= /usr/bin/env python3

TRAIN_CSV		= dataset_train.csv

TEST_CSV		= dataset_test.csv

TRAIN_RESULT	= after_train.csv

PREDICT_RESULT	= houses.csv

.check_csv:
	@test -f $(TRAIN_CSV) || { echo "ERROR: $(TRAIN_CSV) not found."; exit 2; }

.check_csv2:
	@test -f $(TEST_CSV) || { echo "ERROR: $(TEST_CSV) not found."; exit 3; }

.check_train_result:
	@test -f $(TRAIN_RESULT) || { echo "ERROR: $(TRAIN_RESULT) not found."; exit 4; }

.check_predict_result:
	@test -f $(PREDICT_RESULT) || { echo "ERROR: $(PREDICT_RESULT) not found."; exit 5; }

prep:
	-@chmod +x ./prep_python.sh
	./prep_python.sh

describe: .check_csv
	$(PY) ./describe.py $(TRAIN_CSV)

histogram: .check_csv
	$(PY) ./histogram.py $(TRAIN_CSV)

scatter: .check_csv
	$(PY) ./scatter_plot.py $(TRAIN_CSV)

pairplot: .check_csv
	$(PY) ./pair_plot.py $(TRAIN_CSV)

train: .check_csv
	$(PY) ./logreg_train.py $(TRAIN_CSV)
	$(.check_train_result)

predict: .check_csv2 .check_train_result
	$(PY) ./logreg_predict.py $(TEST_CSV)
	$(.check_predict_result)

fclean:
	-@rm -rf .venv after_train.csv scatter_global_view.png \
	histogram.png scatter_plot.png __pycache__ houses.csv
