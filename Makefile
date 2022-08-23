M:=""

setup-env:
	conda env create --file=environment.yml

import-to-nb:
	python src/kaggle_util/import_to_nb.py -f $(F) -m "$(M)"

submit-to-kaggle: import-to-nb
	python src/kaggle_util/submit_kaggle.py -f $(F)

unittest:
	pytest -v tests/ --sample_level 1

unittest-small:
	pytest -v tests/ --sample_level 2