.ONESHELL:

CONDAPATH = $$(conda info --base)

install:
	conda env create -f iba.yml
	${CONDAPATH}/envs/iba/bin/pip install -r requirements.txt
	# ${CONDAPATH}/envs/iba/bin/pip install -e .[${enable}]

install-mac:
	conda env create -f iba.yml
	conda install nomkl
	${CONDAPATH}/envs/iba/bin/pip install -r requirements.txt
	# ${CONDAPATH}/envs/iba/bin/pip install -e .[${enable}]

update:
	conda env update --prune -f iba.yml
	${CONDAPATH}/envs/iba/bin/pip install -r requirements.txt --upgrade
	# ${CONDAPATH}/envs/iba/bin/pip install -U .[${enable}]

clean:
	conda env remove --name iba