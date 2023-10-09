.ONESHELL:

CONDAPATH = $$(conda info --base)

install:
	conda env create -f environment.yml
	${CONDAPATH}/envs/iba/bin/pip install -r requirements.txt
	# ${CONDAPATH}/envs/iba/bin/pip install -e .[${enable}]

install-mac:
	conda env create -f environment.yml
	conda install nomkl
	${CONDAPATH}/envs/iba/bin/pip install -r requirements.txt
	# ${CONDAPATH}/envs/iba/bin/pip install -e .[${enable}]

update:
	conda env update --prune -f environment.yml
	${CONDAPATH}/envs/iba/bin/pip install -r requirements.txt --upgrade
	# ${CONDAPATH}/envs/iba/bin/pip install -U .[${enable}]

clean:
	conda env remove --name iba