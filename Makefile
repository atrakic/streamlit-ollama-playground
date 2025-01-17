MAKEFLAGS += --silent

OPTIONS ?= --build --remove-orphans #--force-recreate
APP ?= app

.PHONY: local docker healthcheck clean

local:
	streamlit run streamlit_app.py

docker:
	docker-compose up $(OPTIONS) -d

%:
	docker-compose up $(OPTIONS) $@ -d
	docker-compose ps -a

healthcheck:
	docker inspect $(APP) --format "{{ (index (.State.Health.Log) 0).Output }}"

clean:
	docker-compose down --remove-orphans -v --rmi local

-include .env
