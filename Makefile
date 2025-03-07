MAKEFLAGS += --silent

OPTIONS ?= --build --remove-orphans #--force-recreate
APP ?= app

.PHONY: docker healthcheck local sync test clean

docker:
	docker-compose up $(OPTIONS) -d

%:
	docker-compose up $(OPTIONS) $@ -d
	docker-compose ps -a

healthcheck:
	docker inspect $(APP) --format "{{ (index (.State.Health.Log) 0).Output }}"

local:
	python streamlit_app.py --browser.gatherUsageStats=false &

sync:
	uv sync

test:
	uv run pytest --verbose

clean:
	docker-compose down --remove-orphans -v --rmi local

-include .env
