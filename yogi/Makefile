down:
	sudo docker compose down --remove-orphans

build-py:
	sudo docker build -t yogi-py .

db:
	sudo docker compose up -d db

yogi-shell:
	sudo docker compose run --rm --no-deps --entrypoint=/bin/bash yogi-shell

