.PHONY: install check lint type test serve train cluster

install:        ## Install deps and git hooks
	poetry install
	poetry run pre-commit install

lint:           ## Run flake8
	poetry run flake8 src tests

type:           ## Run mypy
	poetry run mypy src

test:           ## Run the test suite
	poetry run pytest

check: lint type test  ## Run all quality gates

serve:          ## Start the API
	poetry run log-lens serve

train:          ## Train a classifier
	poetry run log-lens train

cluster:        ## Cluster logs in the model's embedding space
	poetry run log-lens cluster

publish-model:  ## Upload a checkpoint to HF Hub: make publish-model MODEL_REPO=user/name MODEL_DIR=models/...
	poetry run python -c "from huggingface_hub import create_repo, upload_folder; \
create_repo('$(MODEL_REPO)', repo_type='model', exist_ok=True); \
upload_folder(repo_id='$(MODEL_REPO)', folder_path='$(MODEL_DIR)', repo_type='model')"

docker-serve:   ## Build the CPU inference image: make docker-serve MODEL_REPO=user/name
	docker build --build-arg MODEL_REPO=$(MODEL_REPO) -t log-lens .

docker-train:   ## Build the GPU training image
	docker build -f Dockerfile.train -t log-lens-train .
