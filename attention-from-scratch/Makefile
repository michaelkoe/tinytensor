.PHONY: test test-fast train baseline plot clean

test:        ## full suite including the ~15 s end-to-end test
	pytest -q

test-fast:   ## suite without the slow end-to-end test
	pytest -q -m "not slow"

train:       ## main result: 100% exact match, ~1 min
	python scripts/train_reverse.py --steps 250 --eval-every 10

baseline:    ## position-free ablation, lands at its ~15% ceiling
	python scripts/baseline_bag.py --steps 300

plot:
	python scripts/plot_curve.py --hist results/history.json --hist results/history_1layer.json --labels "2 layers" --labels "1 layer" --out results/learning_curve.png

clean:
	rm -rf .pytest_cache */__pycache__
