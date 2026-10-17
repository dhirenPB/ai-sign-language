# TEST EVALUATION — once, with the BEST validation checkpoint, only after training has finished (selection_done.json must exist).
# Needs 10_setup_training.py. Teacher-forced score on the whole test split; the slow free-running score on a limited number of batches.
import json

FREE_RUNNING_BATCHES = _opt("FREE_RUNNING_BATCHES", 20)
test_results = it.final_test_evaluation(model, test_loader, criterion, RUN_DIR, DEVICE, pad_id, free_running=True,
                                        free_running_max_batches=FREE_RUNNING_BATCHES)
print(json.dumps(test_results, indent=1))
print("saved to", RUN_DIR / "test_results.json")
