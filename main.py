import sys
sys.path.append("src")

from src.BatvisionV2_Dataset import *
from src.train import *
from src.test import *

if __name__ == "__main__":
    params = yaml.safe_load(open("params.yaml"))

    if params["dataset"]["name"] == "batvisionv2":
        train_ds = load_batvisionv2_dataset(params, version="train")
        val_ds = load_batvisionv2_dataset(params, version="val")
    else:
        raise ValueError("this batvision ds version is not implemented yet")

    history, model = trainer(params, train_ds, val_ds)

    test_ds = load_batvisionv2_dataset(params, version="test")
    test_spec = test_ds.map(lambda x, y: x)
    test_spec = np.array(list(test_spec.as_numpy_iterator()))

    test_depth = test_ds.map(lambda x, y: y)
    test_depth = np.array(list(test_depth.as_numpy_iterator()))

    predictions = predict_depth(test_spec, model)
    test_loss = compute_test_loss(test_depth, predictions, loss="mae")

    print("Test Mean Absolute Error = ", test_loss)






