import hydra
from omegaconf import DictConfig, OmegaConf
from recommender.utils.module_loader import load_module

@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    DataLoaderClass = load_module(cfg.data.module)
    data_loader = DataLoaderClass(**cfg.data.params)
    train_data, val_data, test_data = data_loader.get_train_val_test_data(cfg.data.options.force)
    print("Loaded data")

    ModelClass = load_module(cfg.model.module)
    model = ModelClass(**cfg.model.params)
    print("Loaded model")

    LossClass = load_module(cfg.loss.module)
    loss_fn = LossClass(**cfg.loss.params)
    print("Loaded loss function:", cfg.loss.name)

    TrainerClass = load_module(cfg.trainer.module)
    trainer = TrainerClass(**cfg.trainer.params)
    print("Loaded trainer")

    print("Training model...")
    trainer.fit(model, train_data, val_data, loss_fn=loss_fn)
    print("Training completed")

if __name__ == "__main__":
    main()