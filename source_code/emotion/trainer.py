from dataloader import EmotionDataLoader
from model import EmotionDetectionModel
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

if __name__ == "__main__":
	model = EmotionDetectionModel(
		lr = 1e-3,
		momentum = 0.9,
		weight_decay = 1e-4,        
	)

	data = EmotionDataLoader(batch_size=64, workers=4)

	trainer = Trainer(
		# accelerator="cpu",
		accelerator="gpu",
		# checkpoint_callback=True,
		callbacks = [
		    LearningRateMonitor(logging_interval='step'),
            ModelCheckpoint(filename='{epoch}-{val_loss:.4f}', save_top_k=2, monitor='val_loss', mode='min'),
		], 
		# check_val_every_n_epoch=1,
		fast_dev_run=False,
		default_root_dir='checkpoint',
		max_epochs=25,       
	)

	trainer.fit(model, data)