from dataloader import EmotionDataLoader
from model import EmotionDetectionModel
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint, EarlyStopping

if __name__ == "__main__":
	max_epochs = 100
	model = EmotionDetectionModel(
		lr = 1e-3,
		momentum = 0.9,
		weight_decay = 1e-4,
		max_epochs = max_epochs    
	)

	data = EmotionDataLoader(batch_size=64, workers=4)

	trainer = Trainer(
		# accelerator="cpu",
		accelerator="gpu",
		# checkpoint_callback=True,
		callbacks = [
            ModelCheckpoint(filename='{epoch}-{val_loss:.4f}', save_top_k=2, monitor='val_loss', mode='min'),
			EarlyStopping(monitor='val_loss', patience=11, min_delta=0.00005, verbose=True),
		    LearningRateMonitor(logging_interval='step'),
		], 
		# check_val_every_n_epoch=1,
		fast_dev_run=False,
		default_root_dir='checkpoint',
		max_epochs=max_epochs,       
	)

	trainer.fit(model, data)