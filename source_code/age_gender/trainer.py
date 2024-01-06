from dataloader import AgeGenderDataLoader
from model import AgeGenderDetectionModel
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

if __name__ == "__main__":
	model = AgeGenderDetectionModel(
		lr = 1e-4,
		momentum = 0.9,
		weight_decay = 1e-4,        
	)#.load_from_checkpoint('checkpoint/epoch=9-step=3940.ckpt')

	data = AgeGenderDataLoader(batch_size=64, workers=5)

	trainer = Trainer(
		accelerator="cpu",
		# accelerator="mps",
		# checkpoint_callback=True,
		callbacks = [
		    LearningRateMonitor(logging_interval='step'),
		    # ModelCheckpoint(dirpath='', filename='{epoch}-{val_acc:.4f}', save_top_k=5, monitor='val_acc', mode='max'),
			ModelCheckpoint(filename='{epoch}-{val_acc:.4f}', save_top_k=5, monitor='val_acc', mode='max'),
		], 
		# check_val_every_n_epoch=1,
		# fast_dev_run=True,
		default_root_dir='checkpoint',
		max_epochs=1,       
	)

	# trainer.test(model, data)
	trainer.fit(model, data)
		