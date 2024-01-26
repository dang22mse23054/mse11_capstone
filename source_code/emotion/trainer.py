from dataloader import EmotionDataLoader
from model import EmotionDetectionModel
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint, EarlyStopping

if __name__ == "__main__":
	max_epochs = 100
	
	# modelv1 = EmotionDetectionModel(
	# 	lr = 1e-3,
	# 	momentum = 0.9,
	# 	weight_decay = 1e-4,
	# 	max_epochs = max_epochs    
	# )
	
	# modelv2 = EmotionDetectionModel(
	# 	lr = 1e-3,
	# 	momentum = 0.9,
	# 	weight_decay = 1e-4,
	# 	max_epochs = max_epochs    
	# )

	modelv3 = EmotionDetectionModel(
		lr = 5e-4,
		weight_decay = 1e-6,
		max_epochs = max_epochs    
	)

	data = EmotionDataLoader(batch_size=64, workers=4)

	trainer = Trainer(
		# accelerator="cpu",
		accelerator="gpu",
		# checkpoint_callback=True,
		callbacks = [
		    LearningRateMonitor(logging_interval='step'),
            # ModelCheckpoint(filename='{epoch}-{val_loss:.4f}', save_top_k=2, monitor='val_loss', mode='min'),
			
			# version 1
			# EarlyStopping(monitor='val_loss', patience=11, min_delta=0.00005, verbose=True),
			
			# version 2
			EarlyStopping(monitor='val_loss', patience=5, verbose=True),

			# version 3
            ModelCheckpoint(filename='{epoch}-{val_acc:.4f}', save_top_k=2, monitor='val_acc', mode='max'),
			EarlyStopping(monitor='val_acc', patience=7, min_delta=0.0001, verbose=True),
		], 
		# check_val_every_n_epoch=1,
		fast_dev_run=False,
		default_root_dir='checkpoint',
		max_epochs=max_epochs,       
	)

	# trainer.fit(modelv1, data)
	# trainer.fit(modelv2, data)
	trainer.fit(modelv3, data)