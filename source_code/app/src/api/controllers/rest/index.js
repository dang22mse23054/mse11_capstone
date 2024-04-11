const FileController = require('./FileController');
const AdsAdviceController = require('./AdsAdviceController');

module.exports = {
	fileController: new FileController(),
	adsAdviceController: new AdsAdviceController(),
};