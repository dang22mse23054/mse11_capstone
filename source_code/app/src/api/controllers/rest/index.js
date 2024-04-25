const FileController = require('./FileController');
const AdsController = require('./AdsController');

module.exports = {
	fileController: new FileController(),
	adsController: new AdsController(),
};