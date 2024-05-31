const wz = require('w-zip');
const sevenZip = require('7zip-min');
const fs = require('fs');
const { Common } = require('commonDir/constants');
const ErrorObject = require('apiDir/dto/ErrorObject');
const { _7Z, ZIP } = Common.FileTypes;
const DEFAULT_PASSWORD = 'YourPassword';
const BAD_PASSWORD_ERR = 'BAD_PASSWORD';

module.exports = {

	isPassProtect: async (filePath, extension = ZIP.ext) => {
		const timestamp = new Date().getTime();
		const TMP_FOLDER = `./tmp/${timestamp}`;
		try {
			switch (extension) {
				case ZIP.ext:
					await wz.mZip.unzip(filePath, TMP_FOLDER, { pw: DEFAULT_PASSWORD });
					break;

				case _7Z.ext:
					await new Promise((resolve, reject) => {
						sevenZip.cmd(['e', filePath, `-o${TMP_FOLDER}`, `-p${DEFAULT_PASSWORD}`], (err) => {
							// done
							if (err) {
								if (err.message.includes('ERROR: Data Error in encrypted file. Wrong password')) {
									reject(new Error(BAD_PASSWORD_ERR));

								} else {
									console.error(err);
									reject(err);
								}
							}
							resolve(true);
						});
					});
					break;

				default:
					throw new ErrorObject('Wrong ZIP file Format');

			}
		} catch (ex) {
			if (ex.message == BAD_PASSWORD_ERR) {
				return true;
			}
			throw ex;

		} finally {
			fs.rmSync(TMP_FOLDER, { recursive: true });
		}
		return false;
	},
};