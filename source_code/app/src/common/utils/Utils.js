const url = require('url');
const crypto = require('crypto');
const moment = require('moment-timezone');
const { Common } = require('../constants');
const xlsx = require('xlsx');

module.exports = {
	removeNullData: (obj, excludeKeys = []) => {
		let newObj = { ...obj };
		Object.keys(newObj).forEach((key) => {
			if (!excludeKeys.includes(key)) {
				return (newObj[key] == null) && delete newObj[key];
			}
			return false;
		});
		return newObj;
	},

	removeKeys: (obj, keys = []) => {
		let newObj = { ...obj };
		Object.keys(newObj).forEach((key) => {
			if (keys.includes(key)) {
				return delete newObj[key];
			}
			return false;
		});
		return newObj;
	},

	getCurrentUrl: (req) => {
		return url.format({
			protocol: req.protocol,
			host: req.get('host'),
			pathname: req.originalUrl
		});
	},

	isNotEmpty: (value) => {
		if (typeof value == typeof true) {
			return true;
		}

		return value ? true : false;
	},

	strNormalize: (str, options = { removeSpace: false, toLowerCase: false, trim: true, toFullWidth: true, replaceSpaceWith: ' ', removeSpecialChar: false }) => {
		if (str) {
			str = str.replace('\u3000', ' ');
			if (options.toFullWidth != false) { str = str.normalize('NFKC'); }
			if (options.removeSpace == true) { str = str.replace(/\s{1,}/g, ''); }
			if (options.replaceSpaceWith) { str = str.replace(/\s{1,}/g, options.replaceSpaceWith); }
			if (options.removeSpecialChar != false) { str = str.replace(/[&/\\#,+()$~%.。・〜'":*?<>{}]/g, ''); }
			if (options.toLowerCase != false) { str = str.toLowerCase(); }
			if (options.trim != false) { str = str.trim(); }
		}
		return str;
	},

	md5: (str) => crypto.createHash('md5').update(str).digest('hex'),

	cnvDateToNum: function (obj) {
		let newObj = { ...obj };
		if (obj instanceof Array) {
			newObj = [...obj];
		}
		Object.keys(newObj).forEach((key) => {
			if (newObj[key] instanceof Date) {
				newObj[key] = newObj[key].getTime();
			} else if (newObj[key] instanceof Object) {
				newObj[key] = this.cnvDateToNum(newObj[key]);
			}
		});
		return newObj;
	},

	convertDispTime(time, timeZone = Common.TIME_ZONE, format = 'YYYY/MM/DD HH:mm:ss') {
		return moment(time).tz(timeZone).format(format);
	},

	isEqualArray: (oldArray, newArray) => {
		const isOldArray = Array.isArray(oldArray);
		const isNewArray = Array.isArray(newArray);
		// if all old and new is Array type
		if (isOldArray && isNewArray) {
			const mergedSet = new Set([...oldArray, ...newArray]);

			if (mergedSet.size > 0) {
				if (mergedSet.size == oldArray.length && mergedSet.size == newArray.length) {
					return true;
				}
				return false;
			}
		} else if (isOldArray || isNewArray) {
			// if old OR new is Array type
			return false;
		}

		// eg: undefined or old=new=[]
		return true;
	},

	/**
	 * 
	 * @param filePath 
	 * @returns boolean: { true: has password, false: none password or not excel } 
	 */
	checkPassExcel: (filePath) => {
		const xlsErrorMsg = 'Encryption Flags/AlgID mismatch';
		const xlsxErrorMsg = 'File is password-protected';
		try {
			xlsx.readFile(filePath);
		} catch (err) {
			if ([xlsErrorMsg, xlsxErrorMsg].includes(err.message)) {
				return true;
			}
		}
		return false;
	},
};