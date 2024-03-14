const { check, validationResult } = require('express-validator');
const BaseResponse = require('../dto/BaseResponse');
const { ErrorCodes } = require('commonDir/constants');
const moment = require('moment-timezone');

const ValidatorType = {
	TASK_DOWNLOAD: {
		SEARCH_RESULT: 1,
		OTHER: 2
	},
	SCHED_DOWNLOAD_CSV: 3,
};

const ValidationMiddleware = {

	[ValidatorType.TASK_DOWNLOAD.SEARCH_RESULT]: [

		check('ownerIds')
			// ("", 0, false, null) will also be considered optional
			.optional({ checkFalsy: true })
			.custom((value, { req, loc, path }) => {
				// if not a number
				if (isNaN(value)) {
					// then, it must be array
					// if not
					if (Array.isArray(value)) {
						if (value.length == 0) {
							throw new Error('Must be non-empty array');
						}
						return true;
					}
					throw new Error('Must be integer');
				}
				return true;
			})
			.toArray(),
		check('ownerIds.*')
			// ("", 0, false, null) will also be considered optional
			.optional({ checkFalsy: true })
			.isNumeric().withMessage('Must be integer')
			.toInt(),
		check('keyword')
			.optional({ checkFalsy: true })
			// 	.isLength({ max: COMMON.VALIDATION.SHORT_URL.DES_LENGTH }).withMessage(`Max length ${COMMON.VALIDATION.SHORT_URL.DES_LENGTH} characters`)
			.trim().escape(),

		check('type')
			.optional({ checkFalsy: false })
			.isNumeric().withMessage('Must be digit')
			.trim(),

		check('mode')
			.optional({ checkFalsy: false })
			.isNumeric().withMessage('Must be digit')
			.toInt(),

		check('largeCategories')
			// ("", 0, false, null) will also be considered optional
			.optional({ checkFalsy: true })
			.custom((value, { req, loc, path }) => {
				// if not a number
				if (isNaN(value)) {
					// then, it must be array
					// if not
					if (Array.isArray(value)) {
						if (value.length == 0) {
							throw new Error('Must be non-empty array');
						}
						return true;
					}
					throw new Error('Must be integer');
				}
				return true;
			})
			.toArray(),
		check('largeCategories.*')
			// ("", 0, false, null) will also be considered optional
			.optional({ checkFalsy: true })
			.isNumeric().withMessage('Must be integer')
			.toInt(),
		check('smallCategories')
			// ("", 0, false, null) will also be considered optional
			.optional({ checkFalsy: true })
			.custom((value, { req, loc, path }) => {
				// if not a number
				if (isNaN(value)) {
					// then, it must be array
					// if not
					if (Array.isArray(value)) {
						if (value.length == 0) {
							throw new Error('Must be non-empty array');
						}
						return true;
					}
					throw new Error('Must be integer');
				}
				return true;
			})
			.toArray(),
		check('smallCategories.*')
			// ("", 0, false, null) will also be considered optional
			.optional({ checkFalsy: true })
			.isNumeric().withMessage('Must be integer')
			.toInt(),

		check('destType')
			.optional({ checkFalsy: false })
			.isNumeric().withMessage('Must be digit')
			.trim(),

		check('destIds')
			// ("", 0, false, null) will also be considered optional
			.optional({ checkFalsy: true })
			.custom((value, { req, loc, path }) => {
				// it must be int | string | array
				// if is Array => check length
				if (Array.isArray(value)) {
					if (value.length == 0) {
						throw new Error('Must be non-empty array');
					}
					return true;
				}
				return true;
			})
			.toArray(),

		check('media')
			// ("", 0, false, null) will also be considered optional
			.optional({ checkFalsy: true })
			.custom((value, { req, loc, path }) => {
				// if not a number
				if (isNaN(value)) {
					// then, it must be array
					// if not
					if (Array.isArray(value)) {
						if (value.length == 0) {
							throw new Error('Must be non-empty array');
						}
						return true;
					}
					throw new Error('Must be integer');
				}
				return true;
			})
			.toArray(),
		check('media.*')
			// ("", 0, false, null) will also be considered optional
			.optional({ checkFalsy: true })
			.isNumeric().withMessage('Must be integer')
			.toInt(),

		check('procMasters')
			// ("", 0, false, null) will also be considered optional
			.optional({ checkFalsy: true })
			.custom((value, { req, loc, path }) => {
				// if not a number
				if (isNaN(value)) {
					// then, it must be array
					// if not
					if (Array.isArray(value)) {
						if (value.length == 0) {
							throw new Error('Must be non-empty array');
						}
						return true;
					}
					throw new Error('Must be integer');
				}
				return true;
			})
			.toArray(),
		check('procMasters.*')
			// ("", 0, false, null) will also be considered optional
			.optional({ checkFalsy: true })
			.isNumeric().withMessage('Must be integer')
			.toInt(),

		// // TODO: startDateとendDateは本来intでparamとして来るようにvalidationしていたが、今はstrで来ているためvalidationを一旦comment outしている
		// check('startDate')
		// 	.exists({ checkFalsy: true }).withMessage('Required field')
		// 	.isISO8601().withMessage('Must be datetime format'),
		// check('endDate')
		// 	.exists({ checkFalsy: true }).withMessage('Required field')
		// 	.isISO8601().withMessage('Must be datetime format'),

		// check('startDate')
		// .optional({ checkFalsy: true })
		// // 	.isLength({ max: COMMON.VALIDATION.SHORT_URL.DES_LENGTH }).withMessage(`Max length ${COMMON.VALIDATION.SHORT_URL.DES_LENGTH} characters`)
		// .trim().escape(),
		// check('endDate')
		// .optional({ checkFalsy: true })
		// // 	.isLength({ max: COMMON.VALIDATION.SHORT_URL.DES_LENGTH }).withMessage(`Max length ${COMMON.VALIDATION.SHORT_URL.DES_LENGTH} characters`)
		// .trim().escape(),

		check('status')
			.optional({ checkFalsy: false })
			.isNumeric().withMessage('Must be digit')
			.trim()
			// .toArray(),

	],
	[ValidatorType.TASK_DOWNLOAD.OTHER]: [
		check('mode')
			.optional({ checkFalsy: false })
			.isNumeric().withMessage('Must be digit')
			.toInt(),
		check('status')
			// ("", 0, false, null) will also be considered optional
			.exists({ checkFalsy: true })
			.custom((value, { req, loc, path }) => {
				// if not a number
				if (isNaN(value)) {
					// then, it must be array
					// if not
					if (Array.isArray(value)) {
						if (value.length == 0) {
							throw new Error('Must be non-empty array');
						}
						return true;
					}
					throw new Error('Must be integer');
				}
				return true;
			})
			.toArray(),
		check('status.*')
			// ("", 0, false, null) will also be considered optional
			.exists({ checkFalsy: true })
			.isNumeric().withMessage('Must be integer')
			.toInt(),
		check('startDate')
			.exists({ checkFalsy: true })
			.isNumeric().withMessage('Must be timestamp format')
			// convert to Moment
			.toInt()
			.customSanitizer((value, { req, location, path }) => moment(value)),
		check('endDate')
			.exists({ checkFalsy: true })
			.isNumeric().withMessage('Must be timestamp format')
			// convert to Moment
			.toInt()
			.customSanitizer((value, { req, location, path }) => moment(value)),
	],
	[ValidatorType.SCHED_DOWNLOAD_CSV]: [],

	result: (req, res, next) => {
		let error = validationResult(req);
		if (!error.isEmpty()) {
			let errItem = error.array({ onlyFirstError: true })[0];
			let respObj = new BaseResponse();

			respObj.setError(ErrorCodes.INVALID_REQUEST, 'Invalid request');
			respObj.setData({
				field: errItem.param,
				errMsg: errItem.msg
			});

			return res.status(400).json(respObj);
		}
		next();
	},
};

const Validator = {
	validate: (type) => {
		let result = ValidationMiddleware[type];
		result.push(ValidationMiddleware.result);
		return result;
	}
};

module.exports.RequestValidator = Validator;
module.exports.ValidationType = ValidatorType;