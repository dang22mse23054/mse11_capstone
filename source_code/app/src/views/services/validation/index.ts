import validateVideo from './video';
export interface IValidationErrorMsg {
	message: string
	field: string
	type: string
	expected?: any
	actual?: any
}

export interface IValidationResult {
	isValid: boolean
	/**
	 * If correct then errorDetail is undefined
	 */
	errorDetail?: Array<IValidationErrorMsg>
}

const convertToErrorObj = (validationResult) => {
	const errorObj = {};
	const regex = /^(\D*)\[(\d*)\]/g;
	validationResult.map((item, idx) => {
		const parts: Array<any> = item.field.split('.');

		const length = parts.length;

		if (length > 1) {
			let cursor = errorObj;
			let name: string = null;
			for (let i = 0; i < length; i++) {
				name = parts[i];
				const regMatch = regex.exec(name);
				if (regMatch) {
					const objName = regMatch[1];
					if (!cursor[objName]) {
						errorObj[objName] = {};
					}
					cursor = errorObj[objName];
					name = `${regMatch[1]}_${regMatch[2]}`;
				}
				if (i == length - 1) {
					cursor[name] = item.message;
					continue;
				}
				if (!cursor[name]) {
					cursor[name] = {};
					cursor = cursor[name];
					continue;
				}
			}
		} else {
			errorObj[item.field] = item.message;
		}
	});
	return errorObj;
};

export const isValidVideo = (video) => {
	const result = validateVideo(video);
	return result == true || convertToErrorObj(result);
};