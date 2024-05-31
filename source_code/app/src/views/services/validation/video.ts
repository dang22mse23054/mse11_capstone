const Validator = require('fastest-validator');
import { Status, VideoTypeValues, DestinationTypeValues, VideoTypes, TypeNames } from 'constDir';
import { IVideo } from 'interfaceDir';

const validator = new Validator({
	// common error message
	messages: {
		// Register our new error message text
		required: 'Please enter the this field',
		isIncludes: 'The \'{field}\' field be one of these values: {expected}',
		stringMin: 'Please enter the this field'
	}
});

const videoSchema = {
	title: {
		type: 'custom',
		check(value, schema, typeName, parent, context, errors = []) {
			if (!value || !value.trim()) {
				errors.push({ type: 'stringMin' });
				return errors;
			}
			if (value.length > 200) {
				errors.push({ type: 'stringMax' });
				return errors;
			}
		},
		messages: {
			stringMin: 'Please enter the title',
			stringMax: 'Please enter the title less than {expected} characters',
		}
	},

	refFileName: { type: 'string', optional: true },
	refFilePath: { type: 'string', optional: true },

	categories: {
		type: 'custom',
		check(value, schema, typeName, { categories, categoryIds }, context, errors = []) {
			if (!(Array.isArray(categoryIds) && categoryIds.length > 0)) {
				errors.push({ type: 'required' });
				return errors;
			}
			// only can return string/object{}/boolean
			// if return array => error obj

			// media is array => cannot return `value`, but {value} OR true
			return { value };
		},
		messages: {
			required: 'Please select at least one category',
		}
	},
};

export default validator.compile(videoSchema);