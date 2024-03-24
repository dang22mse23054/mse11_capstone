import { IValidatorError } from 'interfaceDir';
import { ICategory } from '.';
// import { IDestination, IDestInfo } from './destination';

export interface IVideoError {
	name?: IValidatorError | string
	owner?: IValidatorError | string
	processes?: Map<string, IValidatorError> | string
}

export interface IVideo {
	id?: number
	title?: string
	path?: string
	isEnabled?: boolean

	categoryIds?: Array<number>
	category?: Array<ICategory>

	updatedAt?: string
}

export interface IVideoSearchOpt {
	keyword?: string
	isEnabled?: boolean
	categories?: Array<number>

	startDate?: any
	endDate?: any
}