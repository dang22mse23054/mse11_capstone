import { IValidatorError } from 'interfaceDir';
import { ICategory } from '.';
// import { IDestination, IDestInfo } from './destination';

export interface IVideoError {
	title?: IValidatorError | string
	processes?: Map<string, IValidatorError> | string
}

export interface IVideo {
	id?: number
	title?: string
	isEnabled?: boolean

	categoryIds?: Array<number>
	category?: Array<ICategory>

	refFileName?: string
	refFilePath?: string

	updatedAt?: string
}

export interface IVideoSettingAO extends IActionObj, IVideo {
	isLoading?: boolean;
	error?: IVideoError;
  }

export interface IVideoSearchOpt {
	keyword?: string
	isEnabled?: boolean
	categories?: Array<number>

	startDate?: any
	endDate?: any
}